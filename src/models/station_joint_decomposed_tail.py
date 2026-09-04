"""Joint spatio-temporal decomposed tail components for Station-24.

This module is deliberately isolated from the Raw ResUNet.  It consumes the
frozen decoder representation and causal power-derived conditions, and returns
an additive epsilon correction.  Setting the route to zero, or keeping the
zero-initialized correction heads at initialization, is therefore an exact
identity with respect to the Raw body path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, requested: int) -> int:
    for value in range(min(channels, requested), 0, -1):
        if channels % value == 0:
            return value
    return 1


def same_length_average(value: torch.Tensor, width: int) -> torch.Tensor:
    """Reflection-padded moving average that preserves the final dimension."""

    if value.ndim < 2:
        raise ValueError("low-pass input must have a time dimension")
    width = int(width)
    length = int(value.shape[-1])
    if not 1 <= width < length:
        raise ValueError("low-pass width must be in [1, sequence_length)")
    left = (width - 1) // 2
    right = width - 1 - left
    flattened = value.reshape(-1, 1, length)
    padded = F.pad(flattened, (left, right), mode="reflect")
    filtered = F.avg_pool1d(padded, kernel_size=width, stride=1)
    return filtered.reshape_as(value)


class ComplementaryTemporalProjection(nn.Module):
    """One canonical low/high boundary with exact algebraic reconstruction."""

    def __init__(self, width: int = 12) -> None:
        super().__init__()
        self.width = int(width)

    def low(self, value: torch.Tensor) -> torch.Tensor:
        return same_length_average(value, self.width)

    def split(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.low(value)
        return low, value - low


def _lag_difference(value: torch.Tensor, lag: int) -> torch.Tensor:
    result = torch.zeros_like(value)
    result[..., lag:] = value[..., lag:] - value[..., :-lag]
    return result


def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    adjacency = adjacency.float()
    degree = adjacency.sum(dim=-1).clamp(min=1e-8)
    inverse = degree.rsqrt()
    return inverse[:, None] * adjacency * inverse[None, :]


@dataclass
class JSTDOutput:
    correction: torch.Tensor
    slow_correction: torch.Tensor
    fast_correction: torch.Tensor
    slow_mask: torch.Tensor
    fast_mask: torch.Tensor
    slow_mask_logit: torch.Tensor
    fast_mask_logit: torch.Tensor
    issue_logit: torch.Tensor


class JointSpatioTemporalDecomposedTail(nn.Module):
    """One joint wind/solar tail with localized slow and fast corrections.

    The three condition groups are intentionally compact:
    forecast geometry, recent observed error state, and fixed-graph/system
    aggregates derived from those two sources.  Forecast revision is excluded
    from V1 so it remains an identifiable later ablation.
    """

    def __init__(
        self,
        hidden_channels: int,
        station_features: torch.Tensor,
        adjacency: torch.Tensor,
        station_capacities: torch.Tensor,
        secondary_adjacency: torch.Tensor | None = None,
        config: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        config = dict(config or {})
        self.station_count = int(station_features.shape[0])
        self.sequence_length = int(config.get("sequence_length", 168))
        channels = int(config.get("jstd_channels", 24))
        modes = int(config.get("jstd_system_modes", 4))
        groups = int(config.get("group_norm_groups", 8))
        self.mask_prior = float(config.get("jstd_mask_prior", 0.12))
        self.secondary_mix = float(config.get("jstd_secondary_graph_mix", 0.20))
        self.use_event_hypothesis = bool(
            config.get("use_jstd_event_hypothesis", False)
        )
        self.hypothesis_edge_temperature = float(
            config.get("jstd_hypothesis_edge_temperature_hours", 1.5)
        )
        if channels <= 0 or modes <= 0:
            raise ValueError("JSTD channels and system modes must be positive")
        if not 0.0 < self.mask_prior < 0.5:
            raise ValueError("jstd_mask_prior must be in (0,0.5)")
        if not 0.0 <= self.secondary_mix <= 1.0:
            raise ValueError("jstd_secondary_graph_mix must be in [0,1]")
        if self.hypothesis_edge_temperature <= 0.0:
            raise ValueError(
                "jstd_hypothesis_edge_temperature_hours must be positive"
            )

        primary = _normalize_adjacency(adjacency)
        if secondary_adjacency is None:
            graph = primary
        else:
            secondary = _normalize_adjacency(secondary_adjacency)
            graph = (1.0 - self.secondary_mix) * primary + self.secondary_mix * secondary
        self.register_buffer("fixed_graph", graph, persistent=True)
        self.register_buffer("station_features", station_features.float(), persistent=True)
        capacity = station_capacities.float().clamp(min=1e-8)
        wind = station_features[:, 0].float()
        solar = station_features[:, 1].float()
        wind_capacity = capacity * wind
        solar_capacity = capacity * solar
        self.register_buffer(
            "wind_weight", wind_capacity / wind_capacity.sum().clamp(min=1e-8),
            persistent=True,
        )
        self.register_buffer(
            "solar_weight", solar_capacity / solar_capacity.sum().clamp(min=1e-8),
            persistent=True,
        )
        self.project12 = ComplementaryTemporalProjection(12)
        self.project24 = ComplementaryTemporalProjection(24)

        self.fast_condition = nn.Sequential(
            nn.Conv1d(5, channels, kernel_size=5, padding=2),
            nn.GroupNorm(_groups(channels, groups), channels),
            nn.SiLU(),
        )
        self.slow_condition = nn.Sequential(
            nn.Conv1d(6, channels, kernel_size=5, padding=2),
            nn.GroupNorm(_groups(channels, groups), channels),
            nn.SiLU(),
        )
        self.hidden_projection = nn.Sequential(
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
            nn.GroupNorm(_groups(channels, groups), channels),
            nn.SiLU(),
        )
        self.fast_fusion = self._fusion(channels, groups)
        self.slow_fusion = self._fusion(channels, groups)
        self.hypothesis_fast_encoder: nn.Module | None = None
        self.hypothesis_slow_encoder: nn.Module | None = None
        if self.use_event_hypothesis:
            self.hypothesis_fast_encoder = self._hypothesis_encoder(
                channels, groups
            )
            self.hypothesis_slow_encoder = self._hypothesis_encoder(
                channels, groups
            )
        self.fast_raw = self._zero_head(channels)
        self.slow_raw = self._zero_head(channels)
        self.fast_mask = self._mask_head(channels, self.mask_prior)
        self.slow_mask = self._mask_head(channels, self.mask_prior)

        self.system_encoder = nn.Sequential(
            nn.Conv1d(8, channels, kernel_size=5, padding=2),
            nn.GroupNorm(_groups(channels, groups), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.slow_modes = nn.Conv1d(channels, modes, kernel_size=1)
        self.fast_modes = nn.Conv1d(channels, modes, kernel_size=1)
        nn.init.zeros_(self.slow_modes.weight)
        nn.init.zeros_(self.slow_modes.bias)
        nn.init.zeros_(self.fast_modes.weight)
        nn.init.zeros_(self.fast_modes.bias)
        self.station_loading = nn.Linear(int(station_features.shape[1]), modes, bias=True)
        nn.init.normal_(self.station_loading.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.station_loading.bias)

        self.issue_head = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        nn.init.zeros_(self.issue_head[-1].weight)
        nn.init.constant_(
            self.issue_head[-1].bias,
            math.log(self.mask_prior / (1.0 - self.mask_prior)),
        )

    @staticmethod
    def _fusion(channels: int, groups: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(2 * channels, channels, kernel_size=5, padding=2),
            nn.GroupNorm(_groups(channels, groups), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(_groups(channels, groups), channels),
            nn.SiLU(),
        )

    @staticmethod
    def _zero_head(channels: int) -> nn.Conv1d:
        head = nn.Conv1d(channels, 1, kernel_size=1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return head

    @staticmethod
    def _hypothesis_encoder(channels: int, groups: int) -> nn.Sequential:
        encoder = nn.Sequential(
            nn.Conv1d(5, channels, kernel_size=5, padding=2),
            nn.GroupNorm(_groups(channels, groups), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        # Loading a V1 checkpoint with the new path enabled remains an exact
        # functional identity before H1 training starts.
        nn.init.zeros_(encoder[-1].weight)
        nn.init.zeros_(encoder[-1].bias)
        return encoder

    @staticmethod
    def _mask_head(channels: int, prior: float) -> nn.Conv1d:
        head = nn.Conv1d(channels, 1, kernel_size=1)
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, math.log(prior / (1.0 - prior)))
        return head

    def _causal_condition_groups(
        self,
        forecast: torch.Tensor,
        recent_error: torch.Tensor | None,
        recent_error_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, stations, length = forecast.shape
        delta1 = _lag_difference(forecast, 1)
        delta3 = _lag_difference(forecast, 3)
        delta6 = _lag_difference(forecast, 6)
        curvature = _lag_difference(delta1, 1)
        neighbor_delta3 = torch.einsum("ij,bjt->bit", self.fixed_graph, delta3)
        fast = torch.stack(
            [delta1, delta3, delta6, curvature, neighbor_delta3], dim=2
        )

        low12 = self.project12.low(forecast)
        low24 = self.project24.low(forecast)
        slow_slope = _lag_difference(low12, 12)
        recent6 = torch.zeros(batch, stations, device=forecast.device, dtype=forecast.dtype)
        recent24 = torch.zeros_like(recent6)
        if recent_error is not None:
            if recent_error.ndim != 3 or recent_error.shape[:2] != (batch, stations):
                raise ValueError("recent_error must be [B,S,H]")
            available = torch.ones_like(recent24)
            if recent_error_mask is not None:
                if recent_error_mask.shape != (batch, stations, 1):
                    raise ValueError("recent_error_mask must be [B,S,1]")
                available = recent_error_mask[..., 0].to(forecast.dtype)
            width6 = min(6, recent_error.shape[-1])
            recent6 = recent_error[..., -width6:].mean(dim=-1) * available
            recent24 = recent_error.mean(dim=-1) * available
        neighbor_recent = torch.einsum("ij,bj->bi", self.fixed_graph, recent24)
        slow = torch.stack(
            [
                low12,
                low24,
                slow_slope,
                recent6[..., None].expand(-1, -1, length),
                recent24[..., None].expand(-1, -1, length),
                neighbor_recent[..., None].expand(-1, -1, length),
            ],
            dim=2,
        )

        wind = torch.einsum("s,bst->bt", self.wind_weight, forecast)
        solar = torch.einsum("s,bst->bt", self.solar_weight, forecast)
        wind_recent = torch.einsum("s,bs->b", self.wind_weight, recent24)
        solar_recent = torch.einsum("s,bs->b", self.solar_weight, recent24)
        system = torch.stack(
            [
                wind,
                solar,
                self.project12.low(wind),
                self.project12.low(solar),
                _lag_difference(wind, 3),
                _lag_difference(solar, 3),
                wind_recent[:, None].expand(-1, length),
                solar_recent[:, None].expand(-1, length),
            ],
            dim=1,
        )
        return fast, slow, system

    def event_hypothesis_fields(
        self,
        hypothesis: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand compact event attributes into smooth station-time fields.

        ``hypothesis`` is [active, onset_fraction, duration_fraction,
        signed_wind_depth, signed_solar_depth, source_synchrony].  This is an
        H1 controllability input, not a future-residual map.
        """

        if hypothesis.ndim != 2 or hypothesis.shape[1] != 6:
            raise ValueError("jstd_event_hypothesis must be [B,6]")
        value = hypothesis.to(dtype=dtype)
        active = value[:, 0].clamp(0.0, 1.0)
        onset = value[:, 1].clamp(0.0, 1.0) * float(self.sequence_length - 1)
        duration = value[:, 2].clamp(
            1.0 / float(self.sequence_length), 1.0
        ) * float(self.sequence_length)
        stop = (onset + duration).clamp(max=float(self.sequence_length))
        time = torch.arange(
            self.sequence_length, device=value.device, dtype=dtype
        )[None, :]
        temperature = self.hypothesis_edge_temperature
        envelope = (
            torch.sigmoid((time - onset[:, None]) / temperature)
            * torch.sigmoid((stop[:, None] - time) / temperature)
            * active[:, None]
        )
        onset_edge = torch.exp(
            -0.5 * ((time - onset[:, None]) / temperature) ** 2
        ) * active[:, None]
        offset_edge = torch.exp(
            -0.5 * ((time - stop[:, None]) / temperature) ** 2
        ) * active[:, None]
        station_type = self.station_features[:, :2].to(dtype)
        amplitude = (
            value[:, 3, None] * station_type[None, :, 0]
            + value[:, 4, None] * station_type[None, :, 1]
        )
        signed_envelope = amplitude[:, :, None] * envelope[:, None, :]
        synchrony = value[:, 5].clamp(0.0, 1.0)
        common_envelope = envelope[:, None, :].expand(
            -1, self.station_count, -1
        )
        fields = torch.stack(
            [
                common_envelope,
                signed_envelope,
                amplitude[:, :, None] * onset_edge[:, None, :],
                amplitude[:, :, None] * offset_edge[:, None, :],
                synchrony[:, None, None] * common_envelope,
            ],
            dim=2,
        )
        return fields, envelope, torch.stack([onset, stop], dim=1)

    def forward(
        self,
        hidden: torch.Tensor,
        forecast: torch.Tensor,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
        route: torch.Tensor | float | None = None,
        condition_strength: torch.Tensor | None = None,
        event_hypothesis: torch.Tensor | None = None,
    ) -> JSTDOutput:
        if hidden.ndim != 4:
            raise ValueError("hidden must be [B,S,C,L]")
        batch, stations, channels, length = hidden.shape
        if (stations, length) != (self.station_count, self.sequence_length):
            raise ValueError("JSTD hidden station/time dimensions are invalid")
        if forecast.shape != (batch, stations, length):
            raise ValueError("forecast must match JSTD [B,S,L]")
        fast_condition, slow_condition, system_condition = (
            self._causal_condition_groups(forecast, recent_error, recent_error_mask)
        )
        if condition_strength is not None:
            strength = condition_strength.to(forecast.dtype)
            if strength.shape != (batch, 1, 1):
                raise ValueError("condition_strength must be [B,1,1]")
            fast_condition = fast_condition * strength[:, None, :, :]
            slow_condition = slow_condition * strength[:, None, :, :]
            system_condition = system_condition * strength

        flat_hidden = hidden.reshape(batch * stations, channels, length)
        hidden_encoded = self.hidden_projection(flat_hidden)
        fast_encoded = self.fast_condition(
            fast_condition.reshape(batch * stations, 5, length)
        )
        slow_encoded = self.slow_condition(
            slow_condition.reshape(batch * stations, 6, length)
        )
        if self.use_event_hypothesis:
            if event_hypothesis is None:
                raise ValueError(
                    "H1 JSTD tail requires jstd_event_hypothesis"
                )
            if (
                self.hypothesis_fast_encoder is None
                or self.hypothesis_slow_encoder is None
            ):
                raise RuntimeError("H1 hypothesis encoders were not initialized")
            hypothesis_fields, _, _ = self.event_hypothesis_fields(
                event_hypothesis, forecast.dtype
            )
            flat_hypothesis = hypothesis_fields.reshape(
                batch * stations, 5, length
            )
            fast_encoded = fast_encoded + self.hypothesis_fast_encoder(
                flat_hypothesis
            )
            slow_encoded = slow_encoded + self.hypothesis_slow_encoder(
                flat_hypothesis
            )
        fast_feature = self.fast_fusion(torch.cat([hidden_encoded, fast_encoded], dim=1))
        slow_feature = self.slow_fusion(torch.cat([hidden_encoded, slow_encoded], dim=1))

        fast_mask_logit = self.fast_mask(fast_feature).reshape(batch, stations, length)
        slow_mask_logit = self.slow_mask(slow_feature).reshape(batch, stations, length)
        fast_mask = torch.sigmoid(fast_mask_logit)
        slow_mask = torch.sigmoid(slow_mask_logit)
        fast_raw = self.fast_raw(fast_feature).reshape(batch, stations, length)
        slow_raw = self.slow_raw(slow_feature).reshape(batch, stations, length)

        system_feature = self.system_encoder(system_condition)
        loading = torch.tanh(self.station_loading(self.station_features))
        fast_raw = fast_raw + torch.einsum("sk,bkt->bst", loading, self.fast_modes(system_feature))
        slow_raw = slow_raw + torch.einsum("sk,bkt->bst", loading, self.slow_modes(system_feature))

        # Projection follows localization.  Reversing this order lets mask edges
        # leak low-frequency energy back into the fast correction.
        slow_correction = self.project12.low(slow_mask * slow_raw)
        _, fast_correction = self.project12.split(fast_mask * fast_raw)
        pooled = torch.cat(
            [system_feature.mean(dim=-1), system_feature.amax(dim=-1)], dim=1
        )
        issue_logit = self.issue_head(pooled)[:, 0]
        if route is None:
            route_tensor = torch.zeros(
                batch, 1, 1, device=hidden.device, dtype=hidden.dtype
            )
        elif isinstance(route, (int, float)):
            route_tensor = torch.full(
                (batch, 1, 1), float(route), device=hidden.device, dtype=hidden.dtype
            )
        else:
            route_tensor = route.to(device=hidden.device, dtype=hidden.dtype)
            if route_tensor.ndim == 1:
                route_tensor = route_tensor[:, None, None]
        if route_tensor.shape != (batch, 1, 1):
            raise ValueError("JSTD route must be scalar, [B], or [B,1,1]")
        correction = route_tensor * (slow_correction + fast_correction)
        return JSTDOutput(
            correction=correction,
            slow_correction=route_tensor * slow_correction,
            fast_correction=route_tensor * fast_correction,
            slow_mask=slow_mask,
            fast_mask=fast_mask,
            slow_mask_logit=slow_mask_logit,
            fast_mask_logit=fast_mask_logit,
            issue_logit=issue_logit,
        )

    def issue_logits(
        self,
        forecast: torch.Tensor,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Issue-level causal tail eligibility without accessing noisy state."""

        _, _, system_condition = self._causal_condition_groups(
            forecast, recent_error, recent_error_mask
        )
        encoded = self.system_encoder(system_condition)
        pooled = torch.cat([encoded.mean(dim=-1), encoded.amax(dim=-1)], dim=1)
        return self.issue_head(pooled)[:, 0]
