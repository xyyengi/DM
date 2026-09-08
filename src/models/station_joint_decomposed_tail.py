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


@dataclass
class SegmentEventPriorOutput:
    """Distribution parameters for ordered continuous event segments."""

    count_logits: torch.Tensor
    onset_logits: torch.Tensor
    attribute_raw: torch.Tensor


class SegmentEventPrior(nn.Module):
    """Causal distribution over event count, onset, duration, and joint marks.

    Slots are ordered by onset in the training target.  They are not event
    classes or experts: each slot uses the same semantic fields and retains an
    arbitrary integer-hour duration through a continuous conditional density.
    """

    def __init__(
        self,
        channels: int,
        sequence_length: int,
        max_events: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.max_events = int(max_events)
        self.encoder = nn.Sequential(
            nn.Conv1d(channels + 2, channels, kernel_size=5, padding=2),
            nn.GroupNorm(_groups(channels, groups), channels),
            nn.SiLU(),
            nn.Conv1d(
                channels, channels, kernel_size=5, padding=4, dilation=2
            ),
            nn.GroupNorm(_groups(channels, groups), channels),
            nn.SiLU(),
            nn.Conv1d(
                channels, channels, kernel_size=5, padding=8, dilation=4
            ),
            nn.GroupNorm(_groups(channels, groups), channels),
            nn.SiLU(),
        )
        self.count_head = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Linear(channels, self.max_events + 1),
        )
        self.onset_head = nn.Conv1d(channels, self.max_events, kernel_size=1)
        # Per slot/hour: duration, wind depth, and solar depth each use
        # location+log-scale; synchrony uses one logit (7 values total).
        self.attribute_head = nn.Conv1d(
            channels, self.max_events * 7, kernel_size=1
        )
        nn.init.zeros_(self.count_head[-1].weight)
        nn.init.zeros_(self.count_head[-1].bias)
        nn.init.zeros_(self.onset_head.weight)
        nn.init.zeros_(self.onset_head.bias)
        nn.init.zeros_(self.attribute_head.weight)
        nn.init.zeros_(self.attribute_head.bias)

    def forward(self, system_feature: torch.Tensor) -> SegmentEventPriorOutput:
        if system_feature.ndim != 3:
            raise ValueError("segment prior system feature must be [B,C,L]")
        batch, _, length = system_feature.shape
        if length != self.sequence_length:
            raise ValueError("segment prior sequence length mismatch")
        lead = torch.linspace(
            0.0, 1.0, length, device=system_feature.device,
            dtype=system_feature.dtype,
        )
        position = torch.stack(
            [lead, torch.sin(2.0 * math.pi * lead)], dim=0
        )[None].expand(batch, -1, -1)
        encoded = self.encoder(torch.cat([system_feature, position], dim=1))
        pooled = torch.cat(
            [encoded.mean(dim=-1), encoded.amax(dim=-1)], dim=1
        )
        attributes = self.attribute_head(encoded).reshape(
            batch, self.max_events, 7, length
        )
        return SegmentEventPriorOutput(
            count_logits=self.count_head(pooled),
            onset_logits=self.onset_head(encoded),
            attribute_raw=attributes,
        )


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
        self.use_segment_prior = bool(
            config.get("use_jstd_segment_prior", False)
        )
        self.segment_max_events = int(
            config.get("jstd_segment_max_events", 2)
        )
        self.segment_tail_fraction = float(
            config.get("jstd_segment_tail_fraction", 0.10)
        )
        # Missing field deliberately retains historical checkpoint semantics.
        self.segment_scale_parameterization = str(
            config.get("jstd_segment_scale_parameterization", "legacy_clamp")
        )
        if self.segment_scale_parameterization not in (
            "legacy_clamp",
            "bounded_sigmoid",
            "transformed_normal_v2",
        ):
            raise ValueError("unknown JSTD segment scale parameterization")
        self.segment_duration_scale_bounds = tuple(
            float(value)
            for value in config.get(
                "jstd_segment_duration_latent_scale_bounds", [0.05, 1.50]
            )
        )
        self.segment_depth_scale_bounds = tuple(
            float(value)
            for value in config.get(
                "jstd_segment_depth_latent_scale_bounds", [0.02, 0.80]
            )
        )
        self.segment_duration_initial_hours = float(
            config.get("jstd_segment_duration_initial_hours", 7.0)
        )
        self.segment_duration_initial_scale = float(
            config.get("jstd_segment_duration_initial_latent_scale", 0.55)
        )
        self.segment_depth_initial_scale = float(
            config.get("jstd_segment_depth_initial_latent_scale", 0.20)
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
        if not 1 <= self.segment_max_events <= 4:
            raise ValueError("jstd_segment_max_events must be in [1,4]")
        if not 0.0 < self.segment_tail_fraction < 0.5:
            raise ValueError(
                "jstd_segment_tail_fraction must be in (0,0.5)"
            )
        if self.use_event_hypothesis and self.use_segment_prior:
            raise ValueError(
                "oracle event hypotheses and causal segment prior are exclusive"
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
        if self.use_event_hypothesis or self.use_segment_prior:
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
        self.segment_prior: SegmentEventPrior | None = None
        if self.use_segment_prior:
            self.segment_prior = SegmentEventPrior(
                channels,
                self.sequence_length,
                self.segment_max_events,
                groups,
            )
            if self.segment_scale_parameterization == "transformed_normal_v2":
                self._initialize_transformed_segment_prior()
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

        squeeze_event = hypothesis.ndim == 2
        if squeeze_event:
            hypothesis = hypothesis[:, None, :]
        if hypothesis.ndim != 3 or hypothesis.shape[2] != 6:
            raise ValueError("jstd_event_hypothesis must be [B,6] or [B,E,6]")
        value = hypothesis.to(dtype=dtype)
        active = value[:, :, 0].clamp(0.0, 1.0)
        onset = value[:, :, 1].clamp(0.0, 1.0) * float(self.sequence_length - 1)
        duration = value[:, :, 2].clamp(
            1.0 / float(self.sequence_length), 1.0
        ) * float(self.sequence_length)
        stop = (onset + duration).clamp(max=float(self.sequence_length))
        time = torch.arange(
            self.sequence_length, device=value.device, dtype=dtype
        )[None, None, :]
        temperature = self.hypothesis_edge_temperature
        envelope = (
            torch.sigmoid((time - onset[:, :, None]) / temperature)
            * torch.sigmoid((stop[:, :, None] - time) / temperature)
            * active[:, :, None]
        )
        onset_edge = torch.exp(
            -0.5 * ((time - onset[:, :, None]) / temperature) ** 2
        ) * active[:, :, None]
        offset_edge = torch.exp(
            -0.5 * ((time - stop[:, :, None]) / temperature) ** 2
        ) * active[:, :, None]
        station_type = self.station_features[:, :2].to(dtype)
        amplitude = (
            value[:, :, 3, None] * station_type[None, None, :, 0]
            + value[:, :, 4, None] * station_type[None, None, :, 1]
        )
        signed_envelope = (
            amplitude[:, :, :, None] * envelope[:, :, None, :]
        ).sum(dim=1)
        synchrony = value[:, :, 5].clamp(0.0, 1.0)
        common_envelope = envelope.amax(dim=1)[:, None, :].expand(
            -1, self.station_count, -1
        )
        onset_field = (
            amplitude[:, :, :, None] * onset_edge[:, :, None, :]
        ).sum(dim=1)
        offset_field = (
            amplitude[:, :, :, None] * offset_edge[:, :, None, :]
        ).sum(dim=1)
        sync_envelope = (
            synchrony[:, :, None, None]
            * envelope[:, :, None, :]
        ).amax(dim=1).expand(-1, self.station_count, -1)
        fields = torch.stack(
            [
                common_envelope,
                signed_envelope,
                onset_field,
                offset_field,
                sync_envelope,
            ],
            dim=2,
        )
        combined_envelope = envelope.amax(dim=1)
        bounds = torch.stack([onset, stop], dim=-1)
        if squeeze_event:
            bounds = bounds[:, 0]
        return fields, combined_envelope, bounds

    def segment_prior_output(
        self,
        forecast: torch.Tensor,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
    ) -> SegmentEventPriorOutput:
        if not self.use_segment_prior or self.segment_prior is None:
            raise RuntimeError("causal JSTD segment prior is disabled")
        _, _, system_condition = self._causal_condition_groups(
            forecast, recent_error, recent_error_mask
        )
        return self.segment_prior(self.system_encoder(system_condition))

    @staticmethod
    def _inverse_bounded_sigmoid(value: float, bounds: tuple[float, float]) -> float:
        lower, upper = bounds
        if not lower < value < upper:
            raise ValueError("initial scale must lie strictly inside its bounds")
        probability = (value - lower) / (upper - lower)
        return math.log(probability / (1.0 - probability))

    def _initialize_transformed_segment_prior(self) -> None:
        if self.segment_prior is None:
            raise RuntimeError("segment prior is unavailable")
        if not 1.0 <= self.segment_duration_initial_hours < self.sequence_length:
            raise ValueError(
                "jstd_segment_duration_initial_hours must be in [1, sequence_length)"
            )
        duration_fraction = self.segment_duration_initial_hours / float(
            self.sequence_length
        )
        duration_location = math.log(
            duration_fraction / (1.0 - duration_fraction)
        )
        duration_scale = self._inverse_bounded_sigmoid(
            self.segment_duration_initial_scale,
            self.segment_duration_scale_bounds,
        )
        depth_scale = self._inverse_bounded_sigmoid(
            self.segment_depth_initial_scale,
            self.segment_depth_scale_bounds,
        )
        with torch.no_grad():
            bias = self.segment_prior.attribute_head.bias.reshape(
                self.segment_max_events, 7
            )
            bias[:, 0] = duration_location
            bias[:, 1] = duration_scale
            bias[:, 2] = 0.0
            bias[:, 3] = depth_scale
            bias[:, 4] = 0.0
            bias[:, 5] = depth_scale
            bias[:, 6] = 0.0

    def segment_scale(
        self,
        raw_scale: torch.Tensor,
        attribute: str = "legacy",
    ) -> torch.Tensor:
        if self.segment_scale_parameterization == "transformed_normal_v2":
            if attribute == "duration":
                lower, upper = self.segment_duration_scale_bounds
            elif attribute == "depth":
                lower, upper = self.segment_depth_scale_bounds
            else:
                raise ValueError(
                    "transformed_normal_v2 scale requires duration or depth"
                )
            if not 0.0 < lower < upper:
                raise ValueError("JSTD transformed scale bounds are invalid")
            return lower + (upper - lower) * torch.sigmoid(raw_scale)
        if self.segment_scale_parameterization == "bounded_sigmoid":
            return 0.02 + 0.33 * torch.sigmoid(raw_scale)
        return F.softplus(raw_scale).clamp(min=0.02, max=0.35)

    def _normal_nll(
        self,
        target: torch.Tensor,
        location: torch.Tensor,
        raw_scale: torch.Tensor,
        attribute: str = "legacy",
    ) -> torch.Tensor:
        scale = self.segment_scale(raw_scale, attribute)
        return 0.5 * ((target - location) / scale).square() + torch.log(scale)

    def _transformed_normal_nll(
        self,
        target: torch.Tensor,
        location: torch.Tensor,
        raw_scale: torch.Tensor,
        attribute: str,
    ) -> torch.Tensor:
        if attribute == "duration":
            epsilon = 0.5 / float(self.sequence_length)
            bounded = target.clamp(epsilon, 1.0 - epsilon)
            latent = torch.logit(bounded)
            log_inverse_jacobian = torch.log(bounded) + torch.log1p(-bounded)
        elif attribute == "depth":
            bounded = target.clamp(-1.0 + 1.0e-4, 1.0 - 1.0e-4)
            latent = torch.atanh(bounded)
            log_inverse_jacobian = torch.log1p(-bounded.square())
        else:
            raise ValueError("transformed normal attribute must be duration or depth")
        return (
            self._normal_nll(latent, location, raw_scale, attribute)
            + log_inverse_jacobian
        )

    def segment_prior_loss(
        self,
        forecast: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Proper structured loss using train/validation labels, never conditions."""

        if target.ndim != 3 or target.shape[1:] != (
            self.segment_max_events, 6
        ):
            raise ValueError("segment prior target must be [B,E,6]")
        output = self.segment_prior_output(
            forecast, recent_error, recent_error_mask
        )
        active = target[:, :, 0].to(forecast.dtype)
        count = active.sum(dim=1).long().clamp(max=self.segment_max_events)
        importance = sample_weight.to(forecast.dtype).reciprocal()
        count_error = F.cross_entropy(
            output.count_logits, count, reduction="none"
        )
        count_loss = (count_error * importance).sum() / importance.sum().clamp(min=1.0)

        onset_index = (
            target[:, :, 1].clamp(0.0, 1.0)
            * float(self.sequence_length - 1)
        ).round().long()
        onset_error = F.cross_entropy(
            output.onset_logits.reshape(-1, self.sequence_length),
            onset_index.reshape(-1),
            reduction="none",
        ).reshape_as(active)
        slot_weight = active * importance[:, None]
        onset_loss = (onset_error * slot_weight).sum() / slot_weight.sum().clamp(min=1.0)

        gather_index = onset_index[:, :, None, None].expand(-1, -1, 7, 1)
        attributes = output.attribute_raw.gather(-1, gather_index)[..., 0]
        if self.segment_scale_parameterization == "transformed_normal_v2":
            duration_error = self._transformed_normal_nll(
                target[:, :, 2],
                attributes[:, :, 0],
                attributes[:, :, 1],
                "duration",
            )
            wind_error = self._transformed_normal_nll(
                target[:, :, 3],
                attributes[:, :, 2],
                attributes[:, :, 3],
                "depth",
            )
            solar_error = self._transformed_normal_nll(
                target[:, :, 4],
                attributes[:, :, 4],
                attributes[:, :, 5],
                "depth",
            )
        else:
            duration_location = torch.sigmoid(attributes[:, :, 0])
            wind_location = torch.tanh(attributes[:, :, 2])
            solar_location = torch.tanh(attributes[:, :, 4])
            duration_error = self._normal_nll(
                target[:, :, 2], duration_location, attributes[:, :, 1]
            )
            wind_error = self._normal_nll(
                target[:, :, 3], wind_location, attributes[:, :, 3]
            )
            solar_error = self._normal_nll(
                target[:, :, 4], solar_location, attributes[:, :, 5]
            )
        sync_error = F.binary_cross_entropy_with_logits(
            attributes[:, :, 6], target[:, :, 5].clamp(0.0, 1.0),
            reduction="none",
        )
        mark_error = duration_error + 0.5 * (wind_error + solar_error) + sync_error
        mark_loss = (mark_error * slot_weight).sum() / slot_weight.sum().clamp(min=1.0)
        total = count_loss + onset_loss + mark_loss
        return total, {
            "count": count_loss.detach(),
            "onset": onset_loss.detach(),
            "mark": mark_loss.detach(),
        }

    def sample_segment_hypotheses(
        self,
        forecast: torch.Tensor,
        members: int,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample member-specific causal segment hypotheses."""

        output = self.segment_prior_output(
            forecast, recent_error, recent_error_mask
        )
        batch = forecast.shape[0]
        count_probability = torch.softmax(output.count_logits, dim=-1)
        learned_event_probability = 1.0 - count_probability[:, 0]
        nonzero_probability = count_probability[:, 1:]
        nonzero_probability = nonzero_probability / nonzero_probability.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        sampled_nonzero_count = 1 + torch.multinomial(
            nonzero_probability, int(members), replacement=True
        )
        tail_count = max(
            1, int(round(int(members) * self.segment_tail_fraction))
        )
        route = torch.zeros(
            batch, int(members), device=forecast.device, dtype=torch.bool
        )
        for batch_index in range(batch):
            order = torch.randperm(int(members), device=forecast.device)
            route[batch_index, order[:tail_count]] = True
        counts = torch.where(
            route, sampled_nonzero_count, torch.zeros_like(sampled_nonzero_count)
        )
        hypotheses = torch.zeros(
            batch, int(members), self.segment_max_events, 6,
            device=forecast.device, dtype=forecast.dtype,
        )
        starts = torch.full(
            (batch, int(members), self.segment_max_events), -1,
            device=forecast.device, dtype=torch.long,
        )
        for slot in range(self.segment_max_events):
            active = counts > slot
            onset_probability = torch.softmax(
                output.onset_logits[:, slot], dim=-1
            )
            sampled_onset = torch.multinomial(
                onset_probability, int(members), replacement=True
            )
            starts[:, :, slot] = torch.where(
                active, sampled_onset, torch.full_like(sampled_onset, -1)
            )
            raw = output.attribute_raw[:, slot]
            gathered = raw.gather(
                -1,
                sampled_onset[:, None, :].expand(-1, 7, -1),
            ).transpose(1, 2)
            def sampled_normal(
                location: torch.Tensor,
                raw_scale: torch.Tensor,
                attribute: str = "legacy",
            ) -> torch.Tensor:
                scale = self.segment_scale(raw_scale, attribute)
                return location + scale * torch.randn_like(location)

            if self.segment_scale_parameterization == "transformed_normal_v2":
                duration = torch.sigmoid(
                    sampled_normal(
                        gathered[:, :, 0], gathered[:, :, 1], "duration"
                    )
                )
                wind = torch.tanh(
                    sampled_normal(
                        gathered[:, :, 2], gathered[:, :, 3], "depth"
                    )
                )
                solar = torch.tanh(
                    sampled_normal(
                        gathered[:, :, 4], gathered[:, :, 5], "depth"
                    )
                )
            else:
                duration = sampled_normal(
                    torch.sigmoid(gathered[:, :, 0]), gathered[:, :, 1]
                ).clamp(1.0 / float(self.sequence_length), 1.0)
                wind = sampled_normal(
                    torch.tanh(gathered[:, :, 2]), gathered[:, :, 3]
                ).clamp(-1.0, 1.0)
                solar = sampled_normal(
                    torch.tanh(gathered[:, :, 4]), gathered[:, :, 5]
                ).clamp(-1.0, 1.0)
            synchrony = torch.sigmoid(gathered[:, :, 6])
            hypotheses[:, :, slot, 0] = active.to(forecast.dtype)
            hypotheses[:, :, slot, 1] = sampled_onset.to(forecast.dtype) / float(
                self.sequence_length - 1
            )
            hypotheses[:, :, slot, 2] = duration
            hypotheses[:, :, slot, 3] = wind
            hypotheses[:, :, slot, 4] = solar
            hypotheses[:, :, slot, 5] = synchrony
        survival = torch.stack(
            [nonzero_probability[:, slot:].sum(dim=1)
             for slot in range(self.segment_max_events)],
            dim=1,
        )
        time_probability = (
            survival[:, :, None]
            * torch.softmax(output.onset_logits, dim=-1)
        ).sum(dim=1)
        time_probability = time_probability / time_probability.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        return hypotheses, {
            "tail_probability": torch.full_like(
                learned_event_probability, self.segment_tail_fraction
            ),
            "learned_event_probability": learned_event_probability,
            "count_probability": count_probability,
            "counts": counts,
            "starts": starts,
            "time_probability": time_probability,
        }

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
        if self.use_event_hypothesis or self.use_segment_prior:
            if event_hypothesis is None:
                raise ValueError(
                    "event-conditioned JSTD tail requires jstd_event_hypothesis"
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
