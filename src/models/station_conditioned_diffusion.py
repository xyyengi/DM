"""Lead-aware conditional ResUNet diffusion for 24 wind/solar stations.

Stations remain an explicit spatial axis.  Temporal convolutions share weights
across stations; optional graph propagation can be sequential or fused through
a lightweight station-and-time-dependent parallel branch.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SPATIAL_MODES = {"none", "fixed_graph", "type_gated_graph"}


def _group_count(channels: int, requested: int) -> int:
    """Choose the largest valid GroupNorm divisor up to ``requested``."""
    for groups in range(min(int(requested), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _sinusoidal_embedding(
    values: torch.Tensor,
    dimension: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    if dimension < 2:
        raise ValueError(f"sinusoidal embedding dimension must be >=2, got {dimension}")
    half = dimension // 2
    exponent = -math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=values.device
    ) / max(half - 1, 1)
    frequencies = torch.exp(exponent)
    angles = values.float().unsqueeze(-1) * frequencies
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class DiffusionTimestepEmbedding(nn.Module):
    """Embed the reverse-diffusion timestep independently of real time."""

    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError(
                f"diffusion timestep must be [B], got {tuple(timestep.shape)}"
            )
        return self.mlp(_sinusoidal_embedding(timestep, self.embedding_dim))


def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    adjacency = adjacency.float()
    degree = adjacency.sum(dim=1).clamp(min=1e-8)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


def _normalize_batched_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    if adjacency.ndim != 3 or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError(
            f"batched adjacency must be [B,S,S], got {tuple(adjacency.shape)}"
        )
    degree = adjacency.sum(dim=-1).clamp(min=1e-8)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt[:, :, None] * adjacency * inverse_sqrt[:, None, :]


def _flatten_stations(value: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if value.ndim != 4:
        raise ValueError(f"expected [B,S,C,L], got {tuple(value.shape)}")
    batch, stations, channels, length = value.shape
    return value.reshape(batch * stations, channels, length), batch, stations


def _restore_stations(value: torch.Tensor, batch: int, stations: int) -> torch.Tensor:
    return value.reshape(batch, stations, value.shape[1], value.shape[2])


class StationConditionEncoder(nn.Module):
    """Encode forecast plus optional ramp, revision, and recent-error context."""

    def __init__(
        self,
        channels: Sequence[int],
        station_count: int,
        station_feature_dim: int,
        groups: int,
        condition_config: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        condition_config = dict(condition_config or {})
        channels = tuple(int(value) for value in channels)
        stem_channels = channels[0]
        self.station_count = int(station_count)
        self.use_forecast_ramps = bool(
            condition_config.get("use_forecast_ramps", False)
        )
        self.use_forecast_revision = bool(
            condition_config.get("use_forecast_revision", False)
        )
        self.use_recent_error = bool(
            condition_config.get("use_recent_error", False)
        )
        self.forecast_ramp_lags = tuple(
            int(value)
            for value in condition_config.get("forecast_ramp_lags", [1, 3, 6])
        )
        self.recent_error_hours = int(
            condition_config.get("recent_error_hours", 24)
        )
        self.forecast_stem = nn.Conv1d(1, stem_channels, kernel_size=3, padding=1)
        self.temporal_stem = nn.Conv1d(10, stem_channels, kernel_size=3, padding=1)
        self.station_feature_projection = nn.Linear(station_feature_dim, stem_channels)
        self.station_embedding = nn.Embedding(self.station_count, stem_channels)
        self.fuse = nn.Sequential(
            nn.Conv1d(stem_channels * 3, stem_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(stem_channels, groups), stem_channels),
            nn.SiLU(),
        )
        self.extra_stems = nn.ModuleDict()
        self.extra_norms = nn.ModuleDict()
        self.condition_gates = nn.ParameterDict()
        gate_init = float(condition_config.get("condition_gate_init", -1.0))
        if self.use_forecast_ramps:
            self.extra_stems["ramp"] = nn.Conv1d(
                len(self.forecast_ramp_lags), stem_channels, kernel_size=3, padding=1
            )
            self.extra_norms["ramp"] = nn.GroupNorm(
                _group_count(stem_channels, groups), stem_channels
            )
            self.condition_gates["ramp"] = nn.Parameter(torch.tensor(gate_init))
        if self.use_forecast_revision:
            self.extra_stems["revision"] = nn.Conv1d(
                2, stem_channels, kernel_size=3, padding=1
            )
            self.extra_norms["revision"] = nn.GroupNorm(
                _group_count(stem_channels, groups), stem_channels
            )
            self.condition_gates["revision"] = nn.Parameter(torch.tensor(gate_init))
        if self.use_recent_error:
            self.extra_stems["recent_error"] = nn.Sequential(
                nn.Conv1d(1, stem_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(stem_channels, stem_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.extra_norms["recent_error"] = nn.GroupNorm(
                _group_count(stem_channels, groups), stem_channels
            )
            self.condition_gates["recent_error"] = nn.Parameter(
                torch.tensor(gate_init)
            )
        self.down_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(_group_count(out_channels, groups), out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                )
            )

    def forward(
        self,
        forecast: torch.Tensor,
        calendar: torch.Tensor,
        lead: torch.Tensor,
        station_features: torch.Tensor,
        forecast_ramps: torch.Tensor | None = None,
        forecast_revision: torch.Tensor | None = None,
        revision_mask: torch.Tensor | None = None,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
        forecast_condition_strength: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if forecast.ndim != 3:
            raise ValueError(f"forecast must be [B,S,L], got {tuple(forecast.shape)}")
        batch, stations, length = forecast.shape
        if stations != self.station_count:
            raise ValueError(f"expected {self.station_count} stations, got {stations}")
        if calendar.shape != (batch, 8, length):
            raise ValueError(f"calendar must be [B,8,L], got {tuple(calendar.shape)}")
        if lead.shape != (batch, 2, length):
            raise ValueError(f"lead must be [B,2,L], got {tuple(lead.shape)}")
        if station_features.shape[0] != stations:
            raise ValueError("station feature order/count does not match forecast")

        if forecast_condition_strength is None:
            forecast_condition_strength = torch.ones(
                batch, 1, 1, device=forecast.device, dtype=forecast.dtype
            )
        if forecast_condition_strength.shape != (batch, 1, 1):
            raise ValueError(
                "forecast_condition_strength must be [B,1,1], got "
                f"{tuple(forecast_condition_strength.shape)}"
            )
        local_strength = forecast_condition_strength[:, None, :, :].expand(
            -1, stations, -1, -1
        ).reshape(batch * stations, 1, 1)

        local = self.forecast_stem(forecast.reshape(batch * stations, 1, length))
        local = local * local_strength
        temporal = self.temporal_stem(torch.cat([calendar, lead], dim=1))
        temporal = temporal[:, None, :, :].expand(-1, stations, -1, -1)
        temporal = temporal.reshape(batch * stations, temporal.shape[2], length)

        station_index = torch.arange(stations, device=forecast.device)
        station = self.station_feature_projection(station_features)
        station = station + self.station_embedding(station_index)
        station = station[None, :, :, None].expand(batch, -1, -1, length)
        station = station.reshape(batch * stations, station.shape[2], length)

        feature = self.fuse(torch.cat([local, temporal, station], dim=1))
        extras: dict[str, torch.Tensor] = {}
        if self.use_forecast_ramps:
            expected = (batch, stations, len(self.forecast_ramp_lags), length)
            if forecast_ramps is None or forecast_ramps.shape != expected:
                raise ValueError(
                    f"forecast_ramps must be {expected}, got "
                    f"{None if forecast_ramps is None else tuple(forecast_ramps.shape)}"
                )
            extras["ramp"] = self.extra_stems["ramp"](
                forecast_ramps.reshape(
                    batch * stations, len(self.forecast_ramp_lags), length
                )
            )
            extras["ramp"] = extras["ramp"] * local_strength
        if self.use_forecast_revision:
            expected = (batch, stations, length)
            if (
                forecast_revision is None
                or revision_mask is None
                or forecast_revision.shape != expected
                or revision_mask.shape != expected
            ):
                raise ValueError("forecast_revision/revision_mask must be [B,S,L]")
            revision_input = torch.stack(
                [forecast_revision, revision_mask], dim=2
            ).reshape(batch * stations, 2, length)
            extras["revision"] = self.extra_stems["revision"](revision_input)
            extras["revision"] = extras["revision"] * local_strength
        if self.use_recent_error:
            expected_error = (batch, stations, self.recent_error_hours)
            expected_mask = (batch, stations, 1)
            if (
                recent_error is None
                or recent_error_mask is None
                or recent_error.shape != expected_error
                or recent_error_mask.shape != expected_mask
            ):
                raise ValueError(
                    "recent_error must be [B,S,H] and recent_error_mask [B,S,1]"
                )
            recent = self.extra_stems["recent_error"](
                recent_error.reshape(batch * stations, 1, self.recent_error_hours)
            )
            recent = 0.5 * (recent.mean(dim=-1) + recent[:, :, -1])
            recent = recent * recent_error_mask.reshape(batch * stations, 1)
            extras["recent_error"] = recent[:, :, None].expand(-1, -1, length)

        for name, extra in extras.items():
            normalized = self.extra_norms[name](extra)
            feature = feature + torch.sigmoid(self.condition_gates[name]) * F.silu(
                normalized
            )
        outputs = [_restore_stations(feature, batch, stations)]
        for block in self.down_blocks:
            feature = block(feature)
            outputs.append(_restore_stations(feature, batch, stations))
        return outputs

    def gate_values(self) -> dict[str, float]:
        return {
            name: float(torch.sigmoid(value.detach()).cpu())
            for name, value in self.condition_gates.items()
        }


class StationForecastCorrectionHead(nn.Module):
    """Predict a causal forecast correction before stochastic residual diffusion.

    The head predicts a correction in normalized physical-power units.  Its
    inputs are restricted to information available at issue time.  The
    ``decomposed`` mode differs from ``direct`` only in the representation of
    the issued forecast, keeping the A1/A2 ablation identifiable.
    """

    MODES = {"direct", "decomposed"}

    def __init__(
        self,
        config: Mapping[str, object],
        station_features: torch.Tensor,
        groups: int,
    ) -> None:
        super().__init__()
        self.mode = str(config.get("forecast_correction_mode", "direct"))
        if self.mode not in self.MODES:
            raise ValueError(
                f"forecast_correction_mode must be one of {sorted(self.MODES)}"
            )
        self.station_count = int(config.get("station_count", 24))
        self.recent_error_hours = int(config.get("recent_error_hours", 24))
        self.use_revision = bool(
            config.get("forecast_correction_use_revision", True)
        )
        self.use_recent_error = bool(
            config.get("forecast_correction_use_recent_error", True)
        )
        self.max_abs = float(config.get("forecast_correction_max_abs", 1.0))
        if self.max_abs <= 0:
            raise ValueError("forecast_correction_max_abs must be positive")
        hidden = int(config.get("forecast_correction_channels", 32))
        if hidden <= 0:
            raise ValueError("forecast_correction_channels must be positive")

        forecast_channels = 1 if self.mode == "direct" else 6
        revision_channels = 2 if self.use_revision else 0
        main_channels = forecast_channels + 10 + revision_channels
        self.main_stem = nn.Sequential(
            nn.Conv1d(main_channels, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(_group_count(hidden, groups), hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(_group_count(hidden, groups), hidden),
            nn.SiLU(),
        )
        self.station_projection = nn.Linear(
            int(station_features.shape[1]), hidden
        )
        self.register_buffer(
            "station_features", station_features.float(), persistent=False
        )
        self.recent_encoder = None
        if self.use_recent_error:
            self.recent_encoder = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(hidden, groups), hidden),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
            )
        fusion_channels = hidden * (3 if self.use_recent_error else 2)
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden, groups), hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(_group_count(hidden, groups), hidden),
            nn.SiLU(),
        )
        self.output = nn.Conv1d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _moving_average(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size % 2 != 1:
            raise ValueError("moving-average kernel must be odd")
        padding = kernel_size // 2
        padded = F.pad(value, (padding, padding), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)

    @staticmethod
    def _difference(value: torch.Tensor, lag: int) -> torch.Tensor:
        output = torch.zeros_like(value)
        output[..., lag:] = value[..., lag:] - value[..., :-lag]
        return output

    def _forecast_components(self, forecast: torch.Tensor) -> torch.Tensor:
        if self.mode == "direct":
            return forecast[:, None, :]
        source = forecast[:, None, :]
        low_day = self._moving_average(source, 25)
        low_multi_day = self._moving_average(source, 73)
        event = source - low_day
        return torch.cat(
            [
                low_day,
                low_multi_day,
                event,
                self._difference(source, 1),
                self._difference(source, 3),
                self._difference(source, 6),
            ],
            dim=1,
        )

    def forward(
        self,
        forecast: torch.Tensor,
        calendar: torch.Tensor,
        lead: torch.Tensor,
        forecast_revision: torch.Tensor | None = None,
        revision_mask: torch.Tensor | None = None,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if forecast.ndim != 3:
            raise ValueError("forecast correction expects forecast [B,S,L]")
        batch, stations, length = forecast.shape
        if stations != self.station_count:
            raise ValueError("forecast correction station count mismatch")
        if calendar.shape != (batch, 8, length):
            raise ValueError("forecast correction calendar must be [B,8,L]")
        if lead.shape != (batch, 2, length):
            raise ValueError("forecast correction lead must be [B,2,L]")

        flat_forecast = forecast.reshape(batch * stations, length)
        pieces = [self._forecast_components(flat_forecast)]
        temporal = torch.cat([calendar, lead], dim=1)
        temporal = temporal[:, None].expand(-1, stations, -1, -1)
        pieces.append(temporal.reshape(batch * stations, 10, length))
        if self.use_revision:
            expected = (batch, stations, length)
            if (
                forecast_revision is None
                or revision_mask is None
                or forecast_revision.shape != expected
                or revision_mask.shape != expected
            ):
                raise ValueError(
                    "forecast correction revision and mask must be [B,S,L]"
                )
            pieces.extend(
                [
                    (forecast_revision * revision_mask).reshape(
                        batch * stations, 1, length
                    ),
                    revision_mask.reshape(batch * stations, 1, length),
                ]
            )
        main = self.main_stem(torch.cat(pieces, dim=1))

        station = self.station_projection(self.station_features)
        station = station[None, :, :, None].expand(batch, -1, -1, length)
        station = station.reshape(batch * stations, station.shape[2], length)
        fusion_inputs = [main, station]
        if self.use_recent_error:
            expected_error = (batch, stations, self.recent_error_hours)
            expected_mask = (batch, stations, 1)
            if (
                recent_error is None
                or recent_error_mask is None
                or recent_error.shape != expected_error
                or recent_error_mask.shape != expected_mask
            ):
                raise ValueError(
                    "forecast correction recent error/mask shape mismatch"
                )
            assert self.recent_encoder is not None
            recent = self.recent_encoder(
                recent_error.reshape(batch * stations, 1, self.recent_error_hours)
            )
            recent = 0.5 * (recent.mean(dim=-1) + recent[:, :, -1])
            recent = recent * recent_error_mask.reshape(batch * stations, 1)
            fusion_inputs.append(recent[:, :, None].expand(-1, -1, length))

        raw = self.output(self.fusion(torch.cat(fusion_inputs, dim=1)))
        correction = self.max_abs * torch.tanh(raw / self.max_abs)
        return correction.reshape(batch, stations, length)


class StationStateEncoder(nn.Module):
    """Lightweight multi-scale encoder for four causal state-v1 features."""

    def __init__(
        self,
        state_channels: Sequence[int],
        station_features: torch.Tensor,
        station_capacities: torch.Tensor,
        groups: int,
        state_dim: int = 4,
        global_gate_init: float = -1.0,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in state_channels)
        if not widths or any(value <= 0 for value in widths):
            raise ValueError(f"invalid state_channels={widths}")
        if station_capacities.shape != (station_features.shape[0],):
            raise ValueError("station capacities must be [S]")
        self.state_dim = int(state_dim)
        self.widths = widths
        self.stem = nn.Sequential(
            nn.Conv1d(self.state_dim, widths[0], kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(widths[0], groups), widths[0]),
            nn.SiLU(),
        )
        self.down_blocks = nn.ModuleList()
        for input_width, output_width in zip(widths[:-1], widths[1:]):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_width,
                        output_width,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(_group_count(output_width, groups), output_width),
                    nn.SiLU(),
                    nn.Conv1d(output_width, output_width, kernel_size=3, padding=1),
                )
            )
        self.global_projections = nn.ModuleList(
            [nn.Conv1d(width * 2, width, kernel_size=1) for width in widths]
        )
        self.global_gates = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(global_gate_init))) for _ in widths]
        )
        wind = station_features[:, 0].float()
        solar = station_features[:, 1].float()
        capacity = station_capacities.float().clamp(min=1e-6)
        wind_weight = wind * capacity
        solar_weight = solar * capacity
        wind_weight = wind_weight / wind_weight.sum().clamp(min=1e-8)
        solar_weight = solar_weight / solar_weight.sum().clamp(min=1e-8)
        self.register_buffer("wind_capacity_weight", wind_weight)
        self.register_buffer("solar_capacity_weight", solar_weight)

    def _fuse_global(self, node: torch.Tensor, level: int) -> torch.Tensor:
        wind = torch.einsum("s,bsct->bct", self.wind_capacity_weight, node)
        solar = torch.einsum("s,bsct->bct", self.solar_capacity_weight, node)
        global_state = self.global_projections[level](torch.cat([wind, solar], dim=1))
        return node + torch.sigmoid(self.global_gates[level]) * global_state[:, None]

    def forward(self, node_state: torch.Tensor) -> list[torch.Tensor]:
        if node_state.ndim != 4:
            raise ValueError(
                f"node_state must be [B,S,D,L], got {tuple(node_state.shape)}"
            )
        batch, stations, state_dim, length = node_state.shape
        if state_dim != self.state_dim:
            raise ValueError(f"expected state_dim={self.state_dim}, got {state_dim}")
        feature = self.stem(node_state.reshape(batch * stations, state_dim, length))
        node = _restore_stations(feature, batch, stations)
        outputs = [self._fuse_global(node, 0)]
        for level, block in enumerate(self.down_blocks, start=1):
            feature = block(feature)
            node = _restore_stations(feature, batch, stations)
            outputs.append(self._fuse_global(node, level))
        return outputs

    def gate_values(self) -> dict[str, float]:
        return {
            f"global_level_{index}": float(torch.sigmoid(value.detach()).cpu())
            for index, value in enumerate(self.global_gates)
        }


class StationResBlock(nn.Module):
    """Shared temporal residual block with diffusion-time and local FiLM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        condition_channels: int,
        groups: int,
        dropout: float,
        state_channels: int | None = None,
        state_gate_init: float = -1.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.norm1 = nn.GroupNorm(
            _group_count(self.in_channels, groups), self.in_channels
        )
        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(
            _group_count(self.out_channels, groups), self.out_channels
        )
        self.time_affine = nn.Linear(time_dim, self.out_channels * 2)
        self.condition_affine = nn.Conv1d(
            condition_channels, self.out_channels * 2, kernel_size=1
        )
        self.state_affine = None
        self.state_gate = None
        if state_channels is not None:
            self.state_affine = nn.Conv1d(
                int(state_channels), self.out_channels * 2, kernel_size=1
            )
            nn.init.zeros_(self.state_affine.weight)
            nn.init.zeros_(self.state_affine.bias)
            self.state_gate = nn.Parameter(torch.tensor(float(state_gate_init)))
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, 3, padding=1)
        self.residual = (
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )

    def forward(
        self,
        value: torch.Tensor,
        time_embedding: torch.Tensor,
        condition: torch.Tensor,
        state_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flattened, batch, stations = _flatten_stations(value)
        condition_flattened, _, _ = _flatten_stations(condition)
        residual = self.residual(flattened)
        hidden = self.conv1(F.silu(self.norm1(flattened)))
        hidden = self.norm2(hidden)
        repeated_time = time_embedding.repeat_interleave(stations, dim=0)
        time_gamma, time_beta = self.time_affine(repeated_time).chunk(2, dim=1)
        condition_gamma, condition_beta = self.condition_affine(
            condition_flattened
        ).chunk(2, dim=1)
        gamma = time_gamma[:, :, None] + condition_gamma
        beta = time_beta[:, :, None] + condition_beta
        if self.state_affine is not None:
            if state_condition is None:
                raise ValueError("state_condition is required for a state-aware ResBlock")
            state_flattened, state_batch, state_stations = _flatten_stations(
                state_condition
            )
            if state_batch != batch or state_stations != stations:
                raise ValueError("state_condition batch/station axes do not match")
            state_gamma, state_beta = self.state_affine(state_flattened).chunk(2, dim=1)
            state_weight = torch.sigmoid(self.state_gate)
            gamma = gamma + state_weight * state_gamma
            beta = beta + state_weight * state_beta
        hidden = hidden * (1.0 + gamma) + beta
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return _restore_stations(residual + hidden, batch, stations)


class StationSpatialBlock(nn.Module):
    """One light graph propagation block with fixed or type-gated edges."""

    def __init__(
        self,
        channels: int,
        adjacency: torch.Tensor,
        station_features: torch.Tensor,
        mode: str,
        groups: int,
        dropout: float,
        gate_init: float = -1.0,
        secondary_adjacency: torch.Tensor | None = None,
        dual_graph_primary_logit_init: float = 2.0,
        dual_graph_secondary_logit_init: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in SPATIAL_MODES:
            raise ValueError(f"unsupported spatial mode={mode!r}")
        self.mode = mode
        normalized = _normalize_adjacency(adjacency)
        self.register_buffer("normalized_adjacency", normalized)
        self.dual_graph_logits = None
        if secondary_adjacency is not None:
            if mode != "fixed_graph":
                raise ValueError("dual fixed graph requires spatial_mode=fixed_graph")
            self.register_buffer(
                "normalized_secondary_adjacency",
                _normalize_adjacency(secondary_adjacency),
            )
            self.dual_graph_logits = nn.Parameter(
                torch.tensor(
                    [
                        float(dual_graph_primary_logit_init),
                        float(dual_graph_secondary_logit_init),
                    ]
                )
            )
        wind = station_features[:, 0].float()
        solar = station_features[:, 1].float()
        masks = {
            "wind_wind": wind[:, None] * wind[None, :],
            "solar_solar": solar[:, None] * solar[None, :],
            "wind_solar": wind[:, None] * solar[None, :]
            + solar[:, None] * wind[None, :],
        }
        for name, mask in masks.items():
            self.register_buffer(f"adjacency_{name}", normalized * mask)

        self.norm = nn.GroupNorm(_group_count(channels, groups), channels)
        self.projection = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(float(dropout))
        if mode == "fixed_graph":
            self.graph_gate = nn.Parameter(torch.tensor(float(gate_init)))
            self.relation_gates = None
        elif mode == "type_gated_graph":
            self.graph_gate = None
            self.relation_gates = nn.ParameterDict(
                {
                    "wind_wind": nn.Parameter(torch.tensor(float(gate_init))),
                    "solar_solar": nn.Parameter(torch.tensor(float(gate_init))),
                    "wind_solar": nn.Parameter(torch.tensor(float(gate_init) - 1.0)),
                }
            )
        else:
            self.graph_gate = None
            self.relation_gates = None

    def gate_values(self) -> dict[str, float]:
        if self.mode == "none":
            return {}
        if self.mode == "fixed_graph":
            values = {"all": float(torch.sigmoid(self.graph_gate.detach()).cpu())}
            if self.dual_graph_logits is not None:
                weights = torch.softmax(self.dual_graph_logits.detach(), dim=0).cpu()
                values.update(
                    {
                        "dual_primary": float(weights[0]),
                        "dual_secondary": float(weights[1]),
                    }
                )
            return values
        return {
            name: float(torch.sigmoid(value.detach()).cpu())
            for name, value in self.relation_gates.items()
        }

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return value
        if self.mode == "fixed_graph":
            adjacency = self.normalized_adjacency
            if self.dual_graph_logits is not None:
                weights = torch.softmax(self.dual_graph_logits, dim=0)
                adjacency = (
                    weights[0] * self.normalized_adjacency
                    + weights[1] * self.normalized_secondary_adjacency
                )
            message = torch.einsum("ij,bjct->bict", adjacency, value)
            message = torch.sigmoid(self.graph_gate) * message
        else:
            message = torch.zeros_like(value)
            for relation, gate in self.relation_gates.items():
                relation_adjacency = getattr(self, f"adjacency_{relation}")
                message = message + torch.sigmoid(gate) * torch.einsum(
                    "ij,bjct->bict", relation_adjacency, value
                )
        flattened, batch, stations = _flatten_stations(message)
        transformed = self.projection(F.silu(self.norm(flattened)))
        transformed = self.dropout(transformed)
        return value + _restore_stations(transformed, batch, stations)


class StationParallelGraphFusion(nn.Module):
    """Fuse temporal and graph branches with a local dynamic gate.

    The temporal branch is produced by the existing FiLM-conditioned ResBlock.
    The spatial branch starts from the same pre-ResBlock hidden state plus the
    already-available forecast condition.  The graph can remain geographic or
    use a small condition/state-dependent residual on top of that physical
    prior.  A zero-initialized local gate learns when each station and lead hour
    should use the graph message.
    """

    def __init__(
        self,
        channels: int,
        adjacency: torch.Tensor,
        station_features: torch.Tensor,
        groups: int,
        dropout: float,
        gate_init: float = -1.0,
        adjacency_mode: str = "fixed",
        state_channels: int | None = None,
        dynamic_embedding_dim: int = 16,
        dynamic_top_k: int = 6,
        dynamic_temperature: float = 1.0,
        dynamic_mix_gate_init: float = -3.0,
        secondary_adjacency: torch.Tensor | None = None,
        dual_graph_primary_logit_init: float = 2.0,
        dual_graph_secondary_logit_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.adjacency_mode = str(adjacency_mode)
        if self.adjacency_mode not in {"fixed", "hybrid_dynamic"}:
            raise ValueError(
                "parallel adjacency mode must be fixed or hybrid_dynamic, got "
                f"{self.adjacency_mode!r}"
            )
        normalized_adjacency = _normalize_adjacency(adjacency)
        self.register_buffer("normalized_adjacency", normalized_adjacency)
        self.dual_graph_logits = None
        if secondary_adjacency is not None:
            if self.adjacency_mode != "fixed":
                raise ValueError("dual fixed graph cannot be combined with dynamic adjacency")
            self.register_buffer(
                "normalized_secondary_adjacency",
                _normalize_adjacency(secondary_adjacency),
            )
            self.dual_graph_logits = nn.Parameter(
                torch.tensor(
                    [
                        float(dual_graph_primary_logit_init),
                        float(dual_graph_secondary_logit_init),
                    ]
                )
            )
        self.register_buffer(
            "off_geographic_mask",
            (adjacency <= 0).float(),
        )
        self.spatial_norm = nn.GroupNorm(
            _group_count(self.channels, groups), self.channels
        )
        self.spatial_projection = nn.Conv1d(
            self.channels, self.channels, kernel_size=1
        )
        self.dropout = nn.Dropout(float(dropout))
        self.gate_prior = nn.Parameter(torch.tensor(float(gate_init)))
        self.gate_projection = nn.Conv1d(self.channels * 3, 1, kernel_size=1)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)

        self.dynamic_embedding_dim = int(dynamic_embedding_dim)
        self.dynamic_top_k = int(dynamic_top_k)
        self.dynamic_temperature = float(dynamic_temperature)
        self.dynamic_state_channels = int(state_channels or 0)
        self.dynamic_node_encoder = None
        self.static_node_projection = None
        self.dynamic_node_norm = None
        self.dynamic_mix_gate = None
        if self.adjacency_mode == "hybrid_dynamic":
            station_count = int(adjacency.shape[0])
            if not 1 <= self.dynamic_top_k < station_count:
                raise ValueError(
                    f"dynamic_top_k must be in [1,{station_count - 1}]"
                )
            if self.dynamic_embedding_dim <= 0:
                raise ValueError("dynamic_embedding_dim must be positive")
            if self.dynamic_temperature <= 0:
                raise ValueError("dynamic_temperature must be positive")
            dynamic_input_channels = self.channels + self.dynamic_state_channels
            self.dynamic_node_encoder = nn.Sequential(
                nn.Linear(dynamic_input_channels * 2, self.dynamic_embedding_dim),
                nn.SiLU(),
                nn.Linear(self.dynamic_embedding_dim, self.dynamic_embedding_dim),
            )
            self.static_node_projection = nn.Linear(
                int(station_features.shape[1]),
                self.dynamic_embedding_dim,
                bias=False,
            )
            self.dynamic_node_norm = nn.LayerNorm(self.dynamic_embedding_dim)
            self.dynamic_mix_gate = nn.Parameter(
                torch.tensor(float(dynamic_mix_gate_init))
            )
            self.register_buffer(
                "static_station_features", station_features.float()
            )

        self.register_buffer(
            "gate_observed_sum", torch.tensor(0.0, dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "gate_observed_square_sum",
            torch.tensor(0.0, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "gate_observed_count", torch.tensor(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "gate_observed_min", torch.tensor(float("inf")), persistent=False
        )
        self.register_buffer(
            "gate_observed_max", torch.tensor(float("-inf")), persistent=False
        )
        station_count = int(adjacency.shape[0])
        self.register_buffer(
            "adjacency_observed_sum",
            torch.zeros(station_count, station_count, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "adjacency_observed_square_sum",
            torch.zeros(station_count, station_count, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "adjacency_observed_count",
            torch.tensor(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "off_geographic_mass_sum",
            torch.tensor(0.0, dtype=torch.float64),
            persistent=False,
        )

    def reset_gate_statistics(self) -> None:
        self.gate_observed_sum.zero_()
        self.gate_observed_square_sum.zero_()
        self.gate_observed_count.zero_()
        self.gate_observed_min.fill_(float("inf"))
        self.gate_observed_max.fill_(float("-inf"))
        self.adjacency_observed_sum.zero_()
        self.adjacency_observed_square_sum.zero_()
        self.adjacency_observed_count.zero_()
        self.off_geographic_mass_sum.zero_()

    def gate_statistics(self) -> dict[str, float]:
        values = {
            "prior": float(torch.sigmoid(self.gate_prior.detach()).cpu()),
        }
        if self.dual_graph_logits is not None:
            weights = torch.softmax(self.dual_graph_logits.detach(), dim=0).cpu()
            values["dual_primary"] = float(weights[0])
            values["dual_secondary"] = float(weights[1])
        if self.dynamic_mix_gate is not None:
            values["dynamic_mix"] = float(
                torch.sigmoid(self.dynamic_mix_gate.detach()).cpu()
            )
        if int(self.gate_observed_count.detach().cpu()) > 0:
            mean = self.gate_observed_sum / self.gate_observed_count
            variance = (
                self.gate_observed_square_sum / self.gate_observed_count - mean.square()
            ).clamp(min=0.0)
            values.update(
                {
                    "observed_mean": float(mean.cpu()),
                    "observed_std": float(variance.sqrt().cpu()),
                    "observed_min": float(self.gate_observed_min.cpu()),
                    "observed_max": float(self.gate_observed_max.cpu()),
                }
            )
        if int(self.adjacency_observed_count.detach().cpu()) > 0:
            count = self.adjacency_observed_count.double()
            adjacency_mean = self.adjacency_observed_sum / count
            adjacency_variance = (
                self.adjacency_observed_square_sum / count
                - adjacency_mean.square()
            ).clamp(min=0.0)
            values.update(
                {
                    "adjacency_mean": float(adjacency_mean.mean().cpu()),
                    "adjacency_std": float(adjacency_variance.mean().sqrt().cpu()),
                    "off_geographic_mass": float(
                        (self.off_geographic_mass_sum / count).cpu()
                    ),
                }
            )
        return values

    def adjacency_moments(self) -> dict[str, torch.Tensor]:
        if int(self.adjacency_observed_count.detach().cpu()) == 0:
            return {}
        count = self.adjacency_observed_count.double()
        mean = self.adjacency_observed_sum / count
        variance = (
            self.adjacency_observed_square_sum / count - mean.square()
        ).clamp(min=0.0)
        return {
            "mean": mean.detach().float().cpu(),
            "std": variance.sqrt().detach().float().cpu(),
        }

    def _hybrid_adjacency(
        self,
        condition: torch.Tensor,
        state_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.adjacency_mode == "fixed":
            adjacency = self.normalized_adjacency
            if self.dual_graph_logits is not None:
                weights = torch.softmax(self.dual_graph_logits, dim=0)
                adjacency = (
                    weights[0] * self.normalized_adjacency
                    + weights[1] * self.normalized_secondary_adjacency
                )
            return adjacency[None].expand(
                condition.shape[0], -1, -1
            )
        if self.dynamic_node_encoder is None or self.dynamic_mix_gate is None:
            raise RuntimeError("hybrid dynamic graph modules were not initialized")
        dynamic_inputs = [condition]
        if self.dynamic_state_channels:
            if state_condition is None:
                raise ValueError(
                    "state_condition is required by the hybrid dynamic graph"
                )
            if state_condition.shape[:2] != condition.shape[:2] or (
                state_condition.shape[-1] != condition.shape[-1]
            ):
                raise ValueError("dynamic graph condition/state axes do not match")
            if state_condition.shape[2] != self.dynamic_state_channels:
                raise ValueError(
                    "unexpected dynamic graph state channels: "
                    f"{state_condition.shape[2]} != {self.dynamic_state_channels}"
                )
            dynamic_inputs.append(state_condition)
        context = torch.cat(dynamic_inputs, dim=2)
        pooled = torch.cat(
            [context.mean(dim=-1), context.var(dim=-1, unbiased=False).add(1e-6).sqrt()],
            dim=-1,
        )
        dynamic_nodes = self.dynamic_node_encoder(pooled)
        static_nodes = self.static_node_projection(self.static_station_features)
        nodes = self.dynamic_node_norm(dynamic_nodes + static_nodes[None])
        similarity = torch.einsum("bid,bjd->bij", nodes, nodes)
        similarity = similarity / math.sqrt(self.dynamic_embedding_dim)
        similarity = 0.5 * (similarity + similarity.transpose(1, 2))

        station_count = similarity.shape[-1]
        identity = torch.eye(
            station_count, dtype=torch.bool, device=similarity.device
        )[None]
        ranking = similarity.masked_fill(identity, float("-inf"))
        neighbors = ranking.topk(self.dynamic_top_k, dim=-1).indices
        sparse_mask = torch.zeros_like(similarity, dtype=torch.bool)
        sparse_mask.scatter_(-1, neighbors, True)
        sparse_mask = sparse_mask | sparse_mask.transpose(1, 2) | identity
        dynamic = F.softplus(similarity / self.dynamic_temperature)
        dynamic = dynamic * sparse_mask.to(dynamic.dtype)
        dynamic = dynamic + identity.to(dynamic.dtype)
        dynamic = _normalize_batched_adjacency(dynamic)
        mix = torch.sigmoid(self.dynamic_mix_gate)
        return (1.0 - mix) * self.normalized_adjacency[None] + mix * dynamic

    def _observe_adjacency(self, adjacency: torch.Tensor) -> None:
        detached = adjacency.detach().double()
        self.adjacency_observed_sum.add_(detached.sum(dim=0))
        self.adjacency_observed_square_sum.add_(detached.square().sum(dim=0))
        self.adjacency_observed_count.add_(detached.shape[0])
        off_mass = (
            detached * self.off_geographic_mask[None].double()
        ).sum(dim=(1, 2)) / detached.sum(dim=(1, 2)).clamp(min=1e-12)
        self.off_geographic_mass_sum.add_(off_mass.sum())

    def forward(
        self,
        source: torch.Tensor,
        temporal: torch.Tensor,
        condition: torch.Tensor,
        state_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source.shape != temporal.shape or source.shape != condition.shape:
            raise ValueError(
                "parallel fusion inputs must share [B,S,C,L], got "
                f"source={tuple(source.shape)} temporal={tuple(temporal.shape)} "
                f"condition={tuple(condition.shape)}"
            )
        graph_source = source + condition
        adjacency = self._hybrid_adjacency(condition, state_condition)
        message = torch.einsum("bij,bjct->bict", adjacency, graph_source)
        message_flat, batch, stations = _flatten_stations(message)
        spatial_flat = self.spatial_projection(
            F.silu(self.spatial_norm(message_flat))
        )
        spatial_flat = self.dropout(spatial_flat)
        temporal_flat, _, _ = _flatten_stations(temporal)
        condition_flat, _, _ = _flatten_stations(condition)
        gate = torch.sigmoid(
            self.gate_prior
            + self.gate_projection(
                torch.cat([temporal_flat, spatial_flat, condition_flat], dim=1)
            )
        )
        if not self.training:
            with torch.no_grad():
                if self.adjacency_mode == "hybrid_dynamic":
                    self._observe_adjacency(adjacency)
                detached = gate.detach()
                self.gate_observed_sum.add_(detached.double().sum())
                self.gate_observed_square_sum.add_(detached.double().square().sum())
                self.gate_observed_count.add_(detached.numel())
                self.gate_observed_min.copy_(
                    torch.minimum(self.gate_observed_min, detached.min().float())
                )
                self.gate_observed_max.copy_(
                    torch.maximum(self.gate_observed_max, detached.max().float())
                )
        fused_flat = temporal_flat + gate * spatial_flat
        return _restore_stations(fused_flat, batch, stations)


class StationConditionalResUNet1D(nn.Module):
    """Conditional temporal ResUNet with an explicit station axis."""

    def __init__(
        self,
        config: Mapping[str, object],
        station_features: torch.Tensor,
        adjacency: torch.Tensor,
        station_capacities: torch.Tensor | None = None,
        secondary_adjacency: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.sequence_length = int(config.get("sequence_length", 168))
        self.station_count = int(config.get("station_count", 24))
        self.spatial_mode = str(config.get("spatial_mode", "none"))
        self.use_state_encoder = bool(config.get("use_state_encoder", False))
        self.forecast_condition_dropout_prob = float(
            config.get("forecast_condition_dropout_prob", 0.0)
        )
        if not 0.0 <= self.forecast_condition_dropout_prob < 1.0:
            raise ValueError("forecast_condition_dropout_prob must be in [0,1)")
        self.register_buffer(
            "_forecast_condition_sample_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_forecast_condition_drop_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.use_dual_fixed_graph = bool(config.get("use_dual_fixed_graph", False))
        if self.use_dual_fixed_graph != (secondary_adjacency is not None):
            raise ValueError(
                "use_dual_fixed_graph must match the supplied secondary adjacency"
            )
        if self.use_dual_fixed_graph and self.spatial_mode != "fixed_graph":
            raise ValueError("dual fixed graph requires spatial_mode=fixed_graph")
        if self.spatial_mode not in SPATIAL_MODES:
            raise ValueError(f"unsupported spatial_mode={self.spatial_mode!r}")
        if station_features.shape[0] != self.station_count:
            raise ValueError("station_count does not match station_features")

        num_layers = int(config.get("num_layers", 3))
        multipliers = tuple(config.get("channel_multipliers", [1, 2, 4]))
        if len(multipliers) != num_layers:
            raise ValueError("channel_multipliers length must equal num_layers")
        if self.sequence_length % (2 ** (num_layers - 1)) != 0:
            raise ValueError("sequence length is incompatible with UNet scales")
        base_channels = int(config.get("base_channels", 32))
        self.channels = tuple(base_channels * int(value) for value in multipliers)
        groups = int(config.get("group_norm_groups", 8))
        dropout = float(config.get("dropout", 0.10))
        timestep_dim = int(config.get("timestep_embedding_dim", 128))
        configured_spatial_levels = config.get(
            "spatial_mix_levels", ["bottleneck"]
        )
        if not isinstance(configured_spatial_levels, (list, tuple)):
            raise ValueError("spatial_mix_levels must be a list of level names")
        allowed_spatial_levels = {
            f"encoder_{level}" for level in range(num_layers - 1)
        }
        allowed_spatial_levels.add("bottleneck")
        spatial_mix_levels = tuple(str(value) for value in configured_spatial_levels)
        if not spatial_mix_levels:
            raise ValueError("spatial_mix_levels must not be empty")
        if len(set(spatial_mix_levels)) != len(spatial_mix_levels):
            raise ValueError("spatial_mix_levels must not contain duplicates")
        unknown_spatial_levels = set(spatial_mix_levels) - allowed_spatial_levels
        if unknown_spatial_levels:
            raise ValueError(
                f"unsupported spatial_mix_levels={sorted(unknown_spatial_levels)}; "
                f"allowed={sorted(allowed_spatial_levels)}"
            )
        self.spatial_mix_levels = spatial_mix_levels
        configured_parallel_levels = config.get(
            "parallel_spatial_fusion_levels", []
        )
        if not isinstance(configured_parallel_levels, (list, tuple)):
            raise ValueError(
                "parallel_spatial_fusion_levels must be a list of level names"
            )
        parallel_levels = tuple(str(value) for value in configured_parallel_levels)
        if len(set(parallel_levels)) != len(parallel_levels):
            raise ValueError(
                "parallel_spatial_fusion_levels must not contain duplicates"
            )
        unknown_parallel_levels = set(parallel_levels) - allowed_spatial_levels
        if unknown_parallel_levels:
            raise ValueError(
                "unsupported parallel_spatial_fusion_levels="
                f"{sorted(unknown_parallel_levels)}; "
                f"allowed={sorted(allowed_spatial_levels)}"
            )
        overlap = set(parallel_levels) & set(spatial_mix_levels)
        if overlap:
            raise ValueError(
                "sequential and parallel graph fusion cannot share levels: "
                f"{sorted(overlap)}"
            )
        if parallel_levels and self.spatial_mode != "fixed_graph":
            raise ValueError(
                "parallel graph fusion currently requires spatial_mode=fixed_graph"
            )
        self.parallel_spatial_fusion_levels = parallel_levels
        self.parallel_spatial_adjacency_mode = str(
            config.get("parallel_spatial_adjacency_mode", "fixed")
        )
        if self.parallel_spatial_adjacency_mode not in {
            "fixed",
            "hybrid_dynamic",
        }:
            raise ValueError(
                "parallel_spatial_adjacency_mode must be fixed or hybrid_dynamic"
            )
        if not parallel_levels and self.parallel_spatial_adjacency_mode != "fixed":
            raise ValueError(
                "hybrid dynamic adjacency requires a parallel spatial fusion level"
            )

        self.register_buffer("station_features", station_features.float())
        self.timestep_embedding = DiffusionTimestepEmbedding(
            timestep_dim, timestep_dim
        )
        self.condition_encoder = StationConditionEncoder(
            self.channels,
            self.station_count,
            int(station_features.shape[1]),
            groups,
            condition_config=config,
        )
        self.state_encoder = None
        state_widths: tuple[int, ...] = ()
        if self.use_state_encoder:
            state_widths = tuple(
                int(value) for value in config.get("state_channels", [8, 16, 32])
            )
            if len(state_widths) != num_layers:
                raise ValueError("state_channels length must equal num_layers")
            capacities = (
                station_capacities.float()
                if station_capacities is not None
                else torch.ones(self.station_count, dtype=torch.float32)
            )
            self.state_encoder = StationStateEncoder(
                state_widths,
                station_features,
                capacities,
                groups,
                state_dim=int(config.get("state_feature_dim", 4)),
                global_gate_init=float(config.get("state_global_gate_init", -1.0)),
            )
        self.state_stem = nn.Conv1d(1, self.channels[0], kernel_size=3, padding=1)
        self.encoder_blocks = nn.ModuleList(
            [
                StationResBlock(
                    self.channels[level],
                    self.channels[level],
                    timestep_dim,
                    self.channels[level],
                    groups,
                    dropout,
                    state_channels=(state_widths[level] if self.use_state_encoder else None),
                    state_gate_init=float(config.get("state_film_gate_init", -1.0)),
                )
                for level in range(num_layers)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv1d(
                    self.channels[level],
                    self.channels[level + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
                for level in range(num_layers - 1)
            ]
        )
        self.encoder_spatial_blocks = nn.ModuleDict(
            {
                f"encoder_{level}": StationSpatialBlock(
                    self.channels[level],
                    adjacency,
                    station_features,
                    self.spatial_mode,
                    groups,
                    dropout,
                    gate_init=float(config.get("spatial_gate_init", -1.0)),
                    secondary_adjacency=secondary_adjacency,
                    dual_graph_primary_logit_init=float(
                        config.get("dual_graph_primary_logit_init", 2.0)
                    ),
                    dual_graph_secondary_logit_init=float(
                        config.get("dual_graph_secondary_logit_init", 0.0)
                    ),
                )
                for level in range(num_layers - 1)
                if f"encoder_{level}" in self.spatial_mix_levels
            }
        )
        spatial_level_channels = {
            **{
                f"encoder_{level}": self.channels[level]
                for level in range(num_layers - 1)
            },
            "bottleneck": self.channels[-1],
        }
        self.parallel_spatial_blocks = nn.ModuleDict(
            {
                level: StationParallelGraphFusion(
                    spatial_level_channels[level],
                    adjacency,
                    station_features,
                    groups,
                    dropout,
                    gate_init=float(
                        config.get("parallel_spatial_gate_init", -1.0)
                    ),
                    adjacency_mode=self.parallel_spatial_adjacency_mode,
                    state_channels=(
                        state_widths[int(level.split("_")[-1])]
                        if self.use_state_encoder and level.startswith("encoder_")
                        else (
                            state_widths[-1]
                            if self.use_state_encoder and level == "bottleneck"
                            else None
                        )
                    ),
                    dynamic_embedding_dim=int(
                        config.get("dynamic_graph_embedding_dim", 16)
                    ),
                    dynamic_top_k=int(config.get("dynamic_graph_top_k", 6)),
                    dynamic_temperature=float(
                        config.get("dynamic_graph_temperature", 1.0)
                    ),
                    dynamic_mix_gate_init=float(
                        config.get("dynamic_graph_mix_gate_init", -3.0)
                    ),
                    secondary_adjacency=secondary_adjacency,
                    dual_graph_primary_logit_init=float(
                        config.get("dual_graph_primary_logit_init", 2.0)
                    ),
                    dual_graph_secondary_logit_init=float(
                        config.get("dual_graph_secondary_logit_init", 0.0)
                    ),
                )
                for level in self.parallel_spatial_fusion_levels
            }
        )
        self.bottleneck = StationResBlock(
            self.channels[-1],
            self.channels[-1],
            timestep_dim,
            self.channels[-1],
            groups,
            dropout,
            state_channels=(state_widths[-1] if self.use_state_encoder else None),
            state_gate_init=float(config.get("state_film_gate_init", -1.0)),
        )
        self.spatial_block = StationSpatialBlock(
            self.channels[-1],
            adjacency,
            station_features,
            self.spatial_mode,
            groups,
            dropout,
            gate_init=float(config.get("spatial_gate_init", -1.0)),
            secondary_adjacency=secondary_adjacency,
            dual_graph_primary_logit_init=float(
                config.get("dual_graph_primary_logit_init", 2.0)
            ),
            dual_graph_secondary_logit_init=float(
                config.get("dual_graph_secondary_logit_init", 0.0)
            ),
        )

        self.decoder_levels = tuple(reversed(range(num_layers - 1)))
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current_channels = self.channels[-1]
        for level in self.decoder_levels:
            self.upsamples.append(
                nn.ConvTranspose1d(
                    current_channels,
                    self.channels[level],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.decoder_blocks.append(
                StationResBlock(
                    self.channels[level] * 2,
                    self.channels[level],
                    timestep_dim,
                    self.channels[level],
                    groups,
                    dropout,
                    state_channels=(state_widths[level] if self.use_state_encoder else None),
                    state_gate_init=float(config.get("state_film_gate_init", -1.0)),
                )
            )
            current_channels = self.channels[level]
        self.output_norm = nn.GroupNorm(
            _group_count(self.channels[0], groups), self.channels[0]
        )
        self.output = nn.Conv1d(self.channels[0], 1, kernel_size=1)
        self.use_body_tail_experts = bool(
            config.get("use_body_tail_experts", False)
        )
        self.tail_station_adapter: nn.Module | None = None
        self.tail_common_adapter: nn.Module | None = None
        self.tail_common_gate: nn.Parameter | None = None
        self.tail_risk_encoder: nn.Module | None = None
        self.tail_risk_output: nn.Linear | None = None
        self.tail_signal_attention: nn.Module | None = None
        self.use_tail_time_localizer = bool(
            config.get("use_tail_time_localizer", False)
        )
        self.tail_time_output: nn.Module | None = None
        self.tail_time_temperature = float(
            config.get("tail_time_temperature", 1.0)
        )
        self.tail_time_mask_radius_hours = int(
            config.get("tail_time_mask_radius_hours", 9)
        )
        if self.use_tail_time_localizer and not self.use_body_tail_experts:
            raise ValueError("tail time localization requires body-tail experts")
        if self.tail_time_temperature <= 0.0:
            raise ValueError("tail_time_temperature must be positive")
        if not 1 <= self.tail_time_mask_radius_hours < self.sequence_length:
            raise ValueError(
                "tail_time_mask_radius_hours must be in [1, sequence_length)"
            )
        self.use_wind_common_residual_head = bool(
            config.get("use_wind_common_residual_head", False)
        )
        capacities = (
            station_capacities.float()
            if station_capacities is not None
            else torch.ones(self.station_count, dtype=torch.float32)
        )
        wind_mask = station_features[:, 0].float()
        wind_capacity = capacities * wind_mask
        if wind_capacity.sum() <= 0:
            raise ValueError("at least one wind station is required")
        self.register_buffer("wind_station_mask", wind_mask, persistent=False)
        self.register_buffer(
            "wind_capacity_weight",
            wind_capacity / wind_capacity.sum(),
            persistent=False,
        )
        if self.use_body_tail_experts:
            tail_channels = int(config.get("tail_expert_channels", 16))
            gate_channels = int(config.get("tail_gate_channels", 16))
            tail_prior = float(config.get("tail_gate_prior_probability", 0.08))
            if tail_channels <= 0 or gate_channels <= 0:
                raise ValueError("body-tail expert channels must be positive")
            if not 0.0 < tail_prior < 1.0:
                raise ValueError("tail_gate_prior_probability must be in (0,1)")

            def tail_adapter() -> nn.Sequential:
                adapter = nn.Sequential(
                    nn.Conv1d(
                        self.channels[0], tail_channels, kernel_size=3, padding=1
                    ),
                    nn.GroupNorm(_group_count(tail_channels, groups), tail_channels),
                    nn.SiLU(),
                    nn.Conv1d(
                        tail_channels,
                        tail_channels,
                        kernel_size=3,
                        padding=2,
                        dilation=2,
                    ),
                    nn.GroupNorm(_group_count(tail_channels, groups), tail_channels),
                    nn.SiLU(),
                    nn.Conv1d(tail_channels, 1, kernel_size=1),
                )
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                return adapter

            # The station path learns local event morphology; the common path
            # learns a synchronized wind-fleet perturbation.  Both are residual
            # adapters on the frozen baseline denoiser, so route=0 is exactly the
            # historical-spatial body model at initialization and after training.
            self.tail_station_adapter = tail_adapter()
            self.tail_common_adapter = tail_adapter()
            self.tail_common_gate = nn.Parameter(
                torch.tensor(float(config.get("tail_common_gate_init", -1.0)))
            )

            # Six causal-at-generation signals: issued wind forecast, downward
            # forecast ramp, aligned forecast revision, forecast-derived low and
            # down-ramp states, and the most recent observed forecast error.
            self.tail_signal_attention = nn.Sequential(
                nn.Linear(12, 12),
                nn.SiLU(),
                nn.Linear(12, 6),
            )
            nn.init.zeros_(self.tail_signal_attention[-1].weight)
            nn.init.zeros_(self.tail_signal_attention[-1].bias)
            self.tail_risk_encoder = nn.Sequential(
                nn.Conv1d(6, gate_channels, kernel_size=5, padding=2),
                nn.GroupNorm(_group_count(gate_channels, groups), gate_channels),
                nn.SiLU(),
                nn.Conv1d(
                    gate_channels,
                    gate_channels,
                    kernel_size=5,
                    padding=4,
                    dilation=2,
                ),
                nn.GroupNorm(_group_count(gate_channels, groups), gate_channels),
                nn.SiLU(),
            )
            self.tail_risk_output = nn.Linear(2 * gate_channels, 1)
            nn.init.zeros_(self.tail_risk_output.weight)
            nn.init.constant_(
                self.tail_risk_output.bias,
                math.log(tail_prior / (1.0 - tail_prior)),
            )
            if self.use_tail_time_localizer:
                # Reuse the causal hourly representation that currently feeds
                # the issuance-level risk gate.  Only this small head is new:
                # it predicts a conditional distribution over the 168 lead
                # hours instead of another global condition branch.
                self.tail_time_output = nn.Sequential(
                    nn.Conv1d(
                        gate_channels,
                        gate_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GroupNorm(
                        _group_count(gate_channels, groups), gate_channels
                    ),
                    nn.SiLU(),
                    nn.Conv1d(gate_channels, 1, kernel_size=1),
                )
                # A uniform time distribution is the neutral initialization;
                # the inherited body, tail adapters, and global gate are exact.
                nn.init.zeros_(self.tail_time_output[-1].weight)
                nn.init.zeros_(self.tail_time_output[-1].bias)
        self.wind_common_head = None
        self.wind_common_gate = None
        if self.use_wind_common_residual_head:
            common_channels = int(config.get("wind_common_channels", 16))
            if common_channels <= 0:
                raise ValueError("wind_common_channels must be positive")
            self.wind_common_head = nn.Sequential(
                nn.Conv1d(
                    self.channels[0], common_channels, kernel_size=3, padding=1
                ),
                nn.GroupNorm(
                    _group_count(common_channels, groups), common_channels
                ),
                nn.SiLU(),
                nn.Conv1d(
                    common_channels,
                    common_channels,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                ),
                nn.GroupNorm(
                    _group_count(common_channels, groups), common_channels
                ),
                nn.SiLU(),
                nn.Conv1d(common_channels, 1, kernel_size=3, padding=1),
            )
            self.wind_common_gate = nn.Parameter(
                torch.tensor(float(config.get("wind_common_gate_init", -1.0)))
            )
            # Start exactly from the existing 2D denoiser.  The common path is
            # learned gradually instead of perturbing every wind node at epoch 1.
            nn.init.zeros_(self.wind_common_head[-1].weight)
            nn.init.zeros_(self.wind_common_head[-1].bias)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timestep: torch.Tensor,
        forecast: torch.Tensor,
        calendar: torch.Tensor,
        lead: torch.Tensor,
        forecast_ramps: torch.Tensor | None = None,
        forecast_revision: torch.Tensor | None = None,
        revision_mask: torch.Tensor | None = None,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
        node_state: torch.Tensor | None = None,
        forecast_condition_strength: torch.Tensor | float | None = None,
        tail_expert_route: torch.Tensor | float | None = None,
        tail_time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_residual.shape[1:] != (self.station_count, self.sequence_length):
            raise ValueError(
                "noisy residual must be [B,S,L], got "
                f"{tuple(noisy_residual.shape)}"
            )
        batch, stations, length = noisy_residual.shape
        if forecast_condition_strength is None:
            if self.training and self.forecast_condition_dropout_prob > 0.0:
                keep = torch.rand(
                    batch, 1, 1, device=forecast.device, dtype=forecast.dtype
                ) >= self.forecast_condition_dropout_prob
                condition_strength = keep.to(forecast.dtype)
                with torch.no_grad():
                    self._forecast_condition_sample_count.add_(batch)
                    self._forecast_condition_drop_count.add_((~keep).sum())
            else:
                condition_strength = torch.ones(
                    batch, 1, 1, device=forecast.device, dtype=forecast.dtype
                )
        elif isinstance(forecast_condition_strength, (int, float)):
            condition_strength = torch.full(
                (batch, 1, 1),
                float(forecast_condition_strength),
                device=forecast.device,
                dtype=forecast.dtype,
            )
        else:
            condition_strength = forecast_condition_strength.to(
                device=forecast.device, dtype=forecast.dtype
            )
            if condition_strength.ndim == 1:
                condition_strength = condition_strength[:, None, None]
        if condition_strength.shape != (batch, 1, 1):
            raise ValueError(
                "forecast_condition_strength must be scalar, [B], or [B,1,1]"
            )
        if torch.any(condition_strength < 0.0) or torch.any(condition_strength > 1.0):
            raise ValueError("forecast_condition_strength must be in [0,1]")
        time_embedding = self.timestep_embedding(timestep)
        conditions = self.condition_encoder(
            forecast,
            calendar,
            lead,
            self.station_features,
            forecast_ramps=forecast_ramps,
            forecast_revision=forecast_revision,
            revision_mask=revision_mask,
            recent_error=recent_error,
            recent_error_mask=recent_error_mask,
            forecast_condition_strength=condition_strength,
        )
        state_conditions = None
        if self.use_state_encoder:
            if node_state is None:
                raise ValueError("node_state is required when use_state_encoder=true")
            state_conditions = self.state_encoder(node_state)
            state_strength = condition_strength[:, :, None, :]
            state_conditions = [
                state_condition * state_strength
                for state_condition in state_conditions
            ]
        hidden = self.state_stem(noisy_residual.reshape(batch * stations, 1, length))
        hidden = _restore_stations(hidden, batch, stations)
        skips = []
        for level, block in enumerate(self.encoder_blocks):
            branch_input = hidden
            temporal = block(
                branch_input,
                time_embedding,
                conditions[level],
                None if state_conditions is None else state_conditions[level],
            )
            spatial_level = f"encoder_{level}"
            hidden = temporal
            if spatial_level in self.parallel_spatial_blocks:
                hidden = self.parallel_spatial_blocks[spatial_level](
                    branch_input,
                    temporal,
                    conditions[level],
                    None if state_conditions is None else state_conditions[level],
                )
            if spatial_level in self.encoder_spatial_blocks:
                hidden = self.encoder_spatial_blocks[spatial_level](hidden)
            skips.append(hidden)
            if level < len(self.downsamples):
                flattened, _, _ = _flatten_stations(hidden)
                hidden = _restore_stations(
                    self.downsamples[level](flattened), batch, stations
                )
        bottleneck_input = hidden
        temporal = self.bottleneck(
            bottleneck_input,
            time_embedding,
            conditions[-1],
            None if state_conditions is None else state_conditions[-1],
        )
        hidden = temporal
        if "bottleneck" in self.parallel_spatial_blocks:
            hidden = self.parallel_spatial_blocks["bottleneck"](
                bottleneck_input,
                temporal,
                conditions[-1],
                None if state_conditions is None else state_conditions[-1],
            )
        if "bottleneck" in self.spatial_mix_levels:
            hidden = self.spatial_block(hidden)

        for level, upsample, block in zip(
            self.decoder_levels, self.upsamples, self.decoder_blocks
        ):
            flattened, _, _ = _flatten_stations(hidden)
            hidden = _restore_stations(upsample(flattened), batch, stations)
            skip = skips[level]
            if hidden.shape[-1] != skip.shape[-1]:
                flat, _, _ = _flatten_stations(hidden)
                flat = F.interpolate(
                    flat, size=skip.shape[-1], mode="linear", align_corners=False
                )
                hidden = _restore_stations(flat, batch, stations)
            hidden = block(
                torch.cat([hidden, skip], dim=2),
                time_embedding,
                conditions[level],
                None if state_conditions is None else state_conditions[level],
            )
        flattened, _, _ = _flatten_stations(hidden)
        output = self.output(F.silu(self.output_norm(flattened))).reshape(
            batch, stations, length
        )
        if self.wind_common_head is not None:
            pooled = torch.einsum(
                "s,bsct->bct", self.wind_capacity_weight, hidden
            )
            common = self.wind_common_head(pooled)
            output = output + (
                torch.sigmoid(self.wind_common_gate)
                * common
                * self.wind_station_mask[None, :, None]
            )
        if self.use_body_tail_experts:
            if tail_expert_route is None:
                route = torch.zeros(
                    batch, 1, 1, device=output.device, dtype=output.dtype
                )
            elif isinstance(tail_expert_route, (int, float)):
                route = torch.full(
                    (batch, 1, 1),
                    float(tail_expert_route),
                    device=output.device,
                    dtype=output.dtype,
                )
            else:
                route = tail_expert_route.to(device=output.device, dtype=output.dtype)
                if route.ndim == 1:
                    route = route[:, None, None]
            if route.shape != (batch, 1, 1):
                raise ValueError("tail_expert_route must be scalar, [B], or [B,1,1]")
            if torch.any(route < 0.0) or torch.any(route > 1.0):
                raise ValueError("tail_expert_route must be in [0,1]")
            if (
                self.tail_station_adapter is None
                or self.tail_common_adapter is None
                or self.tail_common_gate is None
            ):
                raise RuntimeError("body-tail expert adapters were not initialized")
            station_delta = self.tail_station_adapter(flattened).reshape(
                batch, stations, length
            )
            pooled = torch.einsum("s,bsct->bct", self.wind_capacity_weight, hidden)
            common_delta = self.tail_common_adapter(pooled)
            tail_delta = station_delta + (
                torch.sigmoid(self.tail_common_gate) * common_delta
            )
            tail_delta = tail_delta * self.wind_station_mask[None, :, None]
            if tail_time_mask is None:
                localized_route = route
            else:
                localized_route = tail_time_mask.to(
                    device=output.device, dtype=output.dtype
                )
                if localized_route.ndim == 2:
                    localized_route = localized_route[:, None, :]
                if localized_route.shape != (batch, 1, length):
                    raise ValueError("tail_time_mask must be [B,L] or [B,1,L]")
                if torch.any(localized_route < 0.0) or torch.any(
                    localized_route > 1.0
                ):
                    raise ValueError("tail_time_mask must be in [0,1]")
                localized_route = route * localized_route
            output = output + localized_route * tail_delta
        return output

    def _tail_risk_features(
        self,
        forecast: torch.Tensor,
        forecast_revision: torch.Tensor | None = None,
        revision_mask: torch.Tensor | None = None,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
        node_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return causal hourly tail features and signal attention weights."""

        if not self.use_body_tail_experts:
            raise RuntimeError("tail risk gate is disabled")
        if self.tail_signal_attention is None or self.tail_risk_encoder is None:
            raise RuntimeError("tail risk gate was not initialized")
        if forecast.ndim != 3 or forecast.shape[1:] != (
            self.station_count,
            self.sequence_length,
        ):
            raise ValueError("forecast must be [B,S,L] for tail routing")
        batch, _, length = forecast.shape
        weight = self.wind_capacity_weight.to(forecast.dtype)
        aggregate = torch.einsum("s,bst->bt", weight, forecast)
        ramp = torch.zeros_like(aggregate)
        ramp[:, 3:] = (aggregate[:, :-3] - aggregate[:, 3:]).clamp(min=0.0)

        revision = torch.zeros_like(aggregate)
        if forecast_revision is not None:
            revision_values = forecast_revision
            if revision_mask is not None:
                revision_values = revision_values * revision_mask
            revision = torch.einsum("s,bst->bt", weight, revision_values)

        low_state = torch.zeros_like(aggregate)
        down_state = torch.zeros_like(aggregate)
        if node_state is not None:
            if node_state.shape[:2] != forecast.shape[:2] or (
                node_state.shape[-1] != length or node_state.shape[2] < 4
            ):
                raise ValueError("node_state must be [B,S,F,L] with F>=4")
            low_state = torch.einsum("s,bst->bt", weight, node_state[:, :, 0])
            down_state = torch.einsum("s,bst->bt", weight, node_state[:, :, 3])

        recent = torch.zeros(batch, device=forecast.device, dtype=forecast.dtype)
        if recent_error is not None:
            recent_station = recent_error.mean(dim=-1)
            if recent_error_mask is not None:
                recent_station = recent_station * recent_error_mask[..., 0]
            recent = torch.einsum("s,bs->b", weight, recent_station)
        recent = recent[:, None].expand(-1, length)

        signals = torch.stack(
            [aggregate, ramp, revision, low_state, down_state, recent], dim=1
        )
        signal_statistics = torch.cat(
            [
                signals.mean(dim=-1),
                signals.var(dim=-1, unbiased=False).add(1e-6).sqrt(),
            ],
            dim=1,
        )
        attention = torch.softmax(
            self.tail_signal_attention(signal_statistics), dim=1
        )
        signals = signals * (6.0 * attention[:, :, None])
        return self.tail_risk_encoder(signals), attention

    def tail_risk_logits(
        self,
        forecast: torch.Tensor,
        forecast_revision: torch.Tensor | None = None,
        revision_mask: torch.Tensor | None = None,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
        node_state: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict an issuance-level tail probability from available conditions."""

        if self.tail_risk_output is None:
            raise RuntimeError("tail risk gate was not initialized")
        encoded, attention = self._tail_risk_features(
            forecast,
            forecast_revision=forecast_revision,
            revision_mask=revision_mask,
            recent_error=recent_error,
            recent_error_mask=recent_error_mask,
            node_state=node_state,
        )
        pooled = torch.cat([encoded.mean(dim=-1), encoded.amax(dim=-1)], dim=1)
        logits = self.tail_risk_output(pooled)[:, 0]
        if return_attention:
            return logits, attention
        return logits

    def tail_time_logits(
        self,
        forecast: torch.Tensor,
        forecast_revision: torch.Tensor | None = None,
        revision_mask: torch.Tensor | None = None,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
        node_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict conditional event-location logits for all lead hours."""

        if not self.use_tail_time_localizer or self.tail_time_output is None:
            raise RuntimeError("tail time localizer is disabled")
        encoded, _ = self._tail_risk_features(
            forecast,
            forecast_revision=forecast_revision,
            revision_mask=revision_mask,
            recent_error=recent_error,
            recent_error_mask=recent_error_mask,
            node_state=node_state,
        )
        return self.tail_time_output(encoded)[:, 0]

    def body_tail_parameter_names(self) -> tuple[str, ...]:
        if not self.use_body_tail_experts:
            return ()
        prefixes = (
            "tail_station_adapter.",
            "tail_common_adapter.",
            "tail_common_gate",
            "tail_risk_encoder.",
            "tail_risk_output.",
            "tail_signal_attention.",
            "tail_time_output.",
        )
        return tuple(
            name for name, _ in self.named_parameters() if name.startswith(prefixes)
        )

    def tail_time_parameter_names(self) -> tuple[str, ...]:
        if not self.use_tail_time_localizer:
            return ()
        return tuple(
            name
            for name, _ in self.named_parameters()
            if name.startswith("tail_time_output.")
        )

    def forecast_condition_dropout_statistics(self) -> dict[str, float | int]:
        total = int(self._forecast_condition_sample_count.detach().cpu())
        dropped = int(self._forecast_condition_drop_count.detach().cpu())
        return {
            "configured_probability": self.forecast_condition_dropout_prob,
            "observed_sample_count": total,
            "observed_drop_count": dropped,
            "observed_drop_rate": dropped / total if total else 0.0,
        }


class StationGaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoiser: StationConditionalResUNet1D,
        num_steps: int,
        beta_start: float,
        beta_end: float,
        ramp_auxiliary_loss_weight: float = 0.0,
        ramp_auxiliary_lags: tuple[int, ...] = (1, 3, 6),
        ramp_auxiliary_lag_weights: tuple[float, ...] = (0.5, 0.3, 0.2),
        wind_common_event_loss_weight: float = 0.0,
        wind_common_event_level_fraction: float = 0.5,
        event_x0_magnitude_loss_weight: float = 0.0,
        event_x0_timing_loss_weight: float = 0.0,
        event_x0_sync_loss_weight: float = 0.0,
        event_x0_window_hours: int = 6,
        event_x0_error_scale: float = 0.10,
        event_x0_timing_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.num_steps = int(num_steps)
        beta = torch.linspace(float(beta_start), float(beta_end), self.num_steps)
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)
        self.ramp_auxiliary_loss_weight = float(ramp_auxiliary_loss_weight)
        self.ramp_auxiliary_lags = tuple(int(value) for value in ramp_auxiliary_lags)
        self.ramp_auxiliary_lag_weights = tuple(
            float(value) for value in ramp_auxiliary_lag_weights
        )
        self.wind_common_event_loss_weight = float(
            wind_common_event_loss_weight
        )
        self.wind_common_event_level_fraction = float(
            wind_common_event_level_fraction
        )
        self.event_x0_magnitude_loss_weight = float(
            event_x0_magnitude_loss_weight
        )
        self.event_x0_timing_loss_weight = float(event_x0_timing_loss_weight)
        self.event_x0_sync_loss_weight = float(event_x0_sync_loss_weight)
        self.event_x0_window_hours = int(event_x0_window_hours)
        self.event_x0_error_scale = float(event_x0_error_scale)
        self.event_x0_timing_temperature = float(
            event_x0_timing_temperature
        )
        if self.ramp_auxiliary_loss_weight < 0:
            raise ValueError("ramp auxiliary loss weight must be non-negative")
        if self.wind_common_event_loss_weight < 0:
            raise ValueError("wind common event loss weight must be non-negative")
        if min(
            self.event_x0_magnitude_loss_weight,
            self.event_x0_timing_loss_weight,
            self.event_x0_sync_loss_weight,
        ) < 0:
            raise ValueError("event x0 loss weights must be non-negative")
        if not 1 <= self.event_x0_window_hours <= 24:
            raise ValueError("event x0 window must be in [1,24]")
        if self.event_x0_error_scale <= 0 or self.event_x0_timing_temperature <= 0:
            raise ValueError("event x0 scales must be positive")
        if not 0.0 <= self.wind_common_event_level_fraction <= 1.0:
            raise ValueError("wind common event level fraction must be in [0,1]")
        if (
            not self.ramp_auxiliary_lags
            or len(self.ramp_auxiliary_lags) != len(self.ramp_auxiliary_lag_weights)
            or any(lag <= 0 for lag in self.ramp_auxiliary_lags)
            or any(weight < 0 for weight in self.ramp_auxiliary_lag_weights)
            or sum(self.ramp_auxiliary_lag_weights) <= 0
        ):
            raise ValueError("invalid ramp auxiliary lags or lag weights")

    def add_noise(
        self,
        clean: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        alpha_hat = self.alpha_hat[timestep].view(-1, 1, 1)
        noisy = alpha_hat.sqrt() * clean + (1.0 - alpha_hat).sqrt() * noise
        return noisy, noise

    def training_loss(
        self,
        clean: torch.Tensor,
        forecast: torch.Tensor,
        calendar: torch.Tensor,
        lead: torch.Tensor,
        valid_mask: torch.Tensor,
        forecast_ramps: torch.Tensor | None = None,
        forecast_revision: torch.Tensor | None = None,
        revision_mask: torch.Tensor | None = None,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
        node_state: torch.Tensor | None = None,
        residual_scale: torch.Tensor | None = None,
        loss_weight: torch.Tensor | None = None,
        event_time_weight: torch.Tensor | None = None,
        event_active: torch.Tensor | None = None,
        event_start: torch.Tensor | None = None,
        event_window_mask: torch.Tensor | None = None,
        event_sync_station_weight: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        include_auxiliary: bool = True,
        forecast_condition_strength: torch.Tensor | float | None = None,
        tail_expert_route: torch.Tensor | float | None = None,
        tail_time_mask: torch.Tensor | None = None,
        epsilon_sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timestep is None:
            timestep = torch.randint(
                0, self.num_steps, (clean.shape[0],), device=clean.device
            )
        noisy, noise = self.add_noise(clean, timestep.long(), noise=noise)
        prediction = self.denoiser(
            noisy,
            timestep.long(),
            forecast,
            calendar,
            lead,
            forecast_ramps=forecast_ramps,
            forecast_revision=forecast_revision,
            revision_mask=revision_mask,
            recent_error=recent_error,
            recent_error_mask=recent_error_mask,
            node_state=node_state,
            forecast_condition_strength=forecast_condition_strength,
            tail_expert_route=tail_expert_route,
            tail_time_mask=tail_time_mask,
        )
        squared_error = (prediction - noise) ** 2
        valid_mask = valid_mask.to(squared_error.dtype)
        effective_mask = valid_mask
        if epsilon_sample_weight is not None:
            if epsilon_sample_weight.shape == (clean.shape[0],):
                epsilon_weight = epsilon_sample_weight[:, None, None]
            elif epsilon_sample_weight.shape == clean.shape:
                epsilon_weight = epsilon_sample_weight
            else:
                raise ValueError(
                    "epsilon_sample_weight must be [B] or match [B,S,L]"
                )
            effective_mask = effective_mask * epsilon_weight.to(
                squared_error.dtype
            )
        if include_auxiliary and loss_weight is not None:
            if loss_weight.shape != clean.shape:
                raise ValueError("loss_weight must match residual target [B,S,L]")
            effective_mask = effective_mask * loss_weight.to(squared_error.dtype)
        epsilon_loss = (squared_error * effective_mask).sum() / effective_mask.sum().clamp(min=1.0)
        needs_x0 = include_auxiliary and (
            self.ramp_auxiliary_loss_weight > 0
            or self.wind_common_event_loss_weight > 0
            or self.event_x0_magnitude_loss_weight > 0
            or self.event_x0_timing_loss_weight > 0
            or self.event_x0_sync_loss_weight > 0
        )
        if not needs_x0:
            return epsilon_loss
        if residual_scale is None or residual_scale.shape != clean.shape:
            raise ValueError(
                "residual_scale [B,S,L] is required for auxiliary losses"
            )

        alpha_hat = self.alpha_hat[timestep.long()].view(-1, 1, 1)
        predicted_clean = (
            noisy - (1.0 - alpha_hat).sqrt() * prediction
        ) / alpha_hat.sqrt().clamp(min=1e-6)
        physical_scale = residual_scale.to(predicted_clean.dtype)
        predicted_actual = forecast + predicted_clean * physical_scale
        target_actual = forecast + clean * physical_scale
        # Suppress unstable x0 reconstruction at very noisy diffusion steps.
        snr_weight = torch.sqrt(
            alpha_hat / (1.0 - alpha_hat).clamp(min=1e-6)
        ).clamp(max=1.0)
        lag_weight_sum = sum(self.ramp_auxiliary_lag_weights)
        ramp_loss = torch.zeros((), device=clean.device, dtype=clean.dtype)
        if self.ramp_auxiliary_loss_weight > 0:
            for lag, lag_weight in zip(
                self.ramp_auxiliary_lags, self.ramp_auxiliary_lag_weights
            ):
                if lag >= clean.shape[-1]:
                    raise ValueError(f"ramp auxiliary lag={lag} exceeds sequence length")
                predicted_ramp = predicted_actual[:, :, lag:] - predicted_actual[:, :, :-lag]
                target_ramp = target_actual[:, :, lag:] - target_actual[:, :, :-lag]
                pair_mask = valid_mask[:, :, lag:] * valid_mask[:, :, :-lag]
                weighted_error = (
                    torch.abs(predicted_ramp - target_ramp)
                    * pair_mask
                    * snr_weight
                )
                current = weighted_error.sum() / pair_mask.sum().clamp(min=1.0)
                ramp_loss = ramp_loss + float(lag_weight) * current
            ramp_loss = ramp_loss / float(lag_weight_sum)

        common_event_loss = torch.zeros((), device=clean.device, dtype=clean.dtype)
        if self.wind_common_event_loss_weight > 0:
            wind_weight = self.denoiser.wind_capacity_weight.to(clean.dtype)
            predicted_wind = torch.einsum("s,bst->bt", wind_weight, predicted_actual)
            target_wind = torch.einsum("s,bst->bt", wind_weight, target_actual)
            wind_valid = (
                valid_mask
                * self.denoiser.wind_station_mask[None, :, None]
            ).sum(dim=1) / self.denoiser.wind_station_mask.sum().clamp(min=1.0)
            time_weight = torch.ones_like(wind_valid)
            if event_time_weight is not None:
                if event_time_weight.shape != wind_valid.shape:
                    raise ValueError("event_time_weight must be [B,L]")
                time_weight = event_time_weight.to(clean.dtype)
            level_mask = wind_valid * time_weight * snr_weight[:, 0]
            level_error = F.smooth_l1_loss(
                predicted_wind, target_wind, reduction="none"
            )
            level_loss = (level_error * level_mask).sum() / level_mask.sum().clamp(min=1.0)
            common_ramp = torch.zeros_like(level_loss)
            for lag, lag_weight in zip(
                self.ramp_auxiliary_lags, self.ramp_auxiliary_lag_weights
            ):
                predicted_delta = predicted_wind[:, lag:] - predicted_wind[:, :-lag]
                target_delta = target_wind[:, lag:] - target_wind[:, :-lag]
                pair_mask = (
                    wind_valid[:, lag:]
                    * wind_valid[:, :-lag]
                    * torch.maximum(time_weight[:, lag:], time_weight[:, :-lag])
                    * snr_weight[:, 0]
                )
                error = F.smooth_l1_loss(
                    predicted_delta, target_delta, reduction="none"
                )
                common_ramp = common_ramp + float(lag_weight) * (
                    (error * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)
                )
            common_ramp = common_ramp / float(lag_weight_sum)
            fraction = self.wind_common_event_level_fraction
            common_event_loss = fraction * level_loss + (1.0 - fraction) * common_ramp

        event_magnitude_loss = torch.zeros(
            (), device=clean.device, dtype=clean.dtype
        )
        event_timing_loss = torch.zeros_like(event_magnitude_loss)
        event_sync_loss = torch.zeros_like(event_magnitude_loss)
        event_x0_enabled = (
            self.event_x0_magnitude_loss_weight > 0
            or self.event_x0_timing_loss_weight > 0
            or self.event_x0_sync_loss_weight > 0
        )
        if event_x0_enabled:
            batch, stations, length = clean.shape
            if (
                event_active is None
                or event_start is None
                or event_window_mask is None
                or event_sync_station_weight is None
            ):
                raise ValueError("event x0 loss requires replay labels in the training batch")
            if event_active.shape != (batch,) or event_start.shape != (batch,):
                raise ValueError("event_active/event_start must be [B]")
            if event_window_mask.shape != (batch, length):
                raise ValueError("event_window_mask must be [B,L]")
            if event_sync_station_weight.shape != (batch, stations):
                raise ValueError("event_sync_station_weight must be [B,S]")
            window = self.event_x0_window_hours
            if window >= length:
                raise ValueError("event x0 window must be shorter than sequence")

            predicted_residual = predicted_clean * physical_scale
            target_residual = clean * physical_scale
            wind_weight = self.denoiser.wind_capacity_weight.to(clean.dtype)
            predicted_wind = torch.einsum(
                "s,bst->bt", wind_weight, predicted_residual
            )
            target_wind = torch.einsum(
                "s,bst->bt", wind_weight, target_residual
            )
            wind_mask = self.denoiser.wind_station_mask.to(clean.dtype)
            wind_valid = (
                valid_mask * wind_mask[None, :, None]
            ).sum(dim=1) / wind_mask.sum().clamp(min=1.0)
            active_weight = (
                event_active.to(clean.dtype)
                * snr_weight[:, 0, 0]
            )

            window_mask = (
                event_window_mask.to(clean.dtype) * wind_valid
            )
            window_count = window_mask.sum(dim=1).clamp(min=1.0)
            predicted_level = (predicted_wind * window_mask).sum(dim=1) / window_count
            target_level = (target_wind * window_mask).sum(dim=1) / window_count
            magnitude_error = torch.abs(predicted_level - target_level) / self.event_x0_error_scale
            event_magnitude_loss = (
                magnitude_error * active_weight
            ).sum() / active_weight.sum().clamp(min=1.0)

            predicted_curve = -F.avg_pool1d(
                predicted_wind[:, None, :], kernel_size=window, stride=1
            )[:, 0]
            rolling_valid = F.avg_pool1d(
                wind_valid[:, None, :], kernel_size=window, stride=1
            )[:, 0]
            timing_logits = (
                predicted_curve / self.event_x0_timing_temperature
            ).clamp(min=-30.0, max=30.0)
            timing_logits = timing_logits.masked_fill(rolling_valid < 0.999, -1.0e4)
            target_start = event_start.long().clamp(
                min=0, max=timing_logits.shape[1] - 1
            )
            timing_error = F.cross_entropy(
                timing_logits, target_start, reduction="none"
            )
            event_timing_loss = (
                timing_error * active_weight
            ).sum() / active_weight.sum().clamp(min=1.0)

            station_window_mask = (
                valid_mask * event_window_mask[:, None, :].to(clean.dtype)
            )
            station_window_count = station_window_mask.sum(dim=2).clamp(min=1.0)
            predicted_station_level = (
                predicted_residual * station_window_mask
            ).sum(dim=2) / station_window_count
            target_station_level = (
                target_residual * station_window_mask
            ).sum(dim=2) / station_window_count
            sync_weight = (
                event_sync_station_weight.to(clean.dtype)
                * wind_mask[None]
                * active_weight[:, None]
            )
            sync_error = torch.abs(
                predicted_station_level - target_station_level
            ) / self.event_x0_error_scale
            event_sync_loss = (
                sync_error * sync_weight
            ).sum() / sync_weight.sum().clamp(min=1.0)

        return (
            epsilon_loss
            + self.ramp_auxiliary_loss_weight * ramp_loss
            + self.wind_common_event_loss_weight * common_event_loss
            + self.event_x0_magnitude_loss_weight * event_magnitude_loss
            + self.event_x0_timing_loss_weight * event_timing_loss
            + self.event_x0_sync_loss_weight * event_sync_loss
        )

    def reverse_variance(self, timestep: torch.Tensor) -> torch.Tensor:
        beta = self.beta[timestep]
        previous = torch.clamp(timestep - 1, min=0)
        posterior = beta * (1.0 - self.alpha_hat[previous]) / (
            1.0 - self.alpha_hat[timestep]
        )
        return torch.where(
            timestep > 0, posterior.clamp(min=0.0), torch.zeros_like(posterior)
        )

    def denoise_step(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        forecast: torch.Tensor,
        calendar: torch.Tensor,
        lead: torch.Tensor,
        forecast_ramps: torch.Tensor | None = None,
        forecast_revision: torch.Tensor | None = None,
        revision_mask: torch.Tensor | None = None,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
        node_state: torch.Tensor | None = None,
        forecast_guidance_scale: float = 1.0,
        tail_expert_route: torch.Tensor | float | None = None,
        tail_time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forecast_guidance_scale = float(forecast_guidance_scale)
        if not 0.0 <= forecast_guidance_scale <= 1.0:
            raise ValueError("forecast_guidance_scale must be in [0,1]")
        denoiser_kwargs = dict(
            forecast_ramps=forecast_ramps,
            forecast_revision=forecast_revision,
            revision_mask=revision_mask,
            recent_error=recent_error,
            recent_error_mask=recent_error_mask,
            node_state=node_state,
        )
        predicted_conditional = self.denoiser(
            noisy,
            timestep,
            forecast,
            calendar,
            lead,
            forecast_condition_strength=1.0,
            tail_expert_route=tail_expert_route,
            tail_time_mask=tail_time_mask,
            **denoiser_kwargs,
        )
        if forecast_guidance_scale == 1.0:
            predicted_noise = predicted_conditional
        else:
            predicted_neutral = self.denoiser(
                noisy,
                timestep,
                forecast,
                calendar,
                lead,
                forecast_condition_strength=0.0,
                tail_expert_route=tail_expert_route,
                tail_time_mask=tail_time_mask,
                **denoiser_kwargs,
            )
            predicted_noise = predicted_neutral + forecast_guidance_scale * (
                predicted_conditional - predicted_neutral
            )
        alpha = self.alpha[timestep].view(-1, 1, 1)
        alpha_hat = self.alpha_hat[timestep].view(-1, 1, 1)
        coefficient = (1.0 - alpha) / (1.0 - alpha_hat).sqrt()
        mean = (noisy - coefficient * predicted_noise) / alpha.sqrt()
        variance = self.reverse_variance(timestep).view(-1, 1, 1)
        random_noise = torch.randn_like(noisy)
        nonzero = (timestep > 0).to(noisy.dtype).view(-1, 1, 1)
        return mean + nonzero * variance.sqrt() * random_noise

    @torch.no_grad()
    def sample(
        self,
        forecast: torch.Tensor,
        calendar: torch.Tensor,
        lead: torch.Tensor,
        n_samples: int,
        forecast_ramps: torch.Tensor | None = None,
        forecast_revision: torch.Tensor | None = None,
        revision_mask: torch.Tensor | None = None,
        recent_error: torch.Tensor | None = None,
        recent_error_mask: torch.Tensor | None = None,
        node_state: torch.Tensor | None = None,
        forecast_guidance_scale: float = 1.0,
        tail_expert_route: torch.Tensor | None = None,
        tail_time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, stations, length = forecast.shape
        n_samples = int(n_samples)
        forecast = forecast.repeat_interleave(n_samples, dim=0)
        calendar = calendar.repeat_interleave(n_samples, dim=0)
        lead = lead.repeat_interleave(n_samples, dim=0)
        optional_conditions = {
            "forecast_ramps": forecast_ramps,
            "forecast_revision": forecast_revision,
            "revision_mask": revision_mask,
            "recent_error": recent_error,
            "recent_error_mask": recent_error_mask,
            "node_state": node_state,
        }
        optional_conditions = {
            name: (
                value.repeat_interleave(n_samples, dim=0)
                if value is not None
                else None
            )
            for name, value in optional_conditions.items()
        }
        if tail_expert_route is not None:
            route = tail_expert_route.to(device=forecast.device, dtype=forecast.dtype)
            if route.shape == (batch, n_samples):
                route = route.reshape(batch * n_samples)
            elif route.shape != (batch * n_samples,):
                raise ValueError(
                    "tail_expert_route must be [B,K] or [B*K] during sampling"
                )
        else:
            route = None
        if tail_time_mask is not None:
            time_mask = tail_time_mask.to(
                device=forecast.device, dtype=forecast.dtype
            )
            if time_mask.shape == (batch, n_samples, length):
                time_mask = time_mask.reshape(batch * n_samples, length)
            elif time_mask.shape != (batch * n_samples, length):
                raise ValueError(
                    "tail_time_mask must be [B,K,L] or [B*K,L] during sampling"
                )
        else:
            time_mask = None
        noisy = torch.randn(
            batch * n_samples,
            stations,
            length,
            device=forecast.device,
        )
        for step in range(self.num_steps - 1, -1, -1):
            timestep = torch.full(
                (batch * n_samples,), step, device=forecast.device, dtype=torch.long
            )
            noisy = self.denoise_step(
                noisy,
                timestep,
                forecast,
                calendar,
                lead,
                forecast_guidance_scale=forecast_guidance_scale,
                tail_expert_route=route,
                tail_time_mask=time_mask,
                **optional_conditions,
            )
        return noisy.reshape(batch, n_samples, stations, length)


class Station24DiffusionModel(nn.Module):
    """High-level wrapper used by the station-specific trainer and generator."""

    architecture = "station24_resunet"

    def __init__(
        self,
        config: Mapping[str, object],
        station_features: torch.Tensor,
        adjacency: torch.Tensor,
        station_capacities: torch.Tensor | None = None,
        secondary_adjacency: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.use_body_tail_experts = bool(
            self.config.get("use_body_tail_experts", False)
        )
        self.use_tail_time_localizer = bool(
            self.config.get("use_tail_time_localizer", False)
        )
        self.train_tail_time_localizer_only = bool(
            self.config.get("train_tail_time_localizer_only", False)
        )
        if self.train_tail_time_localizer_only and not self.use_tail_time_localizer:
            raise ValueError(
                "train_tail_time_localizer_only requires use_tail_time_localizer"
            )
        self.tail_epsilon_context_hours = int(
            self.config.get("tail_epsilon_context_hours", 6)
        )
        if self.tail_epsilon_context_hours < 0 or self.tail_epsilon_context_hours > 24:
            raise ValueError("tail_epsilon_context_hours must be in [0,24]")
        self.tail_gate_loss_weight = float(
            self.config.get("tail_gate_loss_weight", 0.0)
        )
        if self.use_body_tail_experts and self.tail_gate_loss_weight <= 0.0:
            raise ValueError(
                "tail_gate_loss_weight must be positive for body-tail experts"
            )
        if not self.use_body_tail_experts and self.tail_gate_loss_weight != 0.0:
            raise ValueError(
                "tail_gate_loss_weight must be zero when body-tail experts are disabled"
            )
        self.forecast_correction_mode = (
            str(self.config.get("forecast_correction_mode", "direct"))
            if bool(self.config.get("use_forecast_correction", False))
            else "none"
        )
        self.forecast_correction_loss_weight = float(
            self.config.get("forecast_correction_loss_weight", 0.0)
        )
        self.forecast_correction_huber_beta = float(
            self.config.get("forecast_correction_huber_beta", 0.05)
        )
        if self.forecast_correction_mode != "none":
            if self.forecast_correction_loss_weight <= 0:
                raise ValueError(
                    "forecast_correction_loss_weight must be positive when enabled"
                )
            if self.forecast_correction_huber_beta <= 0:
                raise ValueError("forecast_correction_huber_beta must be positive")
            self.forecast_correction_head: StationForecastCorrectionHead | None = (
                StationForecastCorrectionHead(
                    self.config,
                    station_features,
                    int(self.config.get("group_norm_groups", 8)),
                )
            )
        else:
            if self.forecast_correction_loss_weight != 0:
                raise ValueError(
                    "forecast correction loss weight must be zero when disabled"
                )
            self.forecast_correction_head = None
        self.denoiser = StationConditionalResUNet1D(
            self.config,
            station_features,
            adjacency,
            station_capacities,
            secondary_adjacency,
        )
        self.diffusion = StationGaussianDiffusion(
            self.denoiser,
            num_steps=int(self.config.get("num_steps", 500)),
            beta_start=float(self.config.get("beta_start", 1e-4)),
            beta_end=float(self.config.get("beta_end", 0.04)),
            ramp_auxiliary_loss_weight=float(
                self.config.get("ramp_auxiliary_loss_weight", 0.0)
            ),
            ramp_auxiliary_lags=tuple(
                int(value)
                for value in self.config.get("ramp_auxiliary_lags", [1, 3, 6])
            ),
            ramp_auxiliary_lag_weights=tuple(
                float(value)
                for value in self.config.get(
                    "ramp_auxiliary_lag_weights", [0.5, 0.3, 0.2]
                )
            ),
            wind_common_event_loss_weight=float(
                self.config.get("wind_common_event_loss_weight", 0.0)
            ),
            wind_common_event_level_fraction=float(
                self.config.get("wind_common_event_level_fraction", 0.5)
            ),
            event_x0_magnitude_loss_weight=float(
                self.config.get("event_x0_magnitude_loss_weight", 0.0)
            ),
            event_x0_timing_loss_weight=float(
                self.config.get("event_x0_timing_loss_weight", 0.0)
            ),
            event_x0_sync_loss_weight=float(
                self.config.get("event_x0_sync_loss_weight", 0.0)
            ),
            event_x0_window_hours=int(
                self.config.get("event_x0_window_hours", 6)
            ),
            event_x0_error_scale=float(
                self.config.get("event_x0_error_scale", 0.10)
            ),
            event_x0_timing_temperature=float(
                self.config.get("event_x0_timing_temperature", 0.05)
            ),
        )

    @property
    def spatial_mode(self) -> str:
        return self.denoiser.spatial_mode

    @property
    def spatial_mix_levels(self) -> tuple[str, ...]:
        return self.denoiser.spatial_mix_levels

    @property
    def parallel_spatial_fusion_levels(self) -> tuple[str, ...]:
        return self.denoiser.parallel_spatial_fusion_levels

    @property
    def parallel_spatial_adjacency_mode(self) -> str:
        return self.denoiser.parallel_spatial_adjacency_mode

    @property
    def spatial_gate_values(self) -> dict[str, float]:
        if self.denoiser.spatial_mix_levels == ("bottleneck",):
            # Preserve the legacy metadata shape for old single-scale runs.
            return self.denoiser.spatial_block.gate_values()
        values: dict[str, float] = {}
        for level, block in self.denoiser.encoder_spatial_blocks.items():
            for relation, value in block.gate_values().items():
                values[f"{level}/{relation}"] = value
        if "bottleneck" in self.denoiser.spatial_mix_levels:
            for relation, value in self.denoiser.spatial_block.gate_values().items():
                values[f"bottleneck/{relation}"] = value
        return values

    @property
    def parallel_spatial_gate_statistics(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for level, block in self.denoiser.parallel_spatial_blocks.items():
            for name, value in block.gate_statistics().items():
                values[f"{level}/{name}"] = value
        return values

    def reset_parallel_spatial_gate_statistics(self) -> None:
        for block in self.denoiser.parallel_spatial_blocks.values():
            block.reset_gate_statistics()

    @property
    def parallel_spatial_adjacency_moments(
        self,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return {
            level: moments
            for level, block in self.denoiser.parallel_spatial_blocks.items()
            if (moments := block.adjacency_moments())
        }

    @property
    def condition_gate_values(self) -> dict[str, float]:
        return self.denoiser.condition_encoder.gate_values()

    @property
    def forecast_condition_dropout_statistics(self) -> dict[str, float | int]:
        return self.denoiser.forecast_condition_dropout_statistics()

    @property
    def state_gate_values(self) -> dict[str, float]:
        if not self.denoiser.use_state_encoder:
            return {}
        values = self.denoiser.state_encoder.gate_values()
        blocks = list(self.denoiser.encoder_blocks)
        blocks.append(self.denoiser.bottleneck)
        blocks.extend(self.denoiser.decoder_blocks)
        for index, block in enumerate(blocks):
            values[f"film_block_{index}"] = float(
                torch.sigmoid(block.state_gate.detach()).cpu()
            )
        return values

    @property
    def wind_common_gate_value(self) -> float | None:
        gate = self.denoiser.wind_common_gate
        if gate is None:
            return None
        return float(torch.sigmoid(gate.detach()).cpu())

    @property
    def body_tail_trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"denoiser.{name}" for name in self.denoiser.body_tail_parameter_names()
        )

    @property
    def body_tail_state_dict_keys(self) -> tuple[str, ...]:
        """All serialized aliases for the shared denoiser's tail parameters."""

        parameter_suffixes = self.denoiser.body_tail_parameter_names()
        prefixes = ("denoiser.", "diffusion.denoiser.")
        return tuple(
            f"{prefix}{suffix}"
            for prefix in prefixes
            for suffix in parameter_suffixes
        )

    @property
    def tail_time_trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"denoiser.{name}" for name in self.denoiser.tail_time_parameter_names()
        )

    @property
    def tail_time_state_dict_keys(self) -> tuple[str, ...]:
        """All serialized aliases for the hourly tail-location head."""

        parameter_suffixes = self.denoiser.tail_time_parameter_names()
        prefixes = ("denoiser.", "diffusion.denoiser.")
        return tuple(
            f"{prefix}{suffix}"
            for prefix in prefixes
            for suffix in parameter_suffixes
        )

    @property
    def tail_common_gate_value(self) -> float | None:
        gate = self.denoiser.tail_common_gate
        if gate is None:
            return None
        return float(torch.sigmoid(gate.detach()).cpu())

    def configure_body_tail_training(self) -> tuple[str, ...]:
        """Freeze the body model and leave only tail adapters/gate trainable."""

        if not self.use_body_tail_experts:
            raise RuntimeError("body-tail expert mode is disabled")
        allowed = set(self.body_tail_trainable_parameter_names)
        if not allowed:
            raise RuntimeError("body-tail expert has no trainable parameters")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in allowed)
        actual = {name for name, value in self.named_parameters() if value.requires_grad}
        if actual != allowed:
            raise RuntimeError("body-tail parameter isolation failed")
        return tuple(sorted(actual))

    def configure_tail_time_training(self) -> tuple[str, ...]:
        """Freeze inherited Raw body/tail/gate and train only time localization."""

        if not self.use_tail_time_localizer:
            raise RuntimeError("tail time localizer is disabled")
        allowed = set(self.tail_time_trainable_parameter_names)
        if not allowed:
            raise RuntimeError("tail time localizer has no trainable parameters")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in allowed)
        actual = {name for name, value in self.named_parameters() if value.requires_grad}
        if actual != allowed:
            raise RuntimeError("tail time parameter isolation failed")
        return tuple(sorted(actual))

    def tail_risk_logits(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        logits = self.denoiser.tail_risk_logits(
            batch["forecast"],
            forecast_revision=batch.get("forecast_revision"),
            revision_mask=batch.get("revision_mask"),
            recent_error=batch.get("recent_error"),
            recent_error_mask=batch.get("recent_error_mask"),
            node_state=batch.get("node_state"),
        )
        if isinstance(logits, tuple):
            raise RuntimeError("tail risk logits unexpectedly returned attention")
        return logits

    def tail_condition_attention(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        result = self.denoiser.tail_risk_logits(
            batch["forecast"],
            forecast_revision=batch.get("forecast_revision"),
            revision_mask=batch.get("revision_mask"),
            recent_error=batch.get("recent_error"),
            recent_error_mask=batch.get("recent_error_mask"),
            node_state=batch.get("node_state"),
            return_attention=True,
        )
        if not isinstance(result, tuple):
            raise RuntimeError("tail condition attention was not returned")
        return result[1]

    def tail_risk_probability(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.sigmoid(self.tail_risk_logits(batch))

    def tail_time_logits(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.denoiser.tail_time_logits(
            batch["forecast"],
            forecast_revision=batch.get("forecast_revision"),
            revision_mask=batch.get("revision_mask"),
            recent_error=batch.get("recent_error"),
            recent_error_mask=batch.get("recent_error_mask"),
            node_state=batch.get("node_state"),
        )

    def tail_time_probability(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        logits = self.tail_time_logits(batch) / self.denoiser.tail_time_temperature
        return torch.softmax(logits, dim=-1)

    def tail_time_localization_loss(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Soft cross entropy on train-labelled physical event windows only."""

        event_active = batch.get("event_active")
        event_window = batch.get("event_window_mask")
        if event_active is None or event_window is None:
            raise ValueError("tail time training requires event replay labels")
        logits = self.tail_time_logits(batch) / self.denoiser.tail_time_temperature
        if event_active.shape != (logits.shape[0],) or event_window.shape != logits.shape:
            raise ValueError("tail time event labels have invalid shapes")
        target = event_window.to(logits.dtype)
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=1.0)
        per_sample = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        active = event_active.to(per_sample.dtype)
        return (per_sample * active).sum() / active.sum().clamp(min=1.0)

    def sample_tail_time_masks(
        self,
        probability: torch.Tensor,
        tail_route: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one synchronized wind-event center and tapered mask per member."""

        if probability.ndim != 2:
            raise ValueError("tail time probability must be [B,L]")
        if tail_route.ndim != 2 or tail_route.shape[0] != probability.shape[0]:
            raise ValueError("tail route must be [B,K]")
        batch, length = probability.shape
        members = tail_route.shape[1]
        starts = torch.multinomial(probability, members, replacement=True)
        lead_index = torch.arange(
            length, device=probability.device, dtype=probability.dtype
        )[None, None, :]
        distance = torch.abs(lead_index - starts.to(probability.dtype)[..., None])
        radius = float(self.denoiser.tail_time_mask_radius_hours)
        mask = torch.where(
            distance <= radius,
            0.5 * (1.0 + torch.cos(math.pi * distance / (radius + 1.0))),
            torch.zeros_like(distance),
        )
        route = tail_route.to(mask.dtype)
        mask = mask * route[..., None]
        starts = torch.where(
            tail_route.bool(), starts, torch.full_like(starts, -1)
        )
        if mask.shape != (batch, members, length):
            raise RuntimeError("tail time mask construction failed")
        return starts, mask

    def body_tail_epsilon_weight(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Select wind-event support for the specialized tail denoising loss."""

        event_active = batch.get("event_active")
        event_window = batch.get("event_window_mask")
        if event_active is None or event_window is None:
            raise ValueError(
                "body-tail training requires event_active and event_window_mask"
            )
        batch_size = batch["forecast"].shape[0]
        if event_active.shape != (batch_size,) or event_window.shape != (
            batch_size,
            self.denoiser.sequence_length,
        ):
            raise ValueError("body-tail event targets have invalid shapes")
        support = event_window.to(batch["forecast"].dtype)[:, None, :]
        context = self.tail_epsilon_context_hours
        if context > 0:
            support = F.max_pool1d(
                support,
                kernel_size=2 * context + 1,
                stride=1,
                padding=context,
            )
        wind = self.denoiser.wind_station_mask.to(support.dtype)[None, :, None]
        return support * wind * event_active.to(support.dtype)[:, None, None]

    def predict_forecast_correction(
        self, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return the causal correction center in normalized power units."""

        forecast = batch["forecast"]
        if self.forecast_correction_head is None:
            return torch.zeros_like(forecast)
        return self.forecast_correction_head(
            forecast,
            batch["calendar"],
            batch["lead"],
            forecast_revision=batch.get("forecast_revision"),
            revision_mask=batch.get("revision_mask"),
            recent_error=batch.get("recent_error"),
            recent_error_mask=batch.get("recent_error_mask"),
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        include_auxiliary: bool = True,
        forecast_condition_strength: torch.Tensor | float | None = None,
        body_tail_event_masking: bool = False,
    ) -> torch.Tensor:
        correction = self.predict_forecast_correction(batch)
        clean = batch["residual_target"]
        condition_forecast = batch["forecast"]
        if self.forecast_correction_head is not None:
            scale = batch.get("residual_scale")
            if scale is None or scale.shape != correction.shape:
                raise ValueError(
                    "forecast correction requires residual_scale [B,S,L]"
                )
            # The correction head is trained by its explicit supervised loss.
            # Detaching here prevents an unidentifiable trade between the
            # deterministic center and the stochastic diffusion residual.
            detached = correction.detach()
            clean = clean - detached / scale
            condition_forecast = condition_forecast + detached
        tail_route: torch.Tensor | float | None = None
        epsilon_sample_weight: torch.Tensor | None = None
        if self.use_body_tail_experts:
            # Training isolates tail gradients to replayed physical events.  The
            # fixed-noise validation pass evaluates the complete tail denoiser on
            # natural validation data, preventing an unconstrained event adapter.
            tail_route = 1.0
            if include_auxiliary or body_tail_event_masking:
                epsilon_sample_weight = self.body_tail_epsilon_weight(batch)
        diffusion_loss = self.diffusion.training_loss(
            clean,
            condition_forecast,
            batch["calendar"],
            batch["lead"],
            batch["valid_mask"],
            forecast_ramps=batch.get("forecast_ramps"),
            forecast_revision=batch.get("forecast_revision"),
            revision_mask=batch.get("revision_mask"),
            recent_error=batch.get("recent_error"),
            recent_error_mask=batch.get("recent_error_mask"),
            node_state=batch.get("node_state"),
            residual_scale=batch.get("residual_scale"),
            loss_weight=batch.get("loss_weight"),
            event_time_weight=batch.get("event_time_weight"),
            event_active=batch.get("event_active"),
            event_start=batch.get("event_start"),
            event_window_mask=batch.get("event_window_mask"),
            event_sync_station_weight=batch.get(
                "event_sync_station_weight"
            ),
            timestep=timestep,
            noise=noise,
            include_auxiliary=include_auxiliary,
            forecast_condition_strength=forecast_condition_strength,
            tail_expert_route=tail_route,
            epsilon_sample_weight=epsilon_sample_weight,
        )
        if self.use_body_tail_experts and include_auxiliary:
            target = batch["event_active"].to(diffusion_loss.dtype)
            logits = self.tail_risk_logits(batch)
            replay_weight = batch.get("event_replay_weight")
            if replay_weight is None or replay_weight.shape != target.shape:
                raise ValueError(
                    "body-tail gate training requires event_replay_weight [B]"
                )
            # WeightedRandomSampler changes the observed class prior.  Inverse
            # replay weighting recovers the natural issuance distribution for
            # the causal gate instead of calibrating it to the oversampled data.
            importance = replay_weight.to(diffusion_loss.dtype).reciprocal()
            gate_error = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            gate_loss = (gate_error * importance).sum() / importance.sum().clamp(
                min=1.0
            )
            diffusion_loss = diffusion_loss + self.tail_gate_loss_weight * gate_loss
        if self.forecast_correction_head is None:
            return diffusion_loss
        correction_error = F.smooth_l1_loss(
            correction,
            batch["residual"],
            reduction="none",
            beta=self.forecast_correction_huber_beta,
        )
        valid = batch["valid_mask"].to(correction_error.dtype)
        correction_loss = (correction_error * valid).sum() / valid.sum().clamp(
            min=1.0
        )
        return diffusion_loss + self.forecast_correction_loss_weight * correction_loss

    def generate(
        self,
        batch: Mapping[str, torch.Tensor],
        n_samples: int,
        forecast_guidance_scale: float = 1.0,
        return_expert_audit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        correction = self.predict_forecast_correction(batch)
        tail_probability = None
        tail_route = None
        tail_time_probability = None
        tail_time_start = None
        tail_time_mask = None
        if self.use_body_tail_experts:
            tail_probability = self.tail_risk_probability(batch)
            tail_attention = self.tail_condition_attention(batch)
            tail_route = torch.bernoulli(
                tail_probability[:, None].expand(-1, int(n_samples))
            )
            if self.use_tail_time_localizer:
                tail_time_probability = self.tail_time_probability(batch)
                tail_time_start, tail_time_mask = self.sample_tail_time_masks(
                    tail_time_probability, tail_route
                )
        else:
            tail_attention = None
        samples = self.diffusion.sample(
            batch["forecast"] + correction,
            batch["calendar"],
            batch["lead"],
            n_samples,
            forecast_guidance_scale=forecast_guidance_scale,
            forecast_ramps=batch.get("forecast_ramps"),
            forecast_revision=batch.get("forecast_revision"),
            revision_mask=batch.get("revision_mask"),
            recent_error=batch.get("recent_error"),
            recent_error_mask=batch.get("recent_error_mask"),
            node_state=batch.get("node_state"),
            tail_expert_route=tail_route,
            tail_time_mask=tail_time_mask,
        )
        if not return_expert_audit:
            return samples
        batch_size = batch["forecast"].shape[0]
        if tail_probability is None or tail_route is None:
            tail_probability = torch.zeros(
                batch_size,
                device=samples.device,
                dtype=samples.dtype,
            )
            tail_route = torch.zeros(
                batch_size,
                int(n_samples),
                device=samples.device,
                dtype=samples.dtype,
            )
            tail_attention = torch.zeros(
                batch_size,
                6,
                device=samples.device,
                dtype=samples.dtype,
            )
        if tail_time_probability is None:
            tail_time_probability = torch.zeros(
                batch_size,
                self.denoiser.sequence_length,
                device=samples.device,
                dtype=samples.dtype,
            )
        if tail_time_start is None:
            tail_time_start = torch.full(
                (batch_size, int(n_samples)),
                -1,
                device=samples.device,
                dtype=torch.long,
            )
        if tail_attention is None:
            raise RuntimeError("tail condition attention audit is unavailable")
        return samples, {
            "tail_probability": tail_probability,
            "tail_route": tail_route,
            "tail_condition_attention": tail_attention,
            "tail_time_probability": tail_time_probability,
            "tail_time_start": tail_time_start,
        }
