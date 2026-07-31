"""Lead-aware conditional ResUNet diffusion for 24 wind/solar stations.

Stations remain an explicit spatial axis.  Temporal convolutions share weights
across stations; optional graph propagation is applied once at the bottleneck.
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


def _flatten_stations(value: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if value.ndim != 4:
        raise ValueError(f"expected [B,S,C,L], got {tuple(value.shape)}")
    batch, stations, channels, length = value.shape
    return value.reshape(batch * stations, channels, length), batch, stations


def _restore_stations(value: torch.Tensor, batch: int, stations: int) -> torch.Tensor:
    return value.reshape(batch, stations, value.shape[1], value.shape[2])


class StationConditionEncoder(nn.Module):
    """Encode local forecast, temporal marks, lead marks, and station metadata."""

    def __init__(
        self,
        channels: Sequence[int],
        station_count: int,
        station_feature_dim: int,
        groups: int,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in channels)
        stem_channels = channels[0]
        self.station_count = int(station_count)
        self.forecast_stem = nn.Conv1d(1, stem_channels, kernel_size=3, padding=1)
        self.temporal_stem = nn.Conv1d(10, stem_channels, kernel_size=3, padding=1)
        self.station_feature_projection = nn.Linear(station_feature_dim, stem_channels)
        self.station_embedding = nn.Embedding(self.station_count, stem_channels)
        self.fuse = nn.Sequential(
            nn.Conv1d(stem_channels * 3, stem_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(stem_channels, groups), stem_channels),
            nn.SiLU(),
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

        local = self.forecast_stem(forecast.reshape(batch * stations, 1, length))
        temporal = self.temporal_stem(torch.cat([calendar, lead], dim=1))
        temporal = temporal[:, None, :, :].expand(-1, stations, -1, -1)
        temporal = temporal.reshape(batch * stations, temporal.shape[2], length)

        station_index = torch.arange(stations, device=forecast.device)
        station = self.station_feature_projection(station_features)
        station = station + self.station_embedding(station_index)
        station = station[None, :, :, None].expand(batch, -1, -1, length)
        station = station.reshape(batch * stations, station.shape[2], length)

        feature = self.fuse(torch.cat([local, temporal, station], dim=1))
        outputs = [_restore_stations(feature, batch, stations)]
        for block in self.down_blocks:
            feature = block(feature)
            outputs.append(_restore_stations(feature, batch, stations))
        return outputs


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
    ) -> None:
        super().__init__()
        if mode not in SPATIAL_MODES:
            raise ValueError(f"unsupported spatial mode={mode!r}")
        self.mode = mode
        normalized = _normalize_adjacency(adjacency)
        self.register_buffer("normalized_adjacency", normalized)
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
            return {"all": float(torch.sigmoid(self.graph_gate.detach()).cpu())}
        return {
            name: float(torch.sigmoid(value.detach()).cpu())
            for name, value in self.relation_gates.items()
        }

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return value
        if self.mode == "fixed_graph":
            message = torch.einsum(
                "ij,bjct->bict", self.normalized_adjacency, value
            )
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


class StationConditionalResUNet1D(nn.Module):
    """Conditional temporal ResUNet with an explicit station axis."""

    def __init__(
        self,
        config: Mapping[str, object],
        station_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        self.sequence_length = int(config.get("sequence_length", 168))
        self.station_count = int(config.get("station_count", 24))
        self.spatial_mode = str(config.get("spatial_mode", "none"))
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

        self.register_buffer("station_features", station_features.float())
        self.timestep_embedding = DiffusionTimestepEmbedding(
            timestep_dim, timestep_dim
        )
        self.condition_encoder = StationConditionEncoder(
            self.channels,
            self.station_count,
            int(station_features.shape[1]),
            groups,
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
        self.bottleneck = StationResBlock(
            self.channels[-1],
            self.channels[-1],
            timestep_dim,
            self.channels[-1],
            groups,
            dropout,
        )
        self.spatial_block = StationSpatialBlock(
            self.channels[-1],
            adjacency,
            station_features,
            self.spatial_mode,
            groups,
            dropout,
            gate_init=float(config.get("spatial_gate_init", -1.0)),
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
                )
            )
            current_channels = self.channels[level]
        self.output_norm = nn.GroupNorm(
            _group_count(self.channels[0], groups), self.channels[0]
        )
        self.output = nn.Conv1d(self.channels[0], 1, kernel_size=1)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timestep: torch.Tensor,
        forecast: torch.Tensor,
        calendar: torch.Tensor,
        lead: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_residual.shape[1:] != (self.station_count, self.sequence_length):
            raise ValueError(
                "noisy residual must be [B,S,L], got "
                f"{tuple(noisy_residual.shape)}"
            )
        batch, stations, length = noisy_residual.shape
        time_embedding = self.timestep_embedding(timestep)
        conditions = self.condition_encoder(
            forecast, calendar, lead, self.station_features
        )
        hidden = self.state_stem(noisy_residual.reshape(batch * stations, 1, length))
        hidden = _restore_stations(hidden, batch, stations)
        skips = []
        for level, block in enumerate(self.encoder_blocks):
            hidden = block(hidden, time_embedding, conditions[level])
            skips.append(hidden)
            if level < len(self.downsamples):
                flattened, _, _ = _flatten_stations(hidden)
                hidden = _restore_stations(
                    self.downsamples[level](flattened), batch, stations
                )
        hidden = self.bottleneck(hidden, time_embedding, conditions[-1])
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
            )
        flattened, _, _ = _flatten_stations(hidden)
        output = self.output(F.silu(self.output_norm(flattened)))
        return output.reshape(batch, stations, length)


class StationGaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoiser: StationConditionalResUNet1D,
        num_steps: int,
        beta_start: float,
        beta_end: float,
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
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timestep is None:
            timestep = torch.randint(
                0, self.num_steps, (clean.shape[0],), device=clean.device
            )
        noisy, noise = self.add_noise(clean, timestep.long(), noise=noise)
        prediction = self.denoiser(noisy, timestep.long(), forecast, calendar, lead)
        squared_error = (prediction - noise) ** 2
        valid_mask = valid_mask.to(squared_error.dtype)
        return (squared_error * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

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
    ) -> torch.Tensor:
        predicted_noise = self.denoiser(
            noisy, timestep, forecast, calendar, lead
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
    ) -> torch.Tensor:
        batch, stations, length = forecast.shape
        n_samples = int(n_samples)
        forecast = forecast.repeat_interleave(n_samples, dim=0)
        calendar = calendar.repeat_interleave(n_samples, dim=0)
        lead = lead.repeat_interleave(n_samples, dim=0)
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
            noisy = self.denoise_step(noisy, timestep, forecast, calendar, lead)
        return noisy.reshape(batch, n_samples, stations, length)


class Station24DiffusionModel(nn.Module):
    """High-level wrapper used by the station-specific trainer and generator."""

    architecture = "station24_resunet"

    def __init__(
        self,
        config: Mapping[str, object],
        station_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.denoiser = StationConditionalResUNet1D(
            self.config, station_features, adjacency
        )
        self.diffusion = StationGaussianDiffusion(
            self.denoiser,
            num_steps=int(self.config.get("num_steps", 500)),
            beta_start=float(self.config.get("beta_start", 1e-4)),
            beta_end=float(self.config.get("beta_end", 0.04)),
        )

    @property
    def spatial_mode(self) -> str:
        return self.denoiser.spatial_mode

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.diffusion.training_loss(
            batch["residual_target"],
            batch["forecast"],
            batch["calendar"],
            batch["lead"],
            batch["valid_mask"],
            timestep=timestep,
            noise=noise,
        )

    def generate(
        self,
        batch: Mapping[str, torch.Tensor],
        n_samples: int,
    ) -> torch.Tensor:
        return self.diffusion.sample(
            batch["forecast"], batch["calendar"], batch["lead"], n_samples
        )
