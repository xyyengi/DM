"""91-node heterogeneous Raw Body for CPU/GPU preflight and later training.

This is a versioned sibling of the Station-24 conditional ResUNet.  It reuses
the temporal ResBlocks, FiLM, timestep embedding, and fixed-graph propagation,
while preserving an explicit node/resource layout.  It intentionally has no
lead-time branch and no Tail/auxiliary/event machinery.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from src.models.station_conditioned_diffusion import (
    DiffusionTimestepEmbedding,
    StationParallelGraphFusion,
    StationResBlock,
    StationSpatialBlock,
    _flatten_stations,
    _group_count,
    _restore_stations,
)


class Shandong91ConditionEncoder(nn.Module):
    """Forecast/calendar/static conditioning with no lead-time input."""

    def __init__(
        self,
        channels: Sequence[int],
        node_count: int,
        node_feature_dim: int,
        groups: int,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in channels)
        self.node_count = int(node_count)
        stem = widths[0]
        # Three normalized resource forecasts plus their three validity flags.
        self.forecast_projection = nn.Conv1d(6, stem, kernel_size=3, padding=1)
        self.calendar_projection = nn.Conv1d(8, stem, kernel_size=3, padding=1)
        self.node_feature_projection = nn.Linear(node_feature_dim, stem)
        self.node_embedding = nn.Embedding(self.node_count, stem)
        self.fuse = nn.Sequential(
            nn.Conv1d(stem * 3, stem, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(stem, groups), stem),
            nn.SiLU(),
        )
        self.down_blocks = nn.ModuleList(
            [
                nn.Conv1d(left, right, kernel_size=4, stride=2, padding=1)
                for left, right in zip(widths[:-1], widths[1:])
            ]
        )

    def forward(
        self,
        forecast: torch.Tensor,
        forecast_valid_mask: torch.Tensor,
        calendar: torch.Tensor,
        node_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        if forecast.ndim != 4:
            raise ValueError("forecast must be [B,T,N,3]")
        batch, length, nodes, resources = forecast.shape
        if (nodes, resources) != (self.node_count, 3):
            raise ValueError("forecast must retain [N=91,C=3] semantics")
        if forecast_valid_mask.shape != forecast.shape:
            raise ValueError("forecast_valid_mask must match forecast")
        if calendar.shape != (batch, length, 8):
            raise ValueError("calendar must be [B,T,8]")
        if node_features.shape[0] != nodes:
            raise ValueError("node feature order/count mismatch")
        if forecast.device != forecast_valid_mask.device or forecast.device != calendar.device:
            raise ValueError("dynamic condition tensors must share a device")
        if forecast.dtype != calendar.dtype:
            raise ValueError("forecast and calendar must share a floating dtype")

        valid = forecast_valid_mask.to(forecast.dtype)
        local_input = torch.cat([forecast * valid, valid], dim=-1)
        local_input = local_input.permute(0, 2, 3, 1).reshape(
            batch * nodes, 6, length
        )
        local = self.forecast_projection(local_input)
        temporal = self.calendar_projection(calendar.transpose(1, 2))
        temporal = temporal[:, None].expand(-1, nodes, -1, -1).reshape(
            batch * nodes, temporal.shape[1], length
        )
        node_index = torch.arange(nodes, device=forecast.device)
        static = self.node_feature_projection(node_features) + self.node_embedding(node_index)
        static = static[None, :, :, None].expand(batch, -1, -1, length).reshape(
            batch * nodes, static.shape[-1], length
        )
        hidden = self.fuse(torch.cat([local, temporal, static], dim=1))
        outputs = [_restore_stations(hidden, batch, nodes)]
        for block in self.down_blocks:
            hidden = block(hidden)
            outputs.append(_restore_stations(hidden, batch, nodes))
        return outputs


class Shandong91HeterogeneousRawBody(nn.Module):
    """Explicit `[B,T,91,3]` fixed-graph conditional epsilon predictor."""

    architecture = "shandong91_heterogeneous_raw_body_v1_no_lead"

    def __init__(
        self,
        config: Mapping[str, object],
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.node_count = int(self.config.get("node_count", 91))
        self.resource_count = int(self.config.get("resource_count", 3))
        self.sequence_length = int(self.config.get("sequence_length", 168))
        if self.node_count != 91 or self.resource_count != 3:
            raise ValueError("v1 contract requires exactly 91 nodes and 3 resources")
        if node_features.shape[0] != self.node_count:
            raise ValueError("node_features must have 91 rows")
        if adjacency.shape != (self.node_count, self.node_count):
            raise ValueError("physical adjacency must be [91,91]")
        if bool(self.config.get("use_lead_condition", False)):
            raise ValueError("v1 explicitly forbids the legacy lead branch")
        forbidden = (
            "use_body_tail_experts", "use_jstd_tail", "use_joint_multiresidual_tail",
            "use_event_balanced_sampling", "use_self_localization",
            "use_protected_partial", "use_auxiliary_losses",
        )
        if any(bool(self.config.get(name, False)) for name in forbidden):
            raise ValueError("Tail, event, auxiliary, and partial mechanisms are forbidden")

        num_layers = int(self.config.get("num_layers", 3))
        multipliers = tuple(int(v) for v in self.config.get("channel_multipliers", [1, 2, 4]))
        if len(multipliers) != num_layers:
            raise ValueError("channel_multipliers length mismatch")
        base = int(self.config.get("base_channels", 32))
        self.channels = tuple(base * value for value in multipliers)
        groups = int(self.config.get("group_norm_groups", 8))
        dropout = float(self.config.get("dropout", 0.1))
        time_dim = int(self.config.get("timestep_embedding_dim", 128))
        self.register_buffer("node_features", node_features.float())
        self.timestep_embedding = DiffusionTimestepEmbedding(time_dim, time_dim)
        self.condition_encoder = Shandong91ConditionEncoder(
            self.channels, self.node_count, node_features.shape[1], groups
        )
        self.input_projection = nn.Conv1d(3, self.channels[0], kernel_size=3, padding=1)
        self.encoder_blocks = nn.ModuleList(
            [StationResBlock(width, width, time_dim, width, groups, dropout)
             for width in self.channels]
        )
        self.downsamples = nn.ModuleList(
            [nn.Conv1d(left, right, 4, stride=2, padding=1)
             for left, right in zip(self.channels[:-1], self.channels[1:])]
        )
        self.early_graph = StationParallelGraphFusion(
            self.channels[0], adjacency, node_features, groups, dropout,
            gate_init=float(self.config.get("parallel_spatial_gate_init", -1.0)),
            adjacency_mode="fixed",
        )
        self.bottleneck = StationResBlock(
            self.channels[-1], self.channels[-1], time_dim,
            self.channels[-1], groups, dropout,
        )
        self.bottleneck_graph = StationSpatialBlock(
            self.channels[-1], adjacency, node_features, "fixed_graph", groups,
            dropout, gate_init=float(self.config.get("spatial_gate_init", -1.0)),
        )
        self.decoder_levels = tuple(reversed(range(num_layers - 1)))
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current = self.channels[-1]
        for level in self.decoder_levels:
            self.upsamples.append(nn.ConvTranspose1d(
                current, self.channels[level], 4, stride=2, padding=1
            ))
            self.decoder_blocks.append(StationResBlock(
                self.channels[level] * 2, self.channels[level], time_dim,
                self.channels[level], groups, dropout,
            ))
            current = self.channels[level]
        self.output_norm = nn.GroupNorm(_group_count(self.channels[0], groups), self.channels[0])
        self.output_projection = nn.Conv1d(self.channels[0], 3, kernel_size=1)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timestep: torch.Tensor,
        forecast: torch.Tensor,
        forecast_valid_mask: torch.Tensor,
        calendar: torch.Tensor,
    ) -> torch.Tensor:
        expected = (forecast.shape[0], self.sequence_length, self.node_count, 3)
        if forecast.shape != expected or noisy_residual.shape != expected:
            raise ValueError(f"dynamic tensors must be {expected}")
        if timestep.shape != (forecast.shape[0],):
            raise ValueError("timestep must be [B]")
        if not noisy_residual.is_floating_point() or noisy_residual.dtype != forecast.dtype:
            raise ValueError("noisy residual and forecast must share floating dtype")
        if noisy_residual.device != next(self.parameters()).device:
            raise ValueError("model and dynamic inputs must share a device")

        batch, length, nodes, _ = forecast.shape
        time_embedding = self.timestep_embedding(timestep)
        conditions = self.condition_encoder(
            forecast, forecast_valid_mask, calendar, self.node_features
        )
        local = noisy_residual.permute(0, 2, 3, 1).reshape(batch * nodes, 3, length)
        hidden = _restore_stations(self.input_projection(local), batch, nodes)
        skips = []
        for level, block in enumerate(self.encoder_blocks):
            branch_input = hidden
            temporal = block(hidden, time_embedding, conditions[level])
            hidden = self.early_graph(branch_input, temporal, conditions[level]) if level == 0 else temporal
            skips.append(hidden)
            if level < len(self.downsamples):
                flat, _, _ = _flatten_stations(hidden)
                hidden = _restore_stations(self.downsamples[level](flat), batch, nodes)
        hidden = self.bottleneck(hidden, time_embedding, conditions[-1])
        hidden = self.bottleneck_graph(hidden)
        for level, upsample, block in zip(
            self.decoder_levels, self.upsamples, self.decoder_blocks
        ):
            flat, _, _ = _flatten_stations(hidden)
            hidden = _restore_stations(upsample(flat), batch, nodes)
            skip = skips[level]
            if hidden.shape[-1] != skip.shape[-1]:
                flat, _, _ = _flatten_stations(hidden)
                hidden = _restore_stations(
                    F.interpolate(flat, size=skip.shape[-1], mode="linear", align_corners=False),
                    batch, nodes,
                )
            hidden = block(torch.cat([hidden, skip], dim=2), time_embedding, conditions[level])
        flat, _, _ = _flatten_stations(hidden)
        output = self.output_projection(F.silu(self.output_norm(flat)))
        return output.reshape(batch, nodes, 3, length).permute(0, 3, 1, 2).contiguous()


class Shandong91MaskedDiffusion(nn.Module):
    """Minimal epsilon diffusion objective; no generation or auxiliary losses."""

    def __init__(self, denoiser: Shandong91HeterogeneousRawBody, config: Mapping[str, object]) -> None:
        super().__init__()
        self.denoiser = denoiser
        steps = int(config.get("num_steps", 500))
        beta = torch.linspace(float(config.get("beta_start", 1e-4)),
                              float(config.get("beta_end", 0.04)), steps)
        self.register_buffer("alpha_hat", torch.cumprod(1.0 - beta, dim=0))

    def add_noise(self, clean: torch.Tensor, timestep: torch.Tensor,
                  noise: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_hat[timestep].view(-1, 1, 1, 1)
        return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

    def prediction_and_error(self, batch: Mapping[str, torch.Tensor],
                             timestep: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clean = batch["residual"]
        noisy = self.add_noise(clean, timestep, noise)
        prediction = self.denoiser(
            noisy, timestep, batch["forecast"], batch["forecast_valid_mask"], batch["time_mark"]
        )
        return prediction, (prediction - noise).square()

    @staticmethod
    def masked_loss(element_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(element_loss.dtype)
        return (element_loss * weight).sum() / weight.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample_ddim(
        self,
        forecast: torch.Tensor,
        forecast_valid_mask: torch.Tensor,
        calendar: torch.Tensor,
        generation_mask: torch.Tensor,
        *,
        inference_steps: int,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic DDIM sampling under the training epsilon convention.

        Invalid, nonexistent, and training-disabled channels are projected to a
        zero residual after every reverse step.  The returned layout is always
        ``[B,T,N,3]``.
        """
        if generation_mask.shape != forecast.shape:
            raise ValueError("generation_mask must match [B,T,N,3] forecast")
        if inference_steps < 2 or inference_steps > self.alpha_hat.numel():
            raise ValueError("inference_steps must be in [2, diffusion_steps]")
        mask = generation_mask.bool()
        current = (
            torch.randn_like(forecast) if initial_noise is None else initial_noise.clone()
        )
        if current.shape != forecast.shape:
            raise ValueError("initial_noise must match forecast")
        current.masked_fill_(~mask, 0.0)
        indices = torch.linspace(
            self.alpha_hat.numel() - 1, 0, inference_steps,
            device=forecast.device, dtype=torch.float64,
        ).round().long().unique_consecutive()
        batch = forecast.shape[0]
        for position, timestep_value in enumerate(indices):
            timestep = torch.full(
                (batch,), int(timestep_value), device=forecast.device, dtype=torch.long
            )
            epsilon = self.denoiser(
                current, timestep, forecast, forecast_valid_mask, calendar
            )
            alpha = self.alpha_hat[timestep_value].to(current.dtype)
            predicted_clean = (
                current - (1.0 - alpha).sqrt() * epsilon
            ) / alpha.sqrt().clamp_min(1e-12)
            if position + 1 == len(indices):
                current = predicted_clean
            else:
                next_alpha = self.alpha_hat[indices[position + 1]].to(current.dtype)
                current = next_alpha.sqrt() * predicted_clean + (1.0 - next_alpha).sqrt() * epsilon
            current.masked_fill_(~mask, 0.0)
        return current.contiguous()

