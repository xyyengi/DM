"""Shandong91 heterogeneous Raw Body V2 faithful to the mature Station-24 Raw.

The model keeps explicit [B,T,91,3] semantics, restores the mature multi-scale
condition encoder, recent-error gate, forecast-state encoder/FiLM, early
parallel graph fusion, bottleneck graph propagation, and the 500-step epsilon
diffusion.  Lead is intentionally absent: window position is not forecast lead.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from src.models.station_conditioned_diffusion import (
    DiffusionTimestepEmbedding, StationParallelGraphFusion, StationResBlock,
    StationSpatialBlock, _flatten_stations, _group_count, _restore_stations,
)


class Faithful24ConditionEncoder(nn.Module):
    def __init__(self, channels: Sequence[int], node_count: int,
                 node_feature_dim: int, groups: int, gate_init: float = -1.0):
        super().__init__()
        widths = tuple(int(value) for value in channels)
        stem = widths[0]
        self.node_count = int(node_count)
        self.forecast_stem = nn.Conv1d(6, stem, 3, padding=1)
        self.temporal_stem = nn.Conv1d(8, stem, 3, padding=1)
        self.node_projection = nn.Linear(node_feature_dim, stem)
        self.node_embedding = nn.Embedding(node_count, stem)
        self.fuse = nn.Sequential(
            nn.Conv1d(stem * 3, stem, 3, padding=1),
            nn.GroupNorm(_group_count(stem, groups), stem), nn.SiLU(),
        )
        self.recent_stem = nn.Sequential(
            nn.Conv1d(6, stem, 3, padding=1), nn.SiLU(),
            nn.Conv1d(stem, stem, 3, padding=1), nn.SiLU(),
        )
        self.recent_norm = nn.GroupNorm(_group_count(stem, groups), stem)
        self.recent_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(left, right, 4, stride=2, padding=1),
                nn.GroupNorm(_group_count(right, groups), right), nn.SiLU(),
                nn.Conv1d(right, right, 3, padding=1),
            ) for left, right in zip(widths[:-1], widths[1:])
        ])

    def forward(self, forecast, forecast_valid_mask, calendar, node_features,
                recent_error, recent_error_valid_mask):
        batch, length, nodes, resources = forecast.shape
        if (length, nodes, resources) != (168, self.node_count, 3):
            raise ValueError("forecast must be [B,168,91,3]")
        if calendar.shape != (batch, 168, 8):
            raise ValueError("calendar must be [B,168,8]")
        if recent_error.shape != (batch, 24, nodes, 3):
            raise ValueError("recent_error must be [B,24,91,3]")
        valid = forecast_valid_mask.to(forecast.dtype)
        local = torch.cat([forecast * valid, valid], dim=-1)
        local = local.permute(0, 2, 3, 1).reshape(batch * nodes, 6, length)
        local = self.forecast_stem(local)
        temporal = self.temporal_stem(calendar.transpose(1, 2))
        temporal = temporal[:, None].expand(-1, nodes, -1, -1).reshape(
            batch * nodes, temporal.shape[1], length
        )
        node_index = torch.arange(nodes, device=forecast.device)
        static = self.node_projection(node_features) + self.node_embedding(node_index)
        static = static[None, :, :, None].expand(batch, -1, -1, length).reshape(
            batch * nodes, static.shape[-1], length
        )
        feature = self.fuse(torch.cat([local, temporal, static], dim=1))
        recent_valid = recent_error_valid_mask.to(recent_error.dtype)
        recent = torch.cat([recent_error * recent_valid, recent_valid], dim=-1)
        recent = recent.permute(0, 2, 3, 1).reshape(batch * nodes, 6, 24)
        recent = self.recent_stem(recent)
        recent = 0.5 * (recent.mean(dim=-1) + recent[:, :, -1])
        recent_available = recent_error_valid_mask.any(dim=(1, 3)).to(recent.dtype)
        recent = recent * recent_available.reshape(batch * nodes, 1)
        recent = recent[:, :, None].expand(-1, -1, length)
        feature = feature + torch.sigmoid(self.recent_gate) * F.silu(self.recent_norm(recent))
        outputs = [_restore_stations(feature, batch, nodes)]
        for block in self.down_blocks:
            feature = block(feature)
            outputs.append(_restore_stations(feature, batch, nodes))
        return outputs


class ResourceStateEncoder(nn.Module):
    """Mature state-v1 encoder adapted to three explicit resource channels."""
    def __init__(self, widths, node_features, groups, state_dim=12, gate_init=-1.0):
        super().__init__()
        widths = tuple(int(v) for v in widths)
        self.state_dim = int(state_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(state_dim, widths[0], 3, padding=1),
            nn.GroupNorm(_group_count(widths[0], groups), widths[0]), nn.SiLU(),
        )
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(left, right, 4, stride=2, padding=1),
                nn.GroupNorm(_group_count(right, groups), right), nn.SiLU(),
                nn.Conv1d(right, right, 3, padding=1),
            ) for left, right in zip(widths[:-1], widths[1:])
        ])
        self.global_projections = nn.ModuleList(
            [nn.Conv1d(width * 3, width, 1) for width in widths]
        )
        self.global_gates = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(gate_init))) for _ in widths]
        )
        weights = []
        for capacity_column, type_column in zip((1, 2, 3), (4, 5, 6)):
            weight = node_features[:, capacity_column].float().clamp_min(0) * node_features[:, type_column].float()
            weights.append(weight / weight.sum().clamp_min(1e-8))
        self.register_buffer("resource_weights", torch.stack(weights))

    def _fuse(self, node, level):
        global_state = torch.einsum("rs,bsct->brct", self.resource_weights, node)
        batch, resources, channels, length = global_state.shape
        projected = self.global_projections[level](
            global_state.reshape(batch, resources * channels, length)
        )
        return node + torch.sigmoid(self.global_gates[level]) * projected[:, None]

    def forward(self, node_state):
        if node_state.ndim != 4 or node_state.shape[2] != self.state_dim:
            raise ValueError("node_state must be [B,91,12,168]")
        batch, nodes, _, _ = node_state.shape
        feature = self.stem(node_state.reshape(batch * nodes, self.state_dim, 168))
        outputs = [self._fuse(_restore_stations(feature, batch, nodes), 0)]
        for level, block in enumerate(self.down_blocks, 1):
            feature = block(feature)
            outputs.append(self._fuse(_restore_stations(feature, batch, nodes), level))
        return outputs


class Shandong91HeterogeneousRawBodyV2(nn.Module):
    architecture = "shandong91_heterogeneous_raw_body_v2_faithful24"

    def __init__(self, config: Mapping[str, object], node_features, adjacency):
        super().__init__()
        self.config = dict(config)
        self.node_count = int(config.get("node_count", 91))
        self.sequence_length = int(config.get("sequence_length", 168))
        if self.node_count != 91 or int(config.get("resource_count", 3)) != 3:
            raise ValueError("V2 requires explicit 91-node x 3-resource semantics")
        if bool(config.get("use_lead_condition", False)):
            raise ValueError("window_position is not forecast lead; lead must remain disabled")
        forbidden = ("use_body_tail_experts", "use_event_balanced_sampling",
                     "use_self_localization", "use_protected_partial", "use_auxiliary_losses")
        if any(bool(config.get(key, False)) for key in forbidden):
            raise ValueError("V2 faithful Raw forbids Tail/event/partial/auxiliary mechanisms")
        if bool(config.get("use_dual_fixed_graph", False)):
            raise ValueError("no semantically validated Shandong91 secondary graph exists")
        base = int(config.get("base_channels", 32))
        multipliers = tuple(int(v) for v in config.get("channel_multipliers", [1, 2, 4]))
        self.channels = tuple(base * value for value in multipliers)
        groups = int(config.get("group_norm_groups", 8)); dropout = float(config.get("dropout", .1))
        time_dim = int(config.get("timestep_embedding_dim", 128))
        state_widths = tuple(int(v) for v in config.get("state_channels", [8, 16, 32]))
        self.register_buffer("node_features", node_features.float())
        self.timestep_embedding = DiffusionTimestepEmbedding(time_dim, time_dim)
        self.condition_encoder = Faithful24ConditionEncoder(
            self.channels, 91, node_features.shape[1], groups,
            gate_init=float(config.get("condition_gate_init", -1.0)),
        )
        self.state_encoder = ResourceStateEncoder(
            state_widths, node_features, groups, state_dim=12,
            gate_init=float(config.get("state_global_gate_init", -1.0)),
        )
        self.input_projection = nn.Conv1d(3, self.channels[0], 3, padding=1)
        self.encoder_blocks = nn.ModuleList([
            StationResBlock(width, width, time_dim, width, groups, dropout,
                            state_channels=state_widths[level],
                            state_gate_init=float(config.get("state_film_gate_init", -1.0)))
            for level, width in enumerate(self.channels)
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv1d(left, right, 4, stride=2, padding=1)
            for left, right in zip(self.channels[:-1], self.channels[1:])
        ])
        self.early_graph = StationParallelGraphFusion(
            self.channels[0], adjacency, node_features, groups, dropout,
            gate_init=float(config.get("parallel_spatial_gate_init", -1.0)),
            adjacency_mode="fixed", state_channels=state_widths[0],
        )
        self.bottleneck = StationResBlock(
            self.channels[-1], self.channels[-1], time_dim, self.channels[-1],
            groups, dropout, state_channels=state_widths[-1],
            state_gate_init=float(config.get("state_film_gate_init", -1.0)),
        )
        self.bottleneck_graph = StationSpatialBlock(
            self.channels[-1], adjacency, node_features, "fixed_graph", groups,
            dropout, gate_init=float(config.get("spatial_gate_init", -1.0)),
        )
        self.decoder_levels = tuple(reversed(range(len(self.channels) - 1)))
        self.upsamples = nn.ModuleList(); self.decoder_blocks = nn.ModuleList()
        current = self.channels[-1]
        for level in self.decoder_levels:
            self.upsamples.append(nn.ConvTranspose1d(current, self.channels[level], 4, stride=2, padding=1))
            self.decoder_blocks.append(StationResBlock(
                self.channels[level] * 2, self.channels[level], time_dim,
                self.channels[level], groups, dropout,
                state_channels=state_widths[level],
                state_gate_init=float(config.get("state_film_gate_init", -1.0)),
            ))
            current = self.channels[level]
        self.output_norm = nn.GroupNorm(_group_count(self.channels[0], groups), self.channels[0])
        self.output_projection = nn.Conv1d(self.channels[0], 3, 1)

    def forward(self, noisy_residual, timestep, forecast, forecast_valid_mask,
                calendar, recent_error, recent_error_valid_mask, node_state):
        expected = (forecast.shape[0], 168, 91, 3)
        if forecast.shape != expected or noisy_residual.shape != expected:
            raise ValueError("dynamic tensors must be [B,168,91,3]")
        batch, length, nodes, _ = forecast.shape
        time_embedding = self.timestep_embedding(timestep)
        conditions = self.condition_encoder(
            forecast, forecast_valid_mask, calendar, self.node_features,
            recent_error, recent_error_valid_mask,
        )
        states = self.state_encoder(node_state.permute(0, 2, 3, 1))
        local = noisy_residual.permute(0, 2, 3, 1).reshape(batch * nodes, 3, length)
        hidden = _restore_stations(self.input_projection(local), batch, nodes)
        skips = []
        for level, block in enumerate(self.encoder_blocks):
            source = hidden
            temporal = block(hidden, time_embedding, conditions[level], states[level])
            hidden = self.early_graph(source, temporal, conditions[level], states[level]) if level == 0 else temporal
            skips.append(hidden)
            if level < len(self.downsamples):
                flat, _, _ = _flatten_stations(hidden)
                hidden = _restore_stations(self.downsamples[level](flat), batch, nodes)
        hidden = self.bottleneck(hidden, time_embedding, conditions[-1], states[-1])
        hidden = self.bottleneck_graph(hidden)
        for level, upsample, block in zip(self.decoder_levels, self.upsamples, self.decoder_blocks):
            flat, _, _ = _flatten_stations(hidden)
            hidden = _restore_stations(upsample(flat), batch, nodes)
            skip = skips[level]
            if hidden.shape[-1] != skip.shape[-1]:
                flat, _, _ = _flatten_stations(hidden)
                hidden = _restore_stations(F.interpolate(flat, size=skip.shape[-1], mode="linear", align_corners=False), batch, nodes)
            hidden = block(torch.cat([hidden, skip], dim=2), time_embedding, conditions[level], states[level])
        flat, _, _ = _flatten_stations(hidden)
        output = self.output_projection(F.silu(self.output_norm(flat)))
        return output.reshape(batch, nodes, 3, length).permute(0, 3, 1, 2).contiguous()


class Shandong91Faithful24Diffusion(nn.Module):
    def __init__(self, denoiser, config):
        super().__init__(); self.denoiser = denoiser
        steps = int(config.get("num_steps", 500))
        beta = torch.linspace(float(config.get("beta_start", 1e-4)), float(config.get("beta_end", .04)), steps)
        self.register_buffer("beta", beta); self.register_buffer("alpha_hat", torch.cumprod(1 - beta, dim=0))

    def prediction_and_error(self, batch, timestep, noise):
        alpha = self.alpha_hat[timestep].view(-1, 1, 1, 1)
        noisy = alpha.sqrt() * batch["residual"] + (1 - alpha).sqrt() * noise
        prediction = self.denoiser(
            noisy, timestep, batch["forecast"], batch["forecast_valid_mask"],
            batch["time_mark"], batch["recent_error"],
            batch["recent_error_valid_mask"], batch["node_state"],
        )
        return prediction, (prediction - noise).square()

    @staticmethod
    def masked_loss(error, mask):
        weight = mask.to(error.dtype)
        return (error * weight).sum() / weight.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, batch, generation_mask, *, method="ddpm", inference_steps=500,
               initial_noise=None, generator=None):
        forecast = batch["forecast"]; current = torch.randn_like(forecast) if initial_noise is None else initial_noise.clone()
        current.masked_fill_(~generation_mask, 0)
        total = len(self.alpha_hat)
        if method == "ddpm":
            if inference_steps != total:
                raise ValueError("DDPM must use every training-schedule step")
            indices = torch.arange(total - 1, -1, -1, device=forecast.device)
        elif method == "ddim":
            indices = torch.linspace(total - 1, 0, inference_steps, dtype=torch.float64, device=forecast.device).round().long().unique_consecutive()
        else:
            raise ValueError("method must be ddpm or ddim")
        for position, t_value in enumerate(indices):
            t = int(t_value); timestep = torch.full((len(current),), t, device=current.device, dtype=torch.long)
            epsilon = self.denoiser(
                current, timestep, batch["forecast"], batch["forecast_valid_mask"],
                batch["time_mark"], batch["recent_error"],
                batch["recent_error_valid_mask"], batch["node_state"],
            )
            alpha_bar = self.alpha_hat[t].to(current.dtype)
            clean = (current - torch.sqrt(1 - alpha_bar) * epsilon) / torch.sqrt(alpha_bar).clamp_min(1e-12)
            if position + 1 == len(indices):
                current = clean
            else:
                previous = int(indices[position + 1]); previous_bar = self.alpha_hat[previous].to(current.dtype)
                if method == "ddim":
                    current = torch.sqrt(previous_bar) * clean + torch.sqrt(1 - previous_bar) * epsilon
                else:
                    alpha = alpha_bar / previous_bar; beta = 1 - alpha
                    mean = (current - beta / torch.sqrt(1 - alpha_bar) * epsilon) / torch.sqrt(alpha)
                    variance = beta * (1 - previous_bar) / (1 - alpha_bar)
                    noise = torch.randn(current.shape, device=current.device, dtype=current.dtype, generator=generator)
                    current = mean + torch.sqrt(variance.clamp_min(0)) * noise
            current.masked_fill_(~generation_mask, 0)
        return current.contiguous()
