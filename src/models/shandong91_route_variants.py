"""Isolated low-cost Route-A and Route-B variants; V2 semantics remain unchanged."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.shandong91_faithful24_diffusion import Shandong91HeterogeneousRawBodyV2
from src.models.station_conditioned_diffusion import StationSpatialBlock, _flatten_stations, _restore_stations


class Shandong91V2MultiscaleGraph(Shandong91HeterogeneousRawBodyV2):
    architecture = "shandong91_v2_multiscale_physical_graph_v1"

    def __init__(self, config, node_features, adjacency):
        super().__init__(config, node_features, adjacency)
        groups = int(config.get("group_norm_groups", 8)); dropout = float(config.get("dropout", .1))
        gate = float(config.get("multiscale_graph_gate_init", -1.0))
        self.encoder_graphs = nn.ModuleList([
            StationSpatialBlock(width, adjacency, node_features, "fixed_graph", groups, dropout, gate_init=gate)
            for width in self.channels
        ])
        self.decoder_graphs = nn.ModuleList([
            StationSpatialBlock(self.channels[level], adjacency, node_features, "fixed_graph", groups, dropout, gate_init=gate)
            for level in self.decoder_levels
        ])

    def forward(self, noisy_residual, timestep, forecast, forecast_valid_mask,
                calendar, recent_error, recent_error_valid_mask, node_state):
        expected = (forecast.shape[0], 168, 91, 3)
        if forecast.shape != expected or noisy_residual.shape != expected:
            raise ValueError("dynamic tensors must be [B,168,91,3]")
        batch, length, nodes, _ = forecast.shape
        time_embedding = self.timestep_embedding(timestep)
        conditions = self.condition_encoder(forecast, forecast_valid_mask, calendar,
            self.node_features, recent_error, recent_error_valid_mask)
        states = self.state_encoder(node_state.permute(0, 2, 3, 1))
        local = noisy_residual.permute(0, 2, 3, 1).reshape(batch * nodes, 3, length)
        hidden = _restore_stations(self.input_projection(local), batch, nodes)
        skips = []
        for level, block in enumerate(self.encoder_blocks):
            source = hidden
            temporal = block(hidden, time_embedding, conditions[level], states[level])
            hidden = self.early_graph(source, temporal, conditions[level], states[level]) if level == 0 else temporal
            hidden = self.encoder_graphs[level](hidden)
            skips.append(hidden)
            if level < len(self.downsamples):
                flat, _, _ = _flatten_stations(hidden)
                hidden = _restore_stations(self.downsamples[level](flat), batch, nodes)
        hidden = self.bottleneck(hidden, time_embedding, conditions[-1], states[-1])
        hidden = self.bottleneck_graph(hidden)
        for graph, level, upsample, block in zip(self.decoder_graphs, self.decoder_levels, self.upsamples, self.decoder_blocks):
            flat, _, _ = _flatten_stations(hidden)
            hidden = _restore_stations(upsample(flat), batch, nodes)
            skip = skips[level]
            if hidden.shape[-1] != skip.shape[-1]:
                flat, _, _ = _flatten_stations(hidden)
                hidden = _restore_stations(F.interpolate(flat, size=skip.shape[-1], mode="linear", align_corners=False), batch, nodes)
            hidden = block(torch.cat([hidden, skip], dim=2), time_embedding, conditions[level], states[level])
            hidden = graph(hidden)
        flat, _, _ = _flatten_stations(hidden)
        output = self.output_projection(F.silu(self.output_norm(flat)))
        return output.reshape(batch, nodes, 3, length).permute(0, 3, 1, 2).contiguous()


class SystemLoadDiffusionAdapter(nn.Module):
    """One Load diffusion coordinate per time; Wind/Solar remain node-level."""
    architecture = "shandong91_v2_system_load_coordinate_v1"

    def __init__(self, diffusion, transform):
        super().__init__(); self.diffusion = diffusion; self.transform = transform

    def prediction_and_error(self, batch, timestep, node_noise, system_load_noise):
        clean = batch["residual"]
        alpha = self.diffusion.alpha_hat[timestep].view(-1, 1, 1, 1)
        noise = node_noise.clone()
        noise[..., 2] = self.transform.expand(system_load_noise)
        clean_projected = clean.clone()
        system_clean = self.transform.collapse(clean)
        clean_projected[..., 2] = self.transform.expand(system_clean)
        noisy = alpha.sqrt() * clean_projected + (1 - alpha).sqrt() * noise
        prediction = self.diffusion.denoiser(noisy, timestep, batch["forecast"],
            batch["forecast_valid_mask"], batch["time_mark"], batch["recent_error"],
            batch["recent_error_valid_mask"], batch["node_state"])
        predicted_system = self.transform.collapse(prediction)
        return prediction, predicted_system, {
            "wind_solar": (prediction[..., :2] - node_noise[..., :2]).square(),
            "load_system": (predicted_system - system_load_noise).square(),
        }

    @torch.no_grad()
    def sample(self, batch, generation_mask, *, initial_node_noise, initial_system_noise):
        current = initial_node_noise.clone()
        current[..., 2] = self.transform.expand(initial_system_noise)
        current.masked_fill_(~generation_mask, 0)
        for t in range(len(self.diffusion.alpha_hat) - 1, -1, -1):
            timestep = torch.full((len(current),), t, device=current.device, dtype=torch.long)
            epsilon = self.diffusion.denoiser(current, timestep, batch["forecast"],
                batch["forecast_valid_mask"], batch["time_mark"], batch["recent_error"],
                batch["recent_error_valid_mask"], batch["node_state"])
            epsilon[..., 2] = self.transform.expand(self.transform.collapse(epsilon))
            alpha_bar = self.diffusion.alpha_hat[t].to(current)
            clean = (current - torch.sqrt(1 - alpha_bar) * epsilon) / torch.sqrt(alpha_bar).clamp_min(1e-12)
            if t == 0:
                current = clean
            else:
                previous_bar = self.diffusion.alpha_hat[t - 1].to(current)
                alpha = alpha_bar / previous_bar; beta = 1 - alpha
                mean = (current - beta / torch.sqrt(1 - alpha_bar) * epsilon) / torch.sqrt(alpha)
                variance = beta * (1 - previous_bar) / (1 - alpha_bar)
                noise = torch.randn_like(current)
                noise[..., 2] = self.transform.expand(torch.randn_like(initial_system_noise))
                current = mean + torch.sqrt(variance.clamp_min(0)) * noise
            current.masked_fill_(~generation_mask, 0)
        return current

    def masked_loss(self, errors, effective_mask):
        ws_mask = effective_mask[..., :2]
        active = self.transform.active.to(effective_mask.device)
        load_mask = effective_mask[..., 2][..., active].all(dim=-1)
        ws = (errors["wind_solar"] * ws_mask).sum()
        count = ws_mask.sum()
        load = (errors["load_system"] * load_mask).sum()
        return (ws + load) / (count + load_mask.sum()).clamp_min(1)

