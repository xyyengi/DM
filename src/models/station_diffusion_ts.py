"""Conditional full-trajectory Diffusion-TS for joint Station-24 power.

Uses upstream encoder/decoder, polynomial trend, Fourier seasonality and
residual reconstruction. Conditions are issued forecast and observed recent
errors. Neither future actual nor target-derived event labels enter sampling.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .diffusion_ts_vendor.transformer import Transformer


class StationDiffusionTS(nn.Module):
    VERSION = "diffusion_ts_joint_power_v1"

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        if config["version"] != self.VERSION:
            raise ValueError("Unsupported Diffusion-TS checkpoint semantics")
        self.length = int(config["sequence_length"])
        self.stations = int(config["station_count"])
        width = int(config["d_model"])
        self.backbone = Transformer(
            n_feat=self.stations, n_channel=self.length,
            n_layer_enc=config["encoder_layers"], n_layer_dec=config["decoder_layers"],
            n_embd=width, n_heads=config["n_heads"],
            attn_pdrop=config["dropout"], resid_pdrop=config["dropout"],
            max_len=self.length,
        )
        # A time token carries all 24 channels: stations never become separate samples.
        self.forecast_embedding = nn.Linear(self.stations, width)
        self.history_embedding = nn.Linear(2 * self.stations, width)
        self.history_position = nn.Parameter(torch.zeros(1, 24, width))
        steps = int(config["diffusion_steps"])
        u = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        abar = torch.cos(((u / steps + 0.008) / 1.008) * math.pi / 2).square()
        abar = abar / abar[0]
        beta = (1 - abar[1:] / abar[:-1]).clamp(0, 0.999)
        alpha = 1 - beta
        cumulative = alpha.cumprod(0)
        self.register_buffer("alpha_bar", cumulative.float())
        self.register_buffer("loss_weight", (alpha.sqrt() * (1-cumulative).sqrt() / beta / 100).float())
        self.num_steps = steps

    def conditions(self, batch):
        # Explicit allowlist; never inspect batch actual/residual in this function.
        forecast = batch["forecast"].float().transpose(1, 2)
        recent = batch["recent_error"].float().transpose(1, 2)
        mask = batch["recent_error_mask"].float().transpose(1, 2).expand_as(recent)
        if forecast.shape[1:] != (self.length, self.stations):
            raise ValueError("forecast must be [B,24,168]")
        if recent.shape[1:] != (24, self.stations):
            raise ValueError("recent observed error must contain 24 hours")
        return forecast, recent * mask, mask

    def components(self, noisy, timestep, batch):
        forecast, recent, mask = self.conditions(batch)
        emb = self.backbone.emb(noisy)
        condition = self.forecast_embedding(forecast * 2 - 1)
        encoded = self.backbone.encoder(self.backbone.pos_enc(emb + condition), timestep)
        history = self.history_embedding(torch.cat([recent, mask], dim=-1)) + self.history_position
        memory = torch.cat([encoded, history], dim=1)
        output, mean, trend, season = self.backbone.decoder(
            self.backbone.pos_dec(emb + condition), timestep, memory)
        residual = self.backbone.inverse(output)
        residual_mean = residual.mean(dim=1, keepdim=True)
        seasonal = self.backbone.combine_s(season.transpose(1, 2)).transpose(1, 2)
        trend = self.backbone.combine_m(mean) + residual_mean + trend
        return trend.float(), seasonal.float(), (residual - residual_mean).float()

    def predict(self, noisy, timestep, batch, component_weights=(1., 1., 1.)):
        return sum(w * p for w, p in zip(component_weights, self.components(noisy, timestep, batch)))

    def forward(self, batch, timestep=None, noise=None):
        clean = 2 * batch["actual"].float().transpose(1, 2) - 1
        valid = batch["valid_mask"].float().transpose(1, 2)
        if timestep is None:
            timestep = torch.randint(self.num_steps, (len(clean),), device=clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        a = self.alpha_bar[timestep, None, None]
        noisy = a.sqrt() * clean + (1-a).sqrt() * noise
        prediction = self.predict(noisy, timestep, batch)
        with torch.amp.autocast(clean.device.type, enabled=False):
            reconstruction = ((prediction.float()-clean).abs()*valid).sum((1, 2))/valid.sum((1, 2)).clamp(min=1)
            predicted_fft = torch.fft.fft(prediction.float(), dim=1, norm="forward")
            target_fft = torch.fft.fft(clean, dim=1, norm="forward")
            difference = predicted_fft - target_fft
            complete_station = valid.amin(dim=1)[:, None, :]
            fourier = ((difference.real.abs()+difference.imag.abs())*complete_station).sum((1, 2))/(complete_station.sum((1, 2))*self.length).clamp(min=1)
            weight = self.loss_weight[timestep]
            loss = ((reconstruction + self.config["fourier_weight"] * fourier) * weight).mean()
        return loss, {"reconstruction": (reconstruction*weight).mean().detach(),
                      "fourier": (fourier*weight).mean().detach()}

    @torch.no_grad()
    def sample(self, batch, initial_noise, steps=None, component_weights=(1., 1., 1.)):
        # Deterministic DDIM eta=0: fixed initial noise supports paired ablations.
        steps = self.num_steps if steps is None else int(steps)
        if not 1 <= steps <= self.num_steps:
            raise ValueError("invalid sampling steps")
        x = initial_noise.clone()
        times = torch.linspace(-1, self.num_steps-1, steps+1).long().flip(0).tolist()
        for time, following in zip(times[:-1], times[1:]):
            t = torch.full((len(x),), time, device=x.device, dtype=torch.long)
            estimate = self.predict(x, t, batch, component_weights).clamp(-1, 1)
            if following < 0:
                x = estimate
            else:
                a, b = self.alpha_bar[time], self.alpha_bar[following]
                epsilon = (x-a.sqrt()*estimate)/(1-a).sqrt()
                x = b.sqrt()*estimate + (1-b).sqrt()*epsilon
        return (x+1)/2
