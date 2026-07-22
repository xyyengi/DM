"""Stage-1 V5 denoisers with true diffusion-time conditioning.

V5-T uses only the three-channel noisy state and diffusion timestep. V5-TF
adds a state-independent multiscale encoder for forecast, real calendar
sin/cos features, and relative window position. No V5 architecture concatenates
these conditions onto the noisy-state convolution input.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, requested: int) -> int:
    for groups in range(min(int(requested), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_embedding(values: torch.Tensor, dimension: int, max_period: float = 10000.0):
    """Continuous sinusoidal embedding for scalar values of arbitrary shape."""
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
            raise ValueError(f"diffusion timestep must be [B], got {tuple(timestep.shape)}")
        return self.mlp(sinusoidal_embedding(timestep, self.embedding_dim))


class SequenceConditionEncoder(nn.Module):
    """Encode forecast, real calendar, and relative position independently."""

    def __init__(
        self,
        channels: Sequence[int],
        position_dim: int = 32,
        group_norm_groups: int = 8,
    ):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        if not channels:
            raise ValueError("condition encoder requires at least one resolution")
        stem_channels = channels[0]
        self.position_dim = int(position_dim)
        self.forecast_stem = nn.Conv1d(3, stem_channels, kernel_size=3, padding=1)
        self.calendar_stem = nn.Conv1d(8, stem_channels, kernel_size=3, padding=1)
        self.position_stem = nn.Conv1d(self.position_dim, stem_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv1d(stem_channels * 3, stem_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(stem_channels, group_norm_groups), stem_channels),
            nn.SiLU(),
        )
        self.down_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            self.down_blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(_group_count(out_channels, group_norm_groups), out_channels),
                nn.SiLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            ))
        self.last_feature_shapes: list[tuple[int, ...]] = []
        self.last_input_shapes: dict[str, tuple[int, ...]] = {}

    def forward(
        self,
        forecast: torch.Tensor,
        calendar: torch.Tensor,
        relative_positions: torch.Tensor,
    ) -> list[torch.Tensor]:
        if forecast.ndim != 3 or forecast.shape[1] != 3:
            raise ValueError(f"forecast must be [B,3,L], got {tuple(forecast.shape)}")
        if calendar.ndim != 3 or calendar.shape[1] != 8:
            raise ValueError(f"calendar must be [B,8,L], got {tuple(calendar.shape)}")
        if relative_positions.ndim != 2:
            raise ValueError(
                f"relative_positions must be [B,L], got {tuple(relative_positions.shape)}"
            )
        batch, _, length = forecast.shape
        if calendar.shape != (batch, 8, length):
            raise ValueError("calendar shape must match forecast batch and sequence length")
        if relative_positions.shape != (batch, length):
            raise ValueError("relative_positions shape must match forecast [B,L]")

        self.last_input_shapes = {
            "forecast": tuple(forecast.shape),
            "calendar": tuple(calendar.shape),
            "relative_positions": tuple(relative_positions.shape),
        }

        position = sinusoidal_embedding(relative_positions, self.position_dim).permute(0, 2, 1)
        forecast_feature = self.forecast_stem(forecast)
        calendar_feature = self.calendar_stem(calendar)
        position_feature = self.position_stem(position)
        feature = self.fuse(torch.cat(
            [forecast_feature, calendar_feature, position_feature], dim=1
        ))
        outputs = [feature]
        for block in self.down_blocks:
            feature = block(feature)
            outputs.append(feature)
        self.last_feature_shapes = [tuple(value.shape) for value in outputs]
        return outputs


class V5ResBlock(nn.Module):
    """GroupNorm/SiLU residual block with diffusion-time and sequence FiLM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        condition_channels: int | None,
        group_norm_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.norm1 = nn.GroupNorm(
            _group_count(self.in_channels, group_norm_groups), self.in_channels
        )
        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            _group_count(self.out_channels, group_norm_groups), self.out_channels
        )
        self.time_affine = nn.Linear(time_dim, self.out_channels * 2)
        self.condition_affine = (
            nn.Conv1d(int(condition_channels), self.out_channels * 2, kernel_size=1)
            if condition_channels is not None else None
        )
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.residual = (
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
            if self.in_channels != self.out_channels else nn.Identity()
        )
        self.last_modulation_shapes: dict[str, tuple[int, ...] | None] = {}

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        condition_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.residual(x)
        h = self.conv1(F.silu(self.norm1(x)))
        normalized = self.norm2(h)

        time_gamma, time_beta = self.time_affine(time_embedding).chunk(2, dim=1)
        time_gamma = time_gamma.unsqueeze(-1)
        time_beta = time_beta.unsqueeze(-1)
        gamma, beta = time_gamma, time_beta
        condition_shape = None
        if self.condition_affine is not None:
            if condition_feature is None:
                raise ValueError("condition feature is required for a conditioned V5 block")
            if condition_feature.shape[-1] != normalized.shape[-1]:
                condition_feature = F.interpolate(
                    condition_feature,
                    size=normalized.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
            condition_gamma, condition_beta = self.condition_affine(condition_feature).chunk(
                2, dim=1
            )
            gamma = gamma + condition_gamma
            beta = beta + condition_beta
            condition_shape = tuple(condition_feature.shape)

        self.last_modulation_shapes = {
            "time_gamma": tuple(time_gamma.shape),
            "time_beta": tuple(time_beta.shape),
            "condition_feature": condition_shape,
            "gamma": tuple(gamma.shape),
            "beta": tuple(beta.shape),
        }
        h = normalized * (1.0 + gamma) + beta
        h = self.conv2(self.dropout(F.silu(h)))
        return residual + h


class V5ConditionalUNet1D(nn.Module):
    """Three-channel temporal UNet whose conditions never enter the state stem."""

    def __init__(self, config: Mapping[str, object]):
        super().__init__()
        self.sequence_length = int(config.get("sequence_length", 168))
        self.in_channels = int(config.get("in_channels", config.get("input_channels", 3)))
        self.out_channels = int(config.get("out_channels", 3))
        if self.in_channels != 3 or self.out_channels != 3:
            raise ValueError("V5 requires in_channels=3 and out_channels=3")

        self.num_layers = int(config.get("num_layers", 3))
        if self.num_layers < 2:
            raise ValueError("V5 num_layers must be at least 2")
        multipliers = tuple(config.get("channel_multipliers", [2**i for i in range(self.num_layers)]))
        if len(multipliers) != self.num_layers:
            raise ValueError(
                "V5 num_layers must equal len(channel_multipliers); "
                f"got {self.num_layers} and {len(multipliers)}"
            )
        if self.sequence_length % (2 ** (self.num_layers - 1)) != 0:
            raise ValueError(
                "sequence_length must be divisible by 2**(num_layers-1) for exact UNet scales"
            )
        base_channels = int(config.get("base_channels", 64))
        self.channels = tuple(base_channels * int(multiplier) for multiplier in multipliers)
        self.use_sequence_condition = bool(config.get("use_sequence_condition", False))
        groups = int(config.get("group_norm_groups", 8))
        dropout = float(config.get("dropout", 0.0))
        timestep_dim = int(config.get("timestep_embedding_dim", 128))

        self.timestep_embedding = DiffusionTimestepEmbedding(timestep_dim, timestep_dim)
        self.condition_encoder = (
            SequenceConditionEncoder(
                self.channels,
                position_dim=int(config.get("position_embedding_dim", 32)),
                group_norm_groups=groups,
            )
            if self.use_sequence_condition else None
        )
        self.state_stem = nn.Conv1d(3, self.channels[0], kernel_size=3, padding=1)

        condition_channels = self.channels if self.use_sequence_condition else (None,) * self.num_layers
        self.encoder_blocks = nn.ModuleList([
            V5ResBlock(
                self.channels[level], self.channels[level], timestep_dim,
                condition_channels[level], groups, dropout,
            )
            for level in range(self.num_layers)
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv1d(
                self.channels[level], self.channels[level + 1],
                kernel_size=4, stride=2, padding=1,
            )
            for level in range(self.num_layers - 1)
        ])
        self.bottleneck = V5ResBlock(
            self.channels[-1], self.channels[-1], timestep_dim,
            condition_channels[-1], groups, dropout,
        )

        self.decoder_levels = tuple(reversed(range(self.num_layers - 1)))
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        current_channels = self.channels[-1]
        for level in self.decoder_levels:
            self.upsamples.append(nn.ConvTranspose1d(
                current_channels, self.channels[level], kernel_size=4, stride=2, padding=1
            ))
            self.decoder_blocks.append(V5ResBlock(
                self.channels[level] * 2,
                self.channels[level],
                timestep_dim,
                condition_channels[level],
                groups,
                dropout,
            ))
            current_channels = self.channels[level]
        self.output_norm = nn.GroupNorm(
            _group_count(self.channels[0], groups), self.channels[0]
        )
        self.output = nn.Conv1d(self.channels[0], 3, kernel_size=1)
        self.last_noisy_state_shape: tuple[int, ...] | None = None
        self.last_condition_shapes: list[tuple[int, ...]] = []

    @property
    def residual_blocks(self) -> tuple[V5ResBlock, ...]:
        return tuple(self.encoder_blocks) + (self.bottleneck,) + tuple(self.decoder_blocks)

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        forecast: torch.Tensor | None = None,
        calendar: torch.Tensor | None = None,
        relative_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[1:] != (3, self.sequence_length):
            raise ValueError(
                f"V5 noisy state must be [B,3,{self.sequence_length}], got {tuple(x_t.shape)}"
            )
        if timestep.ndim != 1 or timestep.shape[0] != x_t.shape[0]:
            raise ValueError(f"diffusion timestep must be [B], got {tuple(timestep.shape)}")
        self.last_noisy_state_shape = tuple(x_t.shape)
        time_embedding = self.timestep_embedding(timestep)

        if self.use_sequence_condition:
            if forecast is None or calendar is None or relative_positions is None:
                raise ValueError("V5-TF requires forecast, calendar, and relative_positions")
            condition_features = self.condition_encoder(
                forecast, calendar, relative_positions
            )
            self.last_condition_shapes = [tuple(value.shape) for value in condition_features]
        else:
            condition_features = [None] * self.num_layers
            self.last_condition_shapes = []

        h = self.state_stem(x_t)
        skips = []
        for level, block in enumerate(self.encoder_blocks):
            h = block(h, time_embedding, condition_features[level])
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
        h = self.bottleneck(h, time_embedding, condition_features[-1])

        for level, upsample, block in zip(
            self.decoder_levels, self.upsamples, self.decoder_blocks
        ):
            h = upsample(h)
            skip = skips[level]
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = block(torch.cat([h, skip], dim=1), time_embedding, condition_features[level])
        return self.output(F.silu(self.output_norm(h)))


class V5GaussianDiffusion(nn.Module):
    """DDPM training and reverse sampling with mandatory denoiser timesteps."""

    def __init__(
        self,
        denoiser: V5ConditionalUNet1D,
        num_steps: int = 500,
        beta_start: float = 0.0001,
        beta_end: float = 0.04,
        schedule: str = "linear",
        reverse_variance_type: str = "posterior",
    ):
        super().__init__()
        self.denoiser = denoiser
        self.num_steps = int(num_steps)
        self.reverse_variance_type = str(reverse_variance_type)
        if self.reverse_variance_type not in {"beta", "posterior"}:
            raise ValueError("reverse_variance_type must be 'beta' or 'posterior'")
        if schedule == "linear":
            beta = torch.linspace(beta_start, beta_end, self.num_steps)
        elif schedule == "quad":
            beta = torch.linspace(beta_start**0.5, beta_end**0.5, self.num_steps) ** 2
        else:
            raise ValueError(f"Unsupported V5 diffusion schedule={schedule!r}")
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)
        self.last_training_timesteps: torch.Tensor | None = None
        self.last_sampling_timesteps: list[int] = []

    def add_noise(self, x0: torch.Tensor, timestep: torch.Tensor):
        if x0.ndim != 3 or x0.shape[1:] != (3, self.denoiser.sequence_length):
            raise ValueError("V5 diffusion target must be [B,3,L]")
        if timestep.ndim != 1 or timestep.shape[0] != x0.shape[0]:
            raise ValueError("V5 diffusion timestep must be [B]")
        noise = torch.randn_like(x0)
        alpha_hat = self.alpha_hat[timestep].view(-1, 1, 1)
        return alpha_hat.sqrt() * x0 + (1.0 - alpha_hat).sqrt() * noise, noise

    def reverse_variance(self, timestep: torch.Tensor):
        beta = self.beta[timestep]
        if self.reverse_variance_type == "beta":
            return beta
        previous_index = torch.clamp(timestep - 1, min=0)
        alpha_hat_previous = self.alpha_hat[previous_index]
        posterior = beta * (1.0 - alpha_hat_previous) / (1.0 - self.alpha_hat[timestep])
        return torch.where(timestep > 0, posterior.clamp(min=0.0), torch.zeros_like(posterior))

    def predict_noise(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        forecast: torch.Tensor | None,
        calendar: torch.Tensor | None,
        relative_positions: torch.Tensor | None,
    ):
        return self.denoiser(
            x_t,
            timestep,
            forecast=forecast,
            calendar=calendar,
            relative_positions=relative_positions,
        )

    def forward(
        self,
        x0: torch.Tensor,
        forecast: torch.Tensor | None = None,
        calendar: torch.Tensor | None = None,
        relative_positions: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ):
        if timestep is None:
            timestep = torch.randint(
                0, self.num_steps, (x0.shape[0],), device=x0.device
            )
        timestep = timestep.long()
        self.last_training_timesteps = timestep.detach().clone()
        x_t, noise = self.add_noise(x0, timestep)
        predicted = self.predict_noise(
            x_t, timestep, forecast, calendar, relative_positions
        )
        if predicted.shape != noise.shape:
            raise ValueError(
                f"epsilon prediction shape {tuple(predicted.shape)} != noise {tuple(noise.shape)}"
            )
        return F.mse_loss(predicted, noise)

    def denoise_step(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        forecast: torch.Tensor | None,
        calendar: torch.Tensor | None,
        relative_positions: torch.Tensor | None,
    ):
        predicted_noise = self.predict_noise(
            x_t, timestep, forecast, calendar, relative_positions
        )
        alpha = self.alpha[timestep].view(-1, 1, 1)
        alpha_hat = self.alpha_hat[timestep].view(-1, 1, 1)
        coefficient = (1.0 - alpha) / (1.0 - alpha_hat).sqrt()
        mean = (x_t - coefficient * predicted_noise) / alpha.sqrt()
        variance = self.reverse_variance(timestep).view(-1, 1, 1)
        noise = torch.randn_like(x_t)
        nonzero = (timestep > 0).to(x_t.dtype).view(-1, 1, 1)
        return mean + nonzero * variance.sqrt() * noise

    def sample(
        self,
        batch_size: int,
        device,
        n_samples: int = 1,
        forecast: torch.Tensor | None = None,
        calendar: torch.Tensor | None = None,
        relative_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        n_samples = int(n_samples)
        effective_batch = batch_size * n_samples
        if self.denoiser.use_sequence_condition:
            if forecast is None or calendar is None or relative_positions is None:
                raise ValueError("conditioned sampling requires all sequence conditions")
            forecast = forecast.repeat_interleave(n_samples, dim=0)
            calendar = calendar.repeat_interleave(n_samples, dim=0)
            relative_positions = relative_positions.repeat_interleave(n_samples, dim=0)
        else:
            forecast = calendar = relative_positions = None

        x_t = torch.randn(
            effective_batch, 3, self.denoiser.sequence_length, device=device
        )
        self.last_sampling_timesteps = []
        for step in range(self.num_steps - 1, -1, -1):
            timestep = torch.full(
                (effective_batch,), step, device=device, dtype=torch.long
            )
            self.last_sampling_timesteps.append(step)
            x_t = self.denoise_step(
                x_t, timestep, forecast, calendar, relative_positions
            )
        return x_t.reshape(batch_size, n_samples, 3, self.denoiser.sequence_length)


class V5Stage1Model(nn.Module):
    """Training/generation wrapper matching the legacy high-level interface."""

    def __init__(self, config: Mapping[str, object], device):
        super().__init__()
        self.config = dict(config)
        self.device = device
        self.architecture = str(self.config["architecture"])
        if self.architecture not in {"v5_t", "v5_tf"}:
            raise ValueError(f"Unsupported V5 architecture={self.architecture!r}")
        self.target_type = str(self.config.get("target_type", "residual"))
        if self.target_type != "residual":
            raise ValueError("V5 stage 1 supports residual diffusion targets only")
        self.use_sequence_condition = bool(self.config.get("use_sequence_condition", False))
        self.use_forecast = True
        self.use_network_condition = self.use_sequence_condition
        self.use_guidance = False
        self.condition_mode = "sequence_film" if self.use_sequence_condition else "timestep_only"
        self.cond_mask = [1, 1, 1]
        self.denoiser = V5ConditionalUNet1D(self.config)
        self.diffusion = V5GaussianDiffusion(
            self.denoiser,
            num_steps=int(self.config.get("num_steps", 500)),
            beta_start=float(self.config.get("beta_start", 0.0001)),
            beta_end=float(self.config.get("beta_end", 0.04)),
            schedule=str(self.config.get("schedule", "linear")),
            reverse_variance_type=str(self.config.get("reverse_variance_type", "posterior")),
        )

    def _select_target(self, batch) -> torch.Tensor:
        target_key = (
            "residual_target_3ch"
            if "residual_target_3ch" in batch else "residual_3ch"
        )
        x0 = batch[target_key].to(self.device)
        expected = (3, self.denoiser.sequence_length)
        if x0.ndim != 3 or tuple(x0.shape[1:]) != expected:
            raise ValueError(f"V5 residual target must be [B,3,L], got {tuple(x0.shape)}")
        return x0

    def _conditions(self, batch):
        if not self.use_sequence_condition:
            return None, None, None
        forecast = batch["forecast_3ch"].to(self.device)
        calendar_key = "calendar_8ch" if "calendar_8ch" in batch else "time_encoding"
        position_key = (
            "relative_positions" if "relative_positions" in batch else "timepoints"
        )
        calendar = batch[calendar_key].to(self.device)
        relative_positions = batch[position_key].to(self.device)
        return forecast, calendar, relative_positions

    def forward(self, batch):
        x0 = self._select_target(batch)
        forecast, calendar, relative_positions = self._conditions(batch)
        return self.diffusion(
            x0,
            forecast=forecast,
            calendar=calendar,
            relative_positions=relative_positions,
        )

    def generate(self, batch, n_samples: int = 10):
        forecast_batch = batch["forecast_3ch"].to(self.device)
        forecast, calendar, relative_positions = self._conditions(batch)
        with torch.no_grad():
            return self.diffusion.sample(
                batch_size=forecast_batch.shape[0],
                device=self.device,
                n_samples=n_samples,
                forecast=forecast,
                calendar=calendar,
                relative_positions=relative_positions,
            )
