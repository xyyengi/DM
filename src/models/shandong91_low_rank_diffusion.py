"""Minimal Shandong91 V3: joint common factors plus de-commoned local residuals."""
from __future__ import annotations

from collections.abc import Mapping

import torch

from datasets.shandong91_low_rank import FixedPCAFactorTransform
from src.models.shandong91_faithful24_diffusion import (
    Shandong91HeterogeneousRawBodyV2,
)


class Shandong91HeterogeneousRawBodyV3(Shandong91HeterogeneousRawBodyV2):
    """Versioned V2 backbone; factor semantics live in the diffusion wrapper."""

    architecture = "shandong91_heterogeneous_raw_body_v3_low_rank_common_factor"

    def __init__(self, config: Mapping[str, object], node_features, adjacency):
        expected = {"Wind": 5, "Solar": 1, "Load": 1}
        if dict(config.get("common_factor_counts", {})) != expected:
            raise ValueError(f"V3 requires fixed train-only PCA counts {expected}")
        if config.get("load_local_branch", "disabled_strict_rank1") != "disabled_strict_rank1":
            raise ValueError("V3 Load local branch must remain disabled")
        super().__init__(config, node_features, adjacency)


class Shandong91LowRankDiffusion(torch.nn.Module):
    """Diffuse seven factors and a projected Wind/Solar local field jointly.

    The single V2 backbone sees the reconstructed joint state and all legal V2
    conditions.  Its output is deterministically split into factor and local
    coordinates.  Load has no local coordinate, while Wind/Solar local noise
    and predictions are projected away from their fixed PCA subspaces.
    """

    def __init__(self, denoiser, diffusion_config, factor_document):
        super().__init__()
        self.denoiser = denoiser
        self.transform = FixedPCAFactorTransform(factor_document)
        steps = int(diffusion_config.get("num_steps", 500))
        beta = torch.linspace(
            float(diffusion_config.get("beta_start", 1e-4)),
            float(diffusion_config.get("beta_end", .04)), steps,
        )
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_hat", torch.cumprod(1 - beta, dim=0))

    @staticmethod
    def masked_loss(error, mask):
        weight = mask.to(error.dtype)
        return (error * weight).sum() / weight.sum().clamp_min(1)

    def _predict_latents(self, noisy_node, batch, timestep, mask):
        raw = self.denoiser(
            noisy_node, timestep, batch["forecast"], batch["forecast_valid_mask"],
            batch["time_mark"], batch["recent_error"],
            batch["recent_error_valid_mask"], batch["node_state"],
        )
        factor = self.transform.project_factor(raw, mask)
        local = self.transform.project_local(raw, mask)
        return factor, local

    def prediction_and_error(self, batch, timestep, noise=None, *, factor_noise=None,
                             local_noise=None, return_latents=False):
        mask = batch["effective_mask"].bool()
        factor_clean, local_clean = self.transform.decompose(batch["residual"], mask)
        if factor_noise is None:
            factor_noise = torch.randn_like(factor_clean)
        if local_noise is None:
            source = torch.randn_like(batch["residual"]) if noise is None else noise
            local_noise = self.transform.project_local(source, mask)
        alpha = self.alpha_hat[timestep].view(-1, 1, 1)
        noisy_factor = alpha.sqrt() * factor_clean + (1 - alpha).sqrt() * factor_noise
        node_alpha = alpha.unsqueeze(-1)
        noisy_local = node_alpha.sqrt() * local_clean + (1 - node_alpha).sqrt() * local_noise
        noisy_node = self.transform.reconstruct(noisy_factor, noisy_local, mask)
        predicted_factor, predicted_local = self._predict_latents(
            noisy_node, batch, timestep, mask
        )
        prediction = self.transform.reconstruct(
            predicted_factor, predicted_local, mask, include_mean=False
        )
        target = self.transform.reconstruct(
            factor_noise, local_noise, mask, include_mean=False
        )
        error = (prediction - target).square()
        if return_latents:
            return prediction, error, {
                "factor_clean": factor_clean, "local_clean": local_clean,
                "factor_noise": factor_noise, "local_noise": local_noise,
                "predicted_factor_noise": predicted_factor,
                "predicted_local_noise": predicted_local,
                "noisy_node": noisy_node, "target_node_noise": target,
            }
        return prediction, error

    def _step(self, current, epsilon, alpha_bar, previous_bar, method, *, noise):
        clean = (current - torch.sqrt(1 - alpha_bar) * epsilon) / torch.sqrt(alpha_bar).clamp_min(1e-12)
        if previous_bar is None:
            return clean
        if method == "ddim":
            return torch.sqrt(previous_bar) * clean + torch.sqrt(1 - previous_bar) * epsilon
        alpha = alpha_bar / previous_bar
        beta = 1 - alpha
        mean = (current - beta / torch.sqrt(1 - alpha_bar) * epsilon) / torch.sqrt(alpha)
        variance = beta * (1 - previous_bar) / (1 - alpha_bar)
        return mean + torch.sqrt(variance.clamp_min(0)) * noise

    @torch.no_grad()
    def sample(self, batch, generation_mask, *, method="ddpm", inference_steps=500,
               initial_factor_noise=None, initial_local_noise=None, generator=None):
        shape = batch["forecast"].shape
        factors = (
            torch.randn(shape[0], shape[1], self.transform.factor_count,
                        device=batch["forecast"].device,
                        dtype=batch["forecast"].dtype, generator=generator)
            if initial_factor_noise is None else initial_factor_noise.clone()
        )
        local_source = (
            torch.randn(shape, device=batch["forecast"].device,
                        dtype=batch["forecast"].dtype, generator=generator)
            if initial_local_noise is None else initial_local_noise.clone()
        )
        local = self.transform.project_local(local_source, generation_mask)
        total = len(self.alpha_hat)
        if method == "ddpm":
            if inference_steps != total:
                raise ValueError("DDPM must use every training-schedule step")
            indices = torch.arange(total - 1, -1, -1, device=factors.device)
        elif method == "ddim":
            indices = torch.linspace(total - 1, 0, inference_steps, dtype=torch.float64,
                                     device=factors.device).round().long().unique_consecutive()
        else:
            raise ValueError("method must be ddpm or ddim")
        for position, t_value in enumerate(indices):
            timestep = torch.full((shape[0],), int(t_value), device=factors.device,
                                  dtype=torch.long)
            noisy_node = self.transform.reconstruct(factors, local, generation_mask)
            epsilon_factor, epsilon_local = self._predict_latents(
                noisy_node, batch, timestep, generation_mask
            )
            alpha_bar = self.alpha_hat[int(t_value)].to(factors.dtype)
            previous_bar = None
            if position + 1 < len(indices):
                previous_bar = self.alpha_hat[int(indices[position + 1])].to(factors.dtype)
            factor_random = torch.randn(
                factors.shape, device=factors.device, dtype=factors.dtype, generator=generator
            )
            local_random = self.transform.project_local(
                torch.randn(local.shape, device=local.device, dtype=local.dtype,
                            generator=generator), generation_mask
            )
            factors = self._step(
                factors, epsilon_factor, alpha_bar, previous_bar, method,
                noise=factor_random,
            )
            local = self._step(
                local, epsilon_local, alpha_bar, previous_bar, method,
                noise=local_random,
            )
            local = self.transform.project_local(local, generation_mask)
        return self.transform.reconstruct(factors, local, generation_mask).contiguous()
