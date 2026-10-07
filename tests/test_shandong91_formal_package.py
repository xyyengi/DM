import unittest

import torch

from src.models.shandong91_conditioned_diffusion import (
    Shandong91HeterogeneousRawBody,
    Shandong91MaskedDiffusion,
)


class FormalPackageTests(unittest.TestCase):
    def test_ddim_shape_sign_and_inactive_policy(self):
        torch.manual_seed(3)
        config = {
            "sequence_length": 8, "node_count": 91, "resource_count": 3,
            "base_channels": 8, "num_layers": 2, "channel_multipliers": [1, 2],
            "group_norm_groups": 4, "dropout": 0.0, "timestep_embedding_dim": 16,
            "use_lead_condition": False,
        }
        denoiser = Shandong91HeterogeneousRawBody(
            config, torch.randn(91, 10), torch.eye(91, dtype=torch.bool)
        )
        model = Shandong91MaskedDiffusion(
            denoiser, {"num_steps": 8, "beta_start": 1e-4, "beta_end": 0.02}
        )
        forecast = torch.randn(1, 8, 91, 3)
        valid = torch.ones_like(forecast, dtype=torch.bool)
        mask = valid.clone()
        mask[..., 0, 1] = False
        residual = model.sample_ddim(
            forecast, valid, torch.randn(1, 8, 8), mask,
            inference_steps=3, initial_noise=torch.randn_like(forecast),
        )
        self.assertEqual(tuple(residual.shape), (1, 8, 91, 3))
        self.assertTrue(torch.isfinite(residual).all())
        self.assertEqual(float(residual[..., 0, 1].abs().max()), 0.0)
        actual = forecast + residual
        self.assertTrue(torch.equal(actual[..., 0, 1], forecast[..., 0, 1]))


if __name__ == "__main__":
    unittest.main()
