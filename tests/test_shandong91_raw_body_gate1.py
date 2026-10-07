import unittest

import torch

from src.models.shandong91_conditioned_diffusion import (
    Shandong91HeterogeneousRawBody,
    Shandong91MaskedDiffusion,
)


class Shandong91Gate1Test(unittest.TestCase):
    def test_fixed_batch_objective_can_take_two_finite_steps(self):
        torch.manual_seed(7)
        config = {
            "sequence_length": 8, "node_count": 91, "resource_count": 3,
            "base_channels": 8, "num_layers": 2, "channel_multipliers": [1, 2],
            "group_norm_groups": 4, "dropout": 0.0, "timestep_embedding_dim": 16,
            "use_lead_condition": False,
        }
        node_features = torch.randn(91, 10)
        adjacency = torch.eye(91, dtype=torch.bool)
        denoiser = Shandong91HeterogeneousRawBody(config, node_features, adjacency)
        model = Shandong91MaskedDiffusion(denoiser, {"num_steps": 10, "beta_start": 1e-4, "beta_end": 0.02})
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = {
            "residual": torch.randn(1, 8, 91, 3),
            "forecast": torch.randn(1, 8, 91, 3),
            "forecast_valid_mask": torch.ones(1, 8, 91, 3, dtype=torch.bool),
            "time_mark": torch.randn(1, 8, 8),
            "effective_mask": torch.ones(1, 8, 91, 3, dtype=torch.bool),
        }
        timestep = torch.tensor([3])
        noise = torch.randn_like(batch["residual"])
        before = denoiser.encoder_blocks[0].conv1.weight.detach().clone()
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            _, element_loss = model.prediction_and_error(batch, timestep, noise)
            loss = model.masked_loss(element_loss, batch["effective_mask"])
            self.assertTrue(torch.isfinite(loss))
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        update = (denoiser.encoder_blocks[0].conv1.weight.detach() - before).abs().max()
        self.assertGreater(float(update), 0.0)
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
