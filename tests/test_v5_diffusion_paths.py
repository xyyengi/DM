import unittest
import tempfile
from pathlib import Path

import torch


class V5DiffusionPathTests(unittest.TestCase):
    class SpyDenoiser(torch.nn.Module):
        def __init__(self, sequence_length=168):
            super().__init__()
            self.sequence_length = sequence_length
            self.use_sequence_condition = False
            self.scale = torch.nn.Parameter(torch.tensor(0.0))
            self.calls = []

        def forward(
            self,
            x_t,
            timestep,
            forecast=None,
            calendar=None,
            relative_positions=None,
        ):
            self.calls.append((tuple(x_t.shape), timestep.detach().cpu().tolist()))
            return torch.zeros_like(x_t) + self.scale

    def make_diffusion(self, variance_type="posterior"):
        from src.models.v5_conditioned_diffusion import V5GaussianDiffusion

        denoiser = self.SpyDenoiser()
        diffusion = V5GaussianDiffusion(
            denoiser,
            num_steps=4,
            beta_start=1e-4,
            beta_end=0.04,
            schedule="linear",
            reverse_variance_type=variance_type,
        )
        return denoiser, diffusion

    def test_training_passes_the_exact_sampled_timestep_to_denoiser(self):
        denoiser, diffusion = self.make_diffusion()
        x0 = torch.randn(2, 3, 168)
        timestep = torch.tensor([0, 3])

        loss = diffusion(x0, timestep=timestep)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.equal(diffusion.last_training_timesteps, timestep))
        self.assertEqual(denoiser.calls, [((2, 3, 168), [0, 3])])

    def test_sampling_passes_every_reverse_timestep_and_keeps_three_channels(self):
        denoiser, diffusion = self.make_diffusion()

        samples = diffusion.sample(batch_size=2, device="cpu", n_samples=3)

        self.assertEqual(tuple(samples.shape), (2, 3, 3, 168))
        self.assertEqual(diffusion.last_sampling_timesteps, [3, 2, 1, 0])
        self.assertEqual(len(denoiser.calls), 4)
        for expected_timestep, (shape, values) in zip(
            [3, 2, 1, 0], denoiser.calls
        ):
            self.assertEqual(shape, (6, 3, 168))
            self.assertEqual(values, [expected_timestep] * 6)

    def test_posterior_variance_matches_closed_form_and_zero_final_step(self):
        _, diffusion = self.make_diffusion("posterior")
        timestep = torch.tensor([0, 2])

        actual = diffusion.reverse_variance(timestep)
        expected_t2 = (
            diffusion.beta[2]
            * (1.0 - diffusion.alpha_hat[1])
            / (1.0 - diffusion.alpha_hat[2])
        )

        self.assertEqual(actual[0].item(), 0.0)
        self.assertTrue(torch.allclose(actual[1], expected_t2))

    def test_v5_tf_forward_backward_and_strict_state_roundtrip(self):
        from src.models import build_model, load_model_checkpoint

        config = {
            "architecture": "v5_tf",
            "in_channels": 3,
            "out_channels": 3,
            "sequence_length": 168,
            "base_channels": 4,
            "num_layers": 3,
            "channel_multipliers": [1, 2, 4],
            "group_norm_groups": 2,
            "dropout": 0.0,
            "timestep_embedding_dim": 8,
            "position_embedding_dim": 8,
            "use_sequence_condition": True,
            "target_type": "residual",
            "num_steps": 3,
            "reverse_variance_type": "posterior",
        }
        batch = {
            "residual_target_3ch": torch.randn(2, 3, 168),
            "forecast_3ch": torch.randn(2, 3, 168),
            "calendar_8ch": torch.randn(2, 8, 168),
            "relative_positions": torch.arange(168).float().repeat(2, 1),
        }
        model = build_model(config, torch.device("cpu"))

        loss = model(batch)
        loss.backward()

        self.assertTrue(any(
            parameter.grad is not None for parameter in model.parameters()
        ))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "v5_smoke.pt"
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        clone = build_model(config, torch.device("cpu"))
        missing, unexpected = load_model_checkpoint(clone, checkpoint)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        samples = clone.generate(batch, n_samples=2)
        self.assertEqual(tuple(samples.shape), (2, 2, 3, 168))


if __name__ == "__main__":
    unittest.main()
