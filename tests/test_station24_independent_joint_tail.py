import unittest

import torch

from src.models.station_conditioned_diffusion import Station24DiffusionModel


class IndependentJointTailTests(unittest.TestCase):
    def _model(self):
        features = torch.zeros(24, 5)
        features[:13, 0] = 1.0
        features[13:, 1] = 1.0
        config = {
            "architecture": "station24_resunet",
            "spatial_mode": "fixed_graph",
            "station_count": 24,
            "sequence_length": 32,
            "base_channels": 4,
            "num_layers": 3,
            "channel_multipliers": [1, 2, 4],
            "group_norm_groups": 4,
            "dropout": 0.0,
            "timestep_embedding_dim": 8,
            "num_steps": 4,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "use_body_tail_experts": False,
            "ramp_auxiliary_loss_weight": 0.06,
            "ramp_auxiliary_lags": [1, 3, 6],
            "ramp_auxiliary_lag_weights": [0.5, 0.3, 0.2],
            "tail_multiscale_slow_loss_weight": 0.08,
            "tail_multiscale_slow_windows": [12, 24],
        }
        return Station24DiffusionModel(
            config, features, torch.eye(24), torch.linspace(1.0, 2.0, 24)
        )

    @staticmethod
    def _batch(length=32):
        batch = 2
        return {
            "residual_target": torch.randn(batch, 24, length),
            "residual": torch.randn(batch, 24, length),
            "residual_scale": torch.ones(batch, 24, length),
            "forecast": torch.rand(batch, 24, length),
            "calendar": torch.rand(batch, 8, length),
            "lead": torch.rand(batch, 2, length),
            "valid_mask": torch.ones(batch, 24, length),
            "recent_error": torch.randn(batch, 24, 24),
            "recent_error_mask": torch.ones(batch, 24, 1),
        }

    def test_full_joint_model_has_no_legacy_tail_route(self):
        model = self._model()
        self.assertFalse(model.use_body_tail_experts)
        self.assertFalse(model.use_jstd_tail)
        self.assertFalse(model.use_joint_multiresidual_tail)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            sum(parameter.numel() for parameter in model.parameters()),
        )

    def test_fast_and_slow_objectives_backpropagate_through_full_denoiser(self):
        torch.manual_seed(9)
        model = self._model()
        batch = self._batch()
        loss = model(batch, timestep=torch.tensor([1, 2]), noise=torch.randn(2, 24, 32))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.diffusion.last_loss_components["ramp"]), 0.0)
        self.assertGreater(float(model.diffusion.last_loss_components["tail_multiscale_slow"]), 0.0)
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters() if parameter.requires_grad
        ))

    def test_optimizer_updates_full_model(self):
        torch.manual_seed(10)
        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer.zero_grad()
        model(self._batch(), timestep=torch.tensor([1, 2]), noise=torch.randn(2, 24, 32)).backward()
        optimizer.step()
        changed = [
            name for name, value in model.state_dict().items()
            if value.dtype.is_floating_point and not torch.equal(value, before[name])
        ]
        self.assertTrue(all(
            not torch.equal(parameter.detach(), before[name])
            for name, parameter in model.named_parameters()
        ))

    def test_legacy_default_and_explicit_zero_preserve_output(self):
        torch.manual_seed(11)
        model = self._model().eval()
        model.diffusion.tail_multiscale_slow_loss_weight = 0.0
        batch = self._batch()
        noise = torch.randn(2, 24, 32)
        first = model(batch, timestep=torch.tensor([1, 2]), noise=noise)
        del model.config["tail_multiscale_slow_loss_weight"]
        features = torch.zeros(24, 5)
        features[:13, 0] = 1
        features[13:, 1] = 1
        legacy = Station24DiffusionModel(model.config, features, torch.eye(24), torch.ones(24)).eval()
        legacy.load_state_dict(model.state_dict(), strict=True)
        second = legacy(batch, timestep=torch.tensor([1, 2]), noise=noise)
        self.assertTrue(torch.equal(first, second))

    def test_generation_ignores_future_labels_and_roundtrip_is_exact(self):
        import io
        torch.manual_seed(12)
        model = self._model().eval()
        batch = self._batch()
        with torch.no_grad():
            torch.manual_seed(13)
            first = model.generate(batch, 2)
            state = io.BytesIO()
            torch.save(model.state_dict(), state)
            state.seek(0)
            model.load_state_dict(torch.load(state, weights_only=True))
            for key in ("residual", "residual_target", "actual", "jstd_event_hypothesis"):
                batch[key] = torch.randn(2, 24, 32)
            torch.manual_seed(13)
            second = model.generate(batch, 2)
        self.assertTrue(torch.equal(first, second))

    def test_slow_projection_excludes_invalid_neighbors(self):
        from unittest.mock import patch
        model = self._model().eval()
        model.diffusion.ramp_auxiliary_loss_weight = 0.0
        batch = self._batch()
        batch["valid_mask"][:, :, 10:15] = 0
        noise = torch.randn(2, 24, 32)
        changed = noise.clone()
        changed[:, :, 10:15] += 100
        with patch.object(model.denoiser, "forward", return_value=torch.zeros_like(noise)):
            first = model(batch, timestep=torch.tensor([1, 2]), noise=noise)
            second = model(batch, timestep=torch.tensor([1, 2]), noise=changed)
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=0))


if __name__ == "__main__":
    unittest.main()
