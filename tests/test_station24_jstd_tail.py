import unittest

import torch

from src.models.station_joint_decomposed_tail import (
    ComplementaryTemporalProjection,
    JointSpatioTemporalDecomposedTail,
    same_length_average,
)


class JSTDProjectionTests(unittest.TestCase):
    def test_complement_is_exact_for_168_hours(self):
        torch.manual_seed(7)
        value = torch.randn(3, 24, 168)
        low, fast = ComplementaryTemporalProjection(12).split(value)
        self.assertEqual(low.shape, value.shape)
        self.assertLess(float((low + fast - value).abs().max()), 1e-6)

    def test_constant_has_no_fast_component(self):
        value = torch.full((2, 24, 168), 0.37)
        low, fast = ComplementaryTemporalProjection(12).split(value)
        self.assertTrue(torch.allclose(low, value, atol=1e-6))
        self.assertLess(float(fast.abs().max()), 1e-6)

    def test_even_width_filter_keeps_length(self):
        value = torch.randn(2, 24, 168)
        self.assertEqual(same_length_average(value, 12).shape, value.shape)
        self.assertEqual(same_length_average(value, 24).shape, value.shape)


class JSTDIdentityTests(unittest.TestCase):
    def _module(self):
        features = torch.zeros(24, 5)
        features[:13, 0] = 1.0
        features[13:, 1] = 1.0
        adjacency = torch.eye(24)
        return JointSpatioTemporalDecomposedTail(
            32,
            features,
            adjacency,
            torch.ones(24),
            config={"sequence_length": 168, "jstd_channels": 16},
        )

    def test_zero_initialized_tail_is_exact_identity(self):
        module = self._module()
        hidden = torch.randn(2, 24, 32, 168)
        forecast = torch.rand(2, 24, 168)
        recent = torch.randn(2, 24, 24)
        mask = torch.ones(2, 24, 1)
        result = module(hidden, forecast, recent, mask, route=1.0)
        self.assertEqual(result.slow_mask.shape, (2, 24, 168))
        self.assertEqual(result.fast_mask.shape, (2, 24, 168))
        self.assertTrue(torch.equal(result.correction, torch.zeros_like(result.correction)))

    def test_route_zero_is_identity_after_nonzero_parameters(self):
        module = self._module()
        with torch.no_grad():
            module.slow_raw.weight.fill_(0.1)
            module.fast_raw.weight.fill_(0.1)
        result = module(
            torch.randn(1, 24, 32, 168),
            torch.rand(1, 24, 168),
            route=0.0,
        )
        self.assertTrue(torch.equal(result.correction, torch.zeros_like(result.correction)))

    def test_future_actual_cannot_enter_condition_builder(self):
        module = self._module()
        names = set(module._causal_condition_groups.__code__.co_varnames)
        self.assertNotIn("actual", names)
        self.assertNotIn("residual", names)
        self.assertNotIn("forecast_revision", names)


if __name__ == "__main__":
    unittest.main()
