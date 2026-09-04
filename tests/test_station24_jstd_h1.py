import unittest

import torch

from src.models.station_joint_decomposed_tail import (
    JointSpatioTemporalDecomposedTail,
)


class JSTDContinuousHypothesisTests(unittest.TestCase):
    def _module(self, use_hypothesis=True):
        features = torch.zeros(24, 5)
        features[:13, 0] = 1.0
        features[13:, 1] = 1.0
        return JointSpatioTemporalDecomposedTail(
            32,
            features,
            torch.eye(24),
            torch.ones(24),
            config={
                "sequence_length": 168,
                "jstd_channels": 16,
                "use_jstd_event_hypothesis": use_hypothesis,
                "jstd_hypothesis_edge_temperature_hours": 1.5,
            },
        )

    def test_null_hypothesis_creates_no_structured_field(self):
        module = self._module()
        fields, envelope, bounds = module.event_hypothesis_fields(
            torch.zeros(2, 6), torch.float32
        )
        self.assertEqual(fields.shape, (2, 24, 5, 168))
        self.assertEqual(envelope.shape, (2, 168))
        self.assertEqual(bounds.shape, (2, 2))
        self.assertTrue(torch.equal(fields, torch.zeros_like(fields)))

    def test_onset_duration_and_signed_source_are_encoded(self):
        module = self._module()
        hypothesis = torch.tensor(
            [[1.0, 42.0 / 167.0, 12.0 / 168.0, -0.8, 0.5, 0.7]]
        )
        fields, envelope, bounds = module.event_hypothesis_fields(
            hypothesis, torch.float32
        )
        self.assertAlmostEqual(float(bounds[0, 0]), 42.0, places=4)
        self.assertAlmostEqual(float(bounds[0, 1]), 54.0, places=4)
        peak = int(envelope[0].argmax())
        self.assertGreaterEqual(peak, 42)
        self.assertLessEqual(peak, 54)
        self.assertLess(float(fields[0, 0, 1, 48]), 0.0)
        self.assertGreater(float(fields[0, 13, 1, 48]), 0.0)

    def test_h1_starts_as_exact_jstd_v1_identity(self):
        torch.manual_seed(17)
        v1 = self._module(use_hypothesis=False)
        with torch.no_grad():
            for parameter in v1.parameters():
                parameter.uniform_(-0.05, 0.05)
        h1 = self._module(use_hypothesis=True)
        incompatible = h1.load_state_dict(v1.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        hidden = torch.randn(1, 24, 32, 168)
        forecast = torch.rand(1, 24, 168)
        v1_result = v1(hidden, forecast, route=1.0)
        h1_result = h1(
            hidden,
            forecast,
            route=1.0,
            event_hypothesis=torch.tensor(
                [[1.0, 0.25, 0.10, -0.8, 0.0, 0.7]]
            ),
        )
        self.assertTrue(
            torch.equal(v1_result.correction, h1_result.correction)
        )

    def test_h1_forward_rejects_missing_hypothesis(self):
        module = self._module()
        with self.assertRaisesRegex(ValueError, "event_hypothesis"):
            module(
                torch.randn(1, 24, 32, 168),
                torch.rand(1, 24, 168),
                route=1.0,
            )


if __name__ == "__main__":
    unittest.main()
