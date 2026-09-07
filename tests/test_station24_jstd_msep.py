import unittest

import torch

from src.models.station_joint_decomposed_tail import (
    JointSpatioTemporalDecomposedTail,
)


class JSTDMultiscaleSegmentPriorTests(unittest.TestCase):
    def _module(self):
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
                "use_jstd_segment_prior": True,
                "jstd_segment_max_events": 2,
                "jstd_segment_tail_fraction": 0.10,
            },
        )

    def test_prior_shapes_and_finite_structured_loss(self):
        torch.manual_seed(7)
        module = self._module()
        forecast = torch.rand(3, 24, 168)
        output = module.segment_prior_output(forecast)
        self.assertEqual(output.count_logits.shape, (3, 3))
        self.assertEqual(output.onset_logits.shape, (3, 2, 168))
        self.assertEqual(output.attribute_raw.shape, (3, 2, 7, 168))
        target = torch.zeros(3, 2, 6)
        target[0, 0] = torch.tensor([1.0, 0.25, 0.10, -0.8, 0.1, 0.7])
        target[1, 0] = torch.tensor([1.0, 0.40, 0.04, -0.5, -0.2, 0.5])
        target[1, 1] = torch.tensor([1.0, 0.75, 0.20, 0.2, -0.9, 0.8])
        loss, parts = module.segment_prior_loss(
            forecast, target, torch.ones(3)
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"count", "onset", "mark"})
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for name, parameter in module.named_parameters()
                if name.startswith("segment_prior.")
            )
        )

    def test_member_sampling_returns_continuous_hypotheses(self):
        torch.manual_seed(11)
        module = self._module()
        # Force the neutral test model to sample one event per member.
        with torch.no_grad():
            module.segment_prior.count_head[-1].bias.copy_(
                torch.tensor([-20.0, 20.0, -20.0])
            )
        hypothesis, audit = module.sample_segment_hypotheses(
            torch.rand(2, 24, 168), members=7
        )
        self.assertEqual(hypothesis.shape, (2, 7, 2, 6))
        self.assertEqual(audit["counts"].shape, (2, 7))
        self.assertTrue(torch.all((audit["counts"] == 0) | (audit["counts"] == 1)))
        self.assertTrue(torch.equal((audit["counts"] > 0).sum(dim=1), torch.ones(2, dtype=torch.long)))
        self.assertTrue(torch.equal(hypothesis[:, :, 0, 0], (audit["counts"] > 0).float()))
        self.assertTrue(torch.all(hypothesis[:, :, 1, 0] == 0))
        self.assertTrue(torch.all((hypothesis[..., 2] > 0) & (hypothesis[..., 2] <= 1)))

    def test_two_segments_expand_without_collapsing_to_one_interval(self):
        module = self._module()
        hypothesis = torch.tensor(
            [[
                [1.0, 12.0 / 167.0, 3.0 / 168.0, -0.8, 0.0, 0.7],
                [1.0, 96.0 / 167.0, 24.0 / 168.0, -0.6, -0.4, 0.8],
            ]]
        )
        fields, envelope, bounds = module.event_hypothesis_fields(
            hypothesis, torch.float32
        )
        self.assertEqual(fields.shape, (1, 24, 5, 168))
        self.assertEqual(envelope.shape, (1, 168))
        self.assertEqual(bounds.shape, (1, 2, 2))
        self.assertGreater(float(envelope[0, 13]), 0.25)
        self.assertGreater(float(envelope[0, 108]), 0.25)
        self.assertLess(float(envelope[0, 60]), 1e-3)


if __name__ == "__main__":
    unittest.main()
