import unittest

import numpy as np

from tools.diagnose_station24_residual_frequency_decomposition import (
    actual_low_descriptor,
    centered_moving_average,
    decompose,
    high_frequency_descriptors,
    member_low_descriptor,
)
from tools.diagnose_station24_sustained_drop_tail_sweep import Event


class ResidualFrequencyDecompositionTests(unittest.TestCase):
    def test_decomposition_reconstructs_and_does_not_shift_impulse(self):
        values = np.zeros(41)
        values[20] = -10.0
        low, high = decompose(values, 12)
        np.testing.assert_allclose(low + high, values, atol=1e-12)
        support = np.flatnonzero(low < 0.0)
        self.assertAlmostEqual(float(support[0] + support[-1]) / 2.0, 19.5)

    def test_centered_average_preserves_constant(self):
        values = np.full((3, 30), 7.0)
        np.testing.assert_allclose(centered_moving_average(values, 24), values)

    def test_perfect_low_event_has_zero_descriptor_errors(self):
        low = np.zeros(80)
        low[25:37] = -100.0
        event = Event(
            event_id="synthetic",
            issue=0,
            issue_date="2026-01-01",
            onset=25,
            stop=37,
            peak_start=25,
            physical_time=np.datetime64("2026-01-02"),
            depth_mw=100.0,
            mean_shortfall_mw=100.0,
            severity_normalized=1.0,
        )
        reference = actual_low_descriptor(low, event)
        member = member_low_descriptor(low, low, reference)
        self.assertTrue(member["valid"])
        self.assertEqual(member["onset"], reference["onset"])
        self.assertEqual(member["duration"], reference["duration"])
        self.assertAlmostEqual(member["depth_mw"], reference["depth_mw"])
        self.assertAlmostEqual(member["shape_correlation"], 1.0)

    def test_high_frequency_descriptors_capture_lagged_ramps(self):
        values = np.asarray([0.0, 1.0, -1.0, 2.0, 0.0, 1.0, -2.0])
        metrics = high_frequency_descriptors(values)
        self.assertAlmostEqual(float(metrics["ramp_1h_mw"][0]), 3.0)
        self.assertGreater(float(metrics["ramp_3h_mw"][0]), 0.0)
        self.assertGreater(float(metrics["volatility_std_mw"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
