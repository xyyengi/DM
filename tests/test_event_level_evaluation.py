import unittest

import numpy as np

from src.eval.event_level_evaluation import (
    lead_group,
    window_probability_metrics,
)


class LeadGroupTests(unittest.TestCase):
    def test_left_closed_right_open_bins(self):
        self.assertEqual(lead_group(0), "0-24h")
        self.assertEqual(lead_group(23), "0-24h")
        self.assertEqual(lead_group(24), "24-48h")
        self.assertEqual(lead_group(48), "48-72h")
        self.assertEqual(lead_group(72), "72-168h")
        self.assertEqual(lead_group(167), "72-168h")
        self.assertIsNone(lead_group(-1))
        self.assertIsNone(lead_group(168))


class ProbabilityMetricTests(unittest.TestCase):
    def test_perfect_identical_ensemble(self):
        actual = np.arange(12, dtype=float).reshape(3, 4)
        samples = np.repeat(actual[None, :, :], 5, axis=0)

        metrics = window_probability_metrics(samples, actual, np.ones(3))

        self.assertEqual(metrics["total_crps_mw"], 0.0)
        self.assertEqual(metrics["multivariate_energy_score_mw"], 0.0)
        self.assertEqual(metrics["total_coverage_90"], 100.0)
        self.assertEqual(metrics["total_width_90_pct_range"], 0.0)


if __name__ == "__main__":
    unittest.main()
