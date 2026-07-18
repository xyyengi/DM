import unittest

import numpy as np

from evaluation import (
    compute_energy_score,
    compute_multivariate_energy_score,
    evaluate_multichannel,
)
from src.eval.experiment_logger import build_summary_row


class EnergyScoreTests(unittest.TestCase):
    def test_identical_ensemble_equals_distance_to_observation(self):
        samples = np.zeros((1, 3, 2), dtype=np.float64)
        actual = np.array([[3.0, 4.0]], dtype=np.float64)

        self.assertAlmostEqual(compute_energy_score(samples, actual), 5.0)

    def test_symmetric_two_member_ensemble_can_have_zero_unbiased_score(self):
        samples = np.array([[[0.0], [2.0]]], dtype=np.float64)
        actual = np.array([[1.0]], dtype=np.float64)

        self.assertAlmostEqual(compute_energy_score(samples, actual), 0.0)

    def test_single_channel_matches_multivariate_implementation(self):
        rng = np.random.default_rng(2026)
        samples = rng.normal(size=(4, 5, 1, 6))
        actual = rng.normal(size=(4, 1, 6))

        single_channel = compute_energy_score(samples[:, :, 0, :], actual[:, 0, :])
        multivariate = compute_multivariate_energy_score(samples, actual)

        self.assertAlmostEqual(single_channel, multivariate, places=12)

    def test_multichannel_reports_total_nominal_intervals(self):
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(2, 4, 3, 30))
        actual = rng.normal(size=(2, 3, 30))

        metrics = evaluate_multichannel(samples, actual, verbose=False)

        self.assertIn("total_coverage_90%", metrics)
        self.assertIn("total_width_90%", metrics)
        self.assertIn("total_coverage_95%", metrics)
        self.assertIn("total_width_95%", metrics)

    def test_summary_prefers_nominal_90_percent_interval(self):
        row = build_summary_row(
            config={},
            run_id="test-run",
            timestamp="20260718_000000",
            metrics={
                "total_coverage_100%": 99.0,
                "total_coverage_90%": 90.5,
                "total_width_100%": 80.0,
                "total_width_90%": 42.0,
            },
        )

        self.assertEqual(row["coverage"], 90.5)
        self.assertEqual(row["interval_width"], 42.0)


if __name__ == "__main__":
    unittest.main()
