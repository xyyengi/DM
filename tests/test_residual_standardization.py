import unittest

import numpy as np

from src.training.residual_standardization import (
    fit_residual_standardizer,
    inverse_standardize_residual,
    recover_stride_one_unique_hours,
    standardize_residual,
)


def make_windows(hourly, length=4):
    return np.stack([hourly[start : start + length] for start in range(len(hourly) - length + 1)])


class ResidualStandardizationTests(unittest.TestCase):
    def test_recovers_unique_hours_without_overlap_weighting(self):
        hourly = np.arange(24, dtype=np.float64).reshape(8, 3)
        windows = make_windows(hourly)

        recovered = recover_stride_one_unique_hours(windows)

        np.testing.assert_array_equal(recovered, hourly)

    def test_rejects_inconsistent_overlaps(self):
        hourly = np.arange(24, dtype=np.float64).reshape(8, 3)
        windows = make_windows(hourly)
        windows[1, 0, 0] += 1.0

        with self.assertRaisesRegex(ValueError, "not consistent stride-one"):
            recover_stride_one_unique_hours(windows)

    def test_fit_uses_internal_forecast_minus_actual_sign(self):
        actual_minus_forecast = np.array([
            [-2.0, 1.0, 3.0],
            [-1.0, 2.0, 4.0],
            [0.0, 3.0, 5.0],
            [1.0, 4.0, 6.0],
            [2.0, 5.0, 7.0],
        ])
        windows = make_windows(actual_minus_forecast, length=3)

        stats = fit_residual_standardizer(
            windows,
            residual_definition="actual_minus_forecast",
            normalization_divisors=np.ones(3),
        )

        np.testing.assert_allclose(stats["mean"], -actual_minus_forecast.mean(axis=0))
        np.testing.assert_allclose(stats["std"], actual_minus_forecast.std(axis=0))
        self.assertEqual(stats["n_unique_hours"], 5)

    def test_standardize_inverse_roundtrip(self):
        stats = {
            "enabled": True,
            "channel_order": ["wind", "solar", "load"],
            "mean": [0.1, -0.2, 0.03],
            "std": [0.5, 0.25, 0.02],
            "epsilon": 1e-6,
        }
        rng = np.random.default_rng(7)
        residual = rng.normal(size=(2, 5, 3, 4))

        standardized = standardize_residual(residual, stats, channel_axis=2)
        restored = inverse_standardize_residual(standardized, stats, channel_axis=2)

        np.testing.assert_allclose(restored, residual, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
