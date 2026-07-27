import unittest

import numpy as np

from src.training.residual_standardization import (
    SOLAR_CONDITIONAL_MODE,
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

    def test_fits_train_only_solar_forecast_conditional_scale(self):
        rng = np.random.default_rng(19)
        hourly_forecast = np.zeros((36, 3), dtype=np.float64)
        hourly_forecast[:, 0] = np.linspace(0.1, 0.9, 36)
        hourly_forecast[6:, 1] = np.linspace(0.02, 0.95, 30)
        hourly_forecast[:, 2] = np.linspace(0.4, 0.8, 36)
        hourly_residual = rng.normal(size=(36, 3)) * np.array([0.1, 0.1, 0.05])
        hourly_residual[:, 1] *= 0.2 + 3.0 * hourly_forecast[:, 1]
        forecast_windows = make_windows(hourly_forecast, length=6)
        residual_windows = make_windows(hourly_residual, length=6)

        stats = fit_residual_standardizer(
            residual_windows,
            residual_definition="forecast_minus_actual",
            normalization_divisors=np.ones(3),
            train_forecast_windows=forecast_windows,
            mode=SOLAR_CONDITIONAL_MODE,
            solar_conditioning={
                "n_bins": 3,
                "min_bin_count": 8,
                "shrinkage_count": 0,
                "smooth_passes": 0,
                "min_scale_ratio": 0.01,
                "max_scale_ratio": 10.0,
            },
        )

        self.assertEqual(stats["mode"], SOLAR_CONDITIONAL_MODE)
        conditional = stats["solar_conditioning"]
        self.assertEqual(conditional["n_daylight_unique_hours"], 30)
        self.assertEqual(conditional["bin_counts"], [10, 10, 10])
        self.assertGreater(conditional["std_knots"][-1], conditional["std_knots"][0])

    def test_conditional_standardize_inverse_roundtrip_for_dataset_shape(self):
        rng = np.random.default_rng(23)
        residual = rng.normal(size=(4, 7, 3))
        forecast = rng.uniform(size=(4, 7, 3))
        stats = {
            "enabled": True,
            "mode": SOLAR_CONDITIONAL_MODE,
            "channel_order": ["wind", "solar", "load"],
            "mean": [0.1, -0.2, 0.03],
            "std": [0.5, 0.25, 0.02],
            "epsilon": 1e-6,
            "solar_conditioning": {
                "forecast_knots": [0.1, 0.5, 0.9],
                "mean_knots": [-0.05, 0.10, 0.35],
                "std_knots": [0.08, 0.20, 0.45],
                "daylight_forecast_threshold_normalized": 1e-8,
            },
        }

        standardized = standardize_residual(
            residual,
            stats,
            channel_axis=2,
            forecast=forecast,
            forecast_channel_axis=2,
        )
        restored = inverse_standardize_residual(
            standardized,
            stats,
            channel_axis=2,
            forecast=forecast,
            forecast_channel_axis=2,
        )

        np.testing.assert_allclose(restored, residual, atol=1e-12)

    def test_conditional_inverse_roundtrip_for_generated_ensemble_shape(self):
        rng = np.random.default_rng(29)
        standardized = rng.normal(size=(2, 5, 3, 8))
        forecast = rng.uniform(size=(2, 3, 8))
        stats = {
            "enabled": True,
            "mode": SOLAR_CONDITIONAL_MODE,
            "channel_order": ["wind", "solar", "load"],
            "mean": [0.1, -0.2, 0.03],
            "std": [0.5, 0.25, 0.02],
            "epsilon": 1e-6,
            "solar_conditioning": {
                "forecast_knots": [0.1, 0.5, 0.9],
                "mean_knots": [-0.05, 0.10, 0.35],
                "std_knots": [0.08, 0.20, 0.45],
                "daylight_forecast_threshold_normalized": 1e-8,
            },
        }

        residual = inverse_standardize_residual(
            standardized,
            stats,
            channel_axis=2,
            forecast=forecast,
            forecast_channel_axis=1,
        )
        restored = standardize_residual(
            residual,
            stats,
            channel_axis=2,
            forecast=forecast,
            forecast_channel_axis=1,
        )

        np.testing.assert_allclose(restored, standardized, atol=1e-12)

    def test_conditional_mode_requires_forecast(self):
        stats = {
            "enabled": True,
            "mode": SOLAR_CONDITIONAL_MODE,
            "channel_order": ["wind", "solar", "load"],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "epsilon": 1e-6,
            "solar_conditioning": {
                "forecast_knots": [0.1, 0.9],
                "mean_knots": [0.0, 0.0],
                "std_knots": [0.5, 1.5],
                "daylight_forecast_threshold_normalized": 1e-8,
            },
        }

        with self.assertRaisesRegex(ValueError, "forecast"):
            standardize_residual(np.zeros((2, 4, 3)), stats, channel_axis=2)

    def test_generate_actual_path_uses_conditional_inverse_transform(self):
        from generate import model_output_to_actual

        standardized = np.zeros((1, 2, 3, 3), dtype=np.float64)
        forecast = np.array(
            [[[0.4, 0.4, 0.4], [0.2, 0.5, 0.8], [0.6, 0.6, 0.6]]],
            dtype=np.float64,
        )
        stats = {
            "enabled": True,
            "mode": SOLAR_CONDITIONAL_MODE,
            "channel_order": ["wind", "solar", "load"],
            "mean": [0.1, 0.0, 0.03],
            "std": [0.5, 0.25, 0.02],
            "epsilon": 1e-6,
            "solar_conditioning": {
                "forecast_knots": [0.2, 0.5, 0.8],
                "mean_knots": [0.01, 0.10, 0.25],
                "std_knots": [0.05, 0.20, 0.40],
                "daylight_forecast_threshold_normalized": 1e-8,
            },
        }

        actual = model_output_to_actual(
            standardized,
            forecast,
            target_type="residual",
            residual_standardizer=stats,
        )

        expected_residual = np.zeros_like(standardized)
        expected_residual[:, :, 0, :] = 0.1
        expected_residual[:, :, 1, :] = np.array([0.01, 0.10, 0.25])
        expected_residual[:, :, 2, :] = 0.03
        np.testing.assert_allclose(
            actual,
            forecast[:, None, :, :] - expected_residual,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
