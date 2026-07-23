import unittest
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.eval.physical_projection import (
    conservative_shandong_daylight,
    daylight_mask_from_export_metadata,
    daylight_mask_from_train_support,
    project_power_scenarios,
)


class PhysicalProjectionTests(unittest.TestCase):
    def test_projects_bounds_and_preserves_raw_input(self):
        raw = np.array([[
            [[-2.0, 5.0, 12.0], [3.0, -4.0, 15.0], [-1.0, 4.0, 8.0]],
            [[2.0, 7.0, 8.0], [9.0, 4.0, -3.0], [2.0, -6.0, 10.0]],
        ]])
        forecast = np.array([[[1.0, 2.0, 3.0], [0.0, 5.0, 0.5], [8.0, 8.0, 8.0]]])
        original = raw.copy()

        projected, report = project_power_scenarios(
            raw, forecast, np.array([10.0, 10.0, 100.0]), 1.0
        )

        np.testing.assert_array_equal(raw, original)
        self.assertTrue(np.all((projected[:, :, 0] >= 0) & (projected[:, :, 0] <= 10)))
        self.assertTrue(np.all((projected[:, :, 1] >= 0) & (projected[:, :, 1] <= 10)))
        self.assertTrue(np.all(projected[:, :, 2] >= 0))
        np.testing.assert_array_equal(projected[:, :, 1, [0, 2]], 0.0)
        self.assertGreater(report["raw_boundary_rates"]["any_physical_violation_pct"], 0)
        self.assertEqual(
            report["projected_boundary_rates"]["any_physical_violation_pct"], 0.0
        )
        self.assertEqual(
            report["solar_daylight_audit"]["method"],
            "forecast_threshold_fallback",
        )

    def test_timestamp_mask_controls_night_without_forecast_proxy(self):
        raw = np.zeros((1, 2, 3, 3), dtype=np.float32)
        raw[:, :, 1, :] = 5.0
        forecast = np.zeros((1, 3, 3), dtype=np.float32)
        daylight = np.array([[False, True, False]])

        projected, report = project_power_scenarios(
            raw,
            forecast,
            np.array([10.0, 10.0, 100.0]),
            solar_daylight_mask=daylight,
            solar_daylight_metadata={"method": "test_mask"},
        )

        np.testing.assert_array_equal(
            projected[0, :, 1, :],
            np.array([[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]),
        )
        self.assertEqual(report["solar_daylight_audit"]["method"], "test_mask")

    def test_shandong_astronomical_mask_distinguishes_noon_and_midnight(self):
        china_tz = timezone(timedelta(hours=8))
        daylight = conservative_shandong_daylight([
            datetime(2025, 6, 21, 12, 30, tzinfo=china_tz),
            datetime(2025, 12, 21, 12, 30, tzinfo=china_tz),
        ])
        nighttime = conservative_shandong_daylight([
            datetime(2025, 6, 21, 0, 30, tzinfo=china_tz),
            datetime(2025, 12, 21, 0, 30, tzinfo=china_tz),
        ])

        np.testing.assert_array_equal(daylight, [True, True])
        np.testing.assert_array_equal(nighttime, [False, False])

    def test_metadata_mask_reconstructs_stride_one_windows(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "export_metadata.json"
            path.write_text(json.dumps({
                "splits": {
                    "val": {
                        "hours": 4,
                        "windows": 2,
                        "start_local": "2025-06-21 00:00:00+08:00",
                    }
                }
            }), encoding="utf-8")
            mask, audit = daylight_mask_from_export_metadata(
                path, "val", window_count=2, sequence_length=3
            )

        self.assertEqual(mask.shape, (2, 3))
        np.testing.assert_array_equal(mask[0, 1:], mask[1, :-1])
        self.assertEqual(audit["unique_total_hours"], 4)

    def test_train_support_mask_uses_only_train_clock_hours(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_unique = np.zeros((48, 3), dtype=np.float32)
            for index in range(48):
                if 6 <= index % 24 <= 17:
                    train_unique[index, 1] = 0.5
            train_windows = np.stack(
                [train_unique[0:4], train_unique[1:5]], axis=0
            )
            np.save(root / "train_actual.npy", train_windows)
            (root / "export_metadata.json").write_text(json.dumps({
                "splits": {
                    "train": {
                        "hours": 5,
                        "windows": 2,
                        "start_local": "2025-01-01 00:00:00+08:00",
                    },
                    "val": {
                        "hours": 5,
                        "windows": 2,
                        "start_local": "2025-11-01 05:00:00+08:00",
                    },
                }
            }), encoding="utf-8")

            # Rebuild the compact train fixture so its only supported hour is 03.
            compact = np.zeros((5, 3), dtype=np.float32)
            compact[3, 1] = 0.5
            np.save(
                root / "train_actual.npy",
                np.stack([compact[0:4], compact[1:5]], axis=0),
            )
            mask, audit = daylight_mask_from_train_support(
                root, "val", window_count=2, sequence_length=4
            )

        self.assertEqual(audit["supported_local_hours"], [3])
        self.assertFalse(mask.any())
        self.assertFalse(audit["validation_or_test_actual_used_for_fit"])

    def test_rejects_mismatched_forecast_shape(self):
        with self.assertRaisesRegex(ValueError, "forecast_mw"):
            project_power_scenarios(
                np.zeros((2, 3, 3, 4)),
                np.zeros((2, 3, 5)),
                np.ones(3),
            )


if __name__ == "__main__":
    unittest.main()
