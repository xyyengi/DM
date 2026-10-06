import unittest

import numpy as np

from tools.diagnose_station24_wind_direction_recovery import (
    direction_summary,
    recovery_metrics,
    select_strong_negative_anchors,
)


class WindDirectionRecoveryTests(unittest.TestCase):
    def test_direction_summary_separates_sign_and_reports_magnitude(self):
        values = np.array([-10.0, -2.0, 1.0, 5.0])
        positive = direction_summary(values, "positive")
        negative = direction_summary(values, "negative")
        self.assertEqual(positive["count"], 2)
        self.assertEqual(negative["count"], 2)
        self.assertEqual(positive["max"], 5.0)
        self.assertEqual(negative["max"], 10.0)

    def test_recovery_metrics_use_trough_and_future_only(self):
        series = np.array([10.0, 4.0, 5.0, 7.0, 9.0])
        row = recovery_metrics(series, lead_end=1, drop_reference=6.0, horizon=3)
        self.assertEqual(row["any_positive"], 1.0)
        self.assertAlmostEqual(row["max_recovery_mw"], 5.0)
        self.assertAlmostEqual(row["recovery_ratio"], 5.0 / 6.0)
        self.assertEqual(row["time_to_first_positive_h"], 1.0)
        self.assertEqual(row["time_to_half_recovery_h"], 2.0)

    def test_anchor_selection_applies_threshold_and_spacing(self):
        actual = np.array([[10.0, 0.0, 9.0, -2.0, 8.0, 7.0, 6.0, -5.0]])
        event = np.ones((1, actual.shape[1] - 1), dtype=bool)
        anchors = select_strong_negative_anchors(actual, threshold=9.0, event_mask=event, min_spacing=2)
        self.assertEqual(len(anchors), 2)
        self.assertTrue(all(row["event_pm6"] for row in anchors))


if __name__ == "__main__":
    unittest.main()
