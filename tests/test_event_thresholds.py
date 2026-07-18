import unittest

import numpy as np

from src.eval.event_thresholds import (
    bridge_short_gaps,
    daily_daylight_energy_ratio,
    map_windows_to_events,
    merge_intervals,
    reconstruct_sliding_windows,
    ramp_event_summary,
    true_runs,
)


class TimelineReconstructionTests(unittest.TestCase):
    def test_stride_one_windows_reconstruct_original_timeline(self):
        timeline = np.arange(30, dtype=np.float64).reshape(10, 3)
        windows = np.stack([timeline[i : i + 4] for i in range(7)])

        reconstructed, max_error = reconstruct_sliding_windows(windows)

        np.testing.assert_array_equal(reconstructed, timeline)
        self.assertEqual(max_error, 0.0)

    def test_inconsistent_overlap_is_rejected(self):
        timeline = np.arange(24, dtype=np.float64).reshape(8, 3)
        windows = np.stack([timeline[i : i + 4] for i in range(5)])
        windows[2, 1, 0] += 1.0

        with self.assertRaisesRegex(ValueError, "overlap"):
            reconstruct_sliding_windows(windows)


class EventGroupingTests(unittest.TestCase):
    def test_short_bounded_gap_can_be_bridged(self):
        mask = np.array([0, 1, 1, 0, 1, 1, 0], dtype=bool)

        bridged = bridge_short_gaps(mask, max_gap=1)

        self.assertEqual(true_runs(bridged), [(1, 5)])

    def test_long_gap_remains_separate(self):
        mask = np.array([1, 1, 0, 0, 1, 1], dtype=bool)

        bridged = bridge_short_gaps(mask, max_gap=1)

        self.assertEqual(true_runs(bridged), [(0, 1), (4, 5)])

    def test_overlapping_ramp_intervals_merge(self):
        merged = merge_intervals([(4, 10), (5, 11), (20, 26)])

        self.assertEqual(merged, [(4, 11), (20, 26)])

    def test_ramp_summary_counts_points_and_events(self):
        net_load = np.array([0, 0, 0, 0, 0, 0, 10, 11, 0, 0], dtype=float)

        summary = ramp_event_summary(net_load, horizon=6, threshold=9.0)

        self.assertEqual(summary["exceedance_points"], 2)
        self.assertEqual(summary["merged_candidate_events"], 1)

    def test_daily_daylight_energy_uses_twelve_hours(self):
        from datetime import datetime, timedelta, timezone

        timestamps = [
            datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            for i in range(48)
        ]
        solar = np.zeros(48)
        solar[7:19] = 50.0
        solar[31:43] = 100.0

        days, ratios = daily_daylight_energy_ratio(solar, timestamps, 100.0)

        self.assertEqual(len(days), 2)
        np.testing.assert_allclose(ratios, [0.5, 1.0])

    def test_mapping_retains_negative_lead_and_all_overlaps(self):
        from datetime import datetime, timedelta, timezone

        timestamps = [
            datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            for i in range(10)
        ]
        split = {
            "split": "test",
            "timestamps": timestamps,
            "hours": 10,
            "windows": 7,
            "window_length": 4,
        }
        events = [{
            "event_id": "e1",
            "event_type": "high_load",
            "start_time": timestamps[2].isoformat(),
            "end_time": timestamps[5].isoformat(),
        }]

        rows = map_windows_to_events(split, events)

        self.assertEqual(len(rows), 6)
        self.assertTrue(any(row["lead_hours"] < 0 and row["post_onset"] for row in rows))
        self.assertTrue(any(row["fully_contains_event"] for row in rows))


if __name__ == "__main__":
    unittest.main()
