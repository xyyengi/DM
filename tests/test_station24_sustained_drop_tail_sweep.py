import unittest

import numpy as np

from tools.diagnose_station24_sustained_drop_tail_sweep import (
    Event,
    best_member_match,
    contiguous_runs,
)


class SustainedDropTailSweepTests(unittest.TestCase):
    def test_contiguous_event_duration_merges_one_hour_gap(self):
        mask = np.asarray([0, 1, 1, 0, 1, 1, 0, 0, 1], dtype=bool)
        self.assertEqual(contiguous_runs(mask, merge_gap=1), [(1, 6), (8, 9)])

    def test_well_aligned_deep_member_is_a_coverage_hit(self):
        forecast = np.ones(168, dtype=np.float64)
        actual = forecast.copy()
        actual[40:50] -= np.linspace(0.45, 0.70, 10)
        scenario = forecast.copy()
        scenario[42:51] -= np.linspace(0.50, 0.75, 9)
        event = Event(
            event_id="synthetic",
            issue=0,
            issue_date="2026-01-01",
            onset=40,
            stop=50,
            peak_start=42,
            physical_time=np.datetime64("2026-01-02"),
            depth_mw=700.0,
            mean_shortfall_mw=575.0,
            severity_normalized=0.5,
        )
        result = best_member_match(
            event,
            forecast,
            scenario,
            forecast * 1000.0,
            scenario * 1000.0,
            (forecast - actual) * 1000.0,
            threshold=0.36,
            time_tolerance=12,
            shape_time_tolerance=24,
            depth_ratio_required=0.75,
        )
        self.assertTrue(result["has_candidate"])
        self.assertTrue(result["coverage_hit"])
        self.assertLessEqual(result["onset_abs_error_h"], 2.0)
        self.assertGreaterEqual(result["true_interval_recall"], 0.8)


if __name__ == "__main__":
    unittest.main()
