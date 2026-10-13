import unittest

import numpy as np
import pandas as pd

from tools.run_station24_s5_espc import (
    boundary_feasible,
    candidate_rows_for_member,
    interval_bundle,
    percentile_ranks,
)


class S5ESPCTests(unittest.TestCase):
    def setUp(self):
        self.stations = pd.DataFrame({
            "data_type": ["wind", "solar"],
            "capacity_mw": [100.0, 100.0],
        })
        self.stats = {
            "sources": {
                "wind": {"low_q10": .2, "ramp_abs_q90": {"1": .1, "3": .2, "6": .3},
                         "ramp_abs_q99": {"1": .5, "3": .7, "6": .9}},
                "solar": {"low_q10": .1, "ramp_abs_q90": {"1": .1, "3": .2, "6": .3},
                          "ramp_abs_q99": {"1": .5, "3": .7, "6": .9}},
            }
        }

    def test_percentile_rank_is_monotonic(self):
        ranks = percentile_ranks(np.array([3.0, 1.0, 2.0]))
        self.assertGreater(ranks[0], ranks[2])
        self.assertGreater(ranks[2], ranks[1])

    def test_daylight_excludes_nighttime_solar_low(self):
        background = np.full((12, 2), .5)
        tail = background.copy()
        tail[3:7, 0] = np.array([.8, .1, .1, .7])
        tail[:, 1] = 0.0
        daylight = np.zeros((12, 2), bool)
        daylight[7:11, 1] = True
        rows = candidate_rows_for_member(
            np.ones(12), background, tail, daylight, self.stations,
            lengths=[4], taper=1, stats=self.stats,
        )
        chosen = max(rows, key=lambda row: row["structure"])
        self.assertLess(chosen["start"], 7)
        self.assertGreater(chosen["ramp"], 0)

    def test_boundary_constraint_does_not_penalize_existing_raw_ramp(self):
        raw = np.zeros(12); raw[6:] = 1.0
        feasible, excess = boundary_feasible(raw, raw, raw, raw, 4, 9, self.stats)
        self.assertTrue(feasible)
        self.assertEqual(excess, 0.0)

    def test_interval_score_penalizes_missed_narrow_interval(self):
        truth = np.array([[2.0]])
        narrow = np.array([[[0.0]], [[0.1]], [[0.2]]]).transpose(1, 0, 2)
        wide = np.array([[[0.0]], [[1.0]], [[2.0]]]).transpose(1, 0, 2)
        self.assertGreater(interval_bundle(narrow, truth)["interval_score_90"],
                           interval_bundle(wide, truth)["interval_score_90"])


if __name__ == "__main__":
    unittest.main()
