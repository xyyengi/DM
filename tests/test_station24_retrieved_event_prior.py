import unittest

import numpy as np
import pandas as pd

from tools.diagnose_station24_retrieved_event_prior import (
    contiguous_runs,
    predict_one,
    weighted_median,
)


class RetrievedEventPriorTests(unittest.TestCase):
    def test_contiguous_runs_merge_single_hour_gap(self):
        mask = np.asarray([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        self.assertEqual(contiguous_runs(mask, merge_gap=1), [(1, 5), (7, 8)])

    def test_weighted_median(self):
        self.assertEqual(
            weighted_median(np.asarray([1, 6, 12]), np.asarray([0.1, 0.8, 0.1])),
            6.0,
        )

    def test_prediction_uses_event_set_without_trajectory_averaging(self):
        target = pd.Series(
            {
                "event_type": "sustained_deep_drop",
                "direction": "down",
                "onset_hour": 52,
                "duration_hours": 8,
                "depth_mw": 900.0,
                "ramp_severity_mw": 400.0,
            }
        )
        columns = [
            "event_type", "direction", "onset_hour", "duration_hours",
            "depth_mw", "ramp_severity_mw",
        ]
        histories = {
            3: pd.DataFrame(
                [["sustained_deep_drop", "down", 50, 7, 850.0, 390.0]],
                columns=columns,
            ),
            7: pd.DataFrame(
                [["sustained_deep_drop", "down", 120, 12, 500.0, 200.0]],
                columns=columns,
            ),
        }
        result = predict_one(
            target,
            np.asarray([3, 7]),
            np.asarray([0.8, 0.2]),
            histories,
            bandwidth=3.0,
            attribute_radius=12,
        )
        self.assertTrue(result["event_supported"])
        self.assertEqual(result["predicted_onset_hour"], 50)
        self.assertTrue(result["onset_hit_6h"])
        self.assertEqual(result["predicted_duration_hours"], 7.0)
        self.assertEqual(result["predicted_depth_mw"], 850.0)


if __name__ == "__main__":
    unittest.main()
