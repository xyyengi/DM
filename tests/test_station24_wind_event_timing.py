import unittest

import numpy as np

from tools.diagnose_station24_wind_event_timing import (
    _clustered_event_indices,
    _nearest_member_matches,
)


class WindEventTimingDiagnosticTests(unittest.TestCase):
    def test_contiguous_extreme_points_are_one_event(self):
        ramp = np.array([0.0, 1.1, 1.7, 1.2, 0.0, -1.1, -1.8, 0.0])
        self.assertEqual(_clustered_event_indices(ramp, 1.0, "up"), [2])
        self.assertEqual(_clustered_event_indices(ramp, 1.0, "down"), [6])

    def test_positive_offset_means_generated_event_is_late(self):
        member_ramps = np.zeros((3, 20), dtype=float)
        member_ramps[0, 12] = 2.0
        member_ramps[1, 8] = 2.0
        member_ramps[2, 15] = 0.9
        offsets, matched = _nearest_member_matches(
            member_ramps,
            event_index=10,
            threshold=1.0,
            direction="up",
            search_radius=6,
        )
        np.testing.assert_allclose(offsets[:2], [2.0, -2.0])
        self.assertTrue(np.isnan(offsets[2]))
        np.testing.assert_allclose(matched[:2], [2.0, 2.0])

    def test_nearest_event_wins_and_larger_ramp_breaks_tie(self):
        member_ramps = np.zeros((1, 20), dtype=float)
        member_ramps[0, 8] = -1.5
        member_ramps[0, 12] = -2.0
        offsets, matched = _nearest_member_matches(
            member_ramps,
            event_index=10,
            threshold=1.0,
            direction="down",
            search_radius=6,
        )
        self.assertEqual(offsets[0], 2.0)
        self.assertEqual(matched[0], -2.0)


if __name__ == "__main__":
    unittest.main()
