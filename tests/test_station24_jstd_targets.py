import unittest

import numpy as np

from station_jstd_targets import _bridge_short_gaps, _segments_with_seed


class JSTDContinuousEventTests(unittest.TestCase):
    def test_seventeen_hour_event_keeps_actual_duration(self):
        active = np.zeros(40, dtype=bool)
        seed = np.zeros(40, dtype=bool)
        active[8:25] = True
        seed[12] = True
        self.assertEqual(_segments_with_seed(active, seed), [(8, 25)])
        self.assertEqual(25 - 8, 17)

    def test_two_hour_event_is_not_removed(self):
        active = np.zeros(20, dtype=bool)
        seed = np.zeros(20, dtype=bool)
        active[5:7] = True
        seed[5] = True
        self.assertEqual(_segments_with_seed(active, seed), [(5, 7)])

    def test_one_hour_gap_is_bridged_but_not_a_new_event_type(self):
        active = np.zeros(30, dtype=bool)
        active[4:12] = True
        active[13:21] = True
        bridged = _bridge_short_gaps(active, 1)
        seed = np.zeros(30, dtype=bool)
        seed[6] = True
        seed[18] = True
        self.assertEqual(_segments_with_seed(bridged, seed), [(4, 21)])


if __name__ == "__main__":
    unittest.main()
