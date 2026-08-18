import unittest

import numpy as np

from tools.diagnose_station24_forecast_event_attribution import (
    _classify_event,
    _nearest_directional_event,
    _strongest_directional_event,
)


class ForecastEventAttributionTests(unittest.TestCase):
    def test_nearest_directional_event_uses_time_then_amplitude(self):
        ramp = np.zeros(20)
        ramp[8] = -1.5
        ramp[12] = -2.0
        offset, value = _nearest_directional_event(
            ramp, event_index=10, minimum_magnitude=1.0,
            direction="down", search_radius=6,
        )
        self.assertEqual(offset, 2.0)
        self.assertEqual(value, -2.0)

    def test_strongest_directional_event_retains_peak_for_sensitivity(self):
        ramp = np.zeros(20)
        ramp[9] = 1.0
        ramp[14] = 2.0
        offset, value = _strongest_directional_event(
            ramp, event_index=10, direction="up", search_radius=6
        )
        self.assertEqual(offset, 4.0)
        self.assertEqual(value, 2.0)

    def test_category_a_is_forecast_anchor(self):
        category = _classify_event(
            True, forecast_offset=3.0, model_offset=4.0,
            member_hit_rate_3h=0.20, member_hit_rate_6h=0.40,
        )
        self.assertEqual(category, "A_condition_anchor")

    def test_category_b_is_model_delay(self):
        category = _classify_event(
            True, forecast_offset=0.0, model_offset=3.0,
            member_hit_rate_3h=0.20, member_hit_rate_6h=0.40,
        )
        self.assertEqual(category, "B_model_delay")

    def test_category_c_and_d_separate_forecast_presence(self):
        missing = _classify_event(
            False, np.nan, np.nan, member_hit_rate_3h=0.0,
            member_hit_rate_6h=0.01,
        )
        low_mass = _classify_event(
            True, 0.0, np.nan, member_hit_rate_3h=0.01,
            member_hit_rate_6h=0.08,
        )
        self.assertEqual(missing, "C_forecast_omission")
        self.assertEqual(low_mass, "D_low_probability_mass")


if __name__ == "__main__":
    unittest.main()
