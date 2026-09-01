import unittest

from tools.analyze_station24_body_tail_specialization import (
    deep_replay_specification,
)


class BodyTailSpecializationSchemaTests(unittest.TestCase):
    def test_legacy_replay_schema(self):
        replay = {"method": "legacy", "severity_thresholds": [0.1]}
        self.assertIs(deep_replay_specification(replay), replay)

    def test_unified_replay_schema(self):
        deep = {
            "method": "train_independent_wind_event_replay_x0_v1",
            "severity_thresholds": [0.1],
        }
        replay = {
            "method": "train_unified_wind_event_replay_v1",
            "deep_replay": deep,
            "mismatch_replay": {},
        }
        self.assertIs(deep_replay_specification(replay), deep)


if __name__ == "__main__":
    unittest.main()
