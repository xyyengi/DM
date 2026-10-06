import unittest
from pathlib import Path

import pandas as pd

from tools.evaluate_station24_auxiliary_alpha_stage1a import (
    PRIMARY_BODY,
    _pairwise_body_winner,
    _pareto_frontier,
)


def pairwise_rows(intervals):
    rows = []
    for metric in PRIMARY_BODY:
        low, high = intervals.get(metric, (-0.1, 0.1))
        rows.append({
            "family": "body_pairwise", "candidate": "alpha_0.65",
            "reference": "alpha_1.00", "metric": metric,
            "ci_low": low, "ci_high": high, "difference": (low + high) / 2,
        })
    return pd.DataFrame(rows)


class AuxiliaryAlphaProtocolTests(unittest.TestCase):
    def setUp(self):
        self.decisions = {
            "alpha_0.65": {"persistent": "control", "ramp": "control"},
            "alpha_1.00": {"persistent": "control", "ramp": "control"},
        }

    def test_resolved_body_winner_is_not_a_closeness_fallback(self):
        bootstrap = pairwise_rows({"renewable_crps": (-0.2, -0.01)})
        winner, metric = _pairwise_body_winner(
            ["alpha_0.65", "alpha_1.00"], bootstrap
        )
        self.assertEqual(winner, "alpha_0.65")
        self.assertEqual(metric, "renewable_crps")

    def test_pareto_dominance_uses_ci_not_point_estimates(self):
        bootstrap = pairwise_rows({"renewable_crps": (-0.2, -0.01)})
        frontier = _pareto_frontier(
            ["alpha_0.65", "alpha_1.00"], bootstrap, self.decisions
        )
        self.assertEqual(frontier, ["alpha_0.65"])

    def test_one_significant_body_degradation_prevents_dominance(self):
        bootstrap = pairwise_rows({
            "renewable_crps": (-0.2, -0.01),
            "energy_score": (0.01, 0.2),
        })
        frontier = _pareto_frontier(
            ["alpha_0.65", "alpha_1.00"], bootstrap, self.decisions
        )
        self.assertEqual(set(frontier), {"alpha_0.65", "alpha_1.00"})

    def test_formal_launcher_is_stage1a_only_and_has_complete_lifecycle(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "run_station24_auxiliary_alpha_stage1a.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("train_alpha_065", launcher)
        self.assertIn("train_alpha_135", launcher)
        self.assertNotIn("train_alpha_100", launcher)
        self.assertIn("immutable_input_audit.json", launcher)
        self.assertIn("run_station24_auxiliary_alpha_stage1a_finalize.sh", launcher)
        finalize = (root / "run_station24_auxiliary_alpha_stage1a_finalize.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("train_station24.py", finalize)
        shared = (root / "run_station24_lightweight_joint_tail_finalize.sh").read_text(
            encoding="utf-8"
        )
        lifecycle = finalize + shared
        for token in ("mixture_body400_tail100_n500", "continuous_event_evaluation",
                      "joint_wind_solar_evaluation", "ALPHA_SENSITIVITY_REPORT.md",
                      "selected_alpha.json", "tar -czf"):
            self.assertIn(token, lifecycle)


if __name__ == "__main__":
    unittest.main()
