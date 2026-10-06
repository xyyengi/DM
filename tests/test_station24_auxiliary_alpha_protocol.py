import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_station24_auxiliary_alpha_stage1a import (
    PRIMARY_BODY,
    _pairwise_body_winner,
    _pareto_frontier,
    ci_of_paired,
    correlations_from_moments,
    json_default,
    moments_by_issue,
    spatial_rmse_from_correlations,
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

    def test_nested_numpy_decision_evidence_is_json_serializable(self):
        payload = {
            "guardrail": {
                "passed": np.bool_(True),
                "count": np.int64(3),
                "difference": np.float64(0.125),
            }
        }
        restored = json.loads(json.dumps(payload, default=json_default))
        self.assertIs(restored["guardrail"]["passed"], True)
        self.assertEqual(restored["guardrail"]["count"], 3)
        self.assertEqual(restored["guardrail"]["difference"], 0.125)

    def test_spatial_bootstrap_recomputes_after_combining_issue_blocks(self):
        # The solar station is constant in issue 0, so its per-issue
        # correlations are undefined. Every paired draw below contains issue 1;
        # recomputing after concatenation must yield finite spatial metrics.
        actual = np.array([
            [[0., 0., 0.], [1., 2., 0.], [2., 4., 0.]],
            [[3., 6., 1.], [4., 8., 2.], [5., 10., 3.]],
        ])
        generated = np.array([
            [[0., 0.1, 0.], [1.1, 1.9, 0.1], [1.9, 4.2, 0.]],
            [[3.1, 5.8, 1.2], [3.8, 8.1, 1.8], [5.2, 9.9, 3.2]],
        ])
        draws = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int64)
        actual_corr = correlations_from_moments(moments_by_issue(actual), draws)
        generated_corr = correlations_from_moments(moments_by_issue(generated), draws)
        metrics = spatial_rmse_from_correlations(
            actual_corr, generated_corr, np.array(["wind", "wind", "solar"])
        )
        self.assertTrue(all(np.all(np.isfinite(value)) for value in metrics.values()))
        for draw_index, draw in enumerate(draws):
            actual_direct = np.corrcoef(actual[draw].reshape(-1, 3), rowvar=False)
            generated_direct = np.corrcoef(generated[draw].reshape(-1, 3), rowvar=False)
            np.testing.assert_allclose(actual_corr[draw_index], actual_direct, atol=1e-12)
            np.testing.assert_allclose(generated_corr[draw_index], generated_direct, atol=1e-12)

    def test_nonfinite_primary_bootstrap_input_fails_closed(self):
        indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            ci_of_paired([1.0, np.nan], [1.0, 2.0], indices)

    def test_reselect_launcher_is_evaluation_only(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "run_station24_auxiliary_alpha_stage1a_reselect.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("evaluate_station24_auxiliary_alpha_stage1a.py", launcher)
        self.assertIn("stage1a_spatial_bootstrap_v2", launcher)
        self.assertNotIn("train_station24.py", launcher)
        self.assertNotIn("generate_station24.py", launcher)

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
