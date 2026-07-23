import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.compare_v5_stage1_results import (
    add_baseline_deltas,
    compute_saved_diagnostics,
    validate_protocol,
)


class V5Stage1ComparisonTests(unittest.TestCase):
    def test_baseline_deltas_are_matched_by_training_seed(self):
        rows = []
        for seed, baseline_value in ((2026, 10.0), (2027, 20.0)):
            rows.extend([
                {
                    "training_seed": seed,
                    "architecture": "v4_legacy",
                    "checkpoint_rank": 1,
                    "condition_ablation": "none",
                    "total_crps": baseline_value,
                    "multivariate_es": baseline_value,
                    "total_acf_mae": baseline_value,
                },
                {
                    "training_seed": seed,
                    "architecture": "v5_tf",
                    "checkpoint_rank": 1,
                    "condition_ablation": "none",
                    "total_crps": baseline_value / 2,
                    "multivariate_es": baseline_value / 2,
                    "total_acf_mae": baseline_value / 2,
                },
            ])

        add_baseline_deltas(rows)

        for row in rows:
            if row["architecture"] == "v5_tf":
                self.assertEqual(row["total_crps_delta_vs_v4_rank1_pct"], -50.0)

    def test_protocol_rejects_unplanned_training_seed(self):
        row = {
            "training_seed": 2028,
            "generation_seed": 424242,
            "n_samples": 20,
            "reverse_variance_type": "posterior",
            "data_split": "val",
            "result_dir": "test",
        }
        with self.assertRaisesRegex(ValueError, "unsupported training seed"):
            validate_protocol([row])

    def test_saved_diagnostics_report_ramps_correlation_and_raw_boundaries(self):
        actual = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]])
        scenarios = np.repeat(actual[:, None], 2, axis=1)
        scenarios[0, 0, 0, 0] = -1.0
        scenarios[0, 1, 1, 1] = 12.0
        scenarios[0, 0, 2, 2] = -2.0

        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            np.save(result_dir / "actual_data.npy", actual)
            np.save(result_dir / "actual_scenarios.npy", scenarios)
            with (result_dir / "denormalization_used.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump({"scales": [10.0, 10.0, 100.0]}, handle)

            diagnostics = compute_saved_diagnostics(result_dir)
            np.save(
                result_dir / "actual_scenarios_constrained.npy",
                np.maximum(scenarios, 0.0),
            )
            constrained = compute_saved_diagnostics(
                result_dir, "actual_scenarios_constrained.npy"
            )

        self.assertGreater(diagnostics["wind_below_zero_pct"], 0.0)
        self.assertGreater(diagnostics["solar_above_capacity_pct"], 0.0)
        self.assertGreater(diagnostics["load_below_zero_pct"], 0.0)
        self.assertGreater(diagnostics["any_physical_violation_pct"], 0.0)
        self.assertGreaterEqual(diagnostics["net_load_ramp_1h_mae_mw"], 0.0)
        self.assertGreaterEqual(diagnostics["cross_variable_corr_mae"], 0.0)
        self.assertEqual(constrained["wind_below_zero_pct"], 0.0)
        self.assertEqual(constrained["load_below_zero_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
