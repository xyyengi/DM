import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from tools import summarize_station24_generation_seed_stability as summary


class GenerationSeedStabilityTests(unittest.TestCase):
    def write_model(self, root, seed, model, offset):
        slug = model.lower().replace(" ", "_")
        result = root / f"{seed}_{slug}_result"
        event = root / f"{seed}_{slug}_event"
        joint = root / f"{seed}_{slug}_joint"
        result.mkdir(); event.mkdir(); joint.mkdir()
        metrics = {
            "station_average": {"wind": {"crps": .09 + offset}, "solar_daylight": {"crps": .05 + offset}},
            "aggregate_mw": {"renewable": {"crps": 150 + offset, "coverage_90": .88, "width_90": 800 + offset}},
            "joint": {"energy_score_pu": 6.5 + offset, "spatial_corr_rmse_all_pairs": .08 + offset,
                      "spatial_corr_rmse_wind_solar": .09 + offset},
        }
        (result / "metrics.json").write_text(json.dumps(metrics))
        (result / "generation_metadata.json").write_text(json.dumps({"generation_seed": seed, "n_samples": 500}))
        event_rows = []
        for group in ("all", "tail"):
            event_rows.append({"variant": model, "scope": "independent_physical", "member_group": group,
                "standard": "primary", "event_count": 4, "events_with_any_hit": 3 + (model != "Raw"),
                "mean_member_hit_rate": .1 + offset, "median_onset_error_h": 8 - offset,
                "median_duration_error_h": 7 - offset, "median_depth_ratio": .6 + offset})
        pd.DataFrame(event_rows).to_csv(event / "continuous_event_three_standard_summary.csv", index=False)
        ramp_rows = []
        for source in ("wind", "solar"):
            for lag in (1, 3, 6):
                ramp_rows.append({"variant": model, "source": source, "lag_h": lag,
                    "median_ramp_mae": .01 * lag + offset, "ramp_90_coverage": .8,
                    "generated_ramp_std": .02 * lag, "actual_ramp_std": .03 * lag})
        pd.DataFrame(ramp_rows).to_csv(event / "fast_ramp_1_3_6h_summary.csv", index=False)
        pd.DataFrame([
            {"variant": model, "series": series, "wind_solar_correlation_rmse": .2 + offset, "valid_issue_count": 23}
            for series in ("power", "residual")
        ]).to_csv(joint / "same_member_wind_solar_correlation.csv", index=False)
        return {"result_dir": str(result), "event_dir": str(event), "joint_dir": str(joint),
                "event_label": model, "joint_label": model}

    def test_three_seed_summary_and_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for seed in (424242, 271828, 314159):
                models = {
                    "Raw": self.write_model(root, seed, "Raw", 0.0),
                    "Full Independent V2": self.write_model(root, seed, "Full Independent V2", .01),
                    "Lightweight Joint Tail": self.write_model(root, seed, "Lightweight Joint Tail", .005),
                }
                runs.append({"seed": seed, "models": models})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"runs": runs}))
            output = root / "summary"
            with patch.object(sys, "argv", ["summary", "--manifest", str(manifest), "--output-dir", str(output)]):
                summary.main()
            per_seed = pd.read_csv(output / "per_seed_metrics.csv")
            self.assertEqual(len(per_seed), 9)
            aggregate = pd.read_csv(output / "aggregate_mean_std_range.csv")
            self.assertTrue(((aggregate["n"] == 3)).all())
            direction = pd.read_csv(output / "lightweight_vs_raw_event_direction_summary.csv")
            self.assertTrue(direction["direction_consistent_no_worse"].all())
            self.assertTrue((output / "GENERATION_SEED_STABILITY_REPORT.md").is_file())
            self.assertTrue((output / "generation_seed_stability_key_metrics.png").is_file())

    def test_formal_script_is_generation_only_and_fixed_quota(self):
        text = Path("run_station24_generation_seed_stability.sh").read_text(encoding="utf-8")
        self.assertNotIn("train_station24.py", text)
        self.assertIn("271828 314159", text)
        self.assertIn("--body-member-limit 400", text)
        self.assertIn("--tail-member-limit 100", text)
        self.assertIn("--checkpoint-state raw", text)
        self.assertNotIn("self-local", text.lower())


if __name__ == "__main__":
    unittest.main()
