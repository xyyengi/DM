import copy
import unittest
from pathlib import Path

import yaml

from src.eval.stage1_protocol import (
    failed_checks,
    training_protocol_checks,
    validation_protocol_checks,
)


ROOT = Path(__file__).resolve().parents[1]


class Stage1ProtocolTests(unittest.TestCase):
    def load_config(self, name):
        with (ROOT / "configs" / name).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_all_paired_configs_allow_runtime_seeds_2026_and_2027(self):
        cases = (
            ("v4rs_reproducible_no_guidance_168h.yaml", "v4_legacy"),
            ("v5_t_stage1_168h.yaml", "v5_t"),
            ("v5_tf_stage1_168h.yaml", "v5_tf"),
        )
        for filename, architecture in cases:
            config = self.load_config(filename)
            for seed in (2026, 2027):
                checks = training_protocol_checks(
                    config, architecture, seed, 64, 150, 15
                )
                self.assertEqual(failed_checks(checks), [])

    def test_unplanned_training_seed_is_rejected(self):
        config = self.load_config("v5_tf_stage1_168h.yaml")
        checks = training_protocol_checks(config, "v5_tf", 2028, 64, 150, 15)
        self.assertEqual(failed_checks(checks), ["runtime_seed_allowed"])

    def test_saved_2027_run_passes_validation_protocol(self):
        config = copy.deepcopy(self.load_config("v5_tf_stage1_168h.yaml"))
        config["train"]["seed"] = 2027
        self.assertEqual(failed_checks(validation_protocol_checks(config)), [])

    def test_validation_rejects_seed_or_generation_protocol_drift(self):
        config = copy.deepcopy(self.load_config("v5_tf_stage1_168h.yaml"))
        config["train"]["seed"] = 2028
        config["evaluation"]["generation_seed"] = 7
        self.assertEqual(
            failed_checks(validation_protocol_checks(config)),
            ["training_seed_allowed", "generation_seed"],
        )


if __name__ == "__main__":
    unittest.main()
