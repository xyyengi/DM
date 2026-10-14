import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from tools.materialize_station24_s5_crn_single_seed import build_pairings
from tools.validate_station24_s5_crn_stage2 import validate, validate_gate


class Station24S5CRNStage2Tests(unittest.TestCase):
    def config(self):
        return {
            "experiment": {
                "split": "val",
                "generation_seed": 271828,
                "duration_seed": 424242,
                "permutation_seeds": [104729, 130363, 155921],
                "body_members": 400,
                "background_members": 100,
                "tail_members": 100,
            },
            "paths": {"data_path": "data", "raw_run": "raw", "tail_run": "tail"},
            "runtime": {
                "cuda_required": True,
                "gate_amp": True,
                "formal_generation_amp": False,
                "issue_batch_size": 1,
                "member_chunk_size": 100,
                "canonical_member_block_size": 100,
                "checkpoint_state": "raw",
            },
        }

    def test_frozen_config_accepts_one_seed_and_three_permutations(self):
        validate(self.config())
        bad = self.config()
        bad["experiment"]["permutation_seeds"] = [1, 2]
        with self.assertRaises(ValueError):
            validate(bad)

    def test_pairings_reuse_exact_background_pool_and_are_reproducible(self):
        first = build_pairings(3, 400, 100, [104729, 130363, 155921])
        second = build_pairings(3, 400, 100, [104729, 130363, 155921])
        expected = np.arange(400, 500)
        self.assertTrue(np.array_equal(first["crn_identity"][0], expected))
        for name in first:
            self.assertTrue(np.array_equal(first[name], second[name]))
            for row in first[name]:
                self.assertTrue(np.array_equal(np.sort(row), expected))
        self.assertFalse(np.array_equal(first["permutation_1"], first["permutation_2"]))

    def test_gate_requires_cuda_amp_launch_eligibility(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gate.json"
            good = {
                "status": "PASS", "launch_eligible": True,
                "device": "cuda", "amp": True,
                "checks": [{"check": "target_server_CUDA_AMP", "status": "PASS"}],
            }
            path.write_text(json.dumps(good), encoding="utf-8")
            validate_gate(path)
            good["amp"] = False
            path.write_text(json.dumps(good), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_gate(path)


if __name__ == "__main__":
    unittest.main()
