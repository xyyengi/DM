import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tools import merge_station24_independent_tail_members as merge
from tools import check_station24_independent_tail_gate as gate


class IndependentPipelineTests(unittest.TestCase):
    def fixture(self, root):
        for label, members in (("body", 4), ("tail", 2)):
            folder = root / label
            folder.mkdir()
            for name in merge.MEMBER_ARRAYS:
                value = np.ones((2, members, 32, 24), dtype=np.float32) * (1 if label == "body" else 2)
                np.save(folder / name, value)
            for name in merge.STATIC_ARRAYS:
                np.save(folder / name, np.zeros((2, 32, 24), dtype=np.float32))
            (folder / "generation_metadata.json").write_text(json.dumps({
                "n_samples": members, "split": "val", "generation_seed": 42,
                "physical_projection": "clip", "architecture": "station24_resunet",
                "spatial_mode": "fixed_graph", "future_actual_used_as_generation_condition": False,
            }))
        return ["merge", "--body-results", str(root / "body"), "--tail-results", str(root / "tail"),
                "--output-dir", str(root / "output"), "--body-member-limit", "3"]

    def test_fixed_budget_preserves_prefix_and_routes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv = self.fixture(root)
            with patch("sys.argv", argv), patch.object(merge, "evaluate_station_scenarios", return_value=({}, None, None)), patch.object(merge, "save_evaluation"):
                merge.main()
                merge.evaluate_station_scenarios.reset_mock()
            value = np.load(root / "output/actual_scenarios_normalized.npy")
            self.assertEqual(value.shape[1], 5)
            np.testing.assert_array_equal(value[:, :3], 1)
            np.testing.assert_array_equal(value[:, 3:], 2)
            np.testing.assert_array_equal(np.load(root / "output/tail_expert_route.npy"), [[0,0,0,1,1]] * 2)
            metadata = json.loads((root / "output/generation_metadata.json").read_text())
            self.assertEqual(metadata["n_samples"], 5)

    def test_bad_limit_or_data_mismatch_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv = self.fixture(root)
            with patch("sys.argv", argv[:-1] + ["99"]), self.assertRaises(ValueError):
                merge.main()
            self.assertFalse((root / "output").exists())
            np.save(root / "tail/actual_data_normalized.npy", np.ones((2,32,24)))
            with patch("sys.argv", argv), self.assertRaises(ValueError):
                merge.main()
            self.assertFalse((root / "output").exists())

    def test_cpu_evidence_cannot_unlock_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cuda_preflight.json").write_text(json.dumps({"status": "PASS", "cuda_amp_executed": False, "launch_eligible": True}))
            with patch("sys.argv", ["gate", "--pipeline-root", temp, "--pretrain"]), self.assertRaises(ValueError):
                gate.main()

    def test_missing_mandatory_evidence_cannot_unlock_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cuda_preflight.json").write_text(json.dumps({"status": "PASS", "cuda_amp_executed": True, "launch_eligible": False}))
            with patch("sys.argv", ["gate", "--pipeline-root", temp, "--pretrain"]), self.assertRaises(ValueError):
                gate.main()
