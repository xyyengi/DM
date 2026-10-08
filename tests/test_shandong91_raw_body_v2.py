"""Contract tests for the isolated Shandong91 faithful24 V2."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import torch
import yaml

from datasets.shandong91_faithful24 import (
    Shandong91Faithful24Dataset, fit_faithful24_state_thresholds,
    threshold_document, threshold_sha256,
)
from src.models.shandong91_faithful24_diffusion import Shandong91HeterogeneousRawBodyV2


class Shandong91Faithful24ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(Path("configs/shandong91/raw_body_v2_faithful24.yaml").read_text("utf-8"))
        cls.thresholds = fit_faithful24_state_thresholds(cls.config["data"]["data_path"])
        cls.dataset = Shandong91Faithful24Dataset(cls.config["data"]["data_path"], "train", cls.thresholds)

    def test_causal_condition_shapes_and_masks(self):
        sample = self.dataset[0]
        self.assertEqual(tuple(sample["node_state"].shape), (168, 91, 12))
        self.assertEqual(tuple(sample["recent_error"].shape), (24, 91, 3))
        self.assertEqual(tuple(sample["recent_error_valid_mask"].shape), (24, 91, 3))
        self.assertTrue(torch.equal(
            sample["recent_error"].masked_select(~sample["recent_error_valid_mask"]),
            torch.zeros_like(sample["recent_error"].masked_select(~sample["recent_error_valid_mask"])),
        ))
        self.assertFalse(self.dataset.condition_manifest()["recent_error_future_actual_used"])

    def test_threshold_document_roundtrip_hash(self):
        document = threshold_document(self.thresholds)
        self.assertEqual(threshold_sha256(self.thresholds), threshold_sha256(document))

    def test_forbidden_semantic_shortcuts_fail_closed(self):
        for key in ("use_lead_condition", "use_dual_fixed_graph", "use_body_tail_experts"):
            model_config = copy.deepcopy(self.config["model"]); model_config[key] = True
            with self.assertRaises(ValueError):
                Shandong91HeterogeneousRawBodyV2(
                    model_config, self.dataset.node_features,
                    self.dataset.adjacency_with_self.float(),
                )

    def test_residual_inverse_normalization_is_linear(self):
        sample = self.dataset[0]
        restored = self.dataset.denormalize_mw(sample["forecast"] + sample["residual"])
        separated = self.dataset.denormalize_mw(sample["forecast"]) + self.dataset.denormalize_mw(sample["residual"])
        self.assertTrue(torch.allclose(restored, separated, rtol=1e-6, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
