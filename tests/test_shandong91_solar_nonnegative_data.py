"""Contract checks for the Solar-nonnegative Shandong91 modeling data version."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

from datasets.shandong91_faithful24 import (
    Shandong91Faithful24Dataset, fit_faithful24_state_thresholds,
    threshold_sha256,
)
from datasets.shandong91_reliable import validate_data_contract


SOURCE = Path("reliable_channel_training_v1")
PROJECTED = Path("reliable_channel_training_v2_solar_nonnegative")


class SolarNonnegativeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PROJECTED.exists():
            raise unittest.SkipTest("build projected data before running full contract tests")
        cls.node_type = np.load(PROJECTED / "static/node_type_mask.npy").astype(bool)
        cls.train_enabled = np.load(PROJECTED / "static/channel_train_mask.npy").astype(bool)
        cls.active_solar = cls.node_type[:, 1] & cls.train_enabled[:, 1]

    def test_published_contract_and_hashes(self):
        validate_data_contract(PROJECTED, verify_hashes=True)

    def test_projection_and_unchanged_channels(self):
        target = np.load(PROJECTED / "15min/actual_mw.npy", mmap_mode="r")
        source = np.load(SOURCE / "15min/actual_mw.npy", mmap_mode="r")
        valid = np.load(PROJECTED / "15min/actual_valid_mask.npy", mmap_mode="r").astype(bool)
        selected = valid[..., 1] & self.active_solar[None]
        self.assertFalse((np.asarray(target[..., 1])[selected] < 0).any())
        self.assertTrue(np.array_equal(target[..., 0], source[..., 0]))
        self.assertTrue(np.array_equal(target[..., 2], source[..., 2]))
        for name in ("actual_valid_mask", "forecast_valid_mask", "residual_valid_mask"):
            self.assertEqual(
                hashlib.sha256((PROJECTED / f"15min/{name}.npy").read_bytes()).digest(),
                hashlib.sha256((SOURCE / f"15min/{name}.npy").read_bytes()).digest(),
            )
        self.assertEqual(
            hashlib.sha256((PROJECTED / "15min/forecast_mw.npy").read_bytes()).digest(),
            hashlib.sha256((SOURCE / "15min/forecast_mw.npy").read_bytes()).digest(),
        )

    def test_residual_and_roundtrip(self):
        scales = np.zeros((91, 3), dtype=np.float32)
        normal = json.loads((PROJECTED / "normalization_params.json").read_text("utf-8"))
        index = {"Wind": 0, "Solar": 1, "Load": 2}
        for row in normal["records"]:
            if row.get("scale") is not None:
                scales[int(row["node_id"]) - 1, index[row["resource"]]] = row["scale"]
        self.assertEqual(float(scales[47, 1]), 172.0)
        for resolution in ("15min", "hourly"):
            actual = np.load(PROJECTED / resolution / "actual_mw.npy", mmap_mode="r")
            forecast = np.load(PROJECTED / resolution / "forecast_mw.npy", mmap_mode="r")
            residual = np.load(PROJECTED / resolution / "residual_mw.npy", mmap_mode="r")
            normalized = np.load(PROJECTED / resolution / "actual_normalized.npy", mmap_mode="r")
            valid = np.load(PROJECTED / resolution / "residual_valid_mask.npy", mmap_mode="r").astype(bool)
            self.assertEqual(float(np.abs(np.asarray(residual)[valid] - (np.asarray(actual)[valid] - np.asarray(forecast)[valid])).max()), 0.0)
            actual_valid = np.load(PROJECTED / resolution / "actual_valid_mask.npy", mmap_mode="r").astype(bool)
            error = np.abs(np.asarray(normalized) * scales[None] - np.asarray(actual))
            self.assertLessEqual(float(error[actual_valid].max()), 1e-3)

    def test_causal_history_uses_projected_hourly_residual(self):
        thresholds = fit_faithful24_state_thresholds(PROJECTED)
        dataset = Shandong91Faithful24Dataset(PROJECTED, "validation", thresholds)
        sample = dataset[0]
        with (PROJECTED / "hourly/timestamps.csv").open(encoding="utf-8-sig", newline="") as stream:
            hours = [row[next(iter(row))] for row in csv.DictReader(stream)]
        with (PROJECTED / "windows168/validation/window_starts.csv").open(encoding="utf-8-sig", newline="") as stream:
            start = next(csv.DictReader(stream))["window_start"]
        position = hours.index(start)
        expected = torch.from_numpy(np.array(
            np.load(PROJECTED / "hourly/residual_normalized.npy", mmap_mode="r")[position-24:position],
            dtype=np.float32, copy=True,
        ))
        valid = torch.from_numpy(np.array(
            np.load(PROJECTED / "hourly/residual_valid_mask.npy", mmap_mode="r")[position-24:position],
            dtype=bool, copy=True,
        )) & dataset.node_type_mask.unsqueeze(0) & dataset.channel_train_mask.unsqueeze(0)
        expected *= valid
        self.assertEqual(hours[position-1], "2025-10-31T23:00:00+08:00")
        self.assertEqual(start, "2025-11-01T00:00:00+08:00")
        self.assertEqual(float((sample["recent_error"] - expected).abs().max()), 0.0)
        self.assertEqual(thresholds["fit_split"], "train")

    def test_dedicated_config_uses_projected_version(self):
        config = yaml.safe_load(Path(
            "configs/shandong91/raw_body_v2_faithful24_solar_nonnegative.yaml"
        ).read_text("utf-8"))
        self.assertEqual(config["data"]["data_path"], "./reliable_channel_training_v2_solar_nonnegative")
        thresholds = fit_faithful24_state_thresholds(config["data"]["data_path"])
        for split in ("train", "validation", "test"):
            dataset = Shandong91Faithful24Dataset(config["data"]["data_path"], split, thresholds)
            self.assertEqual(threshold_sha256(dataset.thresholds), threshold_sha256(thresholds))


if __name__ == "__main__":
    unittest.main()
