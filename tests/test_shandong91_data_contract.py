from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.shandong91_reliable import (
    EXPECTED_SHAPES, Shandong91ReliableDataset, effective_mask, masked_mse,
    validate_data_contract,
)


ROOT = Path(__file__).resolve().parents[1] / "reliable_channel_training_v1"


class Shandong91ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        validate_data_contract(ROOT)
        cls.datasets = {split: Shandong91ReliableDataset(ROOT, split, validate=False)
                        for split in EXPECTED_SHAPES}

    def test_split_shapes_and_reconstruction(self) -> None:
        for split, expected in EXPECTED_SHAPES.items():
            dataset = self.datasets[split]
            self.assertEqual(dataset.arrays["actual"].shape, expected)
            sample = dataset[0]
            valid = sample["residual_valid_mask"]
            error = (sample["actual"] - sample["forecast"] - sample["residual"])[valid]
            self.assertLessEqual(float(error.abs().max()), 1e-6)

    def test_masked_loss_ignores_invalid_values(self) -> None:
        target = torch.zeros(1, 1, 91, 3)
        prediction = torch.ones_like(target)
        valid = torch.zeros_like(target, dtype=torch.bool)
        valid[..., 8, 0] = True
        node_type = torch.zeros(91, 3, dtype=torch.bool)
        channel_train = torch.zeros(91, 3, dtype=torch.bool)
        node_type[8, 0] = channel_train[8, 0] = True
        mask = effective_mask(valid, channel_train, node_type)
        self.assertEqual(float(masked_mse(prediction, target, mask)), 1.0)
        prediction[~mask] = 9999.0
        self.assertEqual(float(masked_mse(prediction, target, mask)), 1.0)

    def test_e_channels_and_absent_resources_are_excluded(self) -> None:
        dataset = self.datasets["train"]
        for node_id, channel in [(63, 0), (40, 1), (54, 1), (56, 1),
                                 (58, 1), (60, 1), (67, 1)]:
            self.assertFalse(bool(dataset.channel_train_mask[node_id - 1, channel]))
        valid = torch.ones(1, 168, 91, 3, dtype=torch.bool)
        mask = effective_mask(valid, dataset.channel_train_mask, dataset.node_type_mask)
        self.assertFalse(bool(mask[..., ~dataset.node_type_mask].any()))

    def test_inverse_normalization_matches_published_mw(self) -> None:
        dataset = self.datasets["test"]
        sample = dataset[0]
        restored = dataset.denormalize_mw(sample["actual"])
        published_array = np.load(
            dataset.root / "windows168/test/actual_mw.npy", mmap_mode="r"
        )[0]
        published = torch.from_numpy(np.array(published_array, copy=True))
        valid = sample["actual_valid_mask"] & dataset.node_type_mask.unsqueeze(0)
        self.assertLessEqual(float((restored - published)[valid].abs().max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
