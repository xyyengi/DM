import json
import tempfile
import unittest
from pathlib import Path

import torch

from train import fixed_validation_rng, set_reproducible_seed, train


class TinyRandomLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, batch):
        target = batch["target"]
        random_term = torch.rand((), device=self.weight.device)
        return (self.weight + random_term - target).pow(2)


class TrainingReproducibilityTests(unittest.TestCase):
    def test_fixed_validation_rng_repeats_and_restores_training_rng(self):
        torch.manual_seed(7)
        expected_first = torch.rand(3)
        expected_second = torch.rand(3)

        torch.manual_seed(7)
        actual_first = torch.rand(3)
        with fixed_validation_rng(314159):
            validation_draw_1 = torch.rand(5)
        actual_second = torch.rand(3)
        with fixed_validation_rng(314159):
            validation_draw_2 = torch.rand(5)

        torch.testing.assert_close(actual_first, expected_first)
        torch.testing.assert_close(actual_second, expected_second)
        torch.testing.assert_close(validation_draw_1, validation_draw_2)

    def test_same_seed_repeats_training_and_writes_top_three_manifest(self):
        config = {
            "train": {
                "epochs": 5,
                "lr": 0.01,
                "weight_decay": 0.0,
                "seed": 2026,
                "validation_seed": 314159,
                "top_k_checkpoints": 3,
            },
            "model": {
                "num_steps": 5,
                "beta_start": 0.0001,
                "beta_end": 0.01,
                "schedule": "linear",
            },
        }
        train_loader = [{"target": torch.tensor(0.4)}, {"target": torch.tensor(0.6)}]
        val_loader = [{"target": torch.tensor(0.5)}, {"target": torch.tensor(0.7)}]
        manifests = []

        with tempfile.TemporaryDirectory() as tmp:
            for repeat in range(2):
                run_dir = Path(tmp) / f"run_{repeat}"
                (run_dir / "logs").mkdir(parents=True)
                (run_dir / "checkpoints").mkdir()
                set_reproducible_seed(2026)
                model = TinyRandomLossModel()
                train(
                    model,
                    train_loader,
                    val_loader,
                    config,
                    torch.device("cpu"),
                    str(run_dir),
                    patience=10,
                )
                with (run_dir / "checkpoints" / "top_checkpoints.json").open(
                    "r", encoding="utf-8"
                ) as handle:
                    manifests.append(json.load(handle))
                self.assertTrue((run_dir / "checkpoints" / "model_best.pt").exists())

        summary_0 = [(x["epoch"], x["val_loss"]) for x in manifests[0]["checkpoints"]]
        summary_1 = [(x["epoch"], x["val_loss"]) for x in manifests[1]["checkpoints"]]
        self.assertEqual(summary_0, summary_1)
        self.assertEqual(len(summary_0), 3)
        self.assertEqual(summary_0, sorted(summary_0, key=lambda item: (item[1], item[0])))


if __name__ == "__main__":
    unittest.main()
