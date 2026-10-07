"""CPU-only contract and forward dry-run; this script never trains or saves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.shandong91_reliable import Shandong91ReliableDataset, masked_mse


class FlatCompatibilityAdapter(nn.Module):
    """Minimal shape adapter proving [B,T,N,C] round-trips through 273 features."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(273, 273)

    def forward(self, forecast: torch.Tensor, condition_mask: torch.Tensor) -> torch.Tensor:
        condition = forecast.masked_fill(~condition_mask, 0.0)
        batch, hours, nodes, channels = condition.shape
        return self.projection(condition.reshape(batch, hours, nodes * channels)).reshape(
            batch, hours, nodes, channels
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="reliable_channel_training_v1")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    datasets = {
        split: Shandong91ReliableDataset(args.data_root, split)
        for split in ("train", "validation", "test")
    }
    batch = next(iter(DataLoader(datasets["train"], batch_size=args.batch_size)))
    model = FlatCompatibilityAdapter()
    prediction = model(batch["forecast"], batch["forecast_valid_mask"])
    loss = masked_mse(prediction, batch["residual"], batch["effective_mask"])
    if prediction.shape != batch["residual"].shape or not torch.isfinite(loss):
        raise RuntimeError("dry-run forward/loss failed")
    print(json.dumps({
        "status": "PASS", "device": "CPU", "training": "NOT RUN",
        "cuda_amp": "NOT RUN", "split_lengths": {k: len(v) for k, v in datasets.items()},
        "batch_shape": list(batch["forecast"].shape), "output_shape": list(prediction.shape),
        "masked_loss": float(loss.detach()),
    }, indent=2))


if __name__ == "__main__":
    main()

