#!/usr/bin/env python3
"""Audit routing, retrieval concentration, and leakage for the dual-tail model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = Path(args.result_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    metadata = json.loads(
        (result / "generation_metadata.json").read_text(encoding="utf-8")
    )
    probability = np.load(result / "mismatch_expert_probability.npy")
    route = np.load(result / "mismatch_expert_route.npy")
    time_probability = np.load(result / "mismatch_time_probability.npy")
    attention = np.load(result / "retrieval_attention.npy")
    train_index = np.load(result / "retrieval_train_index.npy")
    distance = np.load(result / "retrieval_distance.npy")
    issues = pd.read_csv(Path(args.data_path) / "val_issue_dates.csv")
    train_count = len(np.load(Path(args.data_path) / "train_forecast.npy", mmap_mode="r"))
    if len(probability) != len(issues) or route.shape[0] != len(issues):
        raise ValueError("routing arrays do not match validation issue count")
    if np.any(train_index < 0) or np.any(train_index >= train_count):
        raise ValueError("retrieval index leaves the train split")
    if attention.shape[:2] != train_index.shape:
        raise ValueError("retrieval attention/index shape mismatch")
    entropy = -np.sum(attention * np.log(attention.clip(min=1e-12)), axis=1)
    effective = np.exp(entropy)
    peak_hour = time_probability.argmax(axis=1)
    rows = pd.DataFrame(
        {
            "issue_index": np.arange(len(issues)),
            "issue_date": issues["issue_date"],
            "mismatch_probability": probability,
            "sampled_mismatch_member_fraction": route.mean(axis=1),
            "mismatch_time_peak_hour": peak_hour,
            "mismatch_time_probability_mean": time_probability.mean(axis=1),
            "retrieval_effective_members_mean": effective.mean(axis=1),
            "retrieval_distance_mean": distance.mean(axis=1),
            "retrieval_distance_min": distance.min(axis=1),
        }
    )
    rows.to_csv(output / "per_issue_retrieval_routing.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    image = axes[0].imshow(attention.mean(axis=0), aspect="auto", cmap="viridis")
    axes[0].set_ylabel("Retrieved rank")
    axes[0].set_title("Mean retrieval attention by lead hour")
    fig.colorbar(image, ax=axes[0], fraction=0.02)
    axes[1].plot(time_probability.mean(axis=0), color="#d81b60")
    axes[1].fill_between(
        np.arange(time_probability.shape[1]),
        np.quantile(time_probability, 0.10, axis=0),
        np.quantile(time_probability, 0.90, axis=0),
        color="#f8bbd0",
        alpha=0.7,
    )
    axes[1].set_ylabel("Mismatch time p")
    axes[1].grid(alpha=0.25)
    axes[2].bar(np.arange(len(probability)), probability, color="#00897b")
    axes[2].scatter(
        np.arange(len(probability)), route.mean(axis=1), color="#ef5350", s=15,
        label="sampled fraction",
    )
    axes[2].set_ylabel("Issue route p")
    axes[2].set_xlabel("Validation issue index")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "retrieval_and_routing_audit.png", dpi=180)
    plt.close(fig)

    summary = {
        "method": "station24_retrieval_dual_tail_audit_v1",
        "validation_issue_count": int(len(issues)),
        "ensemble_members": int(route.shape[1]),
        "retrieval_top_k": int(train_index.shape[1]),
        "retrieval_index_train_only": True,
        "future_validation_actual_used_for_retrieval": False,
        "mismatch_probability_mean": float(probability.mean()),
        "mismatch_member_fraction": float(route.mean()),
        "mismatch_time_probability_mean": float(time_probability.mean()),
        "retrieval_effective_members_mean": float(effective.mean()),
        "retrieval_effective_members_min": float(effective.min()),
        "generation_metadata_retrieval_method": metadata.get("retrieval_method"),
    }
    (output / "retrieval_dual_tail_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RETRIEVAL_DUAL_TAIL_AUDIT_COMPLETE output={output}")


if __name__ == "__main__":
    main()
