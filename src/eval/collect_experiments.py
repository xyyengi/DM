#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect outputs/*/metrics.json files into outputs/experiment_summary.csv."""

import argparse
import os
import sys

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.eval.experiment_logger import (  # noqa: E402
    append_summary,
    build_summary_row,
    find_run_dirs,
    load_metrics_json,
)


def load_config(run_dir):
    path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(path):
        return {}, None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, path


def infer_timestamp(run_id):
    parts = run_id.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return "NA"


def collect(outputs_dir):
    summary = os.path.join(outputs_dir, "experiment_summary.csv")
    if os.path.exists(summary):
        os.remove(summary)

    rows = []
    for run_dir in find_run_dirs(outputs_dir):
        run_id = os.path.basename(run_dir)
        config, config_path = load_config(run_dir)
        metrics = load_metrics_json(os.path.join(run_dir, "metrics.json"))
        row = build_summary_row(
            config=config,
            run_id=run_id,
            timestamp=metrics.get("timestamp", infer_timestamp(run_id)),
            config_path=config_path,
            checkpoint_path=metrics.get("checkpoint_path", os.path.join(run_dir, "checkpoints", "model_best.pt")),
            scenario_path=metrics.get("scenario_path", os.path.join(run_dir, "samples", "scenarios.npz")),
            figure_dir=metrics.get("figure_dir", os.path.join(run_dir, "figures")),
            metrics=metrics,
            notes=metrics.get("notes", "collected"),
        )
        append_summary(outputs_dir, row)
        rows.append(row)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description="Collect experiment metrics into experiment_summary.csv")
    parser.add_argument("--outputs_dir", default="outputs")
    args = parser.parse_args()
    summary, rows = collect(args.outputs_dir)
    print(f"Collected {len(rows)} runs into {summary}")


if __name__ == "__main__":
    main()
