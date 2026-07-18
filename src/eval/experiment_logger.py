#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Experiment run directory, metrics, and summary CSV helpers."""

import csv
import json
import os
import re
from datetime import datetime


SUMMARY_FIELDS = [
    "run_id",
    "timestamp",
    "experiment_name",
    "version",
    "config_path",
    "target_type",
    "condition_mode",
    "forecast_in_network",
    "guidance_enable",
    "wind_scale",
    "pv_scale",
    "load_scale",
    "cond_mask",
    "epochs",
    "checkpoint_path",
    "scenario_path",
    "figure_dir",
    "scenario_shape",
    "wind_MAE",
    "pv_MAE",
    "load_MAE",
    "mean_MAE",
    "coverage",
    "interval_width",
    "ramp_MAE",
    "acf_error",
    "notes",
]


def timestamp_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "experiment"


def get_experiment_name(config, fallback="experiment"):
    return config.get("experiment", {}).get("name") or fallback


def create_run_id(experiment_name, timestamp=None):
    timestamp = timestamp or timestamp_now()
    return f"{timestamp}_{safe_name(experiment_name)}"


def ensure_run_dir(outputs_dir, run_id):
    run_dir = os.path.join(outputs_dir, run_id)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def json_safe(value):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save_metrics_json(metrics, run_dir):
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(metrics), f, ensure_ascii=False, indent=2)
    return path


def load_metrics_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summary_path(outputs_dir):
    return os.path.join(outputs_dir, "experiment_summary.csv")


def value_or_na(value):
    if value is None:
        return "NA"
    return value


def metric_first(metrics, keys, default="NA"):
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return default


def scenario_shape_from_path(scenario_path):
    if not scenario_path or not os.path.exists(scenario_path):
        return "NA"
    try:
        import numpy as np

        data = np.load(scenario_path)
        if all(k in data for k in ("wind", "pv", "load")):
            return str([int(data["wind"].shape[0]), 3, int(data["wind"].shape[1])])
    except Exception:
        return "NA"
    return "NA"


def build_summary_row(
    config,
    run_id,
    timestamp,
    config_path=None,
    checkpoint_path=None,
    scenario_path=None,
    figure_dir=None,
    metrics=None,
    notes="",
):
    metrics = metrics or {}
    condition = config.get("condition", {})
    target = config.get("target", {})
    guidance = config.get("guidance", {})
    train = config.get("train", {})
    model = config.get("model", {})
    experiment_name = get_experiment_name(config, fallback=run_id)

    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "version": condition.get("mode", model.get("condition_mode", "NA")),
        "config_path": value_or_na(config_path),
        "target_type": target.get("type", model.get("target_type", "NA")),
        "condition_mode": condition.get("mode", model.get("condition_mode", "NA")),
        "forecast_in_network": condition.get("use_network_condition", model.get("use_network_condition", "NA")),
        "guidance_enable": guidance.get("enable", model.get("use_guidance", "NA")),
        "wind_scale": guidance.get("wind_scale", "NA"),
        "pv_scale": guidance.get("pv_scale", "NA"),
        "load_scale": guidance.get("load_scale", "NA"),
        "cond_mask": json.dumps(condition.get("cond_mask", model.get("cond_mask", "NA"))),
        "epochs": train.get("epochs", "NA"),
        "checkpoint_path": value_or_na(checkpoint_path),
        "scenario_path": value_or_na(scenario_path),
        "figure_dir": value_or_na(figure_dir),
        "scenario_shape": metrics.get("scenario_shape", scenario_shape_from_path(scenario_path)),
        "wind_MAE": metric_first(metrics, ["wind_MAE", "wind_mae"]),
        "pv_MAE": metric_first(metrics, ["pv_MAE", "solar_MAE", "pv_mae", "solar_mae"]),
        "load_MAE": metric_first(metrics, ["load_MAE", "load_mae"]),
        "mean_MAE": metric_first(metrics, ["mean_MAE", "mean_mae"]),
        "coverage": metric_first(metrics, ["coverage", "total_coverage_90%", "total_coverage_100%"]),
        "interval_width": metric_first(metrics, ["interval_width", "total_width_90%", "total_width_100%"]),
        "ramp_MAE": metric_first(metrics, ["ramp_MAE", "ramp_mae"]),
        "acf_error": metric_first(metrics, ["acf_error", "total_acf_mae"]),
        "notes": notes or metrics.get("notes", ""),
    }
    return {field: row.get(field, "NA") for field in SUMMARY_FIELDS}


def append_summary(outputs_dir, row):
    os.makedirs(outputs_dir, exist_ok=True)
    path = summary_path(outputs_dir)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "NA") for field in SUMMARY_FIELDS})
    return path


def find_run_dirs(outputs_dir):
    if not os.path.exists(outputs_dir):
        return []
    dirs = []
    for name in sorted(os.listdir(outputs_dir)):
        path = os.path.join(outputs_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "metrics.json")):
            dirs.append(path)
    return dirs
