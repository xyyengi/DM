"""Re-evaluate saved station24 ensembles without regenerating scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from station_evaluation import evaluate_station_scenarios, save_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--interval-levels", nargs="+", type=float, default=[0.80, 0.90, 0.95]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.result_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    data_path = Path(args.data_path)
    stations = (
        pd.read_csv(data_path / "station_order.csv")
        .sort_values("channel_index")
        .reset_index(drop=True)
    )
    adjacency = np.load(data_path / "station_adjacency.npy")
    samples = np.load(source / "actual_scenarios_normalized.npy", mmap_mode="r")
    raw_samples = np.load(
        source / "actual_scenarios_raw_normalized.npy", mmap_mode="r"
    )
    actual = np.load(source / "actual_data_normalized.npy", mmap_mode="r")
    forecast = np.load(source / "forecast_data_normalized.npy", mmap_mode="r")
    daylight = np.load(source / "station_daylight_mask.npy", mmap_mode="r")

    summary, station_frame, lead_frame = evaluate_station_scenarios(
        samples,
        raw_samples,
        actual,
        forecast,
        stations,
        adjacency,
        daylight_mask=daylight,
        interval_levels=tuple(args.interval_levels),
    )
    original = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    summary["run"] = original["run"]
    summary["reevaluation"] = {
        "source_result_dir": str(source),
        "interval_levels": args.interval_levels,
        "solar_scopes": ["all_hours", "astronomical_daylight"],
        "overwrites_original": False,
    }
    save_evaluation(output, summary, station_frame, lead_frame)
    print(f"REEVALUATION_COMPLETE output={output}")


if __name__ == "__main__":
    main()
