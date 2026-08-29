"""Lead-day audit for dynamic forecast/history conditioned scenarios."""

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
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def ensemble_crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """Exact empirical CRPS using a sorted-ensemble O(K log K) identity."""

    values = np.sort(samples, axis=1)
    members = values.shape[1]
    first = np.mean(np.abs(values - truth[:, None]), axis=1)
    coefficient = 2.0 * np.arange(1, members + 1) - members - 1.0
    pair = np.sum(values * coefficient[None, :, None], axis=1) / members**2
    return float(np.mean(first - pair))


def aggregate_scope(
    values: np.ndarray, indices: np.ndarray, capacities: np.ndarray
) -> np.ndarray:
    return np.sum(values[..., indices] * capacities[indices], axis=-1)


def main() -> None:
    args = parse_args()
    result = Path(args.result_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values(
        "channel_index"
    )
    capacities = stations.capacity_mw.to_numpy(float)
    station_type = stations.data_type.to_numpy()
    scenarios = np.load(result / "actual_scenarios_normalized.npy")
    actual = np.load(result / "actual_data_normalized.npy")
    forecast = np.load(result / "forecast_data_normalized.npy")
    center_path = result / "forecast_center_normalized.npy"
    fraction_path = result / "forecast_history_fraction.npy"
    center = np.load(center_path) if center_path.is_file() else forecast.copy()
    fraction = (
        np.load(fraction_path)
        if fraction_path.is_file()
        else np.zeros_like(forecast, dtype=np.float32)
    )
    if scenarios.ndim != 4 or scenarios.shape[2:] != actual.shape[1:]:
        raise ValueError("expected scenarios [N,K,168,24] and actual [N,168,24]")

    rows: list[dict[str, object]] = []
    for scope, indices in (
        ("wind_aggregate", np.flatnonzero(station_type == "wind")),
        ("solar_aggregate", np.flatnonzero(station_type == "solar")),
    ):
        scenario_scope = aggregate_scope(scenarios, indices, capacities)
        actual_scope = aggregate_scope(actual, indices, capacities)
        forecast_scope = aggregate_scope(forecast, indices, capacities)
        center_scope = aggregate_scope(center, indices, capacities)
        for day in range(7):
            section = slice(day * 24, (day + 1) * 24)
            sample = scenario_scope[:, :, section]
            truth = actual_scope[:, section]
            issued = forecast_scope[:, section]
            body_center = center_scope[:, section]
            median = np.median(sample, axis=1)
            truth_error = truth - issued
            center_change = body_center - issued
            denominator = float(np.sum(truth_error**2))
            row: dict[str, object] = {
                "scope": scope,
                "lead_day": day + 1,
                "forecast_mae_mw": float(np.mean(np.abs(issued - truth))),
                "center_mae_mw": float(np.mean(np.abs(body_center - truth))),
                "median_mae_mw": float(np.mean(np.abs(median - truth))),
                "crps_mw": ensemble_crps(sample, truth),
                "center_correction_capture_slope": (
                    float(np.sum(center_change * truth_error) / denominator)
                    if denominator > 0
                    else 0.0
                ),
                "history_fraction_mean": float(
                    fraction[:, section, :][:, :, indices].mean()
                ),
            }
            for level in (0.80, 0.90, 0.95):
                lower = np.quantile(sample, (1.0 - level) / 2.0, axis=1)
                upper = np.quantile(sample, 1.0 - (1.0 - level) / 2.0, axis=1)
                row[f"coverage_{int(level * 100)}"] = float(
                    np.mean((truth >= lower) & (truth <= upper))
                )
                row[f"width_{int(level * 100)}_mw"] = float(
                    np.mean(upper - lower)
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "lead_day_metrics.csv", index=False)

    station_rows: list[dict[str, object]] = []
    for station_index, station in stations.reset_index(drop=True).iterrows():
        capacity = float(capacities[station_index])
        station_samples = scenarios[..., station_index] * capacity
        station_actual = actual[..., station_index] * capacity
        station_forecast = forecast[..., station_index] * capacity
        station_center = center[..., station_index] * capacity
        for day in range(7):
            section = slice(day * 24, (day + 1) * 24)
            sample = station_samples[:, :, section]
            truth = station_actual[:, section]
            lower = np.quantile(sample, 0.05, axis=1)
            upper = np.quantile(sample, 0.95, axis=1)
            station_rows.append(
                {
                    "station_id": station.get("station_id", station_index),
                    "station_type": station.data_type,
                    "lead_day": day + 1,
                    "forecast_mae_mw": float(
                        np.mean(np.abs(station_forecast[:, section] - truth))
                    ),
                    "center_mae_mw": float(
                        np.mean(np.abs(station_center[:, section] - truth))
                    ),
                    "median_mae_mw": float(
                        np.mean(np.abs(np.median(sample, axis=1) - truth))
                    ),
                    "crps_mw": ensemble_crps(sample, truth),
                    "coverage_90": float(
                        np.mean((truth >= lower) & (truth <= upper))
                    ),
                    "width_90_mw": float(np.mean(upper - lower)),
                    "history_fraction_mean": float(
                        fraction[:, section, station_index].mean()
                    ),
                }
            )
    pd.DataFrame(station_rows).to_csv(
        output / "lead_day_station_metrics.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for scope, group in frame.groupby("scope"):
        label = scope.replace("_aggregate", "").capitalize()
        axes[0, 0].plot(group.lead_day, group.forecast_mae_mw, "--o", label=f"{label} forecast")
        axes[0, 0].plot(group.lead_day, group.center_mae_mw, "-o", label=f"{label} center")
        axes[0, 1].plot(group.lead_day, group.median_mae_mw, "-o", label=label)
        axes[1, 0].plot(group.lead_day, group.coverage_90, "-o", label=label)
        axes[1, 1].plot(group.lead_day, group.history_fraction_mean, "-o", label=label)
    axes[0, 0].set_title("Forecast versus dynamic center MAE")
    axes[0, 1].set_title("Scenario median MAE")
    axes[1, 0].set_title("90% interval coverage")
    axes[1, 0].axhline(0.90, color="black", linestyle=":", linewidth=1)
    axes[1, 1].set_title("Learned historical-center fraction")
    for axis in axes.flat:
        axis.set_xlabel("Lead day")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(output / "lead_day_forecast_trust.png", dpi=180)
    plt.close(fig)

    summary = {
        "result_dir": str(result),
        "issues": int(scenarios.shape[0]),
        "members": int(scenarios.shape[1]),
        "future_actual_used_as_condition": False,
        "test_target_used": False,
        "artifacts": [
            "lead_day_metrics.csv",
            "lead_day_station_metrics.csv",
            "lead_day_forecast_trust.png",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"LEAD_DAY_FORECAST_TRUST_COMPLETE output={output}")


if __name__ == "__main__":
    main()
