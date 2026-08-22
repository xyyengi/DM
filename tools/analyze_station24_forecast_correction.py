#!/usr/bin/env python3
"""Audit deterministic correction centers before judging diffusion spread."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = (
    "geo_history_actual_dual",
    "geo_history_actual_forecast_correction_direct",
    "geo_history_actual_forecast_correction_decomposed",
)
LABELS = {
    VARIANTS[0]: "Historical-spatial baseline",
    VARIANTS[1]: "Direct correction center",
    VARIANTS[2]: "Decomposed correction center",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--direct", required=True)
    parser.add_argument("--decomposed", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-events", type=int, default=5)
    return parser.parse_args()


def load_result(path: Path, expected: str) -> dict[str, object]:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    run = metrics["run"]
    if run.get("condition_variant") != expected:
        raise ValueError(
            f"unexpected variant {run.get('condition_variant')} at {path}"
        )
    if run.get("split") != "val" or bool(run.get("test_used")):
        raise ValueError("forecast correction audit is validation-only")
    actual = np.load(path / "actual_data_normalized.npy")
    forecast = np.load(path / "forecast_data_normalized.npy")
    correction_path = path / "forecast_correction_normalized.npy"
    correction = (
        np.load(correction_path)
        if correction_path.is_file()
        else np.zeros_like(forecast)
    )
    if actual.shape != forecast.shape or correction.shape != forecast.shape:
        raise ValueError(f"array shape mismatch at {path}")
    return {
        "path": path,
        "metrics": metrics,
        "actual": actual,
        "forecast": forecast,
        "correction": correction,
        "center": forecast + correction,
    }


def masked_metrics(
    truth: np.ndarray, center: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    error = center - truth
    selected = error[mask]
    return {
        "mae_pu": float(np.mean(np.abs(selected))),
        "bias_pu": float(np.mean(selected)),
        "rmse_pu": float(np.sqrt(np.mean(np.square(selected)))),
    }


def rolling_mean(value: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.cumsum(value, axis=-1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros(value.shape[:-1] + (1,), dtype=np.float64), cumulative], axis=-1
    )
    return (cumulative[..., width:] - cumulative[..., :-width]) / float(width)


def select_hard_events(
    actual_mw: np.ndarray,
    forecast_mw: np.ndarray,
    count: int,
    width: int = 6,
) -> list[dict[str, float | int]]:
    gap = rolling_mean(forecast_mw - actual_mw, width)
    rows = []
    for issue in range(len(gap)):
        start = int(np.argmax(gap[issue]))
        rows.append(
            {
                "issue_index": issue,
                "lead_start": start,
                "lead_end": start + width - 1,
                "forecast_minus_actual_mw": float(gap[issue, start]),
            }
        )
    return sorted(
        rows,
        key=lambda row: float(row["forecast_minus_actual_mw"]),
        reverse=True,
    )[:count]


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "|" + "|".join(["---"] * len(frame.columns)) + "|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                f"{value:.5f}" if isinstance(value, (float, np.floating)) else str(value)
                for value in row
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    figures = output / "figures"
    figures.mkdir(parents=True)
    paths = [Path(args.baseline), Path(args.direct), Path(args.decomposed)]
    results = {
        variant: load_result(path, variant)
        for variant, path in zip(VARIANTS, paths, strict=True)
    }
    reference = results[VARIANTS[0]]
    actual = reference["actual"]
    forecast = reference["forecast"]
    for result in results.values():
        if not np.array_equal(result["actual"], actual) or not np.array_equal(
            result["forecast"], forecast
        ):
            raise ValueError("paired correction results do not share truth/forecast")

    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values(
        "channel_index"
    )
    station_types = stations.data_type.to_numpy()
    capacities = stations.capacity_mw.to_numpy(dtype=np.float64)
    wind = np.flatnonzero(station_types == "wind")
    solar = np.flatnonzero(station_types == "solar")
    daylight = np.load(paths[0] / "station_daylight_mask.npy").astype(bool)
    common = {
        "all": np.ones_like(actual, dtype=bool),
        "wind": np.broadcast_to(
            np.isin(np.arange(actual.shape[-1]), wind)[None, None, :], actual.shape
        ),
        "solar_daylight": daylight
        & np.broadcast_to(
            np.isin(np.arange(actual.shape[-1]), solar)[None, None, :], actual.shape
        ),
    }
    metric_rows = []
    for variant, result in results.items():
        correction = result["correction"]
        for scope, mask in common.items():
            row = {
                "variant": variant,
                "label": LABELS[variant],
                "scope": scope,
                **masked_metrics(actual, result["center"], mask),
                "correction_mean_abs_pu": float(np.mean(np.abs(correction[mask]))),
                "correction_p95_abs_pu": float(
                    np.quantile(np.abs(correction[mask]), 0.95)
                ),
                "correction_max_abs_pu": float(np.max(np.abs(correction[mask]))),
            }
            metric_rows.append(row)
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(output / "forecast_correction_metrics.csv", index=False)

    actual_wind = np.sum(actual[:, :, wind] * capacities[wind], axis=-1)
    forecast_wind = np.sum(forecast[:, :, wind] * capacities[wind], axis=-1)
    centers_wind = {
        variant: np.sum(result["center"][:, :, wind] * capacities[wind], axis=-1)
        for variant, result in results.items()
    }
    events = select_hard_events(
        actual_wind, forecast_wind, int(args.top_events), width=6
    )
    event_rows = []
    for rank, event in enumerate(events, start=1):
        issue = int(event["issue_index"])
        start = int(event["lead_start"])
        stop = int(event["lead_end"]) + 1
        truth_mean = float(actual_wind[issue, start:stop].mean())
        forecast_mean = float(forecast_wind[issue, start:stop].mean())
        for variant in VARIANTS:
            center_mean = float(centers_wind[variant][issue, start:stop].mean())
            event_rows.append(
                {
                    "event_rank": rank,
                    **event,
                    "variant": variant,
                    "label": LABELS[variant],
                    "actual_window_mean_mw": truth_mean,
                    "forecast_window_mean_mw": forecast_mean,
                    "corrected_center_window_mean_mw": center_mean,
                    "center_minus_actual_mw": center_mean - truth_mean,
                    "absolute_center_error_mw": abs(center_mean - truth_mean),
                }
            )
    event_frame = pd.DataFrame(event_rows)
    event_frame.to_csv(output / "hard_event_correction_centers.csv", index=False)

    fig, axes = plt.subplots(
        len(events), 1, figsize=(15, 3.2 * len(events)), sharex=True, squeeze=False
    )
    lead = np.arange(1, actual.shape[1] + 1)
    colors = {VARIANTS[1]: "#e11d48", VARIANTS[2]: "#2563eb"}
    for axis, event in zip(axes[:, 0], events, strict=True):
        issue = int(event["issue_index"])
        start = int(event["lead_start"])
        stop = int(event["lead_end"]) + 1
        axis.plot(lead, actual_wind[issue], color="#111827", label="actual")
        axis.plot(
            lead,
            forecast_wind[issue],
            color="#0d9488",
            linestyle="--",
            label="issued forecast",
        )
        for variant in VARIANTS[1:]:
            axis.plot(
                lead,
                centers_wind[variant][issue],
                color=colors[variant],
                label=LABELS[variant],
            )
        axis.axvspan(start + 1, stop, color="#f59e0b", alpha=0.15)
        axis.set_ylabel("Wind MW")
        axis.set_title(
            f"issue={issue}, severe 6h lead={start + 1}-{stop}, "
            f"forecast-actual={event['forecast_minus_actual_mw']:.1f} MW"
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=4, frameon=False)
    axes[-1, 0].set_xlabel("Lead hour")
    fig.suptitle("Forecast correction centers on fixed severe validation events")
    fig.tight_layout()
    fig.savefig(figures / "hard_event_correction_centers.png", dpi=180)
    plt.close(fig)

    summary_columns = [
        "label",
        "scope",
        "mae_pu",
        "bias_pu",
        "rmse_pu",
        "correction_mean_abs_pu",
        "correction_p95_abs_pu",
    ]
    report = [
        "# 24场站预测校正中心审计",
        "",
        "本报告只评价确定性校正中心，不以区间变宽代替中心改进。",
        "",
        markdown_table(metric_frame[summary_columns]),
        "",
        "固定严重6 h事件见 `hard_event_correction_centers.csv` 与图件。",
        "测试集未使用。",
    ]
    (output / "forecast_correction_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"FORECAST_CORRECTION_AUDIT_COMPLETE output={output}")


if __name__ == "__main__":
    main()
