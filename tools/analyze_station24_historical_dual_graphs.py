"""Audit three 500-member Station24 runs for spatial and synchronous-tail gains."""

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
    "state_v1_cdsg_2d_conditional_scale",
    "geo_history_actual_dual",
    "geo_history_residual_dual",
)
LABELS = {
    VARIANTS[0]: "Geographic reference",
    VARIANTS[1]: "Geographic + historical actual",
    VARIANTS[2]: "Geographic + standardized residual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs=3)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--prior-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hardest-events", type=int, default=5)
    parser.add_argument("--event-window-hours", type=int, default=6)
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for values in selected.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.6g}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows])


def load_runs(paths: list[str]) -> dict[str, dict[str, object]]:
    runs: dict[str, dict[str, object]] = {}
    signature = set()
    for raw in paths:
        path = Path(raw)
        metadata = json.loads(
            (path / "generation_metadata.json").read_text(encoding="utf-8")
        )
        variant = str(metadata["condition_variant"])
        if variant not in VARIANTS or variant in runs:
            raise ValueError(f"unexpected or duplicate variant {variant}")
        signature.add(
            (
                metadata["split"],
                int(metadata["n_samples"]),
                int(metadata["generation_seed"]),
                bool(metadata.get("test_used", False)),
            )
        )
        runs[variant] = {"path": path, "metadata": metadata}
    if set(runs) != set(VARIANTS) or len(signature) != 1:
        raise ValueError("three variants or generation protocols do not match")
    observed = next(iter(signature))
    if observed != ("val", 500, 424242, False):
        raise ValueError(f"expected sealed-test 500-member validation protocol, got {observed}")
    return runs


def recover_scale(path: Path) -> np.ndarray:
    residual = np.load(path / "generated_residual_normalized.npy", mmap_mode="r")
    standardized = np.load(
        path / "generated_residual_standardized.npy", mmap_mode="r"
    )
    strongest = np.argmax(np.abs(standardized), axis=1)
    numerator = np.take_along_axis(residual, strongest[:, None], axis=1)[:, 0]
    denominator = np.take_along_axis(
        standardized, strongest[:, None], axis=1
    )[:, 0]
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError("could not recover conditional residual scale")
    scale = numerator / denominator
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("recovered conditional residual scale is invalid")
    return scale.astype(np.float32)


def correlation(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    return np.corrcoef(flat, rowvar=False)


def streaming_member_correlation(values: np.ndarray) -> np.ndarray:
    stations = values.shape[-1]
    total = np.zeros(stations, dtype=np.float64)
    cross = np.zeros((stations, stations), dtype=np.float64)
    count = 0
    for issue in range(values.shape[0]):
        block = np.asarray(values[issue], dtype=np.float64).reshape(-1, stations)
        total += block.sum(axis=0)
        cross += block.T @ block
        count += block.shape[0]
    covariance = cross - np.outer(total, total) / count
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    return covariance / np.outer(scale, scale)


def pair_rmse(
    generated: np.ndarray,
    truth: np.ndarray,
    station_types: np.ndarray,
) -> dict[str, float]:
    difference = generated - truth
    result = {}
    masks = {
        "all": np.ones_like(difference, dtype=bool),
        "wind_wind": np.equal.outer(station_types, "wind"),
        "solar_solar": np.equal.outer(station_types, "solar"),
        "wind_solar": np.not_equal.outer(station_types, station_types),
    }
    for name, mask in masks.items():
        selected = np.triu(mask, k=1)
        result[f"residual_spatial_corr_rmse_{name}"] = float(
            np.sqrt(np.mean(np.square(difference[selected])))
        )
    return result


def aggregate_interval_metrics(
    samples: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float]:
    result = {}
    for level in (0.90, 0.95, 0.99):
        alpha = (1.0 - level) / 2.0
        lower = np.quantile(samples, alpha, axis=1)
        upper = np.quantile(samples, 1.0 - alpha, axis=1)
        name = int(round(100 * level))
        result[f"aggregate_wind_coverage_{name}"] = float(
            np.mean((actual >= lower) & (actual <= upper))
        )
        result[f"aggregate_wind_width_{name}_mw"] = float(np.mean(upper - lower))
        result[f"aggregate_wind_below_{name}"] = float(np.mean(actual < lower))
        result[f"aggregate_wind_above_{name}"] = float(np.mean(actual > upper))
    return result


def joint_low_probability(events: np.ndarray) -> np.ndarray:
    flat = np.asarray(events, dtype=np.float64).reshape(-1, events.shape[-1])
    return flat.T @ flat / flat.shape[0]


def select_hard_events(
    actual_mw: np.ndarray,
    forecast_mw: np.ndarray,
    count: int,
    window: int,
) -> list[dict[str, object]]:
    candidates = []
    for issue in range(actual_mw.shape[0]):
        gaps = []
        for start in range(actual_mw.shape[1] - window + 1):
            stop = start + window
            gaps.append(float(np.mean(forecast_mw[issue, start:stop] - actual_mw[issue, start:stop])))
        start = int(np.argmax(gaps))
        candidates.append(
            {
                "issue_index": issue,
                "lead_start": start,
                "lead_end": start + window - 1,
                "forecast_minus_actual_mw": gaps[start],
                "actual_window_mean_mw": float(
                    np.mean(actual_mw[issue, start : start + window])
                ),
                "forecast_window_mean_mw": float(
                    np.mean(forecast_mw[issue, start : start + window])
                ),
            }
        )
    return sorted(
        candidates,
        key=lambda row: float(row["forecast_minus_actual_mw"]),
        reverse=True,
    )[:count]


def event_rows(
    variant: str,
    sample_mw: np.ndarray,
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for rank, event in enumerate(events, start=1):
        issue = int(event["issue_index"])
        start = int(event["lead_start"])
        stop = int(event["lead_end"]) + 1
        member_mean = np.mean(sample_mw[issue, :, start:stop], axis=-1)
        actual_mean = float(event["actual_window_mean_mw"])
        hits = int(np.sum(member_mean <= actual_mean))
        row = {
            "variant": variant,
            "label": LABELS[variant],
            "event_rank": rank,
            **event,
            "minimum_member_mean_mw": float(np.min(member_mean)),
            "hit_members": hits,
            "hit_rate": float(hits / len(member_mean)),
        }
        for level in (90, 95, 99):
            lower = float(np.quantile(member_mean, (1.0 - level / 100.0) / 2.0))
            row[f"lower_{level}_mw"] = lower
            row[f"covered_{level}"] = bool(actual_mean >= lower)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    prior_dir = Path(args.prior_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    runs = load_runs(args.result_dirs)
    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    station_types = stations.data_type.to_numpy()
    capacities = stations.capacity_mw.to_numpy(dtype=np.float32)
    wind = np.flatnonzero(station_types == "wind")

    reference_path = Path(runs[VARIANTS[0]]["path"])
    reference_actual = np.load(
        reference_path / "actual_data_normalized.npy", mmap_mode="r"
    )
    reference_forecast = np.load(
        reference_path / "forecast_data_normalized.npy", mmap_mode="r"
    )
    actual_wind_mw = np.sum(
        reference_actual[..., wind] * capacities[None, None, wind], axis=-1
    )
    forecast_wind_mw = np.sum(
        reference_forecast[..., wind] * capacities[None, None, wind], axis=-1
    )
    hard_events = select_hard_events(
        actual_wind_mw,
        forecast_wind_mw,
        args.hardest_events,
        args.event_window_hours,
    )
    issue_dates = pd.read_csv(data_path / "val_issue_dates.csv").sort_values(
        "sample_index"
    )
    for event in hard_events:
        event["issue_date"] = str(
            issue_dates.iloc[int(event["issue_index"])].issue_date
        )

    thresholds = pd.read_csv(prior_dir / "historical_tail_thresholds.csv")
    thresholds = thresholds.sort_values("channel_index")
    wind_thresholds = thresholds.loc[
        thresholds.station_type.eq("wind"), "low_tail_threshold_standardized"
    ].to_numpy(dtype=np.float64)

    summary_rows = []
    all_event_rows = []
    for variant in VARIANTS:
        path = Path(runs[variant]["path"])
        samples = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")
        actual = np.load(path / "actual_data_normalized.npy", mmap_mode="r")
        forecast = np.load(path / "forecast_data_normalized.npy", mmap_mode="r")
        standardized = np.load(
            path / "generated_residual_standardized.npy", mmap_mode="r"
        )
        if not np.allclose(actual, reference_actual) or not np.allclose(
            forecast, reference_forecast
        ):
            raise ValueError(f"paired validation data mismatch for {variant}")
        scale = recover_scale(path)
        truth_standardized = (np.asarray(actual) - np.asarray(forecast)) / scale
        truth_corr = correlation(truth_standardized)
        generated_corr = streaming_member_correlation(standardized)

        wind_sample_mw = np.sum(
            samples[..., wind] * capacities[None, None, None, wind], axis=-1
        )
        row: dict[str, object] = {
            "variant": variant,
            "label": LABELS[variant],
            "parameter_count": int(runs[variant]["metadata"]["parameter_count"]),
            **aggregate_interval_metrics(wind_sample_mw, actual_wind_mw),
            **pair_rmse(generated_corr, truth_corr, station_types),
        }

        generated_low = standardized[..., wind] < wind_thresholds[None, None, None, :]
        truth_low = truth_standardized[..., wind] < wind_thresholds[None, None, :]
        generated_joint = joint_low_probability(generated_low)
        truth_joint = joint_low_probability(truth_low)
        upper = np.triu_indices(len(wind), k=1)
        row["wind_low_tail_joint_rmse"] = float(
            np.sqrt(np.mean(np.square(generated_joint[upper] - truth_joint[upper])))
        )
        metadata = runs[variant]["metadata"]
        row["bottleneck_dual_primary"] = metadata.get("spatial_gate_values", {}).get(
            "dual_primary"
        )
        row["bottleneck_dual_secondary"] = metadata.get("spatial_gate_values", {}).get(
            "dual_secondary"
        )
        row["encoder0_dual_primary"] = metadata.get(
            "parallel_spatial_gate_statistics", {}
        ).get("encoder_0/dual_primary")
        row["encoder0_dual_secondary"] = metadata.get(
            "parallel_spatial_gate_statistics", {}
        ).get("encoder_0/dual_secondary")
        summary_rows.append(row)
        all_event_rows.extend(event_rows(variant, wind_sample_mw, hard_events))

    summary = pd.DataFrame(summary_rows)
    event_frame = pd.DataFrame(all_event_rows)
    summary.to_csv(output / "historical_dual_graph_summary.csv", index=False)
    event_frame.to_csv(output / "synchronous_deep_drop_events.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    x = np.arange(len(VARIANTS))
    labels = [LABELS[value] for value in VARIANTS]
    axes[0].bar(x, 100 * summary.aggregate_wind_coverage_90, color=["#777777", "#e76f51", "#2878b5"])
    axes[0].axhline(90, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Aggregate wind 90% coverage", ylabel="Coverage (%)", xticks=x, xticklabels=labels)
    hit = event_frame.groupby("variant", sort=False).hit_rate.mean().reindex(VARIANTS)
    axes[1].bar(x, 100 * hit.to_numpy(), color=["#777777", "#e76f51", "#2878b5"])
    axes[1].set(title="Mean hit rate in five hardest 6 h drops", ylabel="Members at/below observed (%)", xticks=x, xticklabels=labels)
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=18)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "historical_dual_graph_key_outcomes.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Historical dual-graph 500-member audit",
        "",
        "All three runs use validation only, 500 members, generation seed 424242, and the same physical projection.",
        "",
        markdown_table(
            summary,
            [
                "variant",
                "aggregate_wind_coverage_90",
                "aggregate_wind_coverage_95",
                "aggregate_wind_coverage_99",
                "aggregate_wind_width_90_mw",
                "residual_spatial_corr_rmse_wind_wind",
                "wind_low_tail_joint_rmse",
                "bottleneck_dual_secondary",
                "encoder0_dual_secondary",
            ],
        ),
        "",
        "## Synchronous deep-drop audit",
        "",
        markdown_table(
            event_frame,
            [
                "variant",
                "event_rank",
                "issue_index",
                "issue_date",
                "lead_start",
                "lead_end",
                "actual_window_mean_mw",
                "forecast_window_mean_mw",
                "minimum_member_mean_mw",
                "hit_members",
                "hit_rate",
                "covered_99",
            ],
        ),
        "",
        "Residual spatial correlation and low-tail joint metrics use the raw generated residual distribution before physical projection; aggregate coverage and deep-drop hits use projected power scenarios.",
    ]
    (output / "historical_dual_graph_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"HISTORICAL_DUAL_GRAPH_AUDIT_COMPLETE output={output}")


if __name__ == "__main__":
    main()
