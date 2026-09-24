"""Aggregate fixed-checkpoint Station-24 generation-seed sensitivity results.

This tool only reads completed generation/evaluation artifacts.  It does not
train, regenerate, relabel events, or change any evaluation threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ("Raw", "Full Independent V2", "Lightweight Joint Tail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def nested(data: dict, *keys: str) -> float:
    value = data
    for key in keys:
        value = value[key]
    return float(value)


def select_variant(frame: pd.DataFrame, requested: str, baseline: bool) -> pd.DataFrame:
    selected = frame[frame["variant"] == requested]
    if not selected.empty:
        return selected
    labels = list(frame["variant"].dropna().unique())
    if len(labels) != 2:
        raise ValueError(f"cannot resolve variant {requested!r} from {labels}")
    fallback = labels[0] if baseline else labels[1]
    return frame[frame["variant"] == fallback]


def event_metrics(event_dir: Path, label: str, baseline: bool) -> dict[str, float]:
    summary = pd.read_csv(event_dir / "continuous_event_three_standard_summary.csv")
    rows = select_variant(summary, label, baseline)
    rows = rows[(rows["scope"] == "independent_physical") & (rows["standard"] == "primary")]
    output: dict[str, float] = {}
    for group in ("all", "tail"):
        row = rows[rows["member_group"] == group]
        if len(row) != 1:
            raise ValueError(f"expected one primary independent {group} row in {event_dir}")
        row = row.iloc[0]
        output.update(
            {
                f"event_{group}_count": float(row["event_count"]),
                f"event_{group}_any_hit": float(row["events_with_any_hit"]),
                f"event_{group}_member_hit_ratio": float(row["mean_member_hit_rate"]),
                f"event_{group}_onset_error_h": float(row["median_onset_error_h"]),
                f"event_{group}_duration_error_h": float(row["median_duration_error_h"]),
                f"event_{group}_depth_ratio": float(row["median_depth_ratio"]),
                f"event_{group}_depth_abs_error_to_one": abs(float(row["median_depth_ratio"]) - 1.0),
            }
        )
    ramps = pd.read_csv(event_dir / "fast_ramp_1_3_6h_summary.csv")
    ramps = select_variant(ramps, label, baseline)
    for source in ("wind", "solar"):
        for lag in (1, 3, 6):
            row = ramps[(ramps["source"] == source) & (ramps["lag_h"] == lag)]
            if len(row) != 1:
                raise ValueError(f"missing {source} {lag}h ramp row in {event_dir}")
            row = row.iloc[0]
            output[f"{source}_ramp_{lag}h_mae"] = float(row["median_ramp_mae"])
            output[f"{source}_ramp_{lag}h_coverage90"] = float(row["ramp_90_coverage"])
            output[f"{source}_ramp_{lag}h_generated_std"] = float(row["generated_ramp_std"])
    return output


def joint_metrics(joint_dir: Path, label: str, baseline: bool) -> dict[str, float]:
    frame = pd.read_csv(joint_dir / "same_member_wind_solar_correlation.csv")
    rows = select_variant(frame, label, baseline)
    output = {}
    for series in ("power", "residual"):
        row = rows[rows["series"] == series]
        if len(row) != 1:
            raise ValueError(f"missing {series} correlation row in {joint_dir}")
        output[f"wind_solar_same_member_{series}_corr_rmse"] = float(
            row.iloc[0]["wind_solar_correlation_rmse"]
        )
    return output


def load_model(seed: int, model: str, entry: dict) -> dict[str, float | int | str]:
    result_dir = Path(entry["result_dir"])
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((result_dir / "generation_metadata.json").read_text(encoding="utf-8"))
    if int(metadata["generation_seed"]) != seed:
        raise ValueError(f"seed mismatch for {model}: {metadata['generation_seed']} != {seed}")
    if int(metadata["n_samples"]) != 500:
        raise ValueError(f"member count mismatch for {model}")
    row: dict[str, float | int | str] = {
        "seed": seed,
        "model": model,
        "wind_crps": nested(metrics, "station_average", "wind", "crps"),
        "solar_daylight_crps": nested(metrics, "station_average", "solar_daylight", "crps"),
        "renewable_crps_mw": nested(metrics, "aggregate_mw", "renewable", "crps"),
        "renewable_coverage90": nested(metrics, "aggregate_mw", "renewable", "coverage_90"),
        "renewable_coverage90_abs_error": abs(nested(metrics, "aggregate_mw", "renewable", "coverage_90") - 0.90),
        "renewable_width90_mw": nested(metrics, "aggregate_mw", "renewable", "width_90"),
        "energy_score": nested(metrics, "joint", "energy_score_pu"),
        "overall_spatial_corr_rmse": nested(metrics, "joint", "spatial_corr_rmse_all_pairs"),
        "wind_solar_spatial_corr_rmse": nested(metrics, "joint", "spatial_corr_rmse_wind_solar"),
    }
    row.update(event_metrics(Path(entry["event_dir"]), entry["event_label"], model == "Raw"))
    row.update(joint_metrics(Path(entry["joint_dir"]), entry["joint_label"], model == "Raw"))
    return row


def markdown(frame: pd.DataFrame, digits: int = 6) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{value:.{digits}f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = [int(entry["seed"]) for entry in manifest["runs"]]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or 424242 not in seeds:
        raise ValueError("summary requires unique seeds including historical 424242 and at least two new seeds")
    rows = []
    for run in manifest["runs"]:
        seed = int(run["seed"])
        if set(run["models"]) != set(MODEL_ORDER):
            raise ValueError(f"seed {seed} does not contain exactly the three declared models")
        for model in MODEL_ORDER:
            rows.append(load_model(seed, model, run["models"][model]))
    per_seed = pd.DataFrame(rows)
    numeric = [c for c in per_seed.columns if c not in ("seed", "model")]
    aggregate = (
        per_seed.groupby("model", sort=False)[numeric]
        .agg(["mean", "std", "min", "max", "count"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "metric", "count": "n"})
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    per_seed.to_csv(output / "per_seed_metrics.csv", index=False)
    aggregate.to_csv(output / "aggregate_mean_std_range.csv", index=False)

    direction_specs = {
        "event_all_any_hit": "max",
        "event_tail_any_hit": "max",
        "event_tail_member_hit_ratio": "max",
        "event_tail_onset_error_h": "min",
        "event_tail_duration_error_h": "min",
        "event_tail_depth_abs_error_to_one": "min",
    }
    paired_rows = []
    for seed in seeds:
        raw = per_seed[(per_seed.seed == seed) & (per_seed.model == "Raw")].iloc[0]
        light = per_seed[(per_seed.seed == seed) & (per_seed.model == "Lightweight Joint Tail")].iloc[0]
        for metric, direction in direction_specs.items():
            raw_value, light_value = float(raw[metric]), float(light[metric])
            benefit = light_value - raw_value if direction == "max" else raw_value - light_value
            paired_rows.append(
                {"seed": seed, "metric": metric, "direction": direction, "raw": raw_value,
                 "lightweight": light_value, "benefit_signed": benefit,
                 "outcome": "improved" if benefit > 1e-12 else "worsened" if benefit < -1e-12 else "tied"}
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "lightweight_vs_raw_event_direction_by_seed.csv", index=False)
    direction = (
        paired.assign(
            improved=(paired.outcome == "improved").astype(int),
            tied=(paired.outcome == "tied").astype(int),
            worsened=(paired.outcome == "worsened").astype(int),
        )
        .groupby(["metric", "direction"], as_index=False)[["improved", "tied", "worsened"]]
        .sum()
    )
    direction["direction_consistent_no_worse"] = direction["worsened"] == 0
    direction.to_csv(output / "lightweight_vs_raw_event_direction_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_metrics = [
        ("wind_crps", "Wind CRPS", False),
        ("event_tail_member_hit_ratio", "Tail event hit ratio", True),
        ("event_tail_onset_error_h", "Tail onset error (h)", False),
    ]
    for ax, (metric, title, _) in zip(axes, plot_metrics):
        for index, model in enumerate(MODEL_ORDER):
            values = per_seed.loc[per_seed.model == model, metric].astype(float)
            ax.errorbar(index, values.mean(), yerr=values.std(ddof=1), fmt="o", capsize=5)
            ax.scatter(np.full(len(values), index), values, s=22, alpha=.65)
        ax.set_xticks(range(len(MODEL_ORDER)), ["Raw", "Full V2", "Lightweight"], rotation=15)
        ax.set_title(title)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "generation_seed_stability_key_metrics.png", dpi=180)
    plt.close(fig)

    key_metrics = [
        "wind_crps", "solar_daylight_crps", "renewable_crps_mw", "renewable_coverage90",
        "energy_score", "overall_spatial_corr_rmse", "wind_solar_spatial_corr_rmse",
        "event_all_any_hit", "event_tail_any_hit", "event_tail_member_hit_ratio",
        "event_tail_onset_error_h", "event_tail_duration_error_h", "event_tail_depth_ratio",
        "wind_solar_same_member_power_corr_rmse", "wind_solar_same_member_residual_corr_rmse",
    ] + [f"{source}_ramp_{lag}h_{suffix}" for source in ("wind", "solar") for lag in (1, 3, 6) for suffix in ("mae", "coverage90")]
    compact = aggregate[aggregate.metric.isin(key_metrics)].copy()
    compact = compact[["model", "metric", "mean", "std", "min", "max", "n"]]
    consistent = bool(direction["direction_consistent_no_worse"].all())
    coverage_consistent = bool(
        direction.loc[direction.metric.isin(["event_all_any_hit", "event_tail_any_hit", "event_tail_member_hit_ratio"]), "direction_consistent_no_worse"].all()
    )
    lines = [
        "# Station-24 generation-seed stability report",
        "",
        f"Seeds: {', '.join(map(str, seeds))}. Checkpoints, data, 500-member budget, 400 Raw + 100 Tail quota and evaluation definitions are fixed.",
        "",
        "## Aggregate mean, sample standard deviation and range",
        "",
        markdown(compact),
        "",
        "## Lightweight versus Raw sustained-event direction",
        "",
        markdown(direction),
        "",
        f"- Coverage evidence (all/tail any-hit and Tail member hit ratio) is directionally no-worse for every seed: **{coverage_consistent}**.",
        f"- Every reported sustained-event location/severity metric is directionally no-worse for every seed: **{consistent}**.",
        "- With only three generation seeds and four independent physical events, this is a Monte-Carlo stability check, not a population-significance claim.",
        "",
        "## Files",
        "",
        "- `per_seed_metrics.csv`: every model/seed measurement.",
        "- `aggregate_mean_std_range.csv`: mean, sample SD, minimum and maximum.",
        "- `lightweight_vs_raw_event_direction_by_seed.csv`: paired event-direction evidence.",
        "- `generation_seed_stability_key_metrics.png`: compact visual audit.",
        "",
    ]
    (output / "GENERATION_SEED_STABILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"GENERATION_SEED_STABILITY_SUMMARY_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
