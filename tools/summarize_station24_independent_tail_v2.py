"""Write a compact, self-contained report for the V2 formal pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def get(data: dict, *keys: str) -> float:
    value = data
    for key in keys:
        value = value[key]
    return float(value)


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--joint-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = json.loads((Path(args.baseline) / "metrics.json").read_text(encoding="utf-8"))
    candidate = json.loads((Path(args.candidate) / "metrics.json").read_text(encoding="utf-8"))
    joint = pd.read_csv(Path(args.joint_dir) / "same_member_wind_solar_correlation.csv")
    fast = pd.read_csv(Path(args.event_dir) / "fast_ramp_1_3_6h_summary.csv")
    rows = []
    metrics = {
        "Wind station CRPS": ("station_average", "wind", "crps"),
        "Solar daylight CRPS": ("station_average", "solar_daylight", "crps"),
        "Renewable aggregate CRPS (MW)": ("aggregate_mw", "renewable", "crps"),
        "Renewable aggregate 90% coverage": ("aggregate_mw", "renewable", "coverage_90"),
        "Energy Score": ("joint", "energy_score_pu"),
        "Spatial correlation RMSE": ("joint", "spatial_corr_rmse_all_pairs"),
        "Wind-solar spatial correlation RMSE": ("joint", "spatial_corr_rmse_wind_solar"),
    }
    for name, path in metrics.items():
        old, new = get(baseline, *path), get(candidate, *path)
        rows.append((name, old, new, 100.0 * (new - old) / max(abs(old), 1e-12)))
    lines = [
        "# Independent joint tail V2 result summary",
        "",
        "The candidate is a fixed 400 Raw-body + 100 independent joint-tail ensemble. "
        "Wind and solar are generated jointly by the same denoiser; future actual/event labels are not generation conditions.",
        "",
        "## Ordinary and joint quality",
        "",
        "| Metric | Raw baseline | V2 mixture | Relative change |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(f"| {name} | {old:.6f} | {new:.6f} | {delta:+.2f}% |" for name, old, new, delta in rows)
    lines += ["", "## Same-member wind-solar correlation", "", markdown(joint), "", "## Fast ramp audit", "", markdown(fast), "", "## Output map", "", "- `comparisons/`: paired ordinary metrics and representative envelopes", "- `continuous_event_evaluation/`: loose/primary/strict event hit, onset, duration and depth", "- `joint_wind_solar_evaluation/`: lead-day and same-member wind-solar/renewable metrics", "- `representative_joint_plots/`: body/tail/all-member wind, solar and renewable-total plots", "- `extreme_wind_tail/` and `wind_event_timing/`: extreme and timing diagnostics", ""]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"INDEPENDENT_TAIL_V2_SUMMARY_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
