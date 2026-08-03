"""Audit aggregate-wind rare-event coverage for paired Station24 results."""

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
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-issues", type=int, default=3)
    return parser.parse_args()


def load_result(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    scenarios = np.load(path / "actual_scenarios_normalized.npy")
    actual = np.load(path / "actual_data_normalized.npy")
    forecast = np.load(path / "forecast_data_normalized.npy")
    metadata = json.loads(
        (path / "generation_metadata.json").read_text(encoding="utf-8")
    )
    if scenarios.ndim != 4 or actual.ndim != 3 or forecast.shape != actual.shape:
        raise ValueError(f"unexpected result shapes in {path}")
    return scenarios, actual, forecast, metadata


def rolling_mean(values: np.ndarray, width: int = 6) -> np.ndarray:
    if values.shape[-1] < width:
        raise ValueError("rolling width exceeds time dimension")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, "valid"), -1, values)


def aggregate_wind(
    scenarios: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    stations: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wind = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    capacity = stations.capacity_mw.to_numpy(dtype=np.float64)[wind]
    scenario_mw = np.einsum("nmts,s->nmt", scenarios[..., wind], capacity)
    actual_mw = np.einsum("nts,s->nt", actual[..., wind], capacity)
    forecast_mw = np.einsum("nts,s->nt", forecast[..., wind], capacity)
    return scenario_mw, actual_mw, forecast_mw


def coverage(scenarios: np.ndarray, actual: np.ndarray, level: float) -> float:
    tail = (1.0 - level) / 2.0
    lower = np.quantile(scenarios, tail, axis=1)
    upper = np.quantile(scenarios, 1.0 - tail, axis=1)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline, actual, forecast, baseline_meta = load_result(baseline_path)
    candidate, candidate_actual, candidate_forecast, candidate_meta = load_result(
        candidate_path
    )
    if actual.shape != candidate_actual.shape or not np.allclose(
        actual, candidate_actual
    ):
        raise ValueError("paired results do not contain the same observations")
    if not np.allclose(forecast, candidate_forecast):
        raise ValueError("paired results do not contain the same forecasts")
    if baseline.shape != candidate.shape:
        raise ValueError("paired results must use the same ensemble size and axes")
    for key in ("split", "generation_seed", "n_samples"):
        if baseline_meta.get(key) != candidate_meta.get(key):
            raise ValueError(f"paired metadata mismatch: {key}")

    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    base_wind, actual_wind, forecast_wind = aggregate_wind(
        baseline, actual, forecast, stations
    )
    candidate_wind, _, _ = aggregate_wind(
        candidate, actual, forecast, stations
    )
    issues = pd.read_csv(
        Path(args.data_path) / f"{candidate_meta['split']}_issue_dates.csv"
    )
    # Select reproducibly from observations: largest six-hour aggregate wind
    # forecast overestimate. No model result participates in issue selection.
    gap_6h = rolling_mean(forecast_wind - actual_wind, 6)
    event_start = np.argmax(gap_6h, axis=1)
    event_score = gap_6h[np.arange(len(gap_6h)), event_start]
    selected = np.argsort(event_score)[::-1][: min(args.top_issues, len(event_score))]

    levels = (0.80, 0.90, 0.95, 0.99)
    summary = {
        "selection_rule": "largest observed six-hour aggregate-wind forecast overestimate",
        "selection_uses_model_output": False,
        "split": candidate_meta["split"],
        "n_samples": int(candidate_meta["n_samples"]),
        "generation_seed": int(candidate_meta["generation_seed"]),
        "baseline": baseline_meta.get("condition_variant"),
        "candidate": candidate_meta.get("condition_variant"),
        "coverage": {
            "baseline": {str(level): coverage(base_wind, actual_wind, level) for level in levels},
            "candidate": {str(level): coverage(candidate_wind, actual_wind, level) for level in levels},
        },
    }
    rows: list[dict[str, object]] = []
    lead = np.arange(actual_wind.shape[-1])
    for rank, issue_index in enumerate(selected, start=1):
        start = int(event_start[issue_index])
        stop = start + 6
        issue_date = str(issues.iloc[issue_index].issue_date)
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
        for axis, name, values in zip(
            axes,
            ("2D baseline", "2F common-event"),
            (base_wind, candidate_wind),
        ):
            ensemble = values[issue_index]
            q005, q05, median, q95, q995 = np.quantile(
                ensemble, [0.005, 0.05, 0.50, 0.95, 0.995], axis=0
            )
            member_event_mean = ensemble[:, start:stop].mean(axis=1)
            extreme_members = np.argsort(member_event_mean)[:3]
            axis.fill_between(lead, q005, q995, color="#f8bbd0", alpha=0.35, label="99% envelope")
            axis.fill_between(lead, q05, q95, color="#ec407a", alpha=0.25, label="90% envelope")
            for member_rank, member in enumerate(extreme_members):
                axis.plot(
                    lead,
                    ensemble[member],
                    color="#7b1fa2",
                    alpha=0.55,
                    linewidth=0.9,
                    label="3 lowest 6h members" if member_rank == 0 else None,
                )
            axis.plot(lead, median, color="#e91e63", linewidth=1.6, label="median")
            axis.plot(lead, forecast_wind[issue_index], "--", color="#009688", linewidth=1.4, label="forecast")
            axis.plot(lead, actual_wind[issue_index], color="#20242a", linewidth=1.5, label="actual")
            axis.axvspan(start, stop - 1, color="#ffb300", alpha=0.10)
            axis.set_title(name)
            axis.set_ylabel("Aggregate wind MW")
            axis.grid(alpha=0.22)
        axes[0].legend(ncol=3, frameon=False, fontsize=8)
        axes[-1].set_xlabel("Lead hour")
        figure.suptitle(
            f"Rare wind event #{rank}: issue={issue_date}, event leads={start}-{stop - 1}"
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"extreme_wind_issue_{issue_index:02d}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

        actual_event = float(actual_wind[issue_index, start:stop].mean())
        for name, values in (("baseline", base_wind), ("candidate", candidate_wind)):
            member_event = values[issue_index, :, start:stop].mean(axis=1)
            row: dict[str, object] = {
                "issue_index": int(issue_index),
                "issue_date": issue_date,
                "event_start_lead": start,
                "event_stop_lead": stop - 1,
                "forecast_minus_actual_6h_mw": float(event_score[issue_index]),
                "model": name,
                "actual_6h_mean_mw": actual_event,
                "minimum_member_6h_mean_mw": float(member_event.min()),
                "lower_tail_gap_mw": float(member_event.min() - actual_event),
                "members_at_or_below_actual_6h": int(np.sum(member_event <= actual_event)),
            }
            for level in levels:
                row[f"coverage_{int(level * 100)}"] = coverage(
                    values[issue_index : issue_index + 1],
                    actual_wind[issue_index : issue_index + 1],
                    level,
                )
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "extreme_wind_tail_metrics.csv", index=False)
    (output_dir / "extreme_wind_tail_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"EXTREME_TAIL_AUDIT_COMPLETE output_dir={output_dir}")


if __name__ == "__main__":
    main()
