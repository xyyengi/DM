#!/usr/bin/env python3
"""Evaluate continuous events and fast ramps for Raw versus JSTD scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from station_jstd_targets import (
    _bridge_short_gaps,
    build_station_jstd_target_arrays,
)


STANDARDS = {
    "loose": (12.0, 0.25, 0.50),
    "primary": (6.0, 0.50, 0.75),
    "strict": (3.0, 0.75, 1.00),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-label", default="Raw body-tail")
    parser.add_argument("--candidate-label", default="JSTD-Tail V1")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="allow deterministic regeneration of files in a partial output directory",
    )
    return parser.parse_args()


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without pandas' optional tabulate package."""

    columns = [str(column) for column in frame.columns]
    if not columns:
        return "(no columns)"

    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    change = np.diff(padded)
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def _aggregate(values: np.ndarray, indices: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    weight = capacities[indices] / capacities[indices].sum()
    return np.einsum("...ts,s->...t", values[..., indices], weight)


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scenarios = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")
    actual = np.load(path / "actual_data_normalized.npy", mmap_mode="r")
    forecast = np.load(path / "forecast_data_normalized.npy", mmap_mode="r")
    route = np.load(path / "tail_expert_route.npy", mmap_mode="r").astype(bool)
    if scenarios.ndim != 4 or scenarios.shape[0] != actual.shape[0]:
        raise ValueError(f"invalid scenario shape in {path}")
    return scenarios, actual, forecast, route


def _independent_rows(catalog: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for key in ((kind, direction) for kind in ("wind", "solar") for direction in ("negative", "positive")):
        rows = [row for row in catalog if (row["source"], row["direction"]) == key]
        rows.sort(key=lambda row: pd.Timestamp(row["physical_onset"]))
        groups: list[list[dict[str, object]]] = []
        for row in rows:
            onset = pd.Timestamp(row["physical_onset"])
            if not groups:
                groups.append([row])
                continue
            group_stop = max(pd.Timestamp(item["physical_stop_exclusive"]) for item in groups[-1])
            if onset <= group_stop + pd.Timedelta(hours=1):
                groups[-1].append(row)
            else:
                groups.append([row])
        selected.extend(max(group, key=lambda row: float(row["depth"])) for group in groups)
    return selected


def _match_member(
    magnitude: np.ndarray,
    onset: int,
    stop: int,
    truth_depth: float,
    entry_threshold: float,
) -> dict[str, float | bool]:
    candidate_mask = _bridge_short_gaps(magnitude >= 0.25 * entry_threshold, 1)
    candidates = _runs(candidate_mask)
    if not candidates:
        return {
            "has_candidate": False,
            "onset_abs_error_h": np.nan,
            "interval_recall": 0.0,
            "duration_abs_error_h": np.nan,
            "depth_ratio": 0.0,
            "depth_abs_error": np.nan,
        }
    truth_duration = stop - onset
    scored = []
    for left, right in candidates:
        intersection = max(0, min(stop, right) - max(onset, left))
        recall = intersection / max(truth_duration, 1)
        onset_error = abs(int(left) - onset)
        duration_error = abs((int(right) - int(left)) - truth_duration)
        score = recall - 0.01 * onset_error - 0.005 * duration_error
        scored.append((score, int(left), int(right), recall))
    _, left, right, recall = max(scored, key=lambda item: item[0])
    depth = float(np.max(magnitude[left:right]))
    return {
        "has_candidate": True,
        "onset_abs_error_h": float(abs(left - onset)),
        "interval_recall": float(recall),
        "duration_abs_error_h": float(abs((right - left) - truth_duration)),
        "depth_ratio": float(depth / max(truth_depth, 1e-8)),
        "depth_abs_error": float(abs(depth - truth_depth)),
    }


def evaluate_events(
    label: str,
    scenarios: np.ndarray,
    forecast: np.ndarray,
    route: np.ndarray,
    catalog: list[dict[str, object]],
    thresholds: dict[str, object],
    stations: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    capacities = stations.capacity_mw.to_numpy(float)
    rows = []
    for event_index, event in enumerate(catalog):
        issue = int(event["sample_index"])
        kind = str(event["source"])
        direction = str(event["direction"])
        indices = stations.index[stations.data_type.eq(kind)].to_numpy(int)
        residual = _aggregate(
            np.asarray(scenarios[issue]) - np.asarray(forecast[issue])[None],
            indices,
            capacities,
        )
        sign = -1.0 if direction == "negative" else 1.0
        magnitude = np.maximum(sign * residual, 0.0)
        onset = int(event["lead_onset"])
        stop = int(event["lead_stop_exclusive"])
        truth_depth = float(event["depth"])
        entry = float(thresholds["type_thresholds"][kind][direction])
        for member in range(magnitude.shape[0]):
            match = _match_member(magnitude[member], onset, stop, truth_depth, entry)
            rows.append(
                {
                    "variant": label,
                    "scope": scope,
                    "event_id": f"{kind}_{direction}_{event_index:03d}",
                    "issue": issue,
                    "source": kind,
                    "direction": direction,
                    "actual_duration_h": stop - onset,
                    "member": member,
                    "member_group": "tail" if bool(route[issue, member]) else "body",
                    **match,
                }
            )
    return pd.DataFrame(rows)


def summarize_hits(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded = pd.concat(
        [matches, matches.assign(member_group="all")], ignore_index=True
    )
    per_event_rows = []
    for standard, (onset, recall, depth) in STANDARDS.items():
        hit = (
            expanded.has_candidate
            & expanded.onset_abs_error_h.le(onset)
            & expanded.interval_recall.ge(recall)
            & expanded.depth_ratio.ge(depth)
        )
        current = expanded.assign(standard=standard, hit=hit)
        for keys, group in current.groupby(
            ["variant", "scope", "member_group", "event_id"], sort=False
        ):
            per_event_rows.append(
                {
                    "variant": keys[0],
                    "scope": keys[1],
                    "member_group": keys[2],
                    "event_id": keys[3],
                    "standard": standard,
                    "member_count": int(len(group)),
                    "hit_count": int(group.hit.sum()),
                    "member_hit_rate": float(group.hit.mean()),
                    "any_hit": bool(group.hit.any()),
                    "median_onset_error_h": float(group.onset_abs_error_h.median()),
                    "median_duration_error_h": float(group.duration_abs_error_h.median()),
                    "median_depth_ratio": float(group.depth_ratio.median()),
                }
            )
    per_event = pd.DataFrame(per_event_rows)
    summary = (
        per_event.groupby(["variant", "scope", "member_group", "standard"], sort=False)
        .agg(
            event_count=("event_id", "nunique"),
            events_with_any_hit=("any_hit", "sum"),
            mean_member_hit_rate=("member_hit_rate", "mean"),
            median_onset_error_h=("median_onset_error_h", "median"),
            median_duration_error_h=("median_duration_error_h", "median"),
            median_depth_ratio=("median_depth_ratio", "median"),
        )
        .reset_index()
    )
    return summary, per_event


def ramp_summary(
    label: str,
    scenarios: np.ndarray,
    actual: np.ndarray,
    stations: pd.DataFrame,
) -> list[dict[str, object]]:
    capacities = stations.capacity_mw.to_numpy(float)
    rows = []
    for kind in ("wind", "solar"):
        indices = stations.index[stations.data_type.eq(kind)].to_numpy(int)
        generated = _aggregate(scenarios, indices, capacities)
        truth = _aggregate(actual, indices, capacities)
        for lag in (1, 3, 6):
            generated_delta = generated[:, :, lag:] - generated[:, :, :-lag]
            truth_delta = truth[:, lag:] - truth[:, :-lag]
            lower = np.quantile(generated_delta, 0.05, axis=1)
            upper = np.quantile(generated_delta, 0.95, axis=1)
            rows.append(
                {
                    "variant": label,
                    "source": kind,
                    "lag_h": lag,
                    "median_ramp_mae": float(
                        np.mean(np.abs(np.median(generated_delta, axis=1) - truth_delta))
                    ),
                    "ramp_90_coverage": float(np.mean((truth_delta >= lower) & (truth_delta <= upper))),
                    "generated_ramp_std": float(np.std(generated_delta)),
                    "actual_ramp_std": float(np.std(truth_delta)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=args.resume_existing)
    target_payload = json.loads(
        (Path(args.candidate_run) / "jstd_event_targets.json").read_text(encoding="utf-8")
    )
    thresholds = target_payload["thresholds"]
    targets = build_station_jstd_target_arrays(args.data_path, "val", thresholds)
    full_catalog = list(targets.catalog)
    independent_catalog = _independent_rows(full_catalog)
    stations = (
        pd.read_csv(Path(args.data_path) / "station_order.csv")
        .sort_values("channel_index")
        .reset_index(drop=True)
    )
    result_sets = {
        args.baseline_label: _load(Path(args.baseline)),
        args.candidate_label: _load(Path(args.candidate)),
    }
    matches = []
    ramp_rows = []
    reference_actual = None
    reference_forecast = None
    for label, (scenarios, actual, forecast, route) in result_sets.items():
        if reference_actual is None:
            reference_actual = np.asarray(actual)
            reference_forecast = np.asarray(forecast)
        elif not np.array_equal(reference_actual, actual) or not np.array_equal(reference_forecast, forecast):
            raise ValueError("baseline and candidate do not share validation targets")
        matches.append(
            evaluate_events(label, scenarios, forecast, route, full_catalog, thresholds, stations, "overlap_windows")
        )
        matches.append(
            evaluate_events(label, scenarios, forecast, route, independent_catalog, thresholds, stations, "independent_physical")
        )
        ramp_rows.extend(ramp_summary(label, scenarios, actual, stations))
    match_frame = pd.concat(matches, ignore_index=True)
    summary, per_event = summarize_hits(match_frame)
    match_frame.to_csv(output / "continuous_event_member_matches.csv", index=False)
    per_event.to_csv(output / "continuous_event_per_event.csv", index=False)
    summary.to_csv(output / "continuous_event_three_standard_summary.csv", index=False)
    pd.DataFrame(ramp_rows).to_csv(output / "fast_ramp_1_3_6h_summary.csv", index=False)
    metadata = {
        "method": "jstd_continuous_event_three_standard_evaluation_v1",
        "standards": {
            name: {
                "onset_tolerance_h": values[0],
                "interval_recall": values[1],
                "depth_ratio": values[2],
            }
            for name, values in STANDARDS.items()
        },
        "event_duration_is_continuous": True,
        "fast_lags_are_observation_scales": [1, 3, 6],
        "event_count_overlap_windows": len(full_catalog),
        "event_count_independent_physical": len(independent_catalog),
    }
    (output / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    primary = summary[
        summary.standard.eq("primary")
        & summary.scope.eq("independent_physical")
        & summary.member_group.eq("all")
    ]
    lines = [
        "# JSTD连续事件与多尺度评价",
        "",
        "持续事件按真实 onset、stop、duration、depth 评价；1/3/6 h仅作为ramp观察尺度。",
        "",
        "## 主要标准（±6 h / 50%区间 / 75%深度）",
        "",
        _markdown_table(primary),
        "",
        "完整三档结果见 `continuous_event_three_standard_summary.csv`，fast结果见 `fast_ramp_1_3_6h_summary.csv`。",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"JSTD_EVENT_EVALUATION_COMPLETE output={output}")


if __name__ == "__main__":
    main()
