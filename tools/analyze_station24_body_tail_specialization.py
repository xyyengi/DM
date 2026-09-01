"""Measure whether routed tail members specialize in sustained wind deficits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-issues", type=int, default=5)
    return parser.parse_args()


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return None
    wins = 0.0
    for value in positive:
        wins += float(np.sum(value > negative))
        wins += 0.5 * float(np.sum(value == negative))
    return wins / float(len(positive) * len(negative))


def deep_replay_specification(replay: dict[str, object]) -> dict[str, object]:
    """Resolve both legacy replay files and the unified deep+mismatch schema."""

    if replay.get("method") == "train_unified_wind_event_replay_v1":
        deep = replay.get("deep_replay")
        if not isinstance(deep, dict):
            raise ValueError("unified event replay is missing deep_replay")
        return deep
    return replay


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    result_dir = Path(args.result_dir)
    data_path = Path(args.data_path)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    scenarios = np.load(result_dir / "actual_scenarios_normalized.npy")
    actual = np.load(result_dir / "actual_data_normalized.npy")
    forecast = np.load(result_dir / "forecast_data_normalized.npy")
    route = np.load(result_dir / "tail_expert_route.npy").astype(bool)
    probability = np.load(result_dir / "tail_expert_probability.npy")
    attention = np.load(result_dir / "tail_condition_attention.npy")
    if scenarios.ndim != 4 or actual.shape != forecast.shape:
        raise ValueError("invalid Station24 generation arrays")
    issues, members, hours, stations_count = scenarios.shape
    if route.shape != (issues, members) or probability.shape != (issues,):
        raise ValueError("routing arrays do not match generated scenarios")
    if not np.any(route) or not np.any(~route):
        raise ValueError("specialization analysis requires body and tail members")

    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    if len(stations) != stations_count:
        raise ValueError("station metadata does not match generation arrays")
    wind = stations.data_type.eq("wind").to_numpy()
    capacities = stations.capacity_mw.to_numpy(dtype=np.float64)
    wind_capacity = capacities[wind]
    wind_capacity_weight = wind_capacity / wind_capacity.sum()
    issue_dates = pd.read_csv(data_path / "val_issue_dates.csv")["issue_date"].astype(str)

    aggregate_scenarios_mw = np.einsum(
        "nkts,s->nkt", scenarios[..., wind], wind_capacity
    )
    aggregate_actual_mw = np.einsum("nts,s->nt", actual[..., wind], wind_capacity)
    aggregate_forecast_mw = np.einsum(
        "nts,s->nt", forecast[..., wind], wind_capacity
    )
    aggregate_scenarios_normalized = np.einsum(
        "nkts,s->nkt", scenarios[..., wind], wind_capacity_weight
    )
    aggregate_forecast_normalized = np.einsum(
        "nts,s->nt", forecast[..., wind], wind_capacity_weight
    )
    aggregate_actual_normalized = np.einsum(
        "nts,s->nt", actual[..., wind], wind_capacity_weight
    )

    replay = json.loads((run_dir / "event_replay.json").read_text(encoding="utf-8"))
    deep_replay = deep_replay_specification(replay)
    window = int(deep_replay["event_window_hours"])
    if not 1 <= window <= hours:
        raise ValueError(f"invalid deep-event window={window}")
    severity_mw = np.zeros(issues, dtype=np.float64)
    severity_normalized = np.zeros(issues, dtype=np.float64)
    event_start = np.zeros(issues, dtype=np.int64)
    for issue in range(issues):
        mismatch_mw = aggregate_forecast_mw[issue] - aggregate_actual_mw[issue]
        curve_mw = rolling_mean(mismatch_mw, window)
        start = int(np.argmax(curve_mw))
        event_start[issue] = start
        severity_mw[issue] = float(curve_mw[start])
        mismatch_normalized = (
            aggregate_forecast_normalized[issue]
            - aggregate_actual_normalized[issue]
        )
        severity_normalized[issue] = float(
            rolling_mean(mismatch_normalized, window).max()
        )

    q80_threshold = float(deep_replay["severity_thresholds"][0])
    validation_event = severity_normalized >= q80_threshold
    gate_auc = binary_auc(validation_event, probability)
    gate_brier = float(np.mean((probability - validation_event.astype(float)) ** 2))

    top = np.argsort(severity_mw)[::-1][: int(args.top_issues)]
    records: list[dict[str, object]] = []
    for rank, issue in enumerate(top, start=1):
        start = int(event_start[issue])
        stop = start + window
        member_level = aggregate_scenarios_mw[issue, :, start:stop].mean(axis=1)
        actual_level = float(aggregate_actual_mw[issue, start:stop].mean())
        body_values = member_level[~route[issue]]
        tail_values = member_level[route[issue]]
        body_hit_count = int(np.sum(body_values <= actual_level))
        tail_hit_count = int(np.sum(tail_values <= actual_level))
        records.append(
            {
                "rank": rank,
                "issue_index": int(issue),
                "issue_date": issue_dates.iloc[issue],
                "lead_start": start,
                "lead_end": stop - 1,
                "event_window_hours": window,
                "forecast_minus_actual_event_mw": float(severity_mw[issue]),
                "actual_event_mean_mw": actual_level,
                "tail_probability": float(probability[issue]),
                "body_member_count": int(len(body_values)),
                "tail_member_count": int(len(tail_values)),
                "body_hit_count": body_hit_count,
                "tail_hit_count": tail_hit_count,
                "body_hit_rate": float(np.mean(body_values <= actual_level))
                if len(body_values)
                else None,
                "tail_hit_rate": float(np.mean(tail_values <= actual_level))
                if len(tail_values)
                else None,
                "body_minimum_event_mw": float(body_values.min())
                if len(body_values)
                else None,
                "tail_minimum_event_mw": float(tail_values.min())
                if len(tail_values)
                else None,
                "tail_minus_body_mean_event_mw": float(
                    tail_values.mean() - body_values.mean()
                )
                if len(tail_values) and len(body_values)
                else None,
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(output / "top5_body_tail_specialization.csv", index=False)

    body_residual = aggregate_scenarios_mw - aggregate_forecast_mw[:, None, :]
    body_values = body_residual[~route]
    tail_values = body_residual[route]
    summary = {
        "method": "member_routed_body_tail_specialization_v1",
        "validation_issue_count": int(issues),
        "members_per_issue": int(members),
        "body_member_count": int((~route).sum()),
        "tail_member_count": int(route.sum()),
        "tail_member_fraction": float(route.mean()),
        "validation_q80_event_count": int(validation_event.sum()),
        "event_window_hours": window,
        "event_replay_method": str(replay.get("method")),
        "deep_replay_method": str(deep_replay.get("method")),
        "gate_q80_roc_auc": gate_auc,
        "gate_q80_brier_score": gate_brier,
        "body_aggregate_residual_mean_mw": float(body_values.mean()),
        "tail_aggregate_residual_mean_mw": float(tail_values.mean()),
        "body_aggregate_residual_q01_mw": float(np.quantile(body_values, 0.01)),
        "tail_aggregate_residual_q01_mw": float(np.quantile(tail_values, 0.01)),
        "condition_attention_names": [
            "issued_wind_level",
            "issued_wind_down_ramp_3h",
            "aligned_forecast_revision",
            "forecast_low_output_state",
            "forecast_down_ramp_state",
            "recent_observed_forecast_error",
        ],
        "condition_attention_mean": [
            float(value) for value in attention.mean(axis=0)
        ],
        "top5_tail_hit_total": int(frame.tail_hit_count.sum()),
        "top5_body_hit_total": int(frame.body_hit_count.sum()),
        "top5_events_with_tail_hit": int((frame.tail_hit_count > 0).sum()),
        "top5_events_with_body_hit": int((frame.body_hit_count > 0).sum()),
        "test_used": False,
    }
    (output / "body_tail_specialization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"BODY_TAIL_SPECIALIZATION_COMPLETE output={output}")


if __name__ == "__main__":
    main()
