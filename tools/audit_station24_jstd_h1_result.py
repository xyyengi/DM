#!/usr/bin/env python3
"""Audit integrity and screening evidence for the non-causal JSTD H1 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--parent-result", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--raw-event-eval", required=True)
    parser.add_argument("--parent-event-eval", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _event_row(path: Path, variant: str, member_group: str) -> dict[str, object]:
    frame = pd.read_csv(path / "continuous_event_three_standard_summary.csv")
    selected = frame[
        frame.variant.eq(variant)
        & frame.scope.eq("independent_physical")
        & frame.member_group.eq(member_group)
        & frame.standard.eq("primary")
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one primary {member_group} row for {variant!r} in {path}"
        )
    return {
        str(key): value.item() if isinstance(value, np.generic) else value
        for key, value in selected.iloc[0].to_dict().items()
    }


def _ordinary_metrics(result: Path) -> dict[str, float]:
    payload = _load_json(result / "metrics.json")
    station = payload["station_average"]
    aggregate = payload["aggregate_mw"]
    joint = payload["joint"]
    return {
        "wind_crps": float(station["wind"]["crps"]),
        "solar_crps": float(station["solar"]["crps"]),
        "aggregate_wind_crps_mw": float(aggregate["wind"]["crps"]),
        "wind_coverage_90": float(station["wind"]["coverage_90"]),
        "wind_width_90": float(station["wind"]["width_90"]),
        "energy_score_pu": float(joint["energy_score_pu"]),
        "spatial_corr_rmse_all_pairs": float(
            joint["spatial_corr_rmse_all_pairs"]
        ),
    }


def _relative(candidate: float, baseline: float) -> float:
    return float((candidate - baseline) / max(abs(baseline), 1e-12))


def main() -> None:
    args = parse_args()
    raw_result = Path(args.raw_result)
    parent_result = Path(args.parent_result)
    candidate_result = Path(args.candidate_result)
    metadata = _load_json(candidate_result / "generation_metadata.json")
    required = {
        "use_jstd_event_hypothesis": True,
        "future_actual_used_as_generation_condition": True,
        "reportable_as_causal_forecast": False,
        "oracle_event_hypothesis_acknowledged": True,
    }
    for key, expected in required.items():
        if metadata.get(key) is not expected:
            raise ValueError(
                f"H1 metadata safety invariant failed: {key}={metadata.get(key)!r}"
            )

    hypothesis = np.load(candidate_result / "jstd_event_hypothesis.npy")
    route = np.load(candidate_result / "tail_expert_route.npy")
    recorded_hypothesis = np.load(
        candidate_result / "tail_condition_attention.npy"
    )
    if hypothesis.ndim != 2 or hypothesis.shape[1] != 6:
        raise ValueError("candidate hypothesis must be [N,6]")
    if route.ndim != 2 or route.shape[0] != hypothesis.shape[0]:
        raise ValueError("candidate route must be [N,K]")
    if not np.array_equal(recorded_hypothesis, hypothesis):
        raise ValueError("saved H1 audit channel does not match consumed hypothesis")
    active = hypothesis[:, 0] > 0.5
    if not np.any(active) or np.all(active):
        raise ValueError("H1 audit requires both event and ordinary validation windows")
    non_event_routes = int(np.count_nonzero(route[~active]))
    if non_event_routes != 0:
        raise ValueError("non-event validation windows contain H1 tail members")
    active_route_rates = np.mean(route[active] > 0.5, axis=1)
    configured_fraction = float(metadata["jstd_h1_tail_fraction"])
    if not np.allclose(
        active_route_rates,
        configured_fraction,
        # The generator enforces a deterministic quota inside each memory
        # chunk. Rounding across fallback chunks can move the full-run rate by
        # a few members while preserving the intended mixture.
        atol=max(0.01, 1.0 / route.shape[1] + 1e-8),
    ):
        raise ValueError("event-window tail quota does not match H1 configuration")

    raw_label = "Raw body-tail"
    parent_label = "JSTD-Tail V1"
    candidate_label = "H1 oracle (non-causal)"
    raw_tail = _event_row(Path(args.raw_event_eval), raw_label, "tail")
    parent_tail = _event_row(Path(args.parent_event_eval), parent_label, "tail")
    candidate_tail = _event_row(
        Path(args.parent_event_eval), candidate_label, "tail"
    )
    candidate_body = _event_row(
        Path(args.parent_event_eval), candidate_label, "body"
    )

    ordinary = {
        "raw": _ordinary_metrics(raw_result),
        "jstd_v1_parent": _ordinary_metrics(parent_result),
        "h1_oracle": _ordinary_metrics(candidate_result),
    }
    relative_to_raw = {
        key: _relative(value, ordinary["raw"][key])
        for key, value in ordinary["h1_oracle"].items()
        if key not in {"wind_coverage_90"}
    }

    improvements = {
        "events_with_any_hit_vs_parent": int(
            candidate_tail["events_with_any_hit"]
            - parent_tail["events_with_any_hit"]
        ),
        "member_hit_rate_vs_parent": float(
            candidate_tail["mean_member_hit_rate"]
            - parent_tail["mean_member_hit_rate"]
        ),
        "median_onset_error_reduction_h_vs_parent": float(
            parent_tail["median_onset_error_h"]
            - candidate_tail["median_onset_error_h"]
        ),
        "median_duration_error_reduction_h_vs_parent": float(
            parent_tail["median_duration_error_h"]
            - candidate_tail["median_duration_error_h"]
        ),
        "median_depth_ratio_distance_reduction_vs_parent": float(
            abs(float(parent_tail["median_depth_ratio"]) - 1.0)
            - abs(float(candidate_tail["median_depth_ratio"]) - 1.0)
        ),
    }
    improved_dimensions = sum(value > 0 for value in improvements.values())
    event_count = int(candidate_tail["event_count"])
    hit_count = int(candidate_tail["events_with_any_hit"])
    ordinary_crps_change = relative_to_raw["wind_crps"]
    if (
        hit_count == event_count
        and improved_dimensions >= 3
        and ordinary_crps_change <= 0.05
    ):
        screen = "h1_positive_controllability_signal"
    elif hit_count >= max(1, event_count - 1) and improved_dimensions >= 2:
        screen = "h1_partial_controllability_signal"
    else:
        screen = "h1_no_clear_controllability_signal"

    audit = {
        "method": "jstd_event_hypothesis_h1_result_audit_v1",
        "interpretation": (
            "non-causal validation-oracle structural upper bound; not a forecast result"
        ),
        "integrity": {
            "validation_issue_count": int(hypothesis.shape[0]),
            "event_issue_count": int(active.sum()),
            "ordinary_issue_count": int((~active).sum()),
            "members_per_issue": int(route.shape[1]),
            "configured_event_tail_fraction": configured_fraction,
            "observed_event_tail_fraction_mean": float(active_route_rates.mean()),
            "non_event_tail_member_count": non_event_routes,
            "saved_hypothesis_matches_consumed_audit_channel": True,
            "causal_report_allowed": False,
        },
        "primary_independent_event_tail": {
            "raw": raw_tail,
            "jstd_v1_parent": parent_tail,
            "h1_oracle": candidate_tail,
            "h1_body_reference": candidate_body,
        },
        "h1_improvements_over_parent": improvements,
        "ordinary_metrics": ordinary,
        "h1_relative_change_vs_raw": relative_to_raw,
        "automatic_screen": screen,
        "decision_rule": {
            "positive": (
                "all independent events hit, at least 3/5 event dimensions improve "
                "over JSTD V1, and wind CRPS worsens by no more than 5% versus Raw"
            ),
            "partial": (
                "at least event_count-1 events hit and at least 2/5 event dimensions improve"
            ),
            "otherwise": "no clear H1 controllability signal",
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "jstd_h1_result_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# JSTD H1 结果完整性与上界判读",
        "",
        "> 本实验使用验证集真实事件属性，只能判断结构可控性，不能作为因果预测结果。",
        "",
        f"- 自动筛查：`{screen}`",
        f"- 事件窗口：{int(active.sum())}；普通窗口：{int((~active).sum())}",
        f"- 事件窗口 Tail 比例：{active_route_rates.mean():.3f}",
        f"- 非事件 Tail 成员数：{non_event_routes}",
        f"- H1 Tail 主标准命中：{hit_count}/{event_count}",
        f"- 相对 JSTD V1 的改善维度数：{improved_dimensions}/5",
        f"- 相对 Raw 的 wind CRPS 变化：{ordinary_crps_change:+.2%}",
        "",
        "详细数值见 `jstd_h1_result_audit.json`。最终结论仍需结合代表性事件图人工复核。",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(
        "JSTD_H1_RESULT_AUDIT_COMPLETE "
        f"screen={screen} output={output}"
    )


if __name__ == "__main__":
    main()
