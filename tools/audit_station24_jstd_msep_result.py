#!/usr/bin/env python3
"""Integrity and mechanism audit for causal JSTD-MSEP generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--h1-result", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--event-eval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-label", default="JSTD-MSEP causal")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def ordinary_metrics(path: Path) -> dict[str, float]:
    payload = load_json(path / "metrics.json")
    return {
        "wind_crps": float(payload["station_average"]["wind"]["crps"]),
        "solar_crps": float(payload["station_average"]["solar"]["crps"]),
        "aggregate_wind_crps_mw": float(payload["aggregate_mw"]["wind"]["crps"]),
        "wind_coverage_90": float(payload["station_average"]["wind"]["coverage_90"]),
        "wind_width_90": float(payload["station_average"]["wind"]["width_90"]),
        "energy_score_pu": float(payload["joint"]["energy_score_pu"]),
        "spatial_corr_rmse_all_pairs": float(
            payload["joint"]["spatial_corr_rmse_all_pairs"]
        ),
    }


def main() -> None:
    args = parse_args()
    raw = Path(args.raw_result)
    h1 = Path(args.h1_result)
    candidate = Path(args.candidate_result)
    metadata = load_json(candidate / "generation_metadata.json")
    required = {
        "use_jstd_segment_prior": True,
        "use_jstd_event_hypothesis": False,
        "future_actual_used_as_generation_condition": False,
        "reportable_as_causal_forecast": True,
    }
    for key, expected in required.items():
        if metadata.get(key) is not expected:
            raise ValueError(
                f"MSEP causality invariant failed: {key}={metadata.get(key)!r}"
            )

    hypothesis = np.load(candidate / "jstd_segment_hypotheses.npy")
    count_probability = np.load(
        candidate / "jstd_segment_count_probability.npy"
    )
    route = np.load(candidate / "tail_expert_route.npy")
    if hypothesis.ndim != 4 or hypothesis.shape[2:] != (2, 6):
        raise ValueError("MSEP hypotheses must be [N,K,2,6]")
    if route.shape != hypothesis.shape[:2]:
        raise ValueError("MSEP route and hypothesis member axes disagree")
    if count_probability.shape != (hypothesis.shape[0], 3):
        raise ValueError("MSEP count distribution must be [N,3]")
    expected_route = hypothesis[..., 0].max(axis=2) > 0.5
    if not np.array_equal(route > 0, expected_route):
        raise ValueError("tail routes do not match sampled event hypotheses")
    active = hypothesis[..., 0] > 0.5
    active_hypothesis = hypothesis[active]
    if active_hypothesis.size == 0:
        raise ValueError("MSEP generated no active event hypotheses")

    event_frame = pd.read_csv(
        Path(args.event_eval) / "continuous_event_three_standard_summary.csv"
    )
    causal_rows = event_frame[
        event_frame.variant.eq(args.candidate_label)
        & event_frame.scope.eq("independent_physical")
        & event_frame.standard.eq("primary")
    ]
    if causal_rows.empty:
        available = sorted(event_frame.variant.dropna().astype(str).unique())
        raise ValueError(
            "MSEP event evaluation lacks primary independent rows for "
            f"candidate label {args.candidate_label!r}; available={available}"
        )

    metrics = {
        "raw": ordinary_metrics(raw),
        "h1_oracle_upper_bound": ordinary_metrics(h1),
        "msep_causal": ordinary_metrics(candidate),
    }
    relative = {
        key: float(
            (metrics["msep_causal"][key] - metrics["raw"][key])
            / max(abs(metrics["raw"][key]), 1e-12)
        )
        for key in metrics["raw"]
        if key != "wind_coverage_90"
    }
    unique_onsets = np.unique(
        np.rint(active_hypothesis[:, 1] * 167).astype(int)
    )
    audit = {
        "method": "jstd_msep_causal_result_audit_v1",
        "integrity": {
            "issues": int(hypothesis.shape[0]),
            "members_per_issue": int(hypothesis.shape[1]),
            "event_slots": int(hypothesis.shape[2]),
            "tail_member_fraction": float(expected_route.mean()),
            "configured_tail_fraction": float(
                metadata.get("jstd_segment_tail_fraction", 0.10)
            ),
            "unique_sampled_onset_hours": int(len(unique_onsets)),
            "sampled_onset_min_h": int(unique_onsets.min()),
            "sampled_onset_max_h": int(unique_onsets.max()),
            "sampled_duration_median_h": float(
                np.median(active_hypothesis[:, 2] * 168)
            ),
            "sampled_signed_wind_depth_median": float(
                np.median(active_hypothesis[:, 3])
            ),
            "sampled_signed_solar_depth_median": float(
                np.median(active_hypothesis[:, 4])
            ),
            "causal_generation_confirmed": True,
        },
        "ordinary_metrics": metrics,
        "relative_change_vs_raw": relative,
        "primary_independent_event_rows": causal_rows.to_dict(orient="records"),
        "decision_guidance": {
            "success": (
                "causal tail improves short-ramp and continuous-event metrics, "
                "while wind CRPS and Energy remain within 5% of Raw"
            ),
            "renderer_limited": (
                "oracle H1 is also weak on the failed event dimension"
            ),
            "prior_limited": (
                "H1 remains strong but causal onset/duration/depth are weak"
            ),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "jstd_msep_result_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# JSTD-MSEP 因果结果审计",
        "",
        f"- 成员级 Tail 比例：{expected_route.mean():.3f}",
        f"- 抽样 onset 覆盖：{unique_onsets.min()}–{unique_onsets.max()} h，共 {len(unique_onsets)} 个小时位置",
        f"- 抽样 duration 中位数：{np.median(active_hypothesis[:, 2] * 168):.2f} h",
        f"- 相对 Raw 风电 CRPS：{relative['wind_crps']:+.2%}",
        f"- 相对 Raw Energy Score：{relative['energy_score_pu']:+.2%}",
        "",
        "事件命中、onset、duration、depth及1/3/6 h指标见配套事件评价目录。",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"JSTD_MSEP_RESULT_AUDIT_COMPLETE output={output}")


if __name__ == "__main__":
    main()
