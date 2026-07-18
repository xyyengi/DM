#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-evaluate saved V3/Vmix/V4 scenarios at event level without generation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.event_level_evaluation import (  # noqa: E402
    CHANNELS,
    aggregate_ordinary_metrics,
    audit_saved_run,
    evaluate_coupling,
    evaluate_event_mappings,
)
from src.eval.event_thresholds import (  # noqa: E402
    build_event_catalog,
    calculate_features,
    fit_event_thresholds,
    load_hourly_split,
    map_windows_to_events,
)


RUNS = {
    "V3": "20260715_195500_v3_actual_forecast_time_encoding_168h",
    "Vmix": "20260715_200506_v_mix_residual_forecast_concat_guidance",
    "V4": "20260718_145118_v4_residual_forecast_time_no_guidance_168h",
}
MAIN_EVENT_TYPES = (
    "low_wind", "low_renewable", "low_solar_daily_energy", "high_load",
    "high_net_load", "high_ramp_6h",
)
DIAGNOSTIC_EVENT_TYPES = ("high_ramp_12h",)
NO_TEST_EVENT_TYPES = (
    "compound_low_renewable_high_net_load",
    "wind_drop_load_rise_1h",
    "solar_drop_load_rise_1h",
)
SPECS = {
    "main_p10_p90_gap1": {"low": "p10", "high": "p90", "gap": 1},
    "strict_p05_p95_gap1": {"low": "p05", "high": "p95", "gap": 1},
    "sensitivity_p10_p90_gap0": {"low": "p10", "high": "p90", "gap": 0},
    "sensitivity_p10_p90_gap3": {"low": "p10", "high": "p90", "gap": 3},
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ramp_diagnostics(split: dict, thresholds: dict, events: list[dict]) -> tuple[list[dict], list[dict]]:
    f = calculate_features(np.asarray(split["hourly"]), split["timestamps"])
    index = {timestamp.isoformat(): idx for idx, timestamp in enumerate(split["timestamps"])}
    diagnostics: list[dict] = []
    alternatives: list[dict] = []
    for horizon in (6, 12):
        event_type = f"high_ramp_{horizon}h"
        threshold = thresholds["absolute_net_load_ramp_mw"][f"{horizon}h"]["p90"]
        ramp = np.abs(f["net_load_mw"][horizon:] - f["net_load_mw"][:-horizon])
        endpoint_indices = np.flatnonzero(ramp >= threshold) + horizon
        for event in [row for row in events if row["event_type"] == event_type]:
            start, end = index[event["start_time"]], index[event["end_time"]]
            endpoints = [int(point) for point in endpoint_indices if point - horizon >= start and point <= end]
            times = [split["timestamps"][point].isoformat() for point in endpoints]
            span = endpoints[-1] - endpoints[0] if len(endpoints) >= 2 else 0
            chain = len(endpoints) > 1 and int(event["duration_hours"]) > horizon + 1
            diagnostics.append({
                "event_id": event["event_id"],
                "event_type": event_type,
                "horizon_hours": horizon,
                "threshold_mw": threshold,
                "superthreshold_endpoint_count": len(endpoints),
                "superthreshold_endpoint_times": ";".join(times),
                "current_start_time": event["start_time"],
                "current_end_time": event["end_time"],
                "current_duration_hours": event["duration_hours"],
                "first_last_endpoint_span_hours": span,
                "chain_overlap_extended": chain,
            })

            # Alternative only: cluster endpoints whose consecutive gap <= 3h.
            clusters: list[list[int]] = []
            for point in endpoints:
                if not clusters or point - clusters[-1][-1] > 3:
                    clusters.append([point])
                else:
                    clusters[-1].append(point)
            for cluster_number, cluster in enumerate(clusters, 1):
                alt_start = cluster[0] - horizon
                alt_end = cluster[-1]
                alternatives.append({
                    "source_event_id": event["event_id"],
                    "alternative_cluster_id": f"{event['event_id']}_endpoint_cluster_{cluster_number:02d}",
                    "event_type": event_type,
                    "endpoint_cluster_gap_hours": 3,
                    "endpoint_count": len(cluster),
                    "endpoint_times": ";".join(split["timestamps"][point].isoformat() for point in cluster),
                    "alternative_start_time": split["timestamps"][alt_start].isoformat(),
                    "alternative_end_time": split["timestamps"][alt_end].isoformat(),
                    "alternative_duration_hours": alt_end - alt_start + 1,
                })
    return diagnostics, alternatives


def ranks(values: dict[str, float], lower_is_better: bool = True) -> dict[str, int]:
    ordered = sorted(values, key=values.get, reverse=not lower_is_better)
    return {model: rank for rank, model in enumerate(ordered, 1)}


def fmt(value, digits=3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:,.{digits}f}"


def build_report(record: dict) -> str:
    lines = [
        "# V3、Vmix、V4 已保存场景的事件级重新评价",
        "",
        "> 本轮只读取已有 `.npy/.npz` 结果；没有调用训练、checkpoint推理或场景生成。",
        "",
        "## 1. 结果对应性审计",
        "",
        "| 模型 | 可评价 | 测试窗口 | 实际场景数/窗口 | actual顺序匹配 | forecast顺序匹配 | 反归一化匹配 | NPZ交叉匹配 | 配置声明场景数 |",
        "|---|---|---:|---:|---|---|---|---|---:|",
    ]
    for model, audit in record["correspondence"].items():
        lines.append(
            f"| {model} | {audit['eligible']} | {audit.get('n_test_windows','NA')} | "
            f"{audit.get('n_scenarios_per_window','NA')} | {audit.get('actual_normalized_matches_source','NA')} | "
            f"{audit.get('forecast_normalized_matches_source','NA')} | "
            f"{audit.get('scenarios_physical_matches_denormalization','NA')} | "
            f"{audit.get('npz_matches_windowed_scenarios','NA')} | {audit.get('config_declared_n_samples','NA')} |"
        )
    lines += [
        "",
        "三个模型的实际保存数组均为 `[577,20,3,168]`，行号严格对应 `test_window_00000` 至 `test_window_00576`。V3/Vmix的旧配置仍声明10个场景，但实际Numpy数组、NPZ压平长度和均匀概率三者一致证明实际保存的是20个；评价采用实际20个，并把配置不一致保留为元数据警告。变量顺序均为wind、solar、load，单位MW。",
        "",
        "## 2. 普通概率评价",
        "",
        "Energy Score原始值是168×3维路径的欧氏范数，天然随维度增大；同时报告除以`sqrt(504)`的缩放值。全部滚动起点577个；每日00:00固定起点25个；非重叠168h辅助窗口4个。",
        "",
        "| 起点口径 | 模型 | n_windows | CRPS(MW) | ES原始(MW) | ES/√维数(MW) | 90%覆盖率 | 90%宽度(%全局极差) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in record["ordinary_metrics"]:
        lines.append(
            f"| {row['start_rule']} | {row['model']} | {row['n_windows']} | "
            f"{fmt(row['total_crps_mw'])} | {fmt(row['multivariate_energy_score_mw'])} | "
            f"{fmt(row['multivariate_es_per_sqrt_dimension_mw'])} | "
            f"{fmt(row['total_coverage_90'])}% | {fmt(row['total_width_90_pct_range'])}% |"
        )
    lines += [
        "",
        "## 3. 事件口径",
        "",
        "主结果：train唯一小时P10/P90、持续至少6h、持续型事件gap=1。严格敏感性：P05/P95且gap=1。gap=0和3另存为敏感性；gap不用于high_ramp。光伏固定使用07:00—18:00日间能量。",
        "",
        "事前指标只用`contains_event_start=true, lead_hours>=0, post_onset=false`；完整过程/峰值/持续时间进一步要求`fully_contains_event=true`。提前期采用左闭右开区间 `[0,24)、[24,48)、[48,72)、[72,168)`。",
        "",
        "每个CSV先按event_id和提前期平均多个窗口，再在event_id之间平均。下表只展示主P10/P90、gap=1；`n_events<3`标为案例性。",
        "完整过程中的peak按事件变量取极值：低风/低新能源取最小值，高负荷/高净负荷取最大值，ramp取最大绝对变化，低光伏取日间能量容量因子。持续时间误差仅用于持续型事件：统计生成场景在真实事件区间内满足同一阈值的小时数，与当前真实事件区间长度比较；光伏为0或12小时，ramp不计算duration MAE。",
        "",
        "| 模型 | event_type | lead | n_events | n_windows | 窗口/事件 | 案例性 | onset CRPS | onset ES/√维数 | onset覆盖90 | full peak MAE | full duration MAE(h) |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in record["main_event_summary"]:
        lines.append(
            f"| {row['model']} | {row['event_type']} | {row['lead_group']} | {row['n_events']} | "
            f"{row['n_windows']} | {fmt(row['windows_per_event_mean'],1)} | {row['case_only']} | "
            f"{fmt(row.get('onset_total_crps_mw'))} | {fmt(row.get('onset_multivariate_es_per_sqrt_dimension_mw'))} | "
            f"{fmt(row.get('onset_total_coverage_90'))} | {fmt(row.get('full_peak_mae'))} | "
            f"{fmt(row.get('full_duration_mae_hours'))} |"
        )
    lines += [
        "",
        "完整主结果、严格尾部、gap敏感性均在 `event_summary_all_specs.csv`；每个event_id的两级汇总在 `event_by_id_all_specs.csv`。严格阈值为0事件时保留n_events=0，没有回退阈值。high_ramp_12h仅诊断展示，不进入综合排名，其duration不解释为真实物理持续时间。",
        "",
        "## 4. 当前测试集无事件的类型",
        "",
        "| 类型 | 主P10/P90事件数 | 处理 |",
        "|---|---:|---|",
    ]
    for event_type, count in record["main_test_event_counts"].items():
        if event_type in NO_TEST_EVENT_TYPES:
            lines.append(f"| {event_type} | {count} | 不降阈值，不参与模型排名 |")
    lines += [
        "",
        "## 5. 源荷耦合描述",
        "",
        "在high_net_load与high_ramp_6h完整包含窗口中，固定使用train P90风/光下降阈值及P90负荷上升阈值。0—3h匹配比例定义为：显著风或光下降终点中，随后0—3小时存在显著负荷上升的比例。详细逐事件结果见 `coupling_by_event.csv`。",
        "",
        "| 模型 | event_type | n_events | 风最大下降差 | 光最大下降差 | 负荷最大上升差 | 同小时重叠差(h) | 0—3h匹配比例差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in record["coupling_summary"]:
        lines.append(
            f"| {row['model']} | {row['event_type']} | {row['n_events']} | "
            f"{fmt(row.get('difference_generated_minus_actual_wind_max_drop_mw'))} | "
            f"{fmt(row.get('difference_generated_minus_actual_solar_max_drop_mw'))} | "
            f"{fmt(row.get('difference_generated_minus_actual_load_max_rise_mw'))} | "
            f"{fmt(row.get('difference_generated_minus_actual_same_hour_source_drop_load_rise_hours'))} | "
            f"{fmt(row.get('difference_generated_minus_actual_lag_0_3h_match_ratio'))} |"
        )
    lines += [
        "",
        "## 6. high_ramp链式合并诊断",
        "",
        f"P90主定义下，high_ramp_6h共{record['ramp_diagnostic_summary']['high_ramp_6h_events']}个事件，其中{record['ramp_diagnostic_summary']['high_ramp_6h_chained']}个由多个重叠[t-6,t]区间形成链；high_ramp_12h共{record['ramp_diagnostic_summary']['high_ramp_12h_events']}个，其中{record['ramp_diagnostic_summary']['high_ramp_12h_chained']}个形成链。逐事件终点和时间见 `high_ramp_chain_diagnostics.csv`。",
        "",
        f"备选结果仅作诊断：按超阈终点时间聚类，相邻终点间隔≤3h归为一簇，再用`[首终点-h,末终点]`表示。6h由当前{record['ramp_diagnostic_summary']['high_ramp_6h_events']}个变为{record['ramp_diagnostic_summary']['high_ramp_6h_alternative_clusters']}簇，平均时长由{record['ramp_diagnostic_summary']['high_ramp_6h_current_mean_duration']:.2f}h变为{record['ramp_diagnostic_summary']['high_ramp_6h_alternative_mean_duration']:.2f}h，最大时长由{record['ramp_diagnostic_summary']['high_ramp_6h_current_max_duration']:.0f}h降至{record['ramp_diagnostic_summary']['high_ramp_6h_alternative_max_duration']:.0f}h；12h由当前{record['ramp_diagnostic_summary']['high_ramp_12h_events']}个变为{record['ramp_diagnostic_summary']['high_ramp_12h_alternative_clusters']}簇，平均时长由{record['ramp_diagnostic_summary']['high_ramp_12h_current_mean_duration']:.2f}h变为{record['ramp_diagnostic_summary']['high_ramp_12h_alternative_mean_duration']:.2f}h，最大时长由{record['ramp_diagnostic_summary']['high_ramp_12h_current_max_duration']:.0f}h降至{record['ramp_diagnostic_summary']['high_ramp_12h_alternative_max_duration']:.0f}h。结果保存在 `high_ramp_endpoint_clusters_gap3.csv`，没有替换本轮正式定义。",
        "",
        "## 7. 排序与口径修正影响",
        "",
        "| 范围 | 指标 | 排序（优→劣） |",
        "|---|---|---|",
    ]
    for row in record["rankings"]:
        lines.append(f"| {row['scope']} | {row['metric']} | {row['order']} |")
    lines += [
        "",
        record["conclusion_change"],
        "",
        "## 8. 输出说明",
        "",
        "- `ordinary_metrics.csv`：三种起点口径。",
        "- `event_summary_all_specs.csv`：主、严格、gap=0/3全部事件汇总。",
        "- `event_by_id_all_specs.csv`：每个event_id、提前期的第一层汇总。",
        "- `coupling_by_event.csv`：真实—生成耦合属性差异。",
        "- `high_ramp_chain_diagnostics.csv`及`high_ramp_endpoint_clusters_gap3.csv`：链式诊断。",
        "- `correspondence_audit.json`：三个模型结果对应性证据。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="diffusion_npy_normalized")
    parser.add_argument("--outputs", default="outputs_shandong")
    parser.add_argument("--output_dir", default="outputs_shandong/event_evaluation/saved_runs_event_reevaluation")
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data_path
    outputs = PROJECT_ROOT / args.outputs
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {name: load_hourly_split(data_path, name) for name in ("train", "test")}
    thresholds = fit_event_thresholds(splits["train"])
    test = splits["test"]
    scales = np.asarray([
        test["scales"]["wind_capacity_mw"], test["scales"]["solar_capacity_mw"],
        test["scales"]["load_denominator_mw"],
    ])
    expected_actual_norm = np.load(data_path / "test_actual.npy").transpose(0, 2, 1)
    expected_forecast_norm = np.load(data_path / "test_forecast.npy").transpose(0, 2, 1)
    global_ranges = np.ptp(np.asarray(test["hourly"]), axis=0)

    spec_data = {}
    all_event_types = MAIN_EVENT_TYPES + DIAGNOSTIC_EVENT_TYPES
    for spec_name, spec in SPECS.items():
        events = build_event_catalog(
            test, thresholds, low_level=spec["low"], high_level=spec["high"],
            max_gap_hours=spec["gap"],
        )
        mappings = map_windows_to_events(test, events)
        spec_data[spec_name] = {"events": events, "mappings": mappings, **spec}
        write_csv(output_dir / f"events_{spec_name}.csv", events)
        write_csv(output_dir / f"window_event_map_{spec_name}.csv", mappings)

    correspondence = {}
    loaded = {}
    for model, run_name in RUNS.items():
        audit, scenarios, actual = audit_saved_run(
            outputs / run_name, expected_actual_norm, expected_forecast_norm, scales
        )
        config_text = (outputs / run_name / "config_used.yaml").read_text(encoding="utf-8")
        declared = None
        in_evaluation = False
        for line in config_text.splitlines():
            if line.startswith("evaluation:"):
                in_evaluation = True
            elif in_evaluation and line and not line.startswith(" "):
                in_evaluation = False
            elif in_evaluation and line.strip().startswith("n_samples:"):
                declared = int(line.split(":", 1)[1].strip())
                break
        audit["config_declared_n_samples"] = declared
        audit["config_vs_actual_n_samples_match"] = declared == audit.get("n_scenarios_per_window")
        correspondence[model] = audit
        if audit["eligible"]:
            loaded[model] = (scenarios, actual)
    (output_dir / "correspondence_audit.json").write_text(
        json.dumps(correspondence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ordinary_rows = []
    daily_ids = [idx for idx, timestamp in enumerate(test["timestamps"][: test["windows"]]) if timestamp.hour == 0]
    rules = {
        "all_rolling_hourly": list(range(test["windows"])),
        "daily_00": daily_ids,
        "nonoverlap_168h_aux": list(range(0, test["windows"], 168)),
    }
    event_by_id_rows = []
    event_summary_rows = []
    coupling_rows = []
    for model, (scenarios, actual) in loaded.items():
        for rule, ids in rules.items():
            ordinary_rows.append({
                "model": model, "start_rule": rule,
                **aggregate_ordinary_metrics(scenarios, actual, ids, global_ranges),
            })
        for spec_name, spec in spec_data.items():
            event_rows, summary_rows = evaluate_event_mappings(
                scenarios, actual, spec["mappings"], spec["events"], thresholds,
                test["scales"], global_ranges, all_event_types, spec["low"], spec["high"],
            )
            event_by_id_rows.extend({"model": model, "spec": spec_name, **row} for row in event_rows)
            event_summary_rows.extend({"model": model, "spec": spec_name, **row} for row in summary_rows)
        coupling_rows.extend(
            {"model": model, **row}
            for row in evaluate_coupling(
                scenarios, actual, spec_data["main_p10_p90_gap1"]["mappings"], thresholds
            )
        )

    write_csv(output_dir / "ordinary_metrics.csv", ordinary_rows)
    write_csv(output_dir / "event_by_id_all_specs.csv", event_by_id_rows)
    write_csv(output_dir / "event_summary_all_specs.csv", event_summary_rows)
    write_csv(output_dir / "coupling_by_event.csv", coupling_rows)

    coupling_summary = []
    groups = defaultdict(list)
    for row in coupling_rows:
        groups[(row["model"], row["event_type"])].append(row)
    for (model, event_type), rows in sorted(groups.items()):
        summary = {"model": model, "event_type": event_type, "n_events": len(rows)}
        difference_keys = [key for key in rows[0] if key.startswith("difference_generated_minus_actual_")]
        for key in difference_keys:
            values = [row[key] for row in rows if np.isfinite(row[key])]
            summary[key] = float(np.mean(values)) if values else np.nan
        coupling_summary.append(summary)
    write_csv(output_dir / "coupling_summary.csv", coupling_summary)

    main_events = spec_data["main_p10_p90_gap1"]["events"]
    ramp_rows, alt_rows = ramp_diagnostics(test, thresholds, main_events)
    write_csv(output_dir / "high_ramp_chain_diagnostics.csv", ramp_rows)
    write_csv(output_dir / "high_ramp_endpoint_clusters_gap3.csv", alt_rows)

    # Transparent, separate rankings; no hidden has_event pooling.
    ordinary_all = {row["model"]: row for row in ordinary_rows if row["start_rule"] == "all_rolling_hourly"}
    rankings = []
    for key, label in (
        ("total_crps_mw", "CRPS"),
        ("multivariate_es_per_sqrt_dimension_mw", "ES/√维数"),
    ):
        order = ranks({model: row[key] for model, row in ordinary_all.items()})
        rankings.append({"scope": "全部滚动起点", "metric": label, "order": " > ".join(sorted(order, key=order.get))})
    coverage_order = ranks({model: abs(row["total_coverage_90"] - 90.0) for model, row in ordinary_all.items()})
    rankings.append({"scope": "全部滚动起点", "metric": "|90%覆盖率-90|", "order": " > ".join(sorted(coverage_order, key=coverage_order.get))})

    stable_main = [
        row for row in event_summary_rows
        if row["spec"] == "main_p10_p90_gap1"
        and row["event_type"] in MAIN_EVENT_TYPES and row["n_events"] >= 3
    ]
    for metric, label in (
        ("onset_total_crps_mw", "稳定事件单元平均CRPS"),
        ("onset_multivariate_es_per_sqrt_dimension_mw", "稳定事件单元平均ES/√维数"),
    ):
        values = {}
        for model in loaded:
            selected = [row[metric] for row in stable_main if row["model"] == model and metric in row and np.isfinite(row[metric])]
            values[model] = float(np.mean(selected))
        order = ranks(values)
        rankings.append({"scope": "主要事件（n_events≥3单元）", "metric": label, "order": " > ".join(sorted(order, key=order.get))})
    values = {}
    for model in loaded:
        selected = [abs(row["onset_total_coverage_90"] - 90.0) for row in stable_main if row["model"] == model and "onset_total_coverage_90" in row]
        values[model] = float(np.mean(selected))
    order = ranks(values)
    rankings.append({"scope": "主要事件（n_events≥3单元）", "metric": "平均|覆盖90-90|", "order": " > ".join(sorted(order, key=order.get))})

    old_crps = {}
    for model, run_name in RUNS.items():
        old = json.loads((outputs / run_name / "metrics.json").read_text(encoding="utf-8"))
        old_crps[model] = old["total_crps"]
    old_order = sorted(old_crps, key=old_crps.get)
    new_order = sorted(ordinary_all, key=lambda model: ordinary_all[model]["total_crps_mw"])
    conclusion_change = (
        f"旧全局结果按CRPS排序为 {' > '.join(old_order)}；按保存数组逐窗口重算后为 {' > '.join(new_order)}。"
        + ("排序未变化，但事件评价显示可靠性与尖锐度的取舍，不能只按CRPS给综合结论。" if old_order == new_order else "排序发生变化，应以本轮可复现口径为准。")
    )

    counts = Counter(row["event_type"] for row in main_events)
    ramp_summary = {}
    for event_type in ("high_ramp_6h", "high_ramp_12h"):
        selected = [row for row in ramp_rows if row["event_type"] == event_type]
        alternative = [row for row in alt_rows if row["event_type"] == event_type]
        ramp_summary[f"{event_type}_events"] = len(selected)
        ramp_summary[f"{event_type}_chained"] = sum(bool(row["chain_overlap_extended"]) for row in selected)
        ramp_summary[f"{event_type}_alternative_clusters"] = len(alternative)
        ramp_summary[f"{event_type}_current_mean_duration"] = float(np.mean([row["current_duration_hours"] for row in selected]))
        ramp_summary[f"{event_type}_current_max_duration"] = float(np.max([row["current_duration_hours"] for row in selected]))
        ramp_summary[f"{event_type}_alternative_mean_duration"] = float(np.mean([row["alternative_duration_hours"] for row in alternative]))
        ramp_summary[f"{event_type}_alternative_max_duration"] = float(np.max([row["alternative_duration_hours"] for row in alternative]))

    record = {
        "generated_at": datetime.now().isoformat(),
        "no_training_or_generation": True,
        "correspondence": correspondence,
        "ordinary_metrics": ordinary_rows,
        "main_event_summary": [row for row in event_summary_rows if row["spec"] == "main_p10_p90_gap1"],
        "main_test_event_counts": {event_type: counts.get(event_type, 0) for event_type in set(MAIN_EVENT_TYPES + DIAGNOSTIC_EVENT_TYPES + NO_TEST_EVENT_TYPES)},
        "coupling_summary": coupling_summary,
        "ramp_diagnostic_summary": ramp_summary,
        "rankings": rankings,
        "conclusion_change": conclusion_change,
    }
    (output_dir / "event_reevaluation_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "event_reevaluation_report.md").write_text(
        build_report(record), encoding="utf-8"
    )
    print(output_dir / "event_reevaluation_report.md")


if __name__ == "__main__":
    main()
