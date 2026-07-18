#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an evidence-based audit of event extraction, mapping, and training use."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import event_thresholds as et


def source_line(obj) -> int:
    return inspect.getsourcelines(obj)[1]


def find_line(path: Path, pattern: str) -> int:
    regex = re.compile(pattern)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if regex.search(line):
            return number
    raise ValueError(f"Pattern {pattern!r} not found in {path}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:,.{digits}f}"


def threshold_rows(thresholds: dict) -> list[dict]:
    rows = [
        {"event_type": "low_wind", "indicator": "wind/capacity", "level": "P10", "threshold": thresholds["low_wind_capacity_ratio"]["p10"], "unit": "capacity_ratio", "formula": "wind_MW / wind_capacity_MW"},
        {"event_type": "low_wind", "indicator": "wind/capacity", "level": "P05", "threshold": thresholds["low_wind_capacity_ratio"]["p05"], "unit": "capacity_ratio", "formula": "wind_MW / wind_capacity_MW"},
        {"event_type": "low_solar_daily_energy", "indicator": "daily daylight solar energy ratio", "level": "P10", "threshold": thresholds["low_solar_daily_daylight_energy_ratio"]["p10"], "unit": "daylight_capacity_factor", "formula": "sum(solar_MW, 07:00..18:00) / (solar_capacity_MW * 12h)"},
        {"event_type": "low_solar_daily_energy", "indicator": "daily daylight solar energy ratio", "level": "P05", "threshold": thresholds["low_solar_daily_daylight_energy_ratio"]["p05"], "unit": "daylight_capacity_factor", "formula": "sum(solar_MW, 07:00..18:00) / (solar_capacity_MW * 12h)"},
        {"event_type": "low_renewable", "indicator": "(wind+solar)/total capacity", "level": "P10", "threshold": thresholds["low_renewable_capacity_ratio"]["p10"], "unit": "capacity_ratio", "formula": "(wind_MW + solar_MW) / (wind_capacity_MW + solar_capacity_MW)"},
        {"event_type": "low_renewable", "indicator": "(wind+solar)/total capacity", "level": "P05", "threshold": thresholds["low_renewable_capacity_ratio"]["p05"], "unit": "capacity_ratio", "formula": "(wind_MW + solar_MW) / (wind_capacity_MW + solar_capacity_MW)"},
        {"event_type": "high_load", "indicator": "load", "level": "P90", "threshold": thresholds["high_load_mw"]["p90"], "unit": "MW", "formula": "load_MW"},
        {"event_type": "high_load", "indicator": "load", "level": "P95", "threshold": thresholds["high_load_mw"]["p95"], "unit": "MW", "formula": "load_MW"},
        {"event_type": "high_net_load", "indicator": "net load", "level": "P90", "threshold": thresholds["high_net_load_mw"]["p90"], "unit": "MW", "formula": "load_MW - wind_MW - solar_MW"},
        {"event_type": "high_net_load", "indicator": "net load", "level": "P95", "threshold": thresholds["high_net_load_mw"]["p95"], "unit": "MW", "formula": "load_MW - wind_MW - solar_MW"},
        {"event_type": "compound_low_renewable_high_net_load", "indicator": "low renewable component", "level": "P10/P90", "threshold": thresholds["low_renewable_capacity_ratio"]["p10"], "unit": "capacity_ratio", "formula": "renewable_ratio <= train P10 AND net_load_MW >= train P90"},
        {"event_type": "compound_low_renewable_high_net_load", "indicator": "high net-load component", "level": "P10/P90", "threshold": thresholds["high_net_load_mw"]["p90"], "unit": "MW", "formula": "renewable_ratio <= train P10 AND net_load_MW >= train P90"},
    ]
    for horizon in (1, 6, 12):
        for level in ("p90", "p95"):
            rows.append({"event_type": f"high_ramp_{horizon}h", "indicator": "absolute net-load change", "level": level.upper(), "threshold": thresholds["absolute_net_load_ramp_mw"][f"{horizon}h"][level], "unit": "MW", "formula": f"abs(net_load[t] - net_load[t-{horizon}])"})
    for source in ("wind", "solar"):
        for level in ("p90", "p95"):
            rows.append({"event_type": f"{source}_drop_load_rise_1h", "indicator": f"positive {source} drop", "level": level.upper(), "threshold": thresholds[f"{source}_drop_1h_mw_positive"][level], "unit": "MW", "formula": f"max({source}[t-1] - {source}[t], 0), percentile fitted among positive drops"})
            rows.append({"event_type": f"{source}_drop_load_rise_1h", "indicator": "positive load rise", "level": level.upper(), "threshold": thresholds["load_rise_1h_mw_positive"][level], "unit": "MW", "formula": "max(load[t] - load[t-1], 0), percentile fitted among positive rises"})
    return rows


def pick_mapping_samples(rows: list[dict], count: int = 12) -> list[dict]:
    selected: list[dict] = []
    seen_types: set[str] = set()
    # First include distinct event types, preferring a pre-onset mapping.
    for row in rows:
        if row["event_type"] not in seen_types and row["lead_hours"] >= 0:
            selected.append(row)
            seen_types.add(row["event_type"])
            if len(selected) >= count:
                return selected
    # Then demonstrate negative leads/post-onset mappings.
    for row in rows:
        if row["lead_hours"] < 0:
            selected.append(row)
            if len(selected) >= count:
                return selected
    return selected[:count]


EVENT_TYPES = (
    "low_wind",
    "low_solar_daily_energy",
    "low_renewable",
    "high_load",
    "high_net_load",
    "compound_low_renewable_high_net_load",
    "high_ramp_1h",
    "high_ramp_6h",
    "high_ramp_12h",
    "wind_drop_load_rise_1h",
    "solar_drop_load_rise_1h",
)


def build_report(record: dict, output_dir: Path) -> str:
    thresholds = record["thresholds_train_only"]
    lines = [
        "# 山东数据极端事件管线核查记录",
        "",
        "> 本记录依据当前实际 Python 代码和本地 NPY 生成，不依据设计文档猜测。阈值只由 train 唯一小时序列拟合。正式模型事件评价仍未执行。",
        "",
        "## 一、唯一小时序列",
        "",
        "当前没有单独的原始小时表可供评价程序直接读取；程序从各 split 的 `*_actual.npy` 168h、步长1的重叠窗口反向还原。还原方式是保留第一个窗口168小时，之后每个窗口只追加最后1小时。代码先逐元素检查相邻窗口的167小时重叠区，只有 `np.allclose(atol=1e-6, rtol=1e-6)` 才允许还原。不是对重复 actual 求平均，也不是任取后不检查。",
        "",
        f"代码位置：`src/eval/event_thresholds.py:{record['code_lines']['reconstruct']}`（还原与一致性检查），`src/eval/event_thresholds.py:{record['code_lines']['load_split']}`（按split加载）。",
        "",
        "| split | 168h窗口数 | 唯一小时数 | 起止时间 | 缺失小时 | 出现重复的不同timestamp数 | 冗余timestamp出现次数 | actual重叠最大误差(归一化) |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        x = record["timeline_audit"][split]
        lines.append(f"| {split} | {x['windows']} | {x['unique_hours']} | {x['start_local']} → {x['end_local']} | {x['missing_hours']} | {x['repeated_distinct_timestamps']} | {x['redundant_timestamp_occurrences']} | {x['overlap_max_abs_error_normalized']:.3g} |")
    lines += [
        "",
        "三个 split 分别加载、分别还原，train 结束于10月31日23时，validation 从11月1日0时开始，test 从12月1日0时开始；没有把相邻 split 拼起来再切分，因此事件不会跨越划分边界。",
        "",
        "注意：NPY本身不保存timestamp，时间轴来自 `export_metadata.json` 的起点并按1小时递增。因此表中“缺失小时=0”表示导出后的窗口与元数据构成完整小时轴；如果要证明导出前15分钟原始表也无缺行，还需要另行审计原始表。",
        "",
        "## 二、阈值拟合",
        "",
        f"入口位于 `src/eval/event_thresholds.py:{record['code_lines']['fit_thresholds']}`。只把 train 的7296个唯一小时传入拟合函数；validation/test只调用同一份阈值进行事件识别。没有在7129×168个高度重复的训练窗口元素上计算分位数。",
        "",
        "主审查级别采用低尾P10、高尾P90；同时保留P05/P95供敏感性判断。光伏已改成每日07:00–18:00的12小时能量容量因子；源荷反向已拆成风电下降+负荷上升、光伏下降+负荷上升，两者不要求同时发生。",
        "",
        "| 事件 | 指标/公式 | 级别 | 阈值 | 单位 |",
        "|---|---|---|---:|---|",
    ]
    for row in record["threshold_table"]:
        lines.append(f"| {row['event_type']} | `{row['formula']}` | {row['level']} | {fmt(row['threshold'])} | {row['unit']} |")
    lines += [
        "",
        "## 三、独立事件识别",
        "",
        f"事件目录代码位于 `src/eval/event_thresholds.py:{record['code_lines']['build_events']}`。流程是先在唯一小时/每日能量序列上识别超阈点，再合并为事件；从不先把168h窗口整体标成极端。",
        "",
        "- `low_wind`、`low_renewable`、`high_load`、`high_net_load`：主规则至少持续6小时，允许中间最多1小时短暂不超阈。",
        "- `compound_low_renewable_high_net_load`：低新能源和高净负荷同时成立至少6小时，缺口规则相同。",
        "- `low_solar_daily_energy`：一个完整自然日的07:00–18:00作为一个12小时事件。",
        "- `high_ramp_1h/6h/12h`：每个超阈变化对应 `[t-horizon,t]`，仅合并互相重叠的区间，额外gap为0。",
        "- `wind_drop_load_rise_1h` 与 `solar_drop_load_rise_1h`：分别判断源侧下降和负荷上升；每个变化覆盖前后两个小时。",
        "",
        "`max_gap_hours`当前主值是1，不是6。设置1是为了容忍单小时测量波动，又避免把相隔半天的过程合成一次事件。下表同时给出0/1/3/6的持续型事件数量敏感性：",
        "",
        "| split | gap | low_wind | low_renewable | high_load | high_net_load | compound |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in record["gap_sensitivity"]:
        lines.append(f"| {row['split']} | {row['max_gap_hours']} | {row.get('low_wind',0)} | {row.get('low_renewable',0)} | {row.get('high_load',0)} | {row.get('high_net_load',0)} | {row.get('compound_low_renewable_high_net_load',0)} |")
    lines += [
        "",
        "事件字段：`start_time/end_time`取合并区间首末小时；`duration_hours=end_idx-start_idx+1`；高事件的`peak_value`取区间最大值，低事件取最小值；光伏取当日日间能量容量因子；源荷反向取两项相对各自阈值倍数的较小者。每条事件有split+类型+序号组成的唯一`event_id`。",
        "",
        "主规则（P10/P90、持续型gap=1）的独立事件数量：",
        "",
        "| split | event_type | 独立事件数 |",
        "|---|---|---:|",
    ]
    for split in ("train", "val", "test"):
        for event_type, count in sorted(record["event_counts"][split].items()):
            lines.append(f"| {split} | {event_type} | {count} |")
    lines += [
        "",
        "完整清单：`events_train.csv`、`events_val.csv`、`events_test.csv`。这些仍是待人工确认的候选事件，不代表已经采用为论文最终定义。",
        "",
        "## 四、事件与168小时窗口映射",
        "",
        f"审计前的训练/评价主链路没有事件—窗口映射。本次新增的映射代码位于 `src/eval/event_thresholds.py:{record['code_lines']['map_windows']}`，所列字段现在均已实际计算。采用规则1：**只要窗口与事件至少有1小时交集，就保留映射**。因此它是多对多：一个事件关联全部相交窗口，一个窗口也可以保存多个事件；没有强制一对一。",
        "",
        "- `lead_hours = event_start - window_start`，不裁剪。负数照常保留，并令`post_onset=true`，表示窗口开始时事件已经发生。",
        "- `event_start_index/event_end_index`是事件相对窗口起点的小时索引，可以小于0或大于167。",
        "- `overlap_hours`只统计窗口和事件实际重叠的小时数。",
        "- `contains_event_start`与`fully_contains_event`分别记录包含起点和完整包含事件。",
        "",
        "真实映射样例（来自本地数据，不是伪代码）：",
        "",
        "| window_id | event_id | type | window_start | event_start | lead | start_idx | end_idx | overlap | contains_start | full | post_onset |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in record["mapping_samples"]:
        lines.append(f"| {row['window_id']} | {row['event_id']} | {row['event_type']} | {row['window_start']} | {row['event_start']} | {row['lead_hours']} | {row['event_start_index']} | {row['event_end_index']} | {row['overlap_hours']} | {row['contains_event_start']} | {row['fully_contains_event']} | {row['post_onset']} |")
    lines += [
        "",
        "完整映射：`window_event_map_train.csv`、`window_event_map_val.csv`、`window_event_map_test.csv`。",
        "",
        "| split | 映射对数 | 被映射的不同窗口数 | 被映射的不同事件数 |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        x = record["mapping_counts"][split]
        lines.append(f"| {split} | {x['pairs']} | {x['unique_windows']} | {x['unique_events']} |")
    lines += [
        "",
        "## 五、事件是否参与当前训练",
        "",
        "结论是 **A：事件标签目前仅用于新增的统计/评价准备，完全没有参与已经完成的V3/Vmix训练**。既不是事件采样，也没有作为扩散条件。依据实际执行链路：",
        "",
        f"1. `Dataset.__getitem__`（`dataset_multivariate.py:{record['code_lines']['dataset_getitem']}`）只返回：`input_14ch`、`actual_3ch`、`residual_3ch`、`forecast_3ch`、`time_encoding`、`cond_matrix`、`timepoints`。不返回event_id/type/lead。",
        f"2. `DataLoader`（`dataset_multivariate.py:{record['code_lines']['dataloader']}`）训练集仅设置`shuffle=True`，没有自定义Sampler或BatchSampler。",
        "3. `event_id`不影响窗口抽样概率；不存在按event_id均衡采样；不限制同一事件每epoch抽取窗口数。",
        f"4. 模型条件组装（`diff_models_multivariate.py:{record['code_lines']['model_input']}`）只有去噪状态、forecast和（Vmix时）8维日历编码。event_type、lead_hours、event_position均未进入condition tensor。",
        f"5. 扩散训练损失（`diff_models_multivariate.py:{record['code_lines']['loss']}`）是普通`F.mse_loss(predicted_noise, noise)`，没有极端窗口权重。",
        f"6. 训练循环（`train.py:{record['code_lines']['train_loop']}`）逐batch直接`loss=model(batch)`，没有事件字段分支。",
        "",
        "所以当前训练仍是普通168h窗口随机打乱采样。新增事件目录和映射只是在训练完成后生成的审查产物，不会追溯改变现有模型。",
        "",
        "## 六、仍需你确认的选择",
        "",
        "1. 持续型事件主缺口是否保持1小时，还是采用3/6小时；建议先看敏感性表，不直接跳到6小时。",
        "2. 主阈值采用P10/P90还是P05/P95。P05/P95更极端，但验证/测试中可能出现0事件。",
        "3. 光伏日间固定为07:00–18:00是否符合山东全年日照定义；后续也可改成逐月日照窗口或太阳高度角。",
        "4. 窗口评价时是否使用全部相交映射，还是只筛`contains_event_start=true`且`lead_hours>=0`的事前预测窗口。建议事件目录保留全部映射，模型评价阶段再筛，避免信息丢失。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="diffusion_npy_normalized")
    parser.add_argument("--output_dir", default="outputs_shandong/event_evaluation/event_pipeline_audit")
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data_path
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {name: et.load_hourly_split(data_path, name) for name in ("train", "val", "test")}
    thresholds = et.fit_event_thresholds(splits["train"])

    events_by_split = {
        name: et.build_event_catalog(split, thresholds, low_level="p10", high_level="p90", max_gap_hours=1)
        for name, split in splits.items()
    }
    maps_by_split = {
        name: et.map_windows_to_events(splits[name], events)
        for name, events in events_by_split.items()
    }
    for name in ("train", "val", "test"):
        write_csv(output_dir / f"events_{name}.csv", events_by_split[name])
        write_csv(output_dir / f"window_event_map_{name}.csv", maps_by_split[name])

    gaps = []
    persistent = {
        "low_wind", "low_renewable", "high_load", "high_net_load",
        "compound_low_renewable_high_net_load",
    }
    for name, split in splits.items():
        for gap in (0, 1, 3, 6):
            catalog = et.build_event_catalog(split, thresholds, max_gap_hours=gap)
            counts = Counter(row["event_type"] for row in catalog if row["event_type"] in persistent)
            gaps.append({"split": name, "max_gap_hours": gap, **dict(counts)})

    dataset_path = PROJECT_ROOT / "dataset_multivariate.py"
    model_path = PROJECT_ROOT / "diff_models_multivariate.py"
    train_path = PROJECT_ROOT / "train.py"
    record = {
        "audit_scope": "actual executing code; event definitions are post-training review candidates",
        "thresholds_train_only": thresholds,
        "threshold_table": threshold_rows(thresholds),
        "timeline_audit": {
            name: {
                "windows": split["windows"],
                "unique_hours": split["hours"],
                "start_local": split["start_local"],
                "end_local": split["end_local"],
                "missing_hours": split["missing_hours"],
                "repeated_distinct_timestamps": split["repeated_distinct_timestamps"],
                "redundant_timestamp_occurrences": split["redundant_timestamp_occurrences"],
                "overlap_max_abs_error_normalized": split["overlap_max_abs_error_normalized"],
            }
            for name, split in splits.items()
        },
        "event_counts": {
            name: {
                event_type: Counter(row["event_type"] for row in events).get(event_type, 0)
                for event_type in EVENT_TYPES
            }
            for name, events in events_by_split.items()
        },
        "gap_sensitivity": gaps,
        "mapping_counts": {
            name: {
                "pairs": len(rows),
                "unique_windows": len({row["window_id"] for row in rows}),
                "unique_events": len({row["event_id"] for row in rows}),
            }
            for name, rows in maps_by_split.items()
        },
        "mapping_samples": pick_mapping_samples(maps_by_split["test"], 12),
        "code_lines": {
            "reconstruct": source_line(et.reconstruct_sliding_windows),
            "load_split": source_line(et.load_hourly_split),
            "fit_thresholds": source_line(et.fit_event_thresholds),
            "build_events": source_line(et.build_event_catalog),
            "map_windows": source_line(et.map_windows_to_events),
            "dataset_getitem": find_line(dataset_path, r"^\s+def __getitem__"),
            "dataloader": find_line(dataset_path, r"^\s+loader = DataLoader"),
            "model_input": find_line(model_path, r"^\s+def build_model_input"),
            "loss": find_line(model_path, r"^\s+loss = F\.mse_loss"),
            "train_loop": find_line(train_path, r"^\s+for batch in train_loader"),
        },
    }
    (output_dir / "event_pipeline_audit.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "event_pipeline_audit.md").write_text(
        build_report(record, output_dir), encoding="utf-8"
    )
    print(output_dir / "event_pipeline_audit.md")


if __name__ == "__main__":
    main()
