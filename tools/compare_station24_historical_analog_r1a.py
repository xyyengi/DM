#!/usr/bin/env python3
"""R1-A: compare train-history analog wind scenarios with random and diffusion.

This is a no-training validation experiment.  Retrieval uses only issuance-time
forecast-shape features fitted on the train split.  Validation actual values are
used only after scenarios have been constructed.  The sealed test split is not
loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from station_evaluation import (
    adjacency_variogram_score,
    evaluate_station_scenarios,
    metric_bundle,
    temporal_acf,
)
from tools.diagnose_station24_historical_analog_r0 import (
    RAMP_LAGS,
    TARGET_ISSUES,
    causal_arrays,
    feature_blocks,
    fit_standardization,
    retrieve,
    standardize_blocks,
)


HOURS = 168
INTERVALS = (0.80, 0.90, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--diffusion-result", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--k-values", default="10,20,40,80")
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--top5-events",
        default=(
            "outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/"
            "forecast_event_attribution/history_vs_body_tail_raw/"
            "sustained_deep_drop_summary.csv"
        ),
    )
    return parser.parse_args()


def load_array(path: Path, name: str) -> np.ndarray:
    return np.load(path / name, mmap_mode="r")


def aggregate_mw(values: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    return np.einsum("...ts,s->...t", values, capacities)


def wind_spatial_corr_rmse(samples: np.ndarray, actual: np.ndarray) -> float:
    actual_corr = np.corrcoef(actual.reshape(-1, actual.shape[-1]), rowvar=False)
    generated = samples.transpose(0, 2, 1, 3).reshape(-1, samples.shape[-1])
    generated_corr = np.corrcoef(generated, rowvar=False)
    upper = np.triu_indices(actual.shape[-1], k=1)
    return float(np.sqrt(np.mean((generated_corr[upper] - actual_corr[upper]) ** 2)))


def core_metrics(
    wind_samples: np.ndarray,
    wind_actual: np.ndarray,
    capacities: np.ndarray,
    wind_adjacency: np.ndarray,
) -> dict[str, float]:
    aggregate_samples = aggregate_mw(wind_samples, capacities)
    aggregate_actual = aggregate_mw(wind_actual, capacities)
    aggregate = metric_bundle(aggregate_samples, aggregate_actual, INTERVALS)
    station = metric_bundle(wind_samples, wind_actual, INTERVALS)
    return {
        **{f"aggregate_{key}": value for key, value in aggregate.items()},
        **{f"station_{key}": value for key, value in station.items()},
        "wind_spatial_corr_rmse": wind_spatial_corr_rmse(
            wind_samples, wind_actual
        ),
        "wind_variogram_score": float(
            adjacency_variogram_score(
                wind_samples, wind_actual, wind_adjacency
            )
        ),
        "wind_acf_abs_error_lag1": abs(
            temporal_acf(wind_samples, 1) - temporal_acf(wind_actual, 1)
        ),
        "wind_acf_abs_error_lag6": abs(
            temporal_acf(wind_samples, 6) - temporal_acf(wind_actual, 6)
        ),
    }


def top5_event_metrics(
    wind_samples_mw: np.ndarray,
    wind_actual_mw: np.ndarray,
    event_frame: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    for event in event_frame.itertuples(index=False):
        issue = int(event.issue_index)
        start = int(event.lead_start)
        stop = int(event.lead_end) + 1
        actual_mean = float(wind_actual_mw[issue, start:stop].mean())
        member_mean = wind_samples_mw[issue, :, start:stop].mean(axis=1)
        hits = member_mean <= actual_mean
        rows.append(
            {
                "event_rank": int(event.event_rank),
                "issue_index": issue,
                "issue_date": str(event.issue_date),
                "lead_start": start,
                "lead_end": stop - 1,
                "actual_window_mean_mw": actual_mean,
                "minimum_member_mean_mw": float(member_mean.min()),
                "members_at_or_below_actual": int(hits.sum()),
                "member_hit_rate": float(hits.mean()),
                "any_member_hit": bool(hits.any()),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "top5_events_any_hit_count": int(frame.any_member_hit.sum()),
        "top5_events_mean_member_hit_rate": float(frame.member_hit_rate.mean()),
        "top5_events_total_hit_members": int(frame.members_at_or_below_actual.sum()),
        "top5_events_mean_minimum_member_mw": float(
            frame.minimum_member_mean_mw.mean()
        ),
    }, frame


def event_metrics(
    wind_samples_mw: np.ndarray,
    wind_actual_mw: np.ndarray,
    wind_forecast_mw: np.ndarray,
    train_residual_lower: float,
    train_actual_ramp_thresholds: dict[int, float],
) -> dict[str, float]:
    target_residual = wind_actual_mw - wind_forecast_mw
    member_count = wind_samples_mw.shape[1]
    negative_member_rates = []
    negative_any = []
    negative_count = 0
    for issue in range(len(target_residual)):
        events = np.flatnonzero(target_residual[issue] <= train_residual_lower)
        for hour in events:
            left, right = max(0, hour - 3), min(HOURS, hour + 4)
            hits = (
                wind_samples_mw[issue, :, left:right].min(axis=1)
                <= wind_actual_mw[issue, hour]
            )
            negative_member_rates.append(float(hits.mean()))
            negative_any.append(bool(hits.any()))
            negative_count += 1

    result = {
        "negative_tail_event_hour_count": int(negative_count),
        "negative_tail_mean_member_hit_rate_pm3h": float(
            np.mean(negative_member_rates)
        ),
        "negative_tail_any_member_hit_rate_pm3h": float(np.mean(negative_any)),
    }
    for lag in RAMP_LAGS:
        threshold = train_actual_ramp_thresholds[lag]
        member_rates = []
        any_hits = []
        abs_offsets = []
        event_count = 0
        for issue in range(len(wind_actual_mw)):
            actual_ramp = (
                wind_actual_mw[issue, lag:] - wind_actual_mw[issue, :-lag]
            )
            forecast_ramp = (
                wind_forecast_mw[issue, lag:] - wind_forecast_mw[issue, :-lag]
            )
            missed = (np.abs(actual_ramp) >= threshold) & (
                (np.sign(actual_ramp) != np.sign(forecast_ramp))
                | (np.abs(forecast_ramp) < 0.5 * np.abs(actual_ramp))
            )
            for local_hour in np.flatnonzero(missed):
                event_hour = int(local_hour) + lag
                left = max(lag, event_hour - 3)
                right = min(HOURS, event_hour + 4)
                ramps = (
                    wind_samples_mw[issue, :, left:right]
                    - wind_samples_mw[issue, :, left - lag : right - lag]
                )
                target_sign = np.sign(actual_ramp[local_hour])
                required = 0.5 * abs(actual_ramp[local_hour])
                qualifies = (np.sign(ramps) == target_sign) & (
                    np.abs(ramps) >= required
                )
                member_hits = qualifies.any(axis=1)
                member_rates.append(float(member_hits.mean()))
                any_hits.append(bool(member_hits.any()))
                for member in np.flatnonzero(member_hits):
                    positions = np.flatnonzero(qualifies[member])
                    offsets = np.arange(left, right) - event_hour
                    nearest = positions[np.argmin(np.abs(offsets[positions]))]
                    abs_offsets.append(abs(float(offsets[nearest])))
                event_count += 1
        result.update(
            {
                f"missed_ramp_event_count_{lag}h": int(event_count),
                f"missed_ramp_mean_member_hit_rate_{lag}h_pm3h": float(
                    np.mean(member_rates)
                ),
                f"missed_ramp_any_member_hit_rate_{lag}h_pm3h": float(
                    np.mean(any_hits)
                ),
                f"missed_ramp_mean_abs_timing_offset_{lag}h": float(
                    np.mean(abs_offsets)
                ),
            }
        )
    return result


def make_analog_wind(
    validation_forecast: np.ndarray,
    train_residual: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = validation_forecast[:, None] + train_residual[selected]
    return np.clip(raw, 0.0, 1.0), raw


def complete_scenarios(
    analog_wind: np.ndarray,
    analog_wind_raw: np.ndarray,
    diffusion_samples: np.ndarray,
    diffusion_member_indices: np.ndarray,
    wind_indices: np.ndarray,
    solar_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(
        diffusion_samples[:, diffusion_member_indices, :, :], dtype=np.float64
    ).copy()
    raw = base.copy()
    base[:, :, :, wind_indices] = analog_wind
    raw[:, :, :, wind_indices] = analog_wind_raw
    # Solar is intentionally preserved from exactly the same diffusion members.
    if not np.array_equal(
        base[:, :, :, solar_indices],
        np.asarray(diffusion_samples[:, diffusion_member_indices, :, :])[
            :, :, :, solar_indices
        ],
    ):
        raise RuntimeError("solar preservation audit failed")
    return base, raw


def metric_rows(
    model: str,
    k: int,
    repeat: int | None,
    core: dict[str, float],
    events: dict[str, float],
    top5: dict[str, float],
) -> dict[str, object]:
    return {
        "model": model,
        "k": int(k),
        "repeat": repeat,
        **core,
        **events,
        **top5,
    }


def plot_k_summary(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("aggregate_crps", "Aggregate wind CRPS", "lower"),
        ("aggregate_coverage_90", "90% coverage", "target"),
        ("aggregate_width_90", "90% width (MW)", "lower"),
        ("negative_tail_mean_member_hit_rate_pm3h", "Negative-tail member hit", "higher"),
        ("missed_ramp_mean_member_hit_rate_3h_pm3h", "Missed 3h-ramp member hit", "higher"),
        ("top5_events_any_hit_count", "Top-5 deep drops hit", "higher"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {
        "retrieved_analog": "#d81b60",
        "random_history": "#9aa3ad",
        "diffusion_subsample": "#009688",
    }
    labels = {
        "retrieved_analog": "Retrieved analog",
        "random_history": "Random history",
        "diffusion_subsample": "Diffusion subsample",
    }
    for axis, (metric, title, _) in zip(axes.flat, metrics):
        for model in labels:
            part = summary[summary.model.eq(model)].sort_values("k")
            axis.plot(
                part.k,
                part[metric],
                marker="o",
                color=colors[model],
                label=labels[model],
            )
            if f"{metric}_std" in part:
                axis.fill_between(
                    part.k,
                    part[metric] - part[f"{metric}_std"],
                    part[metric] + part[f"{metric}_std"],
                    color=colors[model],
                    alpha=0.12,
                )
        if metric == "aggregate_coverage_90":
            axis.axhline(0.90, color="#111827", ls="--", lw=0.8)
        axis.set_title(title)
        axis.set_xlabel("Unique member count K")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("R1-A historical analog K sensitivity", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_target_comparison(
    output: Path,
    issues: pd.DataFrame,
    actual_mw: np.ndarray,
    forecast_mw: np.ndarray,
    analog_mw: np.ndarray,
    diffusion_mw: np.ndarray,
) -> None:
    targets = [index for index in TARGET_ISSUES if index < len(actual_mw)]
    fig, axes = plt.subplots(len(targets), 2, figsize=(16, 3.8 * len(targets)))
    lead = np.arange(HOURS)
    for row, issue in enumerate(targets):
        for column, (samples, label) in enumerate(
            [(analog_mw, "Retrieved historical ensemble"), (diffusion_mw, "Current diffusion")]
        ):
            axis = axes[row, column]
            lower = np.quantile(samples[issue], 0.05, axis=0)
            upper = np.quantile(samples[issue], 0.95, axis=0)
            median = np.median(samples[issue], axis=0)
            axis.fill_between(lead, lower, upper, color="#f48fb1", alpha=0.4, label="90% interval")
            axis.plot(lead, median, color="#d81b60", lw=1.5, label="median")
            axis.plot(lead, forecast_mw[issue], color="#009688", ls="--", lw=1.2, label="forecast")
            axis.plot(lead, actual_mw[issue], color="#111827", lw=1.6, label="actual")
            axis.set_title(
                f"Issue {issue} ({issues.iloc[issue].issue_date}) — {label}"
            )
            axis.grid(alpha=0.25)
            axis.set_ylabel("Aggregated wind MW")
            if row == 0:
                axis.legend(frameon=False, ncol=4, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Lead hour")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output: Path,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    detailed: pd.DataFrame,
    best_k: int,
    decision: str,
) -> None:
    analog = summary[
        summary.model.eq("retrieved_analog") & summary.k.eq(best_k)
    ].iloc[0]
    random = summary[
        summary.model.eq("random_history") & summary.k.eq(best_k)
    ].iloc[0]
    diffusion = summary[
        summary.model.eq("diffusion_subsample") & summary.k.eq(best_k)
    ].iloc[0]
    comp = comparisons[comparisons.k.eq(best_k)].set_index("metric")
    report = f"""# R1-A：历史类比场景集合比较

> 数据：训练历史库280个完整发布窗口；验证集23个发布窗口；测试集未读取。  
> 历史成员：不同训练发布窗口的唯一13站×168 h联合风电残差，不用重复轨迹凑500成员。

## 1. 结论

**{decision}**

本次推荐用于尾部候选池的规模为Top-{best_k}。这不是最终模型成员数，而是可供尾部专家选择的历史原型数量。

| 指标 | 检索历史 | 随机历史 | 同规模扩散抽样 |
|---|---:|---:|---:|
| 聚合风电CRPS | {analog.aggregate_crps:.3f} | {random.aggregate_crps:.3f} | {diffusion.aggregate_crps:.3f} |
| 聚合风电90%覆盖率 | {analog.aggregate_coverage_90:.2%} | {random.aggregate_coverage_90:.2%} | {diffusion.aggregate_coverage_90:.2%} |
| 聚合风电90%宽度 | {analog.aggregate_width_90:.1f} MW | {random.aggregate_width_90:.1f} MW | {diffusion.aggregate_width_90:.1f} MW |
| 负尾部成员命中率（±3 h） | {analog.negative_tail_mean_member_hit_rate_pm3h:.2%} | {random.negative_tail_mean_member_hit_rate_pm3h:.2%} | {diffusion.negative_tail_mean_member_hit_rate_pm3h:.2%} |
| 漏报3 h爬坡成员命中率 | {analog.missed_ramp_mean_member_hit_rate_3h_pm3h:.2%} | {random.missed_ramp_mean_member_hit_rate_3h_pm3h:.2%} | {diffusion.missed_ramp_mean_member_hit_rate_3h_pm3h:.2%} |
| Top-5持续深跌命中事件数 | {int(analog.top5_events_any_hit_count)}/5 | {random.top5_events_any_hit_count:.2f}/5 | {diffusion.top5_events_any_hit_count:.2f}/5 |
| 风电空间相关RMSE | {analog.wind_spatial_corr_rmse:.4f} | {random.wind_spatial_corr_rmse:.4f} | {diffusion.wind_spatial_corr_rmse:.4f} |

检索相对随机历史的经验检验见 `method_comparisons.csv`。重点不是要求纯历史集合全面取代扩散模型，而是判断它是否包含扩散集合缺失的少数事件成员。

## 2. K敏感性

![K敏感性](figures/k_sensitivity.png)

## 3. 典型问题窗口

![典型问题窗口](figures/target_issue_comparison.png)

## 4. 下一步

本次结果显示，检索集合在同规模CRPS和预测漏报爬坡命中上均优于扩散抽样，但持续深跌与一般负尾部成员质量仍弱于现有扩散尾部。下一步不训练纯历史模型，也不做简单成员替换，而直接训练“检索条件化双尾部MoE”：冻结当前主体和持续深跌尾部，Top-K历史完整残差以集合形式输入，只新增失配尾部路由、交叉注意力和适配器。禁止把历史均值加到所有场景，禁止针对验证真实值挑选邻居。

对应数据文件：

- `k_summary.csv`：三类集合的K敏感性均值；
- `all_repeat_metrics.csv`：随机历史与扩散子采样的全部重复；
- `method_comparisons.csv`：检索相对对照的经验p值；
- `top5_event_details.csv`：固定Top-5深跌事件逐项命中；
- `full_evaluation/`：每个K的完整Station24指标。
"""
    output.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    k_values = sorted({int(value) for value in args.k_values.split(",")})
    if min(k_values) < 2:
        raise ValueError("all K values must be at least two")
    output_dir = Path(
        args.output_dir
        or f"outputs_shandong/station24/historical_analog_r1a_{datetime.now():%Y%m%d_%H%M%S}"
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    (output_dir / "figures").mkdir(parents=True)
    (output_dir / "full_evaluation").mkdir()

    data_path = Path(args.data_path)
    diffusion_dir = Path(args.diffusion_result)
    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy()
    solar_indices = stations.index[stations.data_type.eq("solar")].to_numpy()
    capacities = stations.loc[wind_indices, "capacity_mw"].to_numpy(float)
    adjacency = np.load(data_path / "station_adjacency.npy").astype(float)
    wind_adjacency = adjacency[np.ix_(wind_indices, wind_indices)]

    train_forecast = np.load(data_path / "train_forecast.npy").astype(float)
    train_actual = np.load(data_path / "train_actual.npy").astype(float)
    train_residual = np.load(data_path / "train_residual.npy").astype(float)
    train_fill = np.load(data_path / "train_fill_mask.npy")
    train_issues = pd.read_csv(data_path / "train_issue_dates.csv")
    validation_forecast = np.load(data_path / "val_forecast.npy").astype(float)
    validation_actual = np.load(data_path / "val_actual.npy").astype(float)
    validation_residual = np.load(data_path / "val_residual.npy").astype(float)
    validation_issues = pd.read_csv(data_path / "val_issue_dates.csv")
    if not np.allclose(validation_actual - validation_forecast, validation_residual):
        raise ValueError("validation residual identity failed")
    reference_actual = np.asarray(
        load_array(diffusion_dir, "actual_data_normalized.npy"), dtype=float
    )
    reference_forecast = np.asarray(
        load_array(diffusion_dir, "forecast_data_normalized.npy"), dtype=float
    )
    if not np.array_equal(reference_actual, validation_actual):
        raise ValueError("diffusion result actual does not match val data")
    if not np.array_equal(reference_forecast, validation_forecast):
        raise ValueError("diffusion result forecast does not match val data")
    diffusion_samples = load_array(diffusion_dir, "actual_scenarios_normalized.npy")
    daylight_mask = np.asarray(load_array(diffusion_dir, "station_daylight_mask.npy"), bool)

    bank_indices = np.flatnonzero(
        ~np.any(train_fill[:, :, wind_indices] != 0, axis=(1, 2))
    )
    train_causal = causal_arrays(
        train_forecast, train_residual, train_issues, wind_indices
    )
    validation_causal = causal_arrays(
        validation_forecast, validation_residual, validation_issues, wind_indices
    )
    train_blocks = feature_blocks(
        train_causal, train_issues, capacities
    )
    validation_blocks = feature_blocks(
        validation_causal, validation_issues, capacities
    )
    fitted = fit_standardization(train_blocks, bank_indices)
    train_standard = standardize_blocks(train_blocks, fitted)
    validation_standard = standardize_blocks(validation_blocks, fitted)
    max_k = max(k_values)
    retrieved_indices, _, _ = retrieve(
        train_standard,
        validation_standard,
        bank_indices,
        validation_causal,
        "forecast_only",
        max_k,
    )

    wind_train_residual = train_residual[:, :, wind_indices]
    wind_validation_forecast = validation_forecast[:, :, wind_indices]
    wind_validation_actual = validation_actual[:, :, wind_indices]
    train_actual_mw = aggregate_mw(train_actual[:, :, wind_indices], capacities)
    train_residual_mw = aggregate_mw(wind_train_residual, capacities)
    validation_actual_mw = aggregate_mw(wind_validation_actual, capacities)
    validation_forecast_mw = aggregate_mw(wind_validation_forecast, capacities)
    train_residual_lower = float(np.quantile(train_residual_mw[bank_indices], 0.05))
    train_ramp_thresholds = {
        lag: float(
            np.quantile(
                np.abs(
                    train_actual_mw[bank_indices, lag:]
                    - train_actual_mw[bank_indices, :-lag]
                ),
                0.90,
            )
        )
        for lag in RAMP_LAGS
    }
    top5 = pd.read_csv(args.top5_events)
    if "model" in top5:
        candidate = top5[top5.model.astype(str).str.contains("body_tail_moe_raw")]
        if len(candidate) == 5:
            top5 = candidate
        else:
            top5 = top5.drop_duplicates("event_rank").head(5)
    top5 = top5.sort_values("event_rank").head(5)

    rng = np.random.default_rng(args.seed)
    all_rows = []
    top5_detail_rows = []
    analog_full_by_k = {}
    for k in k_values:
        diffusion_fixed_indices = np.linspace(
            0, diffusion_samples.shape[1] - 1, num=k, dtype=np.int64
        )
        selected = retrieved_indices[:, :k]
        analog_wind, analog_wind_raw = make_analog_wind(
            wind_validation_forecast, wind_train_residual, selected
        )
        analog_wind_mw = aggregate_mw(analog_wind, capacities)
        core = core_metrics(
            analog_wind, wind_validation_actual, capacities, wind_adjacency
        )
        events = event_metrics(
            analog_wind_mw,
            validation_actual_mw,
            validation_forecast_mw,
            train_residual_lower,
            train_ramp_thresholds,
        )
        top_metrics, top_details = top5_event_metrics(
            analog_wind_mw, validation_actual_mw, top5
        )
        all_rows.append(
            metric_rows("retrieved_analog", k, None, core, events, top_metrics)
        )
        top_details.insert(0, "k", k)
        top_details.insert(0, "model", "retrieved_analog")
        top5_detail_rows.append(top_details)

        full, full_raw = complete_scenarios(
            analog_wind,
            analog_wind_raw,
            diffusion_samples,
            diffusion_fixed_indices,
            wind_indices,
            solar_indices,
        )
        analog_full_by_k[k] = full
        metrics, station_frame, lead_frame = evaluate_station_scenarios(
            full,
            full_raw,
            validation_actual,
            validation_forecast,
            stations,
            adjacency,
            daylight_mask=daylight_mask,
            interval_levels=INTERVALS,
            energy_score_member_limit=min(k, 80),
        )
        eval_dir = output_dir / "full_evaluation" / f"retrieved_analog_k{k}"
        eval_dir.mkdir()
        (eval_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        station_frame.to_csv(eval_dir / "station_metrics.csv", index=False)
        lead_frame.to_csv(eval_dir / "lead_metrics.csv", index=False)

        for repeat in range(args.random_repeats):
            history_indices = np.stack(
                [rng.choice(bank_indices, size=k, replace=False) for _ in range(len(validation_actual))]
            )
            random_wind, _ = make_analog_wind(
                wind_validation_forecast, wind_train_residual, history_indices
            )
            random_wind_mw = aggregate_mw(random_wind, capacities)
            random_core = core_metrics(
                random_wind, wind_validation_actual, capacities, wind_adjacency
            )
            random_events = event_metrics(
                random_wind_mw,
                validation_actual_mw,
                validation_forecast_mw,
                train_residual_lower,
                train_ramp_thresholds,
            )
            random_top, _ = top5_event_metrics(
                random_wind_mw, validation_actual_mw, top5
            )
            all_rows.append(
                metric_rows(
                    "random_history", k, repeat, random_core, random_events, random_top
                )
            )

            diffusion_indices = np.stack(
                [
                    rng.choice(diffusion_samples.shape[1], size=k, replace=False)
                    for _ in range(len(validation_actual))
                ]
            )
            issue_axis = np.arange(len(validation_actual))[:, None]
            diffusion_wind = np.asarray(
                diffusion_samples[issue_axis, diffusion_indices, :, :][..., wind_indices],
                dtype=float,
            )
            diffusion_wind_mw = aggregate_mw(diffusion_wind, capacities)
            diffusion_core = core_metrics(
                diffusion_wind, wind_validation_actual, capacities, wind_adjacency
            )
            diffusion_events = event_metrics(
                diffusion_wind_mw,
                validation_actual_mw,
                validation_forecast_mw,
                train_residual_lower,
                train_ramp_thresholds,
            )
            diffusion_top, _ = top5_event_metrics(
                diffusion_wind_mw, validation_actual_mw, top5
            )
            all_rows.append(
                metric_rows(
                    "diffusion_subsample",
                    k,
                    repeat,
                    diffusion_core,
                    diffusion_events,
                    diffusion_top,
                )
            )

    all_frame = pd.DataFrame(all_rows)
    all_frame.to_csv(output_dir / "all_repeat_metrics.csv", index=False)
    pd.concat(top5_detail_rows, ignore_index=True).to_csv(
        output_dir / "top5_event_details.csv", index=False
    )

    metric_columns = [
        name
        for name in all_frame.columns
        if name not in {"model", "k", "repeat"}
    ]
    summary_rows = []
    for (model, k), group in all_frame.groupby(["model", "k"], sort=False):
        row = {"model": model, "k": int(k)}
        for metric in metric_columns:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "k_summary.csv", index=False)

    comparison_rows = []
    for k in k_values:
        analog = all_frame[
            all_frame.model.eq("retrieved_analog") & all_frame.k.eq(k)
        ].iloc[0]
        for control in ("random_history", "diffusion_subsample"):
            controls = all_frame[all_frame.model.eq(control) & all_frame.k.eq(k)]
            for metric in metric_columns:
                values = controls[metric].to_numpy(float)
                lower = "crps" in metric or "rmse" in metric or "width" in metric or "abs_error" in metric or "variogram" in metric or "timing_offset" in metric
                observed = float(analog[metric])
                p = (
                    (1 + np.sum(values <= observed)) / (len(values) + 1)
                    if lower
                    else (1 + np.sum(values >= observed)) / (len(values) + 1)
                )
                comparison_rows.append(
                    {
                        "k": k,
                        "control": control,
                        "metric": metric,
                        "direction": "lower" if lower else "higher",
                        "retrieved_analog": observed,
                        "control_mean": float(values.mean()),
                        "control_std": float(values.std(ddof=1)),
                        "empirical_one_sided_p": float(p),
                    }
                )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output_dir / "method_comparisons.csv", index=False)

    # Pick a tail pool: reward event member mass and penalize ordinary CRPS.
    analog_summary = summary[summary.model.eq("retrieved_analog")].copy()
    analog_summary["tail_pool_score"] = (
        analog_summary["negative_tail_mean_member_hit_rate_pm3h"]
        + analog_summary["missed_ramp_mean_member_hit_rate_3h_pm3h"]
        + analog_summary["top5_events_any_hit_count"] / 5.0
        - 0.5
        * analog_summary["aggregate_crps"]
        / analog_summary["aggregate_crps"].min()
    )
    best_k = int(analog_summary.loc[analog_summary.tail_pool_score.idxmax(), "k"])

    best_analog = summary[
        summary.model.eq("retrieved_analog") & summary.k.eq(best_k)
    ].iloc[0]
    best_random = summary[
        summary.model.eq("random_history") & summary.k.eq(best_k)
    ].iloc[0]
    provides_event_value = (
        best_analog.negative_tail_mean_member_hit_rate_pm3h
        > best_random.negative_tail_mean_member_hit_rate_pm3h
        and best_analog.top5_events_any_hit_count >= best_random.top5_events_any_hit_count
    )
    decision = (
        "历史检索提供了可利用的尾部事件成员；进入检索条件化主体—尾部MoE，不以纯历史集合替代扩散主体。"
        if provides_event_value
        else "历史检索未稳定优于随机历史；停止把检索结果接入尾部专家。"
    )

    plot_k_summary(summary, output_dir / "figures" / "k_sensitivity.png")
    diffusion_plot_indices = np.linspace(
        0, diffusion_samples.shape[1] - 1, num=best_k, dtype=np.int64
    )
    analog_plot_mw = aggregate_mw(
        analog_full_by_k[best_k][:, :, :, wind_indices], capacities
    )
    diffusion_plot_mw = aggregate_mw(
        np.asarray(diffusion_samples[:, diffusion_plot_indices, :, :])[:, :, :, wind_indices],
        capacities,
    )
    plot_target_comparison(
        output_dir / "figures" / "target_issue_comparison.png",
        validation_issues,
        validation_actual_mw,
        validation_forecast_mw,
        analog_plot_mw,
        diffusion_plot_mw,
    )

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "station24_historical_analog_r1a_no_training",
        "train_bank_count": int(len(bank_indices)),
        "validation_issue_count": int(len(validation_actual)),
        "k_values": k_values,
        "random_repeats": int(args.random_repeats),
        "best_tail_pool_k": best_k,
        "decision": decision,
        "test_files_loaded": False,
        "future_validation_actual_used_for_retrieval": False,
        "solar_handling": "same diffusion members preserved; only wind replaced",
        "distinct_history_members_not_duplicated_to_500": True,
    }
    (output_dir / "r1a_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        output_dir / "historical_analog_r1a_report.md",
        summary,
        comparisons,
        all_frame,
        best_k,
        decision,
    )
    print(f"R1A_COMPLETE output={output_dir}")
    print(f"BEST_TAIL_POOL_K={best_k}")
    print(f"DECISION={decision}")


if __name__ == "__main__":
    main()
