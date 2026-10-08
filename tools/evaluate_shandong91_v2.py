"""Strict-mask ordinary/joint evaluation for Shandong91 faithful24 V2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHANNELS = ("Wind", "Solar", "Load")


def crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    ordered = np.sort(samples, axis=1)
    members = ordered.shape[1]
    shape = (1, members) + (1,) * (ordered.ndim - 2)
    weights = (2 * np.arange(1, members + 1) - members - 1).reshape(shape)
    return np.mean(np.abs(ordered - truth[:, None]), axis=1) - np.sum(
        weights * ordered, axis=1
    ) / (members * members)


def masked_corr(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    valid = valid & np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    x = left[valid].astype(np.float64); y = right[valid].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator > 0 else float("nan")


def trajectory_acf(truth, scenarios, mask, lags=(1, 3, 6), max_members=100):
    """Compare like with like: one ACF per real/generated trajectory."""
    result = {}; member_count = min(scenarios.shape[1], max_members)
    for lag in lags:
        real_values = []
        generated_values = []
        for issue in range(truth.shape[0]):
            for node in range(truth.shape[2]):
                pair = mask[issue, lag:, node] & mask[issue, :-lag, node]
                value = masked_corr(
                    truth[issue, lag:, node], truth[issue, :-lag, node], pair
                )
                if np.isfinite(value): real_values.append(value)
                for member in range(member_count):
                    value = masked_corr(
                        scenarios[issue, member, lag:, node],
                        scenarios[issue, member, :-lag, node], pair,
                    )
                    if np.isfinite(value): generated_values.append(value)
        real = float(np.mean(real_values)) if real_values else float("nan")
        generated = float(np.mean(generated_values)) if generated_values else float("nan")
        result[str(lag)] = {
            "truth_trajectory_mean": real,
            "generated_member_trajectory_mean": generated,
            "absolute_error": abs(generated - real),
            "generated_members_used": member_count,
        }
    return result


def correlation_matrix(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    nodes = values.shape[-1]; output = np.full((nodes, nodes), np.nan)
    flattened = values.reshape(-1, nodes); valid = mask.reshape(-1, nodes)
    for left in range(nodes):
        output[left, left] = 1.0
        for right in range(left + 1, nodes):
            current = valid[:, left] & valid[:, right]
            value = masked_corr(flattened[:, left], flattened[:, right], current)
            output[left, right] = output[right, left] = value
    return output


def spatial_correlation_rmse(truth, scenarios, mask, max_members=40):
    real = correlation_matrix(truth, mask)
    matrices = []
    for member in range(min(scenarios.shape[1], max_members)):
        matrices.append(correlation_matrix(scenarios[:, member], mask))
    stacked = np.stack(matrices)
    finite_count = np.isfinite(stacked).sum(axis=0)
    generated = np.divide(
        np.nansum(stacked, axis=0), finite_count,
        out=np.full_like(real, np.nan), where=finite_count > 0,
    )
    off_diagonal = ~np.eye(real.shape[0], dtype=bool)
    valid = off_diagonal & np.isfinite(real) & np.isfinite(generated)
    return {
        "rmse": float(np.sqrt(np.mean((real[valid] - generated[valid]) ** 2))),
        "valid_node_pairs": int(valid.sum() // 2),
        "generated_members_used": len(matrices),
    }


def energy_score(system_scenarios, system_truth, max_members=80):
    members = min(system_scenarios.shape[1], max_members); scores = []
    for issue in range(system_scenarios.shape[0]):
        x = system_scenarios[issue, :members].reshape(members, -1).astype(np.float64)
        y = system_truth[issue].reshape(-1).astype(np.float64)
        attraction = np.linalg.norm(x - y, axis=1).mean()
        squares = np.sum(x * x, axis=1)
        pairwise = np.sqrt(np.maximum(squares[:, None] + squares[None] - 2 * x @ x.T, 0))
        scores.append(attraction - 0.5 * pairwise.mean())
    return float(np.mean(scores)), members


def paired_comparison_metrics(scenarios, truth, mask):
    """Core V1/V2 metrics under the exact same truth and effective mask."""
    result = {"channels": {}, "system_aggregate": {}}
    systems = []; system_truths = []
    for channel, name in enumerate(CHANNELS):
        samples = np.asarray(scenarios[..., channel]); target = np.asarray(truth[..., channel])
        current_mask = np.asarray(mask[..., channel])
        lower = np.quantile(samples, .05, axis=1); upper = np.quantile(samples, .95, axis=1)
        interval = upper-lower + 20*(np.maximum(lower-target, 0)+np.maximum(target-upper, 0))
        result["channels"][name] = {
            "crps_mw": float(crps(samples, target)[current_mask].mean()),
            "coverage90": float(((target >= lower) & (target <= upper))[current_mask].mean()),
            "width90_mw": float((upper-lower)[current_mask].mean()),
            "interval_score90_mw": float(interval[current_mask].mean()),
            "ensemble_conditional_std_mw": float(np.std(samples, axis=1)[current_mask].mean()),
            "temporal_acf": trajectory_acf(target, samples, current_mask),
            "spatial_correlation": spatial_correlation_rmse(target, samples, current_mask),
        }
        system_s = np.sum(np.where(current_mask[:, None], samples, 0), axis=3)
        system_y = np.sum(np.where(current_mask, target, 0), axis=2)
        active = current_mask.any(axis=2); lo = np.quantile(system_s,.05,axis=1); hi=np.quantile(system_s,.95,axis=1)
        result["system_aggregate"][name] = {
            "crps_mw": float(crps(system_s, system_y)[active].mean()),
            "coverage90": float(((system_y>=lo)&(system_y<=hi))[active].mean()),
        }
        systems.append(system_s); system_truths.append(system_y)
    result["joint_energy_score_mw"], result["energy_members_used"] = energy_score(
        np.stack(systems, axis=-1), np.stack(system_truths, axis=-1)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-result")
    args = parser.parse_args()
    source = Path(args.result); output = Path(args.output_dir)
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    scenarios = np.load(source / "actual_scenarios_mw.npy", mmap_mode="r")
    truth = np.load(source / "actual_mw.npy", mmap_mode="r")
    forecast = np.load(source / "forecast_mw.npy", mmap_mode="r")
    mask_path = source / "effective_mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError("V2 evaluation requires effective_mask.npy")
    mask = np.load(mask_path, mmap_mode="r").astype(bool)
    if scenarios.shape[0] != truth.shape[0] or scenarios.shape[2:] != truth.shape[1:]:
        raise ValueError("scenario/truth shape mismatch")
    metrics = {
        "evaluation_contract": "strict elementwise effective_mask",
        "acf_contract": "truth trajectory ACF compared with individual generated-member trajectory ACF",
        "shape": list(scenarios.shape),
        "finite_ratio": float(np.isfinite(scenarios).mean()),
        "channels": {}, "system_aggregate": {}, "joint": {}, "physical": {},
    }
    system_scenarios = []; system_truth = []
    for channel, name in enumerate(CHANNELS):
        samples = np.asarray(scenarios[..., channel]); target = np.asarray(truth[..., channel])
        current_mask = np.asarray(mask[..., channel]); prediction = np.asarray(forecast[..., channel])
        lower = np.quantile(samples, .05, axis=1); upper = np.quantile(samples, .95, axis=1)
        median = np.median(samples, axis=1); score = crps(samples, target)
        miss_low = np.maximum(lower - target, 0); miss_high = np.maximum(target - upper, 0)
        interval_score = upper - lower + 20 * (miss_low + miss_high)
        residual_truth = target - prediction; residual_samples = samples - prediction[:, None]
        node_coverage = []
        for node in range(target.shape[2]):
            node_mask = current_mask[..., node]
            if node_mask.any():
                node_coverage.append(float(((target[..., node] >= lower[..., node]) &
                                            (target[..., node] <= upper[..., node]))[node_mask].mean()))
        valid_samples = residual_samples[np.broadcast_to(current_mask[:, None], residual_samples.shape)]
        valid_truth = residual_truth[current_mask]
        metrics["channels"][name] = {
            "effective_elements": int(current_mask.sum()),
            "crps_mw": float(score[current_mask].mean()),
            "median_mae_mw": float(np.abs(median - target)[current_mask].mean()),
            "forecast_mae_mw": float(np.abs(prediction - target)[current_mask].mean()),
            "coverage90": float(((target >= lower) & (target <= upper))[current_mask].mean()),
            "width90_mw": float((upper - lower)[current_mask].mean()),
            "interval_score90_mw": float(interval_score[current_mask].mean()),
            "ensemble_conditional_std_mw": float(np.std(samples, axis=1)[current_mask].mean()),
            "residual_quantiles_mw": {
                key: {"truth": float(np.quantile(valid_truth, q)),
                      "ensemble": float(np.quantile(valid_samples, q))}
                for key, q in (("q01", .01), ("q05", .05), ("q50", .50), ("q95", .95), ("q99", .99))
            },
            "node_coverage90": {
                "count": len(node_coverage), "min": float(np.min(node_coverage)),
                "median": float(np.median(node_coverage)), "max": float(np.max(node_coverage)),
                "values": node_coverage,
            },
            "temporal_acf": trajectory_acf(target, samples, current_mask),
            "spatial_correlation": spatial_correlation_rmse(target, samples, current_mask),
        }
        ramp = {}
        for lag in (1, 3, 6):
            ramp_mask = current_mask[:, lag:] & current_mask[:, :-lag]
            sample_ramp = samples[:, :, lag:] - samples[:, :, :-lag]
            truth_ramp = target[:, lag:] - target[:, :-lag]
            ramp[str(lag)] = {"crps_mw": float(crps(sample_ramp, truth_ramp)[ramp_mask].mean())}
        metrics["channels"][name]["ramp"] = ramp
        system_s = np.sum(np.where(current_mask[:, None], samples, 0.0), axis=3)
        system_y = np.sum(np.where(current_mask, target, 0.0), axis=2)
        active_time = current_mask.any(axis=2)
        sys_low = np.quantile(system_s, .05, axis=1); sys_high = np.quantile(system_s, .95, axis=1)
        sys_interval = sys_high - sys_low + 20 * (np.maximum(sys_low-system_y, 0) + np.maximum(system_y-sys_high, 0))
        metrics["system_aggregate"][name] = {
            "coverage90": float(((system_y >= sys_low) & (system_y <= sys_high))[active_time].mean()),
            "width90_mw": float((sys_high - sys_low)[active_time].mean()),
            "interval_score90_mw": float(sys_interval[active_time].mean()),
            "crps_mw": float(crps(system_s, system_y)[active_time].mean()),
        }
        system_scenarios.append(system_s); system_truth.append(system_y)
        metrics["physical"][name] = {
            "generated_negative_rate": float((samples[current_mask[:, None].repeat(samples.shape[1], axis=1)] < 0).mean()),
            "truth_negative_rate": float((target[current_mask] < 0).mean()),
            "note": "read-only diagnostic; no clipping or label changes",
        }
    joint_s = np.stack(system_scenarios, axis=-1); joint_y = np.stack(system_truth, axis=-1)
    value, used = energy_score(joint_s, joint_y)
    metrics["joint"] = {"system_resource_energy_score_mw": value, "members_used": used}
    if args.baseline_result:
        baseline_path = Path(args.baseline_result)
        baseline_scenarios = np.load(baseline_path / "actual_scenarios_mw.npy", mmap_mode="r")
        if baseline_scenarios.shape[0] != truth.shape[0] or baseline_scenarios.shape[2:] != truth.shape[1:]:
            raise ValueError("baseline is not paired to the same validation windows/layout")
        baseline_metrics = paired_comparison_metrics(baseline_scenarios, truth, mask)
        delta = {name: {
            key: metrics["channels"][name][key] - baseline_metrics["channels"][name][key]
            for key in ("crps_mw", "coverage90", "width90_mw", "interval_score90_mw", "ensemble_conditional_std_mw")
        } for name in CHANNELS}
        metrics["baseline_comparison"] = {
            "baseline_result": args.baseline_result,
            "contract": "paired same validation truth and candidate effective_mask",
            "baseline": baseline_metrics, "candidate_minus_baseline": delta,
        }
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), "utf-8")
    lines = ["# Shandong91 faithful24 V2 evaluation", "",
             f"- Shape: `{metrics['shape']}`", f"- Finite ratio: `{metrics['finite_ratio']}`",
             "- Mask: strict elementwise `effective_mask`", "- Physical clipping: not applied", "",
             "| Resource | CRPS | Coverage90 | Width90 | Interval score90 | System coverage90 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in CHANNELS:
        item = metrics["channels"][name]; system = metrics["system_aggregate"][name]
        lines.append(f"| {name} | {item['crps_mw']:.6f} | {item['coverage90']:.4f} | {item['width90_mw']:.6f} | {item['interval_score90_mw']:.6f} | {system['coverage90']:.4f} |")
    lines += ["", f"Joint system-resource Energy Score: `{value:.6f}` MW.",
              "", "Solar negative values are reported, never silently clipped."]
    (output / "RESULT_SUMMARY.md").write_text("\n".join(lines), "utf-8")


if __name__ == "__main__":
    main()
