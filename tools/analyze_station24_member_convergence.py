"""Diagnose finite-ensemble effects from one large Station24 validation run.

The generated 500-member array is treated as one fixed Monte Carlo pool.  Nested
prefixes provide a reproducible convergence curve, while repeated subsets quantify
member-selection variability without retraining or touching the sealed test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INTERVAL_LEVELS = (0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--member-sizes",
        type=int,
        nargs="+",
        default=[20, 40, 80, 160, 300, 500],
    )
    parser.add_argument("--resamples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--crps-chunk-points", type=int, default=8192)
    return parser.parse_args()


def _flatten_observations(
    samples: np.ndarray,
    actual: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # [N,K,...] -> [observation,K]
    member_last = np.moveaxis(samples, 1, -1)
    if valid_mask is not None:
        return member_last[valid_mask], actual[valid_mask]
    return member_last.reshape(-1, samples.shape[1]), actual.reshape(-1)


def _chunked_crps(
    observations_by_member: np.ndarray,
    actual: np.ndarray,
    chunk_points: int,
) -> float:
    members = observations_by_member.shape[1]
    coefficients = 2 * np.arange(1, members + 1) - members - 1
    total = 0.0
    count = 0
    for start in range(0, actual.size, chunk_points):
        stop = min(start + chunk_points, actual.size)
        chunk = np.asarray(
            observations_by_member[start:stop], dtype=np.float64
        )
        truth = np.asarray(actual[start:stop], dtype=np.float64)
        term1 = np.mean(np.abs(chunk - truth[:, None]), axis=1)
        sorted_chunk = np.sort(chunk, axis=1)
        half_pair = np.sum(sorted_chunk * coefficients[None, :], axis=1) / (
            members**2
        )
        total += float(np.sum(term1 - half_pair))
        count += truth.size
    return total / max(count, 1)


def _metrics(
    samples: np.ndarray,
    actual: np.ndarray,
    valid_mask: np.ndarray | None,
    include_crps: bool,
    chunk_points: int,
) -> dict[str, float]:
    member_values, actual_values = _flatten_observations(
        samples, actual, valid_mask
    )
    result: dict[str, float] = {}
    for nominal in INTERVAL_LEVELS:
        alpha = (1.0 - nominal) / 2.0
        lower = np.quantile(member_values, alpha, axis=1)
        upper = np.quantile(member_values, 1.0 - alpha, axis=1)
        label = int(round(100 * nominal))
        result[f"coverage_{label}"] = float(
            np.mean((actual_values >= lower) & (actual_values <= upper))
        )
        result[f"width_{label}"] = float(np.mean(upper - lower))
        result[f"below_{label}"] = float(np.mean(actual_values < lower))
        result[f"above_{label}"] = float(np.mean(actual_values > upper))
    if include_crps:
        result["crps"] = _chunked_crps(
            member_values, actual_values, chunk_points
        )
    return result


def _targets(
    samples: np.ndarray,
    actual: np.ndarray,
    daylight: np.ndarray,
    station_frame: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, str]]:
    station_types = station_frame.data_type.to_numpy()
    capacities = station_frame.capacity_mw.to_numpy(dtype=np.float32)
    wind = np.flatnonzero(station_types == "wind")
    solar = np.flatnonzero(station_types == "solar")
    renewable_mw = np.sum(
        samples * capacities[None, None, None, :], axis=-1
    )
    renewable_actual_mw = np.sum(actual * capacities[None, None, :], axis=-1)
    wind_mw = np.sum(
        samples[..., wind] * capacities[None, None, None, wind], axis=-1
    )
    wind_actual_mw = np.sum(
        actual[..., wind] * capacities[None, None, wind], axis=-1
    )
    return {
        "wind_station_pu": (
            samples[..., wind],
            actual[..., wind],
            None,
            "p.u.",
        ),
        "solar_daylight_station_pu": (
            samples[..., solar],
            actual[..., solar],
            daylight[..., solar],
            "p.u.",
        ),
        "wind_aggregate_mw": (
            wind_mw,
            wind_actual_mw,
            None,
            "MW",
        ),
        "renewable_aggregate_mw": (
            renewable_mw,
            renewable_actual_mw,
            None,
            "MW",
        ),
    }


def _write_figure(prefix_frame: pd.DataFrame, output_path: Path) -> None:
    targets = list(prefix_frame.target.unique())
    colors = {
        "wind_station_pu": "#2878b5",
        "solar_daylight_station_pu": "#e76f51",
        "wind_aggregate_mw": "#6a4c93",
        "renewable_aggregate_mw": "#2a9d8f",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for target in targets:
        part = prefix_frame.loc[prefix_frame.target.eq(target)].sort_values(
            "member_count"
        )
        axes[0].plot(
            part.member_count,
            100 * part.coverage_90,
            marker="o",
            label=target,
            color=colors.get(target),
        )
        axes[1].plot(
            part.member_count,
            part.width_90,
            marker="o",
            label=target,
            color=colors.get(target),
        )
    axes[0].axhline(90.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Empirical 90% coverage", xlabel="Members", ylabel="Coverage (%)")
    axes[1].set(title="Empirical 90% interval width", xlabel="Members", ylabel="Width")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir or result_dir / "member_convergence")
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = json.loads(
        (result_dir / "generation_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("split") != "val" or metadata.get("test_used") is not False:
        raise SystemExit("member convergence must use an unsealed validation result")
    samples = np.load(result_dir / "actual_scenarios_normalized.npy", mmap_mode="r")
    actual = np.load(result_dir / "actual_data_normalized.npy", mmap_mode="r")
    daylight = np.load(result_dir / "station_daylight_mask.npy", mmap_mode="r")
    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    maximum = int(samples.shape[1])
    sizes = sorted(set(args.member_sizes))
    if any(size <= 1 or size > maximum for size in sizes):
        raise ValueError(f"member sizes must be in [2,{maximum}], got {sizes}")
    if sizes[-1] != maximum:
        raise ValueError(
            f"largest requested member size must equal generated pool {maximum}"
        )
    if args.resamples <= 0:
        raise ValueError("resamples must be positive")

    prefix_rows: list[dict[str, object]] = []
    for size in sizes:
        indices = np.arange(size)
        selected = np.asarray(samples[:, indices, ...])
        for target, (target_samples, target_actual, mask, unit) in _targets(
            selected, actual, daylight, stations
        ).items():
            values = _metrics(
                target_samples,
                target_actual,
                mask,
                include_crps=True,
                chunk_points=args.crps_chunk_points,
            )
            prefix_rows.append(
                {
                    "selection": "nested_prefix",
                    "member_count": size,
                    "target": target,
                    "width_unit": unit,
                    **values,
                }
            )
        print(f"PREFIX_COMPLETE members={size}")
    prefix_frame = pd.DataFrame(prefix_rows)
    prefix_frame.to_csv(output_dir / "member_convergence_prefix.csv", index=False)

    rng = np.random.default_rng(args.seed)
    resample_rows: list[dict[str, object]] = []
    for size in sizes:
        repetitions = 1 if size == maximum else args.resamples
        for repetition in range(repetitions):
            indices = (
                np.arange(maximum)
                if size == maximum
                else np.sort(rng.choice(maximum, size=size, replace=False))
            )
            selected = np.asarray(samples[:, indices, ...])
            for target, (target_samples, target_actual, mask, unit) in _targets(
                selected, actual, daylight, stations
            ).items():
                values = _metrics(
                    target_samples,
                    target_actual,
                    mask,
                    include_crps=False,
                    chunk_points=args.crps_chunk_points,
                )
                resample_rows.append(
                    {
                        "member_count": size,
                        "repetition": repetition,
                        "target": target,
                        "width_unit": unit,
                        **values,
                    }
                )
        print(f"RESAMPLE_COMPLETE members={size} repetitions={repetitions}")
    resample_frame = pd.DataFrame(resample_rows)
    resample_frame.to_csv(output_dir / "member_convergence_resamples.csv", index=False)

    numeric = [
        column
        for column in resample_frame.columns
        if column not in {"member_count", "repetition", "target", "width_unit"}
    ]
    summary = (
        resample_frame.groupby(["member_count", "target"], as_index=False)[numeric]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "member_convergence_resample_summary.csv", index=False)
    _write_figure(prefix_frame, output_dir / "member_convergence.png")

    reference_size = maximum
    original_size = 80 if 80 in sizes else sizes[0]
    findings: dict[str, object] = {}
    lines = [
        "# Station24 validation member-convergence diagnosis",
        "",
        f"- Source: `{result_dir}`",
        f"- Split: validation only; test used: `{metadata.get('test_used')}`",
        f"- Generated pool: {maximum} members; generation seed: {metadata.get('generation_seed')}",
        f"- Original comparison size: {original_size}; reference size: {reference_size}",
        "- All curves reuse one trained checkpoint. Differences therefore isolate ensemble-size Monte Carlo effects, not retraining effects.",
        "",
        "## 90% interval evidence",
        "",
        "| Target | Coverage at original N | Coverage at max N | Gain | Width at original N | Width at max N | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for target in prefix_frame.target.unique():
        part = prefix_frame.loc[prefix_frame.target.eq(target)].set_index(
            "member_count"
        )
        old = part.loc[original_size]
        new = part.loc[reference_size]
        gain = float(new.coverage_90 - old.coverage_90)
        nominal_gap = max(0.90 - float(old.coverage_90), 0.0)
        recovered = gain / nominal_gap if nominal_gap > 1e-12 else 0.0
        if gain >= 0.02 and recovered >= 0.5:
            interpretation = "strong evidence that finite ensemble size explains a substantial share of undercoverage"
        elif gain >= 0.01:
            interpretation = "finite ensemble size contributes, but does not fully explain undercoverage"
        else:
            interpretation = "little finite-size gain; model calibration remains the main limitation"
        findings[target] = {
            "coverage_90_original": float(old.coverage_90),
            "coverage_90_max": float(new.coverage_90),
            "coverage_gain": gain,
            "nominal_gap_recovered_fraction": recovered,
            "width_90_original": float(old.width_90),
            "width_90_max": float(new.width_90),
            "interpretation": interpretation,
        }
        lines.append(
            f"| {target} | {100*old.coverage_90:.2f}% | {100*new.coverage_90:.2f}% | "
            f"{100*gain:+.2f} pp | {old.width_90:.4f} | {new.width_90:.4f} | {interpretation} |"
        )
    lines.extend(
        [
            "",
            "The maximum-member result is a Monte Carlo reference, not ground truth. If coverage remains well below 90% at the maximum N, increasing members alone cannot repair the learned distribution.",
            "",
        ]
    )
    (output_dir / "member_convergence_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    audit = {
        "source_result": str(result_dir),
        "split": metadata.get("split"),
        "test_used": metadata.get("test_used"),
        "generation_seed": metadata.get("generation_seed"),
        "available_member_count": maximum,
        "member_sizes": sizes,
        "resamples": args.resamples,
        "resample_seed": args.seed,
        "interval_levels": list(INTERVAL_LEVELS),
        "findings": findings,
    }
    (output_dir / "member_convergence_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"MEMBER_CONVERGENCE_COMPLETE output_dir={output_dir}")


if __name__ == "__main__":
    main()
