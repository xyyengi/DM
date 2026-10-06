"""Offline Station-24 ramp distribution and residual-scaling diagnosis.

This tool is evaluation-only.  It never changes a checkpoint and never trains a
model.  It compares the canonical 500-member Raw, Full Independent V2 and
Lightweight Joint Tail results against validation actuals, using the unchanged
continuous-event catalogue fitted on train.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

from station_dataset import (
    build_station_daylight_mask,
    get_station_dataloader,
)
from station_jstd_targets import build_station_jstd_target_arrays


LAGS = (1, 3, 6)
QUANTILES = (0.90, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--full-result", required=True)
    parser.add_argument("--lightweight-result", required=True)
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--raw-label", default="Raw")
    parser.add_argument("--full-label", default="Full Independent V2")
    parser.add_argument("--lightweight-label", default="Lightweight Joint Tail")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--event-context-hours", type=int, default=6)
    return parser.parse_args()


def array_header(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        shape, _, dtype = np.lib.format._read_array_header(handle, version)
    return tuple(int(value) for value in shape), np.dtype(dtype)


def require_result(path: Path, issues: int, stations: int) -> None:
    required = (
        "actual_scenarios_normalized.npy",
        "actual_data_normalized.npy",
        "forecast_data_normalized.npy",
        "generation_metadata.json",
        "metrics.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} misses {missing}")
    shape, dtype = array_header(path / "actual_scenarios_normalized.npy")
    expected = (issues, 500, 168, stations)
    if shape != expected or dtype != np.dtype("float32"):
        raise ValueError(
            f"{path} scenario header is {shape}/{dtype}; expected {expected}/float32"
        )
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    actual_bytes = (path / "actual_scenarios_normalized.npy").stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"{path} scenario file is truncated: {actual_bytes} < {expected_bytes}"
        )


def aggregate_scenarios(
    values: np.ndarray, indices: np.ndarray, capacity: np.ndarray
) -> np.ndarray:
    output = np.empty(values.shape[:3], dtype=np.float64)
    weights = capacity[indices]
    for issue in range(values.shape[0]):
        selected = np.take(np.asarray(values[issue]), indices, axis=-1)
        output[issue] = np.einsum(
            "kts,s->kt", selected, weights
        )
    return output


def aggregate_actual(
    values: np.ndarray, indices: np.ndarray, capacity: np.ndarray
) -> np.ndarray:
    selected = np.take(np.asarray(values), indices, axis=-1)
    return np.einsum("nts,s->nt", selected, capacity[indices])


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros_like(mask)
    for shift in range(-radius, radius + 1):
        source_start = max(0, -shift)
        source_stop = min(mask.shape[1], mask.shape[1] - shift)
        target_start = source_start + shift
        target_stop = source_stop + shift
        result[:, target_start:target_stop] |= mask[:, source_start:source_stop]
    return result


def interval_overlap(mask: np.ndarray, lag: int) -> np.ndarray:
    """Whether the closed interval [t-lag,t] intersects a True hour."""

    cumulative = np.pad(np.cumsum(mask.astype(np.int32), axis=1), ((0, 0), (1, 0)))
    end = np.arange(lag, mask.shape[1]) + 1
    start = np.arange(0, mask.shape[1] - lag)
    return (cumulative[:, end] - cumulative[:, start]) > 0


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    absolute = np.abs(values)
    positive = values[values > 0]
    negative = -values[values < 0]
    result: dict[str, float | int] = {
        "count": int(values.size),
        "signed_mean": float(values.mean()),
        "signed_std": float(values.std()),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
    }
    for prefix, selected in (
        ("abs", absolute),
        ("positive", positive),
        ("negative_abs", negative),
    ):
        for quantile in QUANTILES:
            key = f"{prefix}_q{int(100 * quantile)}"
            result[key] = float(np.quantile(selected, quantile)) if selected.size else math.nan
        result[f"{prefix}_max"] = float(selected.max()) if selected.size else math.nan
    return result


def selected_values(delta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if delta.ndim == 2:
        return delta[mask]
    if delta.ndim == 3:
        return delta[np.broadcast_to(mask[:, None, :], delta.shape)]
    raise ValueError(f"unsupported ramp tensor shape={delta.shape}")


def event_masks(
    data_path: Path,
    thresholds: dict[str, object],
    context_hours: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    targets = build_station_jstd_target_arrays(data_path, "val", thresholds)
    issues = targets.time_support.shape[0]
    exact = {
        source: np.zeros((issues, 168), dtype=bool) for source in ("wind", "solar")
    }
    for row in targets.catalog:
        source = str(row["source"])
        sample = int(row["sample_index"])
        start = int(row["lead_onset"])
        stop = int(row["lead_stop_exclusive"])
        exact[source][sample, start:stop] = True
    context = {source: dilate(mask, context_hours) for source, mask in exact.items()}
    audit = {
        "catalog_event_count": int(len(targets.catalog)),
        "event_count_by_source": {
            source: int(sum(str(row["source"]) == source for row in targets.catalog))
            for source in ("wind", "solar")
        },
        "event_context_hours": int(context_hours),
        "event_window_definition": "source-specific continuous event support dilated +/-6h",
    }
    return context, audit


def ramp_rows(
    label: str,
    series: np.ndarray,
    source: str,
    masks: np.ndarray,
    solar_daylight: np.ndarray | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lag in LAGS:
        delta = series[..., lag:] - series[..., :-lag]
        event_pair = interval_overlap(masks, lag)
        groups: list[tuple[str, np.ndarray]] = [
            ("event_pm6", event_pair),
            ("non_event_pm6", ~event_pair),
            ("all", np.ones_like(event_pair, dtype=bool)),
        ]
        if source == "solar" and solar_daylight is not None:
            daylight_pair = solar_daylight[:, lag:] & solar_daylight[:, :-lag]
            groups.extend(
                [
                    ("event_pm6_daylight", event_pair & daylight_pair),
                    ("non_event_pm6_daylight", (~event_pair) & daylight_pair),
                ]
            )
        for group, mask in groups:
            row = {
                "series": label,
                "source": source,
                "lag_h": lag,
                "window": group,
            }
            row.update(distribution(selected_values(delta, mask)))
            rows.append(row)
    return rows


def split_distribution_rows(
    data_path: Path,
    stations: pd.DataFrame,
    capacity: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train", "val"):
        actual = np.load(data_path / f"{split}_actual.npy", mmap_mode="r")
        valid = np.asarray(np.load(data_path / f"{split}_fill_mask.npy", mmap_mode="r")) == 0
        daylight, _ = build_station_daylight_mask(data_path, split)
        for source in ("wind", "solar"):
            indices = stations.index[stations.data_type.eq(source)].to_numpy(int)
            aggregate = aggregate_actual(actual, indices, capacity)
            valid_hour = valid[..., indices].all(axis=-1)
            if source == "solar":
                valid_hour &= daylight[..., indices].mean(axis=-1) > 0.5
            for lag in LAGS:
                pair = valid_hour[:, lag:] & valid_hour[:, :-lag]
                delta = aggregate[:, lag:] - aggregate[:, :-lag]
                row = {"split": split, "source": source, "lag_h": lag}
                row.update(distribution(delta[pair]))
                rows.append(row)
    return rows


def load_scale_tensor(
    data_path: Path, run_dir: Path
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    config = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        run_dir / "checkpoints" / "model_best.pt", map_location="cpu", weights_only=False
    )
    scale_spec = checkpoint["residual_scale"]
    loader, _ = get_station_dataloader(
        data_path,
        "val",
        scale_spec,
        batch_size=1,
        seed=424242,
        num_workers=0,
        condition_config=config["model"],
        state_thresholds=checkpoint.get("state_thresholds"),
    )
    tensors = [batch["residual_scale"].numpy() for batch in loader]
    return np.concatenate(tensors, axis=0), scale_spec, config


def scale_audit_rows(
    data_path: Path,
    stations: pd.DataFrame,
    event_context: dict[str, np.ndarray],
    scale_tensor: np.ndarray,
    scale_spec: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    residual = np.asarray(np.load(data_path / "val_residual.npy", mmap_mode="r"), dtype=np.float64)
    valid = np.asarray(np.load(data_path / "val_fill_mask.npy", mmap_mode="r")) == 0
    daylight, _ = build_station_daylight_mask(data_path, "val")
    residual = residual.transpose(0, 2, 1)
    valid = valid.transpose(0, 2, 1)
    daylight = daylight.transpose(0, 2, 1)
    base = np.asarray(scale_spec["scale"], dtype=np.float64)[None, :, None]
    base_standardized = residual / base
    conditional_standardized = residual / np.asarray(scale_tensor, dtype=np.float64)
    station_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for source in ("wind", "solar"):
        indices = stations.index[stations.data_type.eq(source)].to_numpy(int)
        for lag in LAGS:
            physical = residual[:, indices, lag:] - residual[:, indices, :-lag]
            base_delta = (
                base_standardized[:, indices, lag:]
                - base_standardized[:, indices, :-lag]
            )
            conditional_delta = (
                conditional_standardized[:, indices, lag:]
                - conditional_standardized[:, indices, :-lag]
            )
            pair_valid = valid[:, indices, lag:] & valid[:, indices, :-lag]
            if source == "solar":
                pair_valid &= daylight[:, indices, lag:] & daylight[:, indices, :-lag]
            event_pair = interval_overlap(event_context[source], lag)
            for group, time_mask in (
                ("event_pm6", event_pair),
                ("non_event_pm6", ~event_pair),
                ("all", np.ones_like(event_pair, dtype=bool)),
            ):
                selected = pair_valid & time_mask[:, None, :]
                p = np.abs(physical[selected])
                b = np.abs(base_delta[selected])
                c = np.abs(conditional_delta[selected])
                if not p.size:
                    continue
                threshold_p = np.quantile(p, 0.99)
                physical_top = p >= threshold_p
                threshold_c90 = np.quantile(c, 0.90)
                summary_rows.append(
                    {
                        "source": source,
                        "lag_h": lag,
                        "window": group,
                        "physical_abs_q99_pu": float(threshold_p),
                        "base_standardized_abs_q99": float(np.quantile(b, 0.99)),
                        "conditional_standardized_abs_q99": float(np.quantile(c, 0.99)),
                        "conditional_to_base_q99_ratio": float(
                            np.quantile(c, 0.99) / max(np.quantile(b, 0.99), 1e-12)
                        ),
                        "physical_top1_retained_in_standardized_top10": float(
                            np.mean(c[physical_top] >= threshold_c90)
                        ),
                        "observation_count": int(p.size),
                    }
                )
            for local, station_index in enumerate(indices):
                selected = pair_valid[:, local]
                p = np.abs(physical[:, local][selected])
                b = np.abs(base_delta[:, local][selected])
                c = np.abs(conditional_delta[:, local][selected])
                if p.size:
                    station_rows.append(
                        {
                            "station_index": int(station_index),
                            "station_id": str(stations.iloc[station_index].station_id),
                            "source": source,
                            "lag_h": lag,
                            "physical_abs_q99_pu": float(np.quantile(p, 0.99)),
                            "base_standardized_abs_q99": float(np.quantile(b, 0.99)),
                            "conditional_standardized_abs_q99": float(np.quantile(c, 0.99)),
                            "conditional_to_base_q99_ratio": float(
                                np.quantile(c, 0.99) / max(np.quantile(b, 0.99), 1e-12)
                            ),
                        }
                    )
    return summary_rows, station_rows


def static_loss_audit(config: dict[str, object]) -> dict[str, object]:
    model = config["model"]
    lags = [int(value) for value in model["ramp_auxiliary_lags"]]
    weights = [float(value) for value in model["ramp_auxiliary_lag_weights"]]
    fraction = float(model["event_balanced_ramp_top_fraction"])
    outer = float(model["event_balanced_ramp_loss_weight"])
    return {
        "implementation": "StationGaussianDiffusion.training_loss",
        "target_scale": (
            "normalized physical actual power: forecast + standardized x0 * "
            "causal residual_scale; not standardized residual and not MW"
        ),
        "top_selection_axis": "per batch sample x station, across time independently for each lag",
        "top_fraction": fraction,
        "top_count_by_lag": {
            str(lag): int(math.ceil((168 - lag) * fraction)) for lag in lags
        },
        "wind_solar_separate_weights": False,
        "wind_solar_shared_loss": True,
        "solar_night_removed_from_ramp_loss": False,
        "lags": lags,
        "lag_weights": weights,
        "outer_ramp_weight": outer,
        "nominal_total_loss_coefficients": {
            str(lag): outer * weight / sum(weights)
            for lag, weight in zip(lags, weights)
        },
        "point_loss": "smooth_l1 beta=0.05",
        "event_context_hours": int(model["event_balanced_context_hours"]),
        "event_focus": (
            "maximum(top-10% binary mask, continuous event support dilated +/-6h), "
            "then multiplied by diffusion-SNR weight"
        ),
        "event_support": (
            "max(station support, 0.25 * system time support), so a system event "
            "also weakly supervises every station"
        ),
        "sampling_event_fraction": float(model["independent_tail_event_sampling_fraction"]),
        "actual_per_lag_gradient_contribution": "computed by the CUDA dynamics audit",
    }


def write_report(
    output: Path,
    event_audit: dict[str, object],
    loss_audit: dict[str, object],
) -> None:
    text = [
        "# Station-24 short-ramp distribution and scaling audit",
        "",
        "This is an evaluation-only diagnostic. No checkpoint was changed and no training was run.",
        "",
        "## Event-window definition",
        "",
        f"- Continuous catalogue events: {event_audit['catalog_event_count']}",
        f"- Counts by source: {event_audit['event_count_by_source']}",
        f"- Classification: {event_audit['event_window_definition']}",
        "",
        "## Ramp-loss implementation",
        "",
        f"- Target scale: {loss_audit['target_scale']}",
        f"- Top selection: {loss_audit['top_selection_axis']}",
        f"- Top counts: {loss_audit['top_count_by_lag']}",
        f"- Shared wind/solar loss: {loss_audit['wind_solar_shared_loss']}",
        f"- Solar night removed: {loss_audit['solar_night_removed_from_ramp_loss']}",
        f"- Nominal coefficients: {loss_audit['nominal_total_loss_coefficients']}",
        f"- Event focus: {loss_audit['event_focus']}",
        "",
        "Numerical conclusions are finalized only after the CUDA gradient and denoising-trajectory audit is merged.",
    ]
    (output / "DISTRIBUTION_AND_SCALING_REPORT.md").write_text(
        "\n".join(text) + "\n", encoding="utf-8"
    )


def plot_dispersion(
    frame: pd.DataFrame, output: Path, model_labels: tuple[str, str, str]
) -> None:
    actual = frame[frame.series.eq("Actual")][
        ["source", "lag_h", "window", "signed_std", "abs_q99"]
    ].rename(columns={"signed_std": "actual_std", "abs_q99": "actual_q99"})
    merged = frame[~frame.series.eq("Actual")].merge(
        actual, on=["source", "lag_h", "window"], how="left", validate="many_to_one"
    )
    merged["std_ratio"] = merged.signed_std / merged.actual_std.clip(lower=1e-12)
    merged["q99_ratio"] = merged.abs_q99 / merged.actual_q99.clip(lower=1e-12)
    merged.to_csv(output / "ramp_dispersion_ratios.csv", index=False)
    selected = merged[merged.window.isin(["event_pm6", "non_event_pm6"])]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for row, source in enumerate(("wind", "solar")):
        for col, metric in enumerate(("std_ratio", "q99_ratio")):
            ax = axes[row, col]
            subset = selected[selected.source.eq(source)]
            width = 0.12
            labels = []
            x = np.arange(6)
            for index, series in enumerate(model_labels):
                values = []
                for window in ("event_pm6", "non_event_pm6"):
                    for lag in LAGS:
                        item = subset[
                            subset.series.eq(series)
                            & subset.window.eq(window)
                            & subset.lag_h.eq(lag)
                        ]
                        values.append(float(item.iloc[0][metric]))
                        if index == 0:
                            labels.append(f"{window.replace('_pm6','')}\n{lag}h")
                ax.bar(x + (index - 1) * width, values, width=width, label=series)
            ax.axhline(1.0, color="black", linewidth=1)
            ax.set_xticks(x, labels, rotation=20)
            ax.set_title(f"{source}: {metric}")
            ax.set_ylabel("model / actual")
            ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "ramp_dispersion_ratios.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    stations = (
        pd.read_csv(data_path / "station_order.csv")
        .sort_values("channel_index")
        .reset_index(drop=True)
    )
    capacity = stations.capacity_mw.to_numpy(float)
    actual = np.load(data_path / "val_actual.npy", mmap_mode="r")
    issues, _, station_count = actual.shape
    paths = {
        args.raw_label: Path(args.raw_result),
        args.full_label: Path(args.full_result),
        args.lightweight_label: Path(args.lightweight_result),
    }
    for path in paths.values():
        require_result(path, issues, station_count)

    full_run = Path(args.full_run)
    target_payload = json.loads(
        (full_run / "jstd_event_targets.json").read_text(encoding="utf-8")
    )
    context, event_audit = event_masks(
        data_path, target_payload["thresholds"], int(args.event_context_hours)
    )
    daylight, daylight_audit = build_station_daylight_mask(data_path, "val")
    rows: list[dict[str, object]] = []
    for source in ("wind", "solar"):
        indices = stations.index[stations.data_type.eq(source)].to_numpy(int)
        actual_aggregate = aggregate_actual(actual, indices, capacity)
        solar_daylight = None
        if source == "solar":
            solar_daylight = daylight[..., indices].mean(axis=-1) > 0.5
        rows.extend(
            ramp_rows("Actual", actual_aggregate, source, context[source], solar_daylight)
        )
        for label, path in paths.items():
            scenarios = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")
            aggregate = aggregate_scenarios(scenarios, indices, capacity)
            rows.extend(
                ramp_rows(label, aggregate, source, context[source], solar_daylight)
            )
            del scenarios, aggregate
    ramp_frame = pd.DataFrame(rows)
    ramp_frame.to_csv(output / "ramp_distribution_event_vs_nonevent.csv", index=False)
    plot_dispersion(
        ramp_frame,
        output,
        (args.raw_label, args.full_label, args.lightweight_label),
    )
    pd.DataFrame(split_distribution_rows(data_path, stations, capacity)).to_csv(
        output / "train_val_actual_ramp_distribution.csv", index=False
    )

    scale_tensor, scale_spec, config = load_scale_tensor(data_path, full_run)
    scale_summary, scale_station = scale_audit_rows(
        data_path, stations, context, scale_tensor, scale_spec
    )
    pd.DataFrame(scale_summary).to_csv(output / "residual_scaling_ramp_summary.csv", index=False)
    pd.DataFrame(scale_station).to_csv(output / "residual_scaling_station_detail.csv", index=False)
    loss_audit = static_loss_audit(config)
    audit = {
        "status": "PASS",
        "formal_training": "NOT RUN",
        "models_modified": False,
        "canonical_generation_seed": 424242,
        "members_per_model": 500,
        "event_audit": event_audit,
        "daylight_audit": daylight_audit,
        "residual_scaling_method": scale_spec["method"],
        "loss_audit": loss_audit,
    }
    (output / "distribution_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output, event_audit, loss_audit)
    print(f"RAMP_DISTRIBUTION_DIAGNOSTIC_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
