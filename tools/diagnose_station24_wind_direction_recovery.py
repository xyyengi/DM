"""Read-only Station-24 wind direction and drop-recovery diagnostic.

This diagnostic never constructs an optimizer, never updates a parameter, and
never writes a checkpoint. Validation actuals are used only for evaluation and
event anchoring; they are not passed to generation. Existing generated members
are reused without sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from generate_station24 import move_batch
from station_dataset import get_station_dataloader, load_station_static_data
from station_graph_prior import load_generation_graphs
from station_jstd_targets import build_station_jstd_target_arrays
from src.models.station_conditioned_diffusion import Station24DiffusionModel
from tools.diagnose_station24_ramp_distribution import (
    aggregate_actual,
    aggregate_scenarios,
    dilate,
    interval_overlap,
)


LAGS = (1, 3, 6)
QUANTILES = (0.90, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--lightweight-result", required=True)
    parser.add_argument("--ramp-selection-result", required=True)
    parser.add_argument("--ramp-selection-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lightweight-label", default="Lightweight original")
    parser.add_argument("--candidate-label", default="Lightweight ramp selection")
    parser.add_argument("--gradient-batches", type=int, default=3)
    parser.add_argument("--gradient-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--event-context-hours", type=int, default=6)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def direction_summary(values: np.ndarray, direction: str) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    selected = values[values > 0] if direction == "positive" else -values[values < 0]
    row: dict[str, float | int] = {"count": int(selected.size)}
    if not selected.size:
        return row
    row["std"] = float(selected.std())
    for quantile in QUANTILES:
        row[f"q{int(100 * quantile)}"] = float(np.quantile(selected, quantile))
    row["max"] = float(selected.max())
    return row


def split_direction_rows(
    data_path: Path, wind_indices: np.ndarray, capacities: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train", "val"):
        actual = np.load(data_path / f"{split}_actual.npy", mmap_mode="r")
        valid = np.asarray(np.load(data_path / f"{split}_fill_mask.npy", mmap_mode="r")) == 0
        series = aggregate_actual(actual, wind_indices, capacities)
        hour_valid = valid[..., wind_indices].all(axis=-1)
        for lag in LAGS:
            delta = series[:, lag:] - series[:, :-lag]
            pair = hour_valid[:, lag:] & hour_valid[:, :-lag]
            for direction in ("positive", "negative"):
                row = {"split": split, "scope": "aggregate_wind_mw", "direction": direction, "lag_h": lag}
                row.update(direction_summary(delta[pair], direction))
                rows.append(row)
    return rows


def event_context(
    data_path: Path, run: Path, radius: int
) -> tuple[np.ndarray, dict[str, object]]:
    payload = json.loads((run / "jstd_event_targets.json").read_text(encoding="utf-8"))
    targets = build_station_jstd_target_arrays(data_path, "val", payload["thresholds"])
    exact = np.zeros((targets.time_support.shape[0], 168), dtype=bool)
    count = 0
    for row in targets.catalog:
        if str(row["source"]) != "wind":
            continue
        exact[int(row["sample_index"]), int(row["lead_onset"]):int(row["lead_stop_exclusive"])] = True
        count += 1
    return dilate(exact, radius), {
        "wind_catalog_event_count": count,
        "context_hours": radius,
        "definition": "wind continuous-event support dilated +/- context hours",
    }


def result_direction_rows(
    label: str,
    result: Path,
    wind_indices: np.ndarray,
    capacities: np.ndarray,
    context: np.ndarray,
    actual_reference: dict[tuple[int, str, str], dict[str, float | int]] | None,
) -> tuple[list[dict[str, object]], dict[tuple[int, str, str], dict[str, float | int]]]:
    scenarios = np.load(result / "actual_scenarios_normalized.npy", mmap_mode="r")
    generated = aggregate_scenarios(scenarios, wind_indices, capacities)
    actual = np.asarray(np.load(result / "actual_data_normalized.npy", mmap_mode="r"), dtype=np.float64)
    actual_wind = aggregate_actual(actual, wind_indices, capacities)
    series_items = [("Actual", actual_wind)] if actual_reference is None else []
    series_items.append((label, generated))
    rows: list[dict[str, object]] = []
    reference = {} if actual_reference is None else actual_reference
    for series_label, series in series_items:
        for lag in LAGS:
            delta = series[..., lag:] - series[..., :-lag]
            event_pair = interval_overlap(context, lag)
            for window, mask in (("event_pm6", event_pair), ("non_event_pm6", ~event_pair)):
                values = delta[mask] if delta.ndim == 2 else delta[np.broadcast_to(mask[:, None, :], delta.shape)]
                for direction in ("positive", "negative"):
                    summary = direction_summary(values, direction)
                    key = (lag, window, direction)
                    if series_label == "Actual":
                        reference[key] = summary
                    row = {"series": series_label, "lag_h": lag, "window": window, "direction": direction}
                    row.update(summary)
                    if series_label != "Actual":
                        truth = reference[key]
                        for metric in ("std", "q90", "q95", "q99", "max"):
                            row[f"{metric}_truth_ratio"] = float(summary[metric]) / max(float(truth[metric]), 1e-12)
                    rows.append(row)
    return rows, reference


def load_model(run: Path, data_path: Path, device: torch.device):
    config = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
    checkpoint = torch.load(run / "checkpoints" / "model_best.pt", map_location="cpu", weights_only=False)
    static = load_station_static_data(data_path)
    primary, secondary, graph = load_generation_graphs(data_path, run, config["model"], checkpoint)
    model = Station24DiffusionModel(
        config["model"], static["station_features"], primary,
        static["station_capacities"], secondary,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config, checkpoint, graph


def gradient_loader(data_path: Path, run: Path, config: dict, checkpoint: dict, batch_size: int, seed: int):
    payload = json.loads((run / "jstd_event_targets.json").read_text(encoding="utf-8"))
    targets = build_station_jstd_target_arrays(
        data_path, "train", payload["thresholds"], event_sampling_target_fraction=0.60
    )
    _, dataset = get_station_dataloader(
        data_path, "train", checkpoint["residual_scale"], batch_size=batch_size,
        seed=seed, num_workers=0, condition_config=config["model"],
        state_thresholds=checkpoint.get("state_thresholds"), jstd_targets=targets,
    )
    indices = np.flatnonzero(targets.event_active > 0).tolist()
    if len(indices) < batch_size:
        raise ValueError("insufficient event issues for direction-gradient audit")
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)


def grad_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter], retain_graph: bool) -> float:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    squares = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def direction_gradient_rows(
    model: Station24DiffusionModel,
    run: Path,
    data_path: Path,
    config: dict,
    checkpoint: dict,
    device: torch.device,
    batches: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, object]]:
    loader = gradient_loader(data_path, run, config, checkpoint, batch_size, seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 20588:
        raise ValueError("direction audit requires the 20,588-parameter Lightweight Tail")
    diffusion = model.diffusion
    wind_mask = diffusion.denoiser.wind_station_mask.to(device)
    lag_weights = dict(zip(diffusion.ramp_auxiliary_lags, diffusion.ramp_auxiliary_lag_weights))
    lag_weight_sum = float(sum(diffusion.ramp_auxiliary_lag_weights))
    rows: list[dict[str, object]] = []
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= batches:
            break
        batch = move_batch(raw_batch, device)
        generator = torch.Generator(device=device).manual_seed(seed + batch_index)
        timestep = torch.randint(0, diffusion.num_steps, (batch_size,), generator=generator, device=device)
        noise = torch.randn(batch["residual_target"].shape, generator=generator, device=device, dtype=batch["residual_target"].dtype)
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            captured["prediction"] = output[0] if isinstance(output, tuple) else output

        handle = model.denoiser.register_forward_hook(hook)
        model.zero_grad(set_to_none=True)
        model(batch, timestep=timestep, noise=noise, include_auxiliary=False)
        handle.remove()
        prediction = captured["prediction"]
        clean = batch["residual_target"]
        noisy, _ = diffusion.add_noise(clean, timestep.long(), noise=noise)
        alpha_hat = diffusion.alpha_hat[timestep.long()].view(-1, 1, 1)
        predicted_clean = (noisy - (1.0 - alpha_hat).sqrt() * prediction) / alpha_hat.sqrt().clamp(min=1e-6)
        center, _ = model.predict_forecast_center(batch)
        predicted_actual = center + predicted_clean * batch["residual_scale"]
        target_actual = center + clean * batch["residual_scale"]
        snr = torch.sqrt(alpha_hat / (1.0 - alpha_hat).clamp(min=1e-6)).clamp(max=1.0)
        active = batch["jstd_event_active"].to(clean.dtype)
        time_support = batch["jstd_event_time_support"].to(clean.dtype)
        station_support = batch["jstd_event_station_support"].to(clean.dtype)
        event_support = torch.maximum(station_support, 0.25 * time_support[:, None, :]) * active[:, None, None]
        radius = int(diffusion.event_balanced_context_hours)
        event_context = F.max_pool1d(event_support, 2 * radius + 1, stride=1, padding=radius)
        valid = batch["valid_mask"].to(clean.dtype)
        component_specs: list[tuple[int, str, torch.Tensor, torch.Tensor, int, float]] = []
        for lag in LAGS:
            predicted_delta = predicted_actual[:, :, lag:] - predicted_actual[:, :, :-lag]
            target_delta = target_actual[:, :, lag:] - target_actual[:, :, :-lag]
            pair_valid = valid[:, :, lag:] * valid[:, :, :-lag]
            local_event = torch.maximum(event_context[:, :, lag:], event_context[:, :, :-lag]) * pair_valid
            error = F.smooth_l1_loss(predicted_delta, target_delta, reduction="none", beta=0.05)
            all_focus_sum = torch.zeros((), device=device, dtype=clean.dtype)
            staged = []
            for source, station_mask in (
                ("wind", wind_mask),
                ("solar", diffusion.denoiser.solar_station_mask.to(device)),
            ):
                source_valid = pair_valid * station_mask.to(clean.dtype)[None, :, None]
                if source == "solar":
                    source_valid = source_valid * (
                        batch["daylight_mask"][:, :, lag:]
                        * batch["daylight_mask"][:, :, :-lag]
                    )
                for direction, eligible in (("positive", target_delta.detach() > 0), ("negative", target_delta.detach() < 0)):
                    eligible = (source_valid > 0) & eligible
                    flat = eligible.flatten(1)
                    magnitude = (target_delta.detach() if direction == "positive" else -target_delta.detach()).flatten(1)
                    counts = flat.sum(-1)
                    top_counts = torch.ceil(counts.to(clean.dtype) * diffusion.event_balanced_ramp_top_fraction).long()
                    scores = magnitude.masked_fill(~flat, float("-inf"))
                    sorted_scores = torch.sort(scores, dim=-1, descending=True).values
                    index = (top_counts - 1).clamp(min=0, max=scores.shape[-1] - 1)
                    threshold = torch.gather(sorted_scores, -1, index.unsqueeze(-1))
                    extreme = ((scores >= threshold) & flat & (counts > 0).unsqueeze(-1)).reshape_as(eligible).to(clean.dtype)
                    directed_event = local_event * eligible.to(clean.dtype)
                    focus = torch.maximum(extreme, directed_event) * snr
                    if source == "wind":
                        staged.append((direction, error, focus, int(eligible.sum().detach()), int((focus > 0).sum().detach())))
                    all_focus_sum = all_focus_sum + focus.sum()
            for direction, error_tensor, focus, eligible_count, focus_count in staged:
                standalone = (error_tensor * focus).sum() / focus.sum().clamp(min=1.0)
                pooled_wind_contribution = (error_tensor * focus).sum() / all_focus_sum.clamp(min=1.0)
                coefficient = float(diffusion.event_balanced_ramp_loss_weight) * float(lag_weights[lag]) / lag_weight_sum
                component_specs.append((lag, direction, standalone, pooled_wind_contribution, eligible_count, coefficient))
                rows.append({
                    "batch": batch_index, "lag_h": lag, "direction": direction,
                    "timestep_mean": float(timestep.float().mean()),
                    "eligible_count": eligible_count, "focus_count": focus_count,
                    "standalone_loss": float(standalone.detach()),
                    "pooled_within_wind_loss_contribution": float(pooled_wind_contribution.detach()),
                    "configured_lag_coefficient": coefficient,
                    "standalone_gradient_norm": math.nan,
                    "configured_pooled_gradient_norm": math.nan,
                })
        for index, (lag, direction, standalone, pooled, _eligible_count, coefficient) in enumerate(component_specs):
            retain = index < len(component_specs) - 1
            standalone_norm = grad_norm(standalone, parameters, retain_graph=True)
            configured_norm = grad_norm(pooled * coefficient, parameters, retain_graph=retain)
            row = next(r for r in rows if r["batch"] == batch_index and r["lag_h"] == lag and r["direction"] == direction)
            row["standalone_gradient_norm"] = standalone_norm
            row["configured_pooled_gradient_norm"] = configured_norm
    return rows


def select_strong_negative_anchors(
    actual: np.ndarray,
    threshold: float,
    event_mask: np.ndarray,
    min_spacing: int = 6,
) -> list[dict[str, object]]:
    delta = actual[:, 1:] - actual[:, :-1]
    candidates: list[tuple[float, int, int]] = []
    for issue, offset in zip(*np.where(delta <= -threshold)):
        candidates.append((float(-delta[issue, offset]), int(issue), int(offset + 1)))
    candidates.sort(reverse=True)
    selected: list[dict[str, object]] = []
    for magnitude, issue, lead_end in candidates:
        if any(row["issue"] == issue and abs(int(row["lead_end"]) - lead_end) <= min_spacing for row in selected):
            continue
        selected.append({
            "issue": issue, "lead_end": lead_end, "drop_mw": magnitude,
            "event_pm6": bool(event_mask[issue, lead_end - 1]),
        })
    return selected


def recovery_metrics(series: np.ndarray, lead_end: int, drop_reference: float, horizon: int) -> dict[str, float]:
    stop = min(series.shape[-1] - 1, lead_end + horizon)
    if stop <= lead_end:
        return {"any_positive": math.nan, "max_recovery_mw": math.nan, "recovery_ratio": math.nan,
                "half_recovered": math.nan, "time_to_first_positive_h": math.nan, "time_to_half_recovery_h": math.nan}
    future = series[lead_end:stop + 1]
    steps = np.diff(future)
    positive = np.flatnonzero(steps > 0)
    recovery = future[1:] - future[0]
    half = np.flatnonzero(recovery >= 0.5 * drop_reference)
    maximum = max(float(np.max(recovery)), 0.0)
    return {
        "any_positive": float(positive.size > 0),
        "max_recovery_mw": maximum,
        "recovery_ratio": maximum / max(drop_reference, 1e-12),
        "half_recovered": float(half.size > 0),
        "time_to_first_positive_h": float(positive[0] + 1) if positive.size else math.nan,
        "time_to_half_recovery_h": float(half[0] + 1) if half.size else math.nan,
    }


def recovery_rows(
    label: str,
    generated: np.ndarray | None,
    actual: np.ndarray,
    anchors: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for anchor_id, anchor in enumerate(anchors):
        issue, end, truth_drop = int(anchor["issue"]), int(anchor["lead_end"]), float(anchor["drop_mw"])
        if generated is None:
            members = actual[issue][None]
            member_indices = [-1]
        else:
            members = generated[issue]
            member_indices = range(members.shape[0])
        for member_index, member in zip(member_indices, members):
            generated_drop = max(float(member[end - 1] - member[end]), 0.0)
            drop_conditioned = generated is None or generated_drop >= 0.5 * truth_drop
            for horizon in LAGS:
                metrics = recovery_metrics(member, end, generated_drop if generated is not None else truth_drop, horizon)
                rows.append({
                    "series": label, "anchor_id": anchor_id, "issue": issue, "lead_end": end,
                    "event_pm6": bool(anchor["event_pm6"]), "member_index": member_index,
                    "horizon_h": horizon, "truth_drop_mw": truth_drop,
                    "generated_drop_mw": generated_drop if generated is not None else truth_drop,
                    "drop_conditioned": bool(drop_conditioned), **metrics,
                })
    return rows


def _finite_median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.median()) if len(finite) else math.nan


def summarize_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expanded = pd.concat([frame.assign(scope="all"), frame[frame.event_pm6].assign(scope="event_pm6")])
    for (series, scope, horizon), group in expanded.groupby(["series", "scope", "horizon_h"]):
        for subset_name, subset in (("all_members", group), ("drop_conditioned", group[group.drop_conditioned])):
            rows.append({
                "series": series, "scope": scope, "horizon_h": int(horizon), "member_subset": subset_name,
                "record_count": int(len(subset)), "anchor_count": int(subset.anchor_id.nunique()),
                "drop_condition_rate": float(group.drop_conditioned.mean()),
                "any_positive_probability": float(subset.any_positive.mean()) if len(subset) else math.nan,
                "median_max_recovery_mw": float(subset.max_recovery_mw.median()) if len(subset) else math.nan,
                "median_recovery_ratio": float(subset.recovery_ratio.median()) if len(subset) else math.nan,
                "half_recovery_probability": float(subset.half_recovered.mean()) if len(subset) else math.nan,
                "median_time_to_first_positive_h": _finite_median(subset.time_to_first_positive_h),
                "median_time_to_half_recovery_h": _finite_median(subset.time_to_half_recovery_h),
            })
    return pd.DataFrame(rows)


def write_report(output: Path, gradient: pd.DataFrame, recovery: pd.DataFrame) -> None:
    means = gradient.groupby(["direction", "lag_h"], as_index=False)[
        ["standalone_gradient_norm", "configured_pooled_gradient_norm"]
    ].mean()
    pos1 = means[(means.direction == "positive") & (means.lag_h == 1)].iloc[0]
    neg1 = means[(means.direction == "negative") & (means.lag_h == 1)].iloc[0]
    gradient_ratio = float(pos1.configured_pooled_gradient_norm / max(neg1.configured_pooled_gradient_norm, 1e-12))
    event = recovery[(recovery.scope == "event_pm6") & (recovery.member_subset == "drop_conditioned")]
    lines = [
        "# Wind direction/recovery diagnostic",
        "",
        "No optimizer step, checkpoint write, or new member generation was performed.",
        "Validation actuals were used only for evaluation anchors.",
        "",
        "## Gradient decision evidence",
        "",
        f"- Positive/negative 1h configured pooled gradient-norm ratio: {gradient_ratio:.3f}.",
        "- See `wind_direction_gradient_summary.csv` for all matched batch/timestep/noise components.",
        "",
        "## Recovery evidence",
        "",
    ]
    for row in event.itertuples(index=False):
        lines.append(
            f"- {row.series}, +{row.horizon_h}h: drop-conditioned n={row.record_count}, "
            f"positive probability={row.any_positive_probability:.3f}, "
            f"median recovery ratio={row.median_recovery_ratio:.3f}, "
            f"half-recovery probability={row.half_recovery_probability:.3f}."
        )
    lines += [
        "",
        "## Decision rule",
        "",
        "Compare the positive/negative 1h gradient ratio with the amplitude and recovery gaps. "
        "A modest gradient imbalance does not by itself justify option A when the conditional "
        "drop-to-recovery trajectory remains delayed; report CPU diagnostics as CPU-only and "
        "reserve CUDA/AMP status for an explicit target-server check.",
        "",
        "## Planned evaluation additions (not implemented in this run)",
        "",
        "For every direction/lag/window cell, add the central 90% interval width `q95-q05` and "
        "the 90% Winkler interval score: `width + 20*(lower-y)*I(y<lower) + "
        "20*(y-upper)*I(y>upper)`. Report width and score together with coverage; do not add "
        "either quantity to the training objective in this diagnostic stage.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    run = Path(args.ramp_selection_run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA required for formal gradient audit; --allow-cpu is an explicit CPU diagnostic")
    stations = pd.read_csv(data_path / "station_order.csv").sort_values("channel_index").reset_index(drop=True)
    capacities = stations.capacity_mw.to_numpy(float)
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    split_rows = split_direction_rows(data_path, wind_indices, capacities)
    pd.DataFrame(split_rows).to_csv(output / "train_val_wind_direction_distribution.csv", index=False)
    context, context_audit = event_context(data_path, run, args.event_context_hours)
    results = [
        ("Raw", Path(args.raw_result)),
        (args.lightweight_label, Path(args.lightweight_result)),
        (args.candidate_label, Path(args.ramp_selection_result)),
    ]
    distribution_rows: list[dict[str, object]] = []
    reference = None
    aggregate_members: dict[str, np.ndarray] = {}
    actual_aggregate = None
    for label, result in results:
        current, reference = result_direction_rows(
            label, result, wind_indices, capacities, context, reference
        )
        distribution_rows.extend(current)
        scenarios = np.load(result / "actual_scenarios_normalized.npy", mmap_mode="r")
        aggregate_members[label] = aggregate_scenarios(scenarios, wind_indices, capacities)
        if actual_aggregate is None:
            actual = np.asarray(np.load(result / "actual_data_normalized.npy", mmap_mode="r"), dtype=np.float64)
            actual_aggregate = aggregate_actual(actual, wind_indices, capacities)
    pd.DataFrame(distribution_rows).to_csv(output / "model_wind_direction_event_distribution.csv", index=False)
    model, config, checkpoint, graph = load_model(run, data_path, device)
    before = state_digest(model)
    gradient_rows = direction_gradient_rows(
        model, run, data_path, config, checkpoint, device,
        args.gradient_batches, args.gradient_batch_size, args.seed,
    )
    after = state_digest(model)
    if before != after:
        raise RuntimeError("read-only gradient diagnostic changed model state")
    gradient = pd.DataFrame(gradient_rows)
    gradient.to_csv(output / "wind_direction_gradient_batches.csv", index=False)
    gradient.groupby(["direction", "lag_h"], as_index=False).agg(
        batches=("batch", "nunique"), eligible_count_mean=("eligible_count", "mean"),
        focus_count_mean=("focus_count", "mean"), standalone_loss_mean=("standalone_loss", "mean"),
        standalone_gradient_norm_mean=("standalone_gradient_norm", "mean"),
        configured_pooled_gradient_norm_mean=("configured_pooled_gradient_norm", "mean"),
        configured_pooled_gradient_norm_std=("configured_pooled_gradient_norm", "std"),
    ).to_csv(output / "wind_direction_gradient_summary.csv", index=False)
    train_negative_1h = next(
        row for row in split_rows
        if row["split"] == "train" and row["direction"] == "negative" and row["lag_h"] == 1
    )
    anchors = select_strong_negative_anchors(
        actual_aggregate, float(train_negative_1h["q90"]), interval_overlap(context, 1)
    )
    pd.DataFrame(anchors).to_csv(output / "strong_negative_anchors.csv", index=False)
    rec_rows = recovery_rows("Actual", None, actual_aggregate, anchors)
    for label, generated in aggregate_members.items():
        rec_rows.extend(recovery_rows(label, generated, actual_aggregate, anchors))
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(output / "drop_recovery_member_records.csv", index=False)
    recovery_summary = summarize_recovery(rec)
    recovery_summary.to_csv(output / "drop_recovery_summary.csv", index=False)
    write_report(output, gradient, recovery_summary)
    audit = {
        "status": "PASS", "device": str(device), "formal_training": "NOT RUN",
        "optimizer_steps": 0, "checkpoint_writes": 0, "new_generation": "NOT RUN",
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "state_hash_before_after_equal": True, "gradient_batches": args.gradient_batches,
        "gradient_batch_size": args.gradient_batch_size, "matched_timestep_noise": True,
        "event_context": context_audit, "strong_negative_definition": (
            "validation aggregate-wind negative 1h ramp >= train negative-1h q90; "
            "greedy 6h within-issue nonmaximum suppression"
        ),
        "generated_drop_condition": "generated 1h drop >= 50% of anchor truth drop",
        "graph": json.loads(json.dumps(graph, default=str)),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WIND_DIRECTION_RECOVERY_DIAGNOSTIC_COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
