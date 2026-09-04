"""Generate and evaluate validation/test scenarios from a station24 checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_graph_prior import load_generation_graphs
from station_dataset import (
    build_station_daylight_mask,
    get_station_dataloader,
    load_station_static_data,
)
from station_evaluation import evaluate_station_scenarios, save_evaluation
from station_retrieval_memory import build_retrieval_arrays
from station_discrete_event_memory import build_discrete_event_arrays
from station_forecast_trust import build_forecast_trust_arrays
from station_jstd_targets import build_station_jstd_target_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--issue-batch-size", type=int, default=None)
    parser.add_argument("--member-chunk-size", type=int, default=None)
    parser.add_argument(
        "--auto-tune-member-chunk",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Probe CUDA memory before generation and select the largest safe member chunk",
    )
    parser.add_argument(
        "--member-chunk-candidates",
        default=None,
        help="Comma-separated descending chunk candidates used by the CUDA preflight",
    )
    parser.add_argument("--max-generation-memory-fraction", type=float, default=None)
    parser.add_argument("--generation-probe-steps", type=int, default=None)
    parser.add_argument(
        "--checkpoint-state",
        choices=["ema", "raw"],
        default="ema",
        help=(
            "Choose the saved parameter state used for generation. The default "
            "keeps historical EMA behavior; raw uses model_state_dict from the "
            "same selected checkpoint."
        ),
    )
    parser.add_argument(
        "--result-variant",
        default=None,
        help=(
            "Optional evaluation identity for an inference-only ablation. The "
            "trained condition variant is retained separately in metadata."
        ),
    )
    parser.add_argument(
        "--forecast-guidance-scale",
        type=float,
        default=1.0,
        help=(
            "Interpolate denoiser predictions between the forecast-neutral path "
            "(0) and full forecast conditioning (1)"
        ),
    )
    parser.add_argument(
        "--energy-score-member-limit",
        type=int,
        default=None,
        help=(
            "Evaluate the high-cost multivariate energy score on a deterministic "
            "subset of at most this many members; all other metrics use every member"
        ),
    )
    parser.add_argument(
        "--tail-route-probability",
        type=float,
        default=None,
        help=(
            "Inference-only Bernoulli routing override for the two-way Raw "
            "body-tail generator. It does not modify checkpoint parameters."
        ),
    )
    parser.add_argument(
        "--allow-oracle-event-hypothesis",
        action="store_true",
        help=(
            "Required safety acknowledgement for the H1 validation-only "
            "controllability upper bound. This mode derives compact event "
            "attributes from validation actual residuals and is not causal."
        ),
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _member_chunk_candidates(
    configured: object,
    requested: int,
    n_samples: int,
) -> list[int]:
    if isinstance(configured, str):
        values = [int(value.strip()) for value in configured.split(",") if value.strip()]
    elif isinstance(configured, (list, tuple)):
        values = [int(value) for value in configured]
    elif configured is None:
        values = [requested, 192, 128, 96, 64, 48, 32, 16]
    else:
        raise ValueError("member chunk candidates must be a list or comma-separated string")
    upper_bound = min(int(requested), int(n_samples))
    values.append(upper_bound)
    candidates = sorted(
        {
            int(value)
            for value in values
            if 0 < int(value) <= upper_bound
        },
        reverse=True,
    )
    if not candidates:
        raise ValueError("member chunk candidate list is empty")
    return candidates


def tune_member_chunk_size(
    model: Station24DiffusionModel,
    batch: dict[str, torch.Tensor],
    candidates: list[int],
    forecast_guidance_scale: float,
    max_memory_fraction: float,
    probe_steps: int,
    tail_route_probability: float | None = None,
) -> tuple[int, dict[str, object]]:
    """Select the largest CUDA member chunk below a recorded memory ceiling.

    The probe temporarily executes only a few reverse steps. Tensor shapes and
    peak activation memory are representative, while the formal generation is
    restarted from the original seed and still uses every configured step.
    """

    device = batch["forecast"].device
    if device.type != "cuda":
        return candidates[0], {
            "enabled": False,
            "reason": "non_cuda_device",
            "selected_member_chunk": candidates[0],
        }
    if not 0.1 <= float(max_memory_fraction) < 1.0:
        raise ValueError("max generation memory fraction must be in [0.1,1.0)")
    if int(probe_steps) <= 0:
        raise ValueError("generation probe steps must be positive")

    original_steps = int(model.diffusion.num_steps)
    model.diffusion.num_steps = min(int(probe_steps), original_steps)
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    attempts: list[dict[str, object]] = []
    selected = None
    try:
        for candidate in candidates:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    probe = model.generate(
                        batch,
                        n_samples=int(candidate),
                        forecast_guidance_scale=forecast_guidance_scale,
                        return_expert_audit=False,
                        tail_route_probability_override=tail_route_probability,
                    )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                peak_bytes = int(torch.cuda.max_memory_allocated(device))
                peak_fraction = peak_bytes / max(total_memory, 1)
                attempts.append(
                    {
                        "member_chunk": int(candidate),
                        "status": "ok" if peak_fraction <= max_memory_fraction else "above_limit",
                        "elapsed_seconds": float(elapsed),
                        "peak_memory_gb": float(peak_bytes / 1024**3),
                        "peak_memory_fraction": float(peak_fraction),
                    }
                )
                del probe
                if peak_fraction <= max_memory_fraction:
                    selected = int(candidate)
                    break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                attempts.append(
                    {
                        "member_chunk": int(candidate),
                        "status": "cuda_oom",
                    }
                )
            finally:
                torch.cuda.empty_cache()
    finally:
        model.diffusion.num_steps = original_steps
    if selected is None:
        raise RuntimeError(
            "CUDA member-chunk preflight found no safe candidate; "
            f"attempts={attempts}"
        )
    return selected, {
        "enabled": True,
        "probe_diffusion_steps": min(int(probe_steps), original_steps),
        "formal_diffusion_steps": original_steps,
        "max_memory_fraction": float(max_memory_fraction),
        "issue_batch_size": int(batch["forecast"].shape[0]),
        "selected_member_chunk": selected,
        "attempts": attempts,
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "sample_index"
    }


def select_checkpoint_state(
    checkpoint: dict[str, object], source: str
) -> tuple[dict[str, torch.Tensor], str]:
    """Select an explicit checkpoint state without silently changing semantics."""

    if source == "raw":
        key = "model_state_dict"
    elif source == "ema":
        key = "ema_model_state_dict"
        if key not in checkpoint:
            key = "model_state_dict"
    else:
        raise ValueError(f"unsupported checkpoint state source: {source}")
    state = checkpoint.get(key)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint lacks a valid {key}")
    return state, key


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test split is sealed; pass --allow-test only after model lock")
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config_used.yaml"
    checkpoint_path = run_dir / "checkpoints" / "model_best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("run directory lacks config_used.yaml or model_best.pt")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluation_config = config["evaluation"]
    data_path = Path(args.data_path or config["data"]["data_path"])
    n_samples = int(args.n_samples or evaluation_config.get("n_samples", 80))
    seed = int(args.seed or evaluation_config.get("generation_seed", 424242))
    issue_batch_size = int(
        args.issue_batch_size or evaluation_config.get("issue_batch_size", 1)
    )
    member_chunk_size = int(
        args.member_chunk_size or evaluation_config.get("member_chunk_size", 10)
    )
    auto_tune_member_chunk = (
        bool(args.auto_tune_member_chunk)
        if args.auto_tune_member_chunk is not None
        else bool(evaluation_config.get("auto_tune_member_chunk", False))
    )
    configured_candidates = (
        args.member_chunk_candidates
        if args.member_chunk_candidates is not None
        else evaluation_config.get("member_chunk_candidates")
    )
    member_chunk_candidates = _member_chunk_candidates(
        configured_candidates,
        member_chunk_size,
        n_samples,
    )
    max_generation_memory_fraction = float(
        args.max_generation_memory_fraction
        if args.max_generation_memory_fraction is not None
        else evaluation_config.get("max_generation_memory_fraction", 0.82)
    )
    generation_probe_steps = int(
        args.generation_probe_steps
        if args.generation_probe_steps is not None
        else evaluation_config.get("generation_probe_steps", 8)
    )
    if n_samples <= 0 or issue_batch_size <= 0 or member_chunk_size <= 0:
        raise ValueError("n_samples, issue_batch_size, and member_chunk_size must be positive")
    forecast_guidance_scale = float(args.forecast_guidance_scale)
    if not 0.0 <= forecast_guidance_scale <= 1.0:
        raise ValueError("forecast-guidance-scale must be in [0,1]")
    tail_route_probability = args.tail_route_probability
    if tail_route_probability is not None and not 0.0 <= tail_route_probability <= 1.0:
        raise ValueError("tail-route-probability must be in [0,1]")
    output_dir = Path(
        args.output_dir
        or run_dir / f"generation_{args.split}_n{n_samples}_seed{seed}"
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory {output_dir}")
    output_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA is required for full generation; use --allow-cpu for smoke tests")
    set_seed(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    residual_scale = checkpoint["residual_scale"]
    static = load_station_static_data(data_path)
    primary_adjacency, secondary_adjacency, graph_manifest = load_generation_graphs(
        data_path,
        run_dir,
        config["model"],
        checkpoint,
    )
    model = Station24DiffusionModel(
        config["model"],
        static["station_features"],
        primary_adjacency,
        static["station_capacities"],
        secondary_adjacency,
    ).to(device)
    if checkpoint.get("architecture") != model.architecture:
        raise ValueError("checkpoint architecture does not match config")
    if checkpoint.get("spatial_mode") != model.spatial_mode:
        raise ValueError("checkpoint spatial mode does not match config")
    if checkpoint.get("spatial_mix_levels", ["bottleneck"]) != list(
        model.spatial_mix_levels
    ):
        raise ValueError("checkpoint spatial levels do not match config")
    if checkpoint.get("parallel_spatial_fusion_levels", []) != list(
        model.parallel_spatial_fusion_levels
    ):
        raise ValueError("checkpoint parallel fusion levels do not match config")
    if checkpoint.get("parallel_spatial_adjacency_mode", "fixed") != (
        model.parallel_spatial_adjacency_mode
    ):
        raise ValueError("checkpoint parallel adjacency mode does not match config")
    if checkpoint.get("forecast_correction_mode", "none") != (
        model.forecast_correction_mode
    ):
        raise ValueError("checkpoint forecast correction mode does not match config")
    if bool(checkpoint.get("use_body_tail_experts", False)) != bool(
        model.use_body_tail_experts
    ):
        raise ValueError("checkpoint body-tail expert mode does not match config")
    if bool(checkpoint.get("use_jstd_event_hypothesis", False)) != bool(
        model.use_jstd_event_hypothesis
    ):
        raise ValueError(
            "checkpoint JSTD event-hypothesis mode does not match config"
        )
    if bool(checkpoint.get("use_tail_time_localizer", False)) != bool(
        model.use_tail_time_localizer
    ):
        raise ValueError("checkpoint tail time localizer mode does not match config")
    if bool(checkpoint.get("use_retrieval_mismatch_expert", False)) != bool(
        model.use_retrieval_mismatch_expert
    ):
        raise ValueError("checkpoint retrieval mismatch mode does not match config")
    if bool(checkpoint.get("use_discrete_event_memory", False)) != bool(
        model.use_discrete_event_memory
    ):
        raise ValueError("checkpoint discrete event-memory mode does not match config")
    if bool(checkpoint.get("use_event_transport_transformer", False)) != bool(
        model.use_event_transport_transformer
    ):
        raise ValueError("checkpoint event-transport Transformer mode does not match config")
    if bool(checkpoint.get("use_forecast_trust_center", False)) != bool(
        model.use_forecast_trust_center
    ):
        raise ValueError("checkpoint forecast-trust center mode does not match config")
    state, checkpoint_state_key = select_checkpoint_state(
        checkpoint, args.checkpoint_state
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    trained_condition_variant = str(
        config.get("experiment", {}).get("variant", "baseline")
    )
    result_variant = str(args.result_variant or trained_condition_variant)

    retrieval_arrays = None
    if model.use_discrete_event_memory:
        retrieval_arrays = build_discrete_event_arrays(
            data_path,
            args.split,
            int(config["model"].get("event_memory_top_k", 48)),
            int(config["model"].get("retrieval_exclusion_days", 6)),
            float(config["model"].get("event_memory_quantile", 0.75)),
            int(config["model"].get("event_memory_target_stride_hours", 3)),
            float(
                config["model"].get(
                    "event_memory_severe_downside_fraction", 0.0
                )
            ),
            tuple(
                int(value)
                for value in config["model"].get(
                    "event_memory_durations", [6, 12, 24]
                )
            ),
        )
    elif model.use_retrieval_mismatch_expert:
        retrieval_arrays = build_retrieval_arrays(
            data_path,
            args.split,
            int(config["model"].get("retrieval_top_k", 40)),
            int(config["model"].get("retrieval_exclusion_days", 6)),
        )
    forecast_trust_arrays = None
    if model.use_forecast_trust_center:
        forecast_trust_arrays = build_forecast_trust_arrays(
            data_path,
            args.split,
            top_k=int(config["model"].get("forecast_trust_top_k", 24)),
            exclusion_days=int(
                config["model"].get("forecast_trust_exclusion_days", 6)
            ),
            temperature=float(
                config["model"].get("forecast_trust_retrieval_temperature", 0.75)
            ),
        )
    jstd_targets = None
    if model.use_jstd_event_hypothesis:
        if not args.allow_oracle_event_hypothesis:
            raise ValueError(
                "H1 generation is an oracle controllability audit; pass "
                "--allow-oracle-event-hypothesis explicitly"
            )
        if args.split != "val":
            raise ValueError(
                "oracle event hypotheses are restricted to validation; test remains locked"
            )
        target_manifest_path = run_dir / "jstd_event_targets.json"
        if not target_manifest_path.is_file():
            raise FileNotFoundError(
                f"missing H1 event-target manifest: {target_manifest_path}"
            )
        target_manifest = json.loads(
            target_manifest_path.read_text(encoding="utf-8")
        )
        jstd_targets = build_station_jstd_target_arrays(
            data_path, args.split, target_manifest["thresholds"]
        )
    loader, dataset = get_station_dataloader(
        data_path,
        args.split,
        residual_scale,
        batch_size=issue_batch_size,
        seed=seed,
        num_workers=0,
        condition_config=config["model"],
        state_thresholds=checkpoint.get("state_thresholds"),
        event_weighting=checkpoint.get("event_weighting"),
        event_replay=checkpoint.get("event_replay"),
        jstd_targets=jstd_targets,
        retrieval_arrays=retrieval_arrays,
        forecast_trust_arrays=forecast_trust_arrays,
    )
    tuning_audit: dict[str, object] = {
        "enabled": False,
        "selected_member_chunk": member_chunk_size,
    }
    if auto_tune_member_chunk:
        probe_batch = move_batch(next(iter(loader)), device)
        member_chunk_size, tuning_audit = tune_member_chunk_size(
            model,
            probe_batch,
            member_chunk_candidates,
            forecast_guidance_scale,
            max_generation_memory_fraction,
            generation_probe_steps,
            tail_route_probability,
        )
        print(f"GENERATION_PREFLIGHT {json.dumps(tuning_audit, ensure_ascii=False)}")
        del probe_batch
        set_seed(seed)
    daylight_mask, daylight_audit = build_station_daylight_mask(data_path, args.split)
    generated_standardized = []
    generated_stochastic_standardized = []
    generated_residual = []
    forecast_corrections = []
    forecast_centers = []
    forecast_history_fractions = []
    raw_actual_scenarios = []
    projected_actual_scenarios = []
    actual_values = []
    forecast_values = []
    tail_probabilities = []
    tail_routes = []
    tail_condition_attentions = []
    tail_time_probabilities = []
    tail_time_starts = []
    mismatch_probabilities = []
    mismatch_routes = []
    mismatch_time_probabilities = []
    retrieval_attentions = []
    event_memory_indices = []
    event_memory_types = []
    event_memory_durations = []
    event_memory_train_indices = []
    event_memory_probabilities = []
    jstd_event_hypotheses = []
    model.reset_parallel_spatial_gate_statistics()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    generation_started = time.perf_counter()

    print(
        f"GENERATION split={args.split} issues={len(loader.dataset)} "
        f"members={n_samples} chunks={member_chunk_size} device={device} "
        f"checkpoint_state={args.checkpoint_state} key={checkpoint_state_key}"
    )
    for batch_index, raw_batch in enumerate(loader, start=1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        issue_started = time.perf_counter()
        batch = move_batch(raw_batch, device)
        with torch.inference_mode():
            correction = model.predict_forecast_correction(batch).cpu().numpy()
            forecast_center, history_fraction = model.predict_forecast_center(batch)
            forecast_center = forecast_center.cpu().numpy()
            history_fraction = history_fraction.cpu().numpy()
        chunks = []
        route_chunks = []
        time_start_chunks = []
        mismatch_route_chunks = []
        event_memory_index_chunks = []
        event_memory_type_chunks = []
        event_memory_duration_chunks = []
        event_memory_train_index_chunks = []
        issue_tail_probability = None
        issue_tail_attention = None
        issue_tail_time_probability = None
        issue_mismatch_probability = None
        issue_mismatch_time_probability = None
        issue_retrieval_attention = None
        issue_event_memory_probability = None
        remaining = n_samples
        while remaining > 0:
            current = min(member_chunk_size, remaining)
            with torch.inference_mode():
                generated = model.generate(
                        batch,
                        n_samples=current,
                        forecast_guidance_scale=forecast_guidance_scale,
                        return_expert_audit=model.use_body_tail_experts,
                        tail_route_probability_override=tail_route_probability,
                    )
                if model.use_body_tail_experts:
                    samples, expert_audit = generated
                    chunks.append(samples.cpu())
                    route_chunks.append(expert_audit["tail_route"].cpu())
                    time_start_chunks.append(
                        expert_audit["tail_time_start"].cpu()
                    )
                    if issue_tail_probability is None:
                        issue_tail_probability = expert_audit[
                            "tail_probability"
                        ].cpu()
                        issue_tail_attention = expert_audit[
                            "tail_condition_attention"
                        ].cpu()
                        issue_tail_time_probability = expert_audit[
                            "tail_time_probability"
                        ].cpu()
                        issue_mismatch_probability = expert_audit[
                            "mismatch_probability"
                        ].cpu()
                        issue_mismatch_time_probability = expert_audit[
                            "mismatch_time_probability"
                        ].cpu()
                        issue_retrieval_attention = expert_audit[
                            "retrieval_attention"
                        ].cpu()
                        issue_event_memory_probability = expert_audit[
                            "event_memory_probability"
                        ].cpu()
                    mismatch_route_chunks.append(
                        expert_audit["mismatch_route"].cpu()
                    )
                    event_memory_index_chunks.append(
                        expert_audit["event_memory_index"].cpu()
                    )
                    event_memory_type_chunks.append(
                        expert_audit["event_memory_type"].cpu()
                    )
                    event_memory_duration_chunks.append(
                        expert_audit["event_memory_duration"].cpu()
                    )
                    event_memory_train_index_chunks.append(
                        expert_audit["event_memory_train_index"].cpu()
                    )
                else:
                    chunks.append(generated.cpu())
            remaining -= current
        stochastic_standardized = torch.cat(chunks, dim=1).numpy()  # [B,K,S,T]
        scale_tensor = raw_batch["residual_scale"].numpy()  # [B,S,T]
        stochastic_residual = (
            stochastic_standardized * scale_tensor[:, None, :, :]
        )
        residual = correction[:, None, :, :] + stochastic_residual
        standardized = residual / scale_tensor[:, None, :, :]
        forecast = raw_batch["forecast"].numpy()  # [B,S,T]
        actual = raw_batch["actual"].numpy()
        raw_scenarios = forecast_center[:, None, :, :] + residual
        projected = np.clip(raw_scenarios, 0.0, 1.0)

        generated_standardized.append(standardized.transpose(0, 1, 3, 2))
        generated_stochastic_standardized.append(
            stochastic_standardized.transpose(0, 1, 3, 2)
        )
        generated_residual.append(residual.transpose(0, 1, 3, 2))
        forecast_corrections.append(correction.transpose(0, 2, 1))
        forecast_centers.append(forecast_center.transpose(0, 2, 1))
        forecast_history_fractions.append(history_fraction.transpose(0, 2, 1))
        raw_actual_scenarios.append(raw_scenarios.transpose(0, 1, 3, 2))
        projected_actual_scenarios.append(projected.transpose(0, 1, 3, 2))
        actual_values.append(actual.transpose(0, 2, 1))
        forecast_values.append(forecast.transpose(0, 2, 1))
        if model.use_jstd_event_hypothesis:
            jstd_event_hypotheses.append(
                raw_batch["jstd_event_hypothesis"].numpy()
            )
        if model.use_body_tail_experts:
            if issue_tail_probability is None or issue_tail_attention is None:
                raise RuntimeError("body-tail generation lacks routing probability")
            tail_probabilities.append(issue_tail_probability.numpy())
            tail_routes.append(torch.cat(route_chunks, dim=1).numpy())
            tail_condition_attentions.append(issue_tail_attention.numpy())
            if issue_tail_time_probability is None:
                raise RuntimeError("body-tail generation lacks time distribution")
            tail_time_probabilities.append(issue_tail_time_probability.numpy())
            tail_time_starts.append(
                torch.cat(time_start_chunks, dim=1).numpy()
            )
            if model.use_discrete_event_memory:
                if issue_event_memory_probability is None:
                    raise RuntimeError("discrete event memory audit is unavailable")
                event_memory_indices.append(
                    torch.cat(event_memory_index_chunks, dim=1).numpy()
                )
                event_memory_types.append(
                    torch.cat(event_memory_type_chunks, dim=1).numpy()
                )
                event_memory_durations.append(
                    torch.cat(event_memory_duration_chunks, dim=1).numpy()
                )
                event_memory_train_indices.append(
                    torch.cat(event_memory_train_index_chunks, dim=1).numpy()
                )
                event_memory_probabilities.append(
                    issue_event_memory_probability.numpy()
                )
            if model.use_retrieval_mismatch_expert:
                if (
                    issue_mismatch_probability is None
                    or issue_mismatch_time_probability is None
                    or issue_retrieval_attention is None
                ):
                    raise RuntimeError("retrieval mismatch generation audit is unavailable")
                mismatch_probabilities.append(issue_mismatch_probability.numpy())
                mismatch_routes.append(
                    torch.cat(mismatch_route_chunks, dim=1).numpy()
                )
                mismatch_time_probabilities.append(
                    issue_mismatch_time_probability.numpy()
                )
                retrieval_attentions.append(issue_retrieval_attention.numpy())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        issue_seconds = time.perf_counter() - issue_started
        issue_count = int(raw_batch["forecast"].shape[0])
        scenario_rate = issue_count * n_samples / max(issue_seconds, 1e-9)
        print(
            f"generated issue batch {batch_index}/{len(loader)} "
            f"issues={issue_count} seconds={issue_seconds:.2f} "
            f"scenarios_s={scenario_rate:.3f}"
        )

    standardized_array = np.concatenate(generated_standardized, axis=0)
    stochastic_standardized_array = np.concatenate(
        generated_stochastic_standardized, axis=0
    )
    residual_array = np.concatenate(generated_residual, axis=0)
    correction_array = np.concatenate(forecast_corrections, axis=0)
    forecast_center_array = np.concatenate(forecast_centers, axis=0)
    history_fraction_array = np.concatenate(forecast_history_fractions, axis=0)
    raw_array = np.concatenate(raw_actual_scenarios, axis=0)
    projected_array = np.concatenate(projected_actual_scenarios, axis=0)
    actual_array = np.concatenate(actual_values, axis=0)
    forecast_array = np.concatenate(forecast_values, axis=0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        generation_peak_memory_gb = float(
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
    else:
        generation_peak_memory_gb = 0.0
    generation_seconds = time.perf_counter() - generation_started
    tail_probability_array = (
        np.concatenate(tail_probabilities, axis=0)
        if tail_probabilities
        else np.zeros((projected_array.shape[0],), dtype=np.float32)
    )
    tail_route_array = (
        np.concatenate(tail_routes, axis=0).astype(np.uint8)
        if tail_routes
        else np.zeros((projected_array.shape[0], n_samples), dtype=np.uint8)
    )
    tail_attention_array = (
        np.concatenate(tail_condition_attentions, axis=0)
        if tail_condition_attentions
        else np.zeros((projected_array.shape[0], 6), dtype=np.float32)
    )
    tail_time_probability_array = (
        np.concatenate(tail_time_probabilities, axis=0)
        if tail_time_probabilities
        else np.zeros(
            (projected_array.shape[0], projected_array.shape[2]),
            dtype=np.float32,
        )
    )
    tail_time_start_array = (
        np.concatenate(tail_time_starts, axis=0).astype(np.int16)
        if tail_time_starts
        else np.full(
            (projected_array.shape[0], n_samples), -1, dtype=np.int16
        )
    )
    mismatch_probability_array = (
        np.concatenate(mismatch_probabilities, axis=0)
        if mismatch_probabilities
        else np.zeros(projected_array.shape[0], dtype=np.float32)
    )
    mismatch_route_array = (
        np.concatenate(mismatch_routes, axis=0).astype(np.uint8)
        if mismatch_routes
        else np.zeros((projected_array.shape[0], n_samples), dtype=np.uint8)
    )
    mismatch_time_probability_array = (
        np.concatenate(mismatch_time_probabilities, axis=0)
        if mismatch_time_probabilities
        else np.zeros(
            (projected_array.shape[0], projected_array.shape[2]), dtype=np.float32
        )
    )
    retrieval_attention_array = (
        np.concatenate(retrieval_attentions, axis=0)
        if retrieval_attentions
        else np.zeros(
            (projected_array.shape[0], 1, projected_array.shape[2]), dtype=np.float32
        )
    )
    event_memory_index_array = (
        np.concatenate(event_memory_indices, axis=0)
        if event_memory_indices
        else np.full((projected_array.shape[0], n_samples), -1, dtype=np.int64)
    )
    event_memory_type_array = (
        np.concatenate(event_memory_types, axis=0)
        if event_memory_types
        else np.full((projected_array.shape[0], n_samples), -1, dtype=np.int64)
    )
    event_memory_duration_array = (
        np.concatenate(event_memory_durations, axis=0)
        if event_memory_durations
        else np.zeros((projected_array.shape[0], n_samples), dtype=np.int64)
    )
    event_memory_train_index_array = (
        np.concatenate(event_memory_train_indices, axis=0)
        if event_memory_train_indices
        else np.full((projected_array.shape[0], n_samples), -1, dtype=np.int64)
    )
    event_memory_probability_array = (
        np.concatenate(event_memory_probabilities, axis=0)
        if event_memory_probabilities
        else np.zeros((projected_array.shape[0], 1), dtype=np.float32)
    )
    if projected_array.shape[0] != daylight_mask.shape[0]:
        raise ValueError("daylight mask issue count does not match generated scenarios")
    projected_array = np.where(
        daylight_mask[:, None, :, :], projected_array, 0.0
    ).astype(np.float32, copy=False)

    np.save(output_dir / "generated_residual_standardized.npy", standardized_array)
    np.save(
        output_dir / "generated_stochastic_residual_standardized.npy",
        stochastic_standardized_array,
    )
    np.save(output_dir / "generated_residual_normalized.npy", residual_array)
    np.save(output_dir / "forecast_correction_normalized.npy", correction_array)
    np.save(output_dir / "forecast_center_normalized.npy", forecast_center_array)
    np.save(
        output_dir / "forecast_history_fraction.npy", history_fraction_array
    )
    np.save(
        output_dir / "corrected_forecast_center_normalized.npy",
        forecast_center_array + correction_array,
    )
    np.save(output_dir / "actual_scenarios_raw_normalized.npy", raw_array)
    np.save(output_dir / "actual_scenarios_normalized.npy", projected_array)
    np.save(output_dir / "actual_data_normalized.npy", actual_array)
    np.save(output_dir / "forecast_data_normalized.npy", forecast_array)
    np.save(output_dir / "station_daylight_mask.npy", daylight_mask)
    np.save(output_dir / "tail_expert_probability.npy", tail_probability_array)
    np.save(output_dir / "tail_expert_route.npy", tail_route_array)
    np.save(output_dir / "tail_condition_attention.npy", tail_attention_array)
    np.save(
        output_dir / "tail_event_time_probability.npy",
        tail_time_probability_array,
    )
    np.save(output_dir / "tail_event_start.npy", tail_time_start_array)
    if jstd_event_hypotheses:
        np.save(
            output_dir / "jstd_event_hypothesis.npy",
            np.concatenate(jstd_event_hypotheses, axis=0),
        )
    np.save(output_dir / "mismatch_expert_probability.npy", mismatch_probability_array)
    np.save(output_dir / "mismatch_expert_route.npy", mismatch_route_array)
    np.save(
        output_dir / "mismatch_time_probability.npy",
        mismatch_time_probability_array,
    )
    np.save(output_dir / "retrieval_attention.npy", retrieval_attention_array)
    np.save(output_dir / "event_memory_selected_index.npy", event_memory_index_array)
    np.save(output_dir / "event_memory_selected_type.npy", event_memory_type_array)
    np.save(output_dir / "event_memory_selected_duration.npy", event_memory_duration_array)
    np.save(
        output_dir / "event_memory_selected_train_index.npy",
        event_memory_train_index_array,
    )
    np.save(
        output_dir / "event_memory_candidate_probability.npy",
        event_memory_probability_array,
    )
    if retrieval_arrays is not None:
        np.save(output_dir / "retrieval_train_index.npy", retrieval_arrays.train_index)
        np.save(output_dir / "retrieval_distance.npy", retrieval_arrays.distance)
    if forecast_trust_arrays is not None:
        np.save(
            output_dir / "forecast_trust_neighbor_index.npy",
            forecast_trust_arrays.neighbor_index,
        )
        np.save(
            output_dir / "forecast_trust_neighbor_distance.npy",
            forecast_trust_arrays.distance,
        )
        np.save(
            output_dir / "forecast_trust_neighbor_weight.npy",
            forecast_trust_arrays.weight,
        )
    for level, moments in model.parallel_spatial_adjacency_moments.items():
        safe_level = level.replace("/", "_")
        np.save(
            output_dir / f"parallel_adjacency_{safe_level}_mean.npy",
            moments["mean"].numpy(),
        )
        np.save(
            output_dir / f"parallel_adjacency_{safe_level}_std.npy",
            moments["std"].numpy(),
        )

    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    adjacency = np.load(data_path / "station_adjacency.npy")
    metrics, station_frame, lead_frame = evaluate_station_scenarios(
        projected_array,
        raw_array,
        actual_array,
        forecast_array,
        stations,
        adjacency,
        daylight_mask=daylight_mask,
        interval_levels=tuple(
            float(value)
            for value in evaluation_config.get("quantiles", [0.80, 0.90, 0.95])
        ),
        energy_score_member_limit=args.energy_score_member_limit,
    )
    metrics["run"] = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_mse": float(checkpoint["val_loss"]),
        "checkpoint_validation_objective": float(checkpoint["val_loss"]),
        "checkpoint_validation_objective_type": (
            "h1_oracle_event_hypothesis_jstd_controllability"
            if model.use_jstd_event_hypothesis
            else
            "jstd_tail_epsilon_plus_decomposition_mask_issue_and_structure"
            if model.use_jstd_tail
            else "dynamic_center_residual_diffusion_plus_unified_event_objectives"
            if model.use_forecast_trust_center
            else "transformer_event_transport_plus_tail_epsilon_and_gate_bce"
            if model.use_event_transport_transformer
            else "localized_tail_energy_plus_temporal_variogram_plus_body_anchor"
            if model.sampler_event_localized
            else "tail_event_epsilon_plus_gate_bce_plus_sampler_energy_score"
            if model.train_sampler_energy_score_only
            else "retrieval_mismatch_epsilon_plus_route_and_hourly_bce"
            if model.train_retrieval_mismatch_only
            else "tail_event_time_soft_cross_entropy"
            if model.train_tail_time_localizer_only
            else (
                "tail_event_epsilon_plus_gate_bce"
                if model.use_body_tail_experts
                else (
                    "diffusion_epsilon_plus_forecast_correction_huber"
                    if model.forecast_correction_mode != "none"
                    else "diffusion_epsilon_mse"
                )
            )
        ),
        "checkpoint_state_source": args.checkpoint_state,
        "checkpoint_state_key": checkpoint_state_key,
        "checkpoint_state_fallback": (
            args.checkpoint_state == "ema"
            and checkpoint_state_key != "ema_model_state_dict"
        ),
        "architecture": model.architecture,
        "spatial_mode": model.spatial_mode,
        "spatial_mix_levels": list(model.spatial_mix_levels),
        "parallel_spatial_fusion_levels": list(
            model.parallel_spatial_fusion_levels
        ),
        "parallel_spatial_adjacency_mode": (
            model.parallel_spatial_adjacency_mode
        ),
        "parameter_count": int(checkpoint["parameter_count"]),
        "spatial_gate_values": model.spatial_gate_values,
        "parallel_spatial_gate_statistics": (
            model.parallel_spatial_gate_statistics
        ),
        "condition_variant": result_variant,
        "trained_condition_variant": trained_condition_variant,
        "condition_gate_values": model.condition_gate_values,
        "forecast_condition_dropout_prob": float(
            model.denoiser.forecast_condition_dropout_prob
        ),
        "training_forecast_condition_dropout_statistics": checkpoint.get(
            "forecast_condition_dropout_statistics"
        ),
        "forecast_guidance_scale": forecast_guidance_scale,
        "forecast_correction_mode": model.forecast_correction_mode,
        "forecast_correction_loss_weight": float(
            model.forecast_correction_loss_weight
        ),
        "forecast_correction_huber_beta": float(
            model.forecast_correction_huber_beta
        ),
        "forecast_correction_mean_abs_normalized": float(
            np.mean(np.abs(correction_array))
        ),
        "forecast_correction_max_abs_normalized": float(
            np.max(np.abs(correction_array))
        ),
        "use_forecast_trust_center": bool(model.use_forecast_trust_center),
        "forecast_trust_history_fraction_mean": float(
            history_fraction_array.mean()
        ),
        "forecast_trust_history_fraction_by_lead_day": [
            float(history_fraction_array[:, day * 24 : (day + 1) * 24].mean())
            for day in range(7)
        ],
        "forecast_trust_center_mae_normalized": float(
            np.mean(np.abs(forecast_center_array - actual_array))
        ),
        "forecast_trust_retrieval_audit": (
            forecast_trust_arrays.audit
            if forecast_trust_arrays is not None
            else None
        ),
        "event_prototype_anchor_strength": float(
            model.event_prototype_anchor_strength
        ),
        "state_gate_values": model.state_gate_values,
        "wind_common_gate_value": model.wind_common_gate_value,
        "use_body_tail_experts": bool(model.use_body_tail_experts),
        "use_jstd_tail": bool(model.use_jstd_tail),
        "use_jstd_event_hypothesis": bool(model.use_jstd_event_hypothesis),
        "jstd_h1_tail_fraction": float(model.jstd_h1_tail_fraction),
        "oracle_event_hypothesis_acknowledged": bool(
            args.allow_oracle_event_hypothesis
        ),
        "future_actual_used_as_generation_condition": bool(
            model.use_jstd_event_hypothesis
        ),
        "reportable_as_causal_forecast": not bool(
            model.use_jstd_event_hypothesis
        ),
        "use_tail_time_localizer": bool(model.use_tail_time_localizer),
        "use_retrieval_mismatch_expert": bool(
            model.use_retrieval_mismatch_expert
        ),
        "use_discrete_event_memory": bool(model.use_discrete_event_memory),
        "use_event_transport_transformer": bool(
            model.use_event_transport_transformer
        ),
        "retrieval_method": (
            retrieval_arrays.audit["method"] if retrieval_arrays is not None else None
        ),
        "retrieval_top_k": (
            int(
                retrieval_arrays.audit.get(
                    "top_k", retrieval_arrays.audit.get("top_k_candidate_pool", 0)
                )
            )
            if retrieval_arrays is not None
            else 0
        ),
        "retrieval_future_actual_used": False,
        "mismatch_probability_mean": float(mismatch_probability_array.mean()),
        "mismatch_member_fraction": float(mismatch_route_array.mean()),
        "mismatch_time_probability_mean": float(
            mismatch_time_probability_array.mean()
        ),
        "retrieval_attention_entropy_mean": float(
            -np.mean(
                np.sum(
                    retrieval_attention_array
                    * np.log(retrieval_attention_array.clip(min=1e-12)),
                    axis=1,
                )
            )
        ),
        "tail_time_temperature": float(model.denoiser.tail_time_temperature),
        "tail_time_mask_radius_hours": int(
            model.denoiser.tail_time_mask_radius_hours
        ),
        "tail_gate_loss_weight": float(model.tail_gate_loss_weight),
        "tail_common_gate_value": model.tail_common_gate_value,
        "tail_probability_mean": float(tail_probability_array.mean()),
        "tail_probability_min": float(tail_probability_array.min()),
        "tail_probability_max": float(tail_probability_array.max()),
        "tail_member_fraction": float(tail_route_array.mean()),
        "tail_route_probability_override": (
            float(tail_route_probability)
            if tail_route_probability is not None
            else None
        ),
        "tail_time_probability_entropy_mean": float(
            -np.mean(
                np.sum(
                    tail_time_probability_array
                    * np.log(tail_time_probability_array.clip(min=1e-12)),
                    axis=1,
                )
            )
        ),
        "tail_time_routed_start_mean": (
            float(tail_time_start_array[tail_time_start_array >= 0].mean())
            if np.any(tail_time_start_array >= 0)
            else None
        ),
        "tail_time_probability_semantics": (
            "oracle_event_hypothesis_smooth_onset_duration_envelope"
            if model.use_jstd_event_hypothesis
            else
            "legacy_placeholder_not_applicable_jstd_uses_internal_station_time_masks"
            if model.use_jstd_tail
            else "standalone_tail_time_localizer_probability"
            if model.use_tail_time_localizer
            else "not_available"
        ),
        "jstd_internal_masks_saved_in_generation_result": False,
        "jstd_internal_mask_audit_tool": (
            None
            if model.use_jstd_event_hypothesis
            else "tools/audit_station24_jstd_mechanism.py"
            if model.use_jstd_tail
            else None
        ),
        "tail_condition_attention_names": (
            [
                "event_active",
                "event_onset_fraction",
                "event_duration_fraction",
                "signed_wind_depth",
                "signed_solar_depth",
                "source_synchrony",
            ]
            if model.use_jstd_event_hypothesis
            else [
                "issued_wind_level",
                "issued_wind_down_ramp_3h",
                "aligned_forecast_revision",
                "forecast_low_output_state",
                "forecast_down_ramp_state",
                "recent_observed_forecast_error",
            ]
        ),
        "tail_condition_attention_mean": [
            float(value) for value in tail_attention_array.mean(axis=0)
        ],
        "tail_routing_method": (
            "validation_oracle_continuous_event_hypothesis_fixed_tail_fraction"
            if model.use_jstd_event_hypothesis
            else
            "jstd_issue_probability_plus_member_bernoulli_with_internal_station_time_masks"
            if model.use_jstd_tail
            else "two_expert_body_plus_transformer_localized_discrete_event_transport"
            if model.use_event_transport_transformer
            else "two_expert_body_plus_member_level_discrete_event_memory_routing"
            if model.use_discrete_event_memory
            else "three_way_body_deep_tail_retrieval_mismatch_categorical_routing"
            if model.use_retrieval_mismatch_expert
            else "causal_condition_gate_with_member_level_bernoulli_routing"
            if model.use_body_tail_experts
            else "disabled"
        ),
        "event_weighting_file": (
            str(run_dir / "event_weighting.json")
            if (run_dir / "event_weighting.json").is_file()
            else None
        ),
        "event_weighting_applied_during_generation": False,
        "event_weighting_method": (
            str(checkpoint["event_weighting"].get("method"))
            if checkpoint.get("event_weighting") is not None
            else None
        ),
        "event_replay_file": (
            str(run_dir / "event_replay.json")
            if (run_dir / "event_replay.json").is_file()
            else None
        ),
        "event_replay_applied_during_generation": False,
        "event_replay_method": (
            str(checkpoint["event_replay"].get("method"))
            if checkpoint.get("event_replay") is not None
            else None
        ),
        "state_thresholds_file": (
            str(run_dir / "state_thresholds.json")
            if (run_dir / "state_thresholds.json").is_file()
            else None
        ),
        "condition_feature_audit": dataset.condition_audit,
        "residual_scaling_method": str(
            residual_scale.get("method", "per_station_std")
        ),
        "ramp_auxiliary_loss_weight": float(
            model.diffusion.ramp_auxiliary_loss_weight
        ),
        "ramp_auxiliary_lags": list(model.diffusion.ramp_auxiliary_lags),
        "ramp_auxiliary_lag_weights": list(
            model.diffusion.ramp_auxiliary_lag_weights
        ),
        "wind_common_event_loss_weight": float(
            model.diffusion.wind_common_event_loss_weight
        ),
        "split": args.split,
        "n_samples": n_samples,
        "generation_seed": seed,
        "evaluation_member_count": n_samples,
        "energy_score_member_count": int(
            metrics["joint"]["energy_score_member_count"]
        ),
        "issue_batch_size": issue_batch_size,
        "member_chunk_size": member_chunk_size,
        "member_chunk_tuning": tuning_audit,
        "generation_seconds": float(generation_seconds),
        "generation_scenarios_per_second": float(
            len(loader.dataset) * n_samples / max(generation_seconds, 1e-9)
        ),
        "generation_peak_cuda_memory_gb": generation_peak_memory_gb,
        "physical_projection": "clip_0_1_and_station_astronomical_solar_night",
        "daylight_audit": daylight_audit,
        "test_used": args.split == "test",
        "graph_manifest": graph_manifest,
    }
    solar_indices = stations.index[stations.data_type.eq("solar")].to_numpy()
    solar_night = ~daylight_mask[:, :, solar_indices]
    raw_solar = raw_array[..., solar_indices]
    projected_solar = projected_array[..., solar_indices]
    metrics["physical"]["raw_solar_night_nonzero_rate"] = float(
        np.mean(np.abs(raw_solar[solar_night[:, None, :, :].repeat(n_samples, axis=1)]) > 1e-6)
    )
    metrics["physical"]["projected_solar_night_nonzero_rate"] = float(
        np.mean(
            np.abs(
                projected_solar[
                    solar_night[:, None, :, :].repeat(n_samples, axis=1)
                ]
            )
            > 1e-6
        )
    )
    save_evaluation(output_dir, metrics, station_frame, lead_frame)
    (output_dir / "generation_metadata.json").write_text(
        json.dumps(metrics["run"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"GENERATION_COMPLETE spatial_mode={model.spatial_mode}")
    print(f"RESULT_DIR={output_dir}")


if __name__ == "__main__":
    main()
