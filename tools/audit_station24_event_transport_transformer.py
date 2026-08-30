"""Fail-fast audit for the Transformer-localized discrete event tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_checkpoint(run: Path) -> dict[str, object]:
    path = run / "checkpoints" / "model_best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = load_checkpoint(Path(args.source_run))
    candidate = load_checkpoint(Path(args.candidate_run))
    result = Path(args.candidate_result)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    if source.get("condition_variant") != "geo_history_actual_body_tail_moe":
        raise ValueError("source is not the Raw body-tail checkpoint")
    if candidate.get("condition_variant") != (
        "geo_history_actual_event_transport_transformer"
    ):
        raise ValueError("candidate variant mismatch")
    if not bool(candidate.get("use_event_transport_transformer", False)):
        raise ValueError("candidate did not enable the event Transformer")
    metadata = json.loads(
        (result / "generation_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("split") != "val" or int(metadata.get("n_samples", 0)) != 500:
        raise ValueError("formal audit requires validation generation with 500 members")
    if bool(metadata.get("test_used", True)):
        raise ValueError("test data were used")
    if bool(metadata.get("event_replay_applied_during_generation", False)):
        raise ValueError("target-derived replay labels leaked into generation")
    if metadata.get("checkpoint_state_source") != "raw":
        raise ValueError("formal comparison must use the raw checkpoint state")

    trainable = set(candidate.get("discrete_event_trainable_parameter_names", []))
    transformer_names = {
        name for name in trainable if name.startswith("event_memory_selector.")
    }
    if not transformer_names:
        raise ValueError("Transformer parameters are absent from the trainable set")
    if any(
        not (
            name.startswith("event_memory_selector.")
            or name.startswith("denoiser.tail_")
            or name.startswith("denoiser.event_prototype_adapter.")
        )
        for name in trainable
    ):
        raise ValueError("parameters outside the tail/selector contract were trainable")

    source_state = source["model_state_dict"]
    candidate_state = candidate["model_state_dict"]
    allowed_source = set(candidate.get("body_tail_trainable_parameter_names", []))
    allowed_source.update(
        name.replace("denoiser.", "diffusion.denoiser.", 1)
        for name in tuple(allowed_source)
        if name.startswith("denoiser.")
    )
    changed_frozen: list[str] = []
    missing_frozen: list[str] = []
    checked = 0
    for key, value in source_state.items():
        if key in allowed_source:
            continue
        checked += 1
        if key not in candidate_state:
            missing_frozen.append(key)
            continue
        observed = candidate_state[key]
        equal = (
            torch.allclose(value, observed, rtol=0.0, atol=1e-7)
            if value.is_floating_point()
            else torch.equal(value, observed)
        )
        if not equal:
            changed_frozen.append(key)
    if missing_frozen or changed_frozen:
        raise ValueError(
            "Raw body protection failed: "
            f"missing={missing_frozen[:5]} changed={changed_frozen[:5]}"
        )

    probability = np.load(result / "event_memory_candidate_probability.npy")
    route = np.load(result / "tail_expert_route.npy").astype(bool)
    selected = np.load(result / "event_memory_selected_index.npy")
    starts = np.load(result / "tail_event_start.npy")
    durations = np.load(result / "event_memory_selected_duration.npy")
    if probability.ndim != 2 or route.shape != (probability.shape[0], 500):
        raise ValueError("invalid probability or routing shapes")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("candidate probabilities do not sum to one")
    effective_k = 1.0 / np.sum(np.square(probability), axis=1)
    candidate_count = probability.shape[1]
    selector_nonuniform = float(effective_k.mean()) < 0.95 * candidate_count
    routed_candidates = np.unique(selected[route]) if route.any() else np.array([])
    routed_days = np.unique(starts[route] // 24) if route.any() else np.array([])
    routed_durations = np.unique(durations[route]) if route.any() else np.array([])
    routing_noncollapsed = (
        route.any() and len(routed_candidates) >= 2 and len(routed_days) >= 2
    )

    transformer_parameter_count = int(
        sum(candidate_state[name].numel() for name in transformer_names)
    )
    audit = {
        "status": "protocol_passed",
        "method": "causal_transformer_discrete_event_transport_v1",
        "raw_body_parameter_tensors_checked": checked,
        "changed_frozen_parameter_keys": changed_frozen,
        "transformer_parameter_tensors": len(transformer_names),
        "transformer_parameter_count": transformer_parameter_count,
        "candidate_count": int(candidate_count),
        "mean_effective_candidate_count": float(effective_k.mean()),
        "mean_max_candidate_probability": float(probability.max(axis=1).mean()),
        "selector_nonuniform": bool(selector_nonuniform),
        "routing_noncollapsed": bool(routing_noncollapsed),
        "tail_member_fraction": float(route.mean()),
        "unique_routed_candidates": int(len(routed_candidates)),
        "unique_routed_lead_days": [int(value + 1) for value in routed_days],
        "unique_routed_durations_hours": [int(value) for value in routed_durations],
        "two_generation_paths_only": True,
        "dynamic_graph_used": False,
        "future_actual_used_as_condition": False,
        "test_used": False,
    }
    (output / "event_transport_transformer_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "event_transport_transformer_audit.md").write_text(
        "# Event transport Transformer audit\n\n"
        "- Protocol status: passed\n"
        f"- Frozen Raw-body tensors checked: {checked}\n"
        f"- Transformer parameters: {transformer_parameter_count:,}\n"
        f"- Mean effective candidate count: {effective_k.mean():.2f}/{candidate_count}\n"
        f"- Selector non-uniform: {selector_nonuniform}\n"
        f"- Routing non-collapsed: {routing_noncollapsed}\n"
        f"- Realized tail-member fraction: {route.mean():.2%}\n"
        f"- Routed lead days: {[int(value + 1) for value in routed_days]}\n"
        "- Top-K trajectory averaging: disabled\n"
        "- Future actual used as condition: false\n",
        encoding="utf-8",
    )
    print(
        "EVENT_TRANSPORT_TRANSFORMER_AUDIT_COMPLETE "
        f"selector_nonuniform={selector_nonuniform} "
        f"routing_noncollapsed={routing_noncollapsed} output={output}"
    )


if __name__ == "__main__":
    main()
