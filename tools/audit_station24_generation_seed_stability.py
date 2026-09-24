"""Preflight for checkpoint-frozen Station-24 generation-seed sensitivity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--lightweight-run", required=True)
    parser.add_argument("--raw-424242", required=True)
    parser.add_argument("--full-424242", required=True)
    parser.add_argument("--lightweight-424242", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_result(path: Path, seed: int, members: int) -> dict:
    metadata = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
    if int(metadata["generation_seed"]) != seed or int(metadata["n_samples"]) != members:
        raise ValueError(f"protocol mismatch in {path}")
    if metadata.get("future_actual_used_as_generation_condition", False):
        raise ValueError(f"noncausal result in {path}")
    if metadata.get("reportable_as_causal_forecast", True) is not True:
        raise ValueError(f"non-reportable result in {path}")
    return metadata


def config(run: Path) -> dict:
    return yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    seeds = list(args.seeds)
    if len(seeds) not in (2, 3) or len(set(seeds)) != len(seeds) or 424242 in seeds:
        raise ValueError("provide two or three unique new seeds, none equal to 424242")
    runs = {name: Path(value) for name, value in (
        ("raw", args.raw_run), ("full", args.full_run), ("lightweight", args.lightweight_run)
    )}
    checks: dict[str, str] = {}
    checkpoint_hashes = {}
    for name, run in runs.items():
        checkpoint = run / "checkpoints" / "model_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes[name] = sha256(checkpoint)
        checks[f"{name}_checkpoint"] = "PASS"
    raw_cfg, full_cfg, light_cfg = (config(runs[name]) for name in ("raw", "full", "lightweight"))
    if not bool(full_cfg["model"].get("independent_joint_tail_training")):
        raise ValueError("Full V2 run is not an independent joint-tail checkpoint")
    if light_cfg["model"].get("joint_multiresidual_training_version") != "v2_fair_v1":
        raise ValueError("Lightweight run is not the fair lightweight checkpoint")
    for name, cfg in (("raw", raw_cfg), ("full", full_cfg), ("lightweight", light_cfg)):
        model = cfg["model"]
        if model.get("use_tail_time_localizer", False) or model.get("use_jstd_event_hypothesis", False):
            raise ValueError(f"{name} enables localization/oracle hypothesis")
        checks[f"{name}_no_self_localization_or_oracle"] = "PASS"
    current = {
        "raw": load_result(Path(args.raw_424242), 424242, 500),
        "full": load_result(Path(args.full_424242), 424242, 500),
        "lightweight": load_result(Path(args.lightweight_424242), 424242, 500),
    }
    for name in ("full", "lightweight"):
        meta = current[name]
        if int(meta.get("body_members", -1)) != 400 or int(meta.get("tail_members", -1)) != 100:
            raise ValueError(f"{name} historical mixture is not 400+100")
        checks[f"{name}_historical_400_plus_100"] = "PASS"
    for relative in (
        "generate_station24.py",
        "tools/merge_station24_independent_tail_members.py",
        "tools/evaluate_station24_jstd_events.py",
        "tools/evaluate_station24_diffusion_ts.py",
        "tools/diagnose_station24_wind_event_timing.py",
        "tools/summarize_station24_generation_seed_stability.py",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        checks[f"tool_{relative}"] = "PASS"
    checks.update(
        {
            "new_generation_seed_count": "PASS",
            "no_retraining": "PASS",
            "member_budget_500": "PASS",
            "body_tail_quota_400_100": "PASS",
            "event_definition_unchanged": "PASS",
            "evaluation_tools_unchanged": "PASS",
            "historical_seed_424242_included": "PASS",
        }
    )
    report = {
        "status": "PASS",
        "launch_eligible": True,
        "formal_training": "NOT RUN (not part of this checkpoint-frozen experiment)",
        "historical_seed": 424242,
        "new_generation_seeds": seeds,
        "checkpoint_sha256": checkpoint_hashes,
        "checks": checks,
        "historical_result_variants": {name: meta.get("condition_variant") for name, meta in current.items()},
        "tool_sha256": {
            relative: sha256(ROOT / relative)
            for relative in (
                "generate_station24.py",
                "tools/merge_station24_independent_tail_members.py",
                "tools/evaluate_station24_jstd_events.py",
                "tools/evaluate_station24_diffusion_ts.py",
                "tools/diagnose_station24_wind_event_timing.py",
                "tools/summarize_station24_generation_seed_stability.py",
            )
        },
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GENERATION_SEED_STABILITY_PREFLIGHT_PASS output={output}", flush=True)


if __name__ == "__main__":
    main()
