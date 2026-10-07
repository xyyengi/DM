"""Read-only mechanism gate: no optimizer, checkpoint write, or generation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from station_dataset import get_station_dataloader, load_station_static_data
from station_jstd_targets import (
    build_station_jstd_target_arrays, fit_station_jstd_event_thresholds,
)
from station_lightweight_tail import expected_config, validate_config
from src.models.station_conditioned_diffusion import Station24DiffusionModel

RAW_RUN = Path("outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027")
PREFIX = "denoiser.joint_multiresidual_tail."


def digest_state(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def grad_norm(loss, params, retain=True):
    gradients = torch.autograd.grad(
        loss, params, retain_graph=retain, allow_unused=True
    )
    return float(sum(
        g.detach().float().square().sum() for g in gradients if g is not None
    ).sqrt())


def grad_vector(loss, params):
    gradients = torch.autograd.grad(
        loss, params, retain_graph=True, allow_unused=True
    )
    return torch.cat([
        (torch.zeros_like(p) if g is None else g).detach().float().reshape(-1)
        for p, g in zip(params, gradients)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/station24_lightweight_joint_tail_aux_timestep_compensation_168h.yaml")
    parser.add_argument("--checkpoint", default=str(RAW_RUN / "checkpoints/model_best.pt"))
    parser.add_argument("--secondary-adjacency", default=str(RAW_RUN / "graphs/secondary_adjacency.npy"))
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"formal_training": "NOT RUN", "generation": "NOT RUN", "optimizer_step": "NOT RUN", "checks": {}}

    def check(name, condition, evidence=None):
        report["checks"][name] = {"status": "PASS" if condition else "FAIL", "evidence": evidence}
        if not condition:
            raise AssertionError(name)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    (output / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    report["config_sha256"] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    wrapper = ROOT / "run_station24_lightweight_joint_tail_aux_timestep_compensation_formal.sh"
    report["wrapper_sha256"] = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    if args.device == "cuda" and not torch.cuda.is_available():
        report["checks"]["CUDA_AMP_forward_backward"] = {"status": "NOT RUN", "evidence": "CUDA unavailable"}
        (output / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    static = load_station_static_data(args.data_path)
    graph = torch.as_tensor(np.load(args.secondary_adjacency), dtype=torch.float32)

    def build(candidate):
        recipe = expected_config(auxiliary_timestep_compensation=candidate)["model"]
        recipe["secondary_adjacency_path"] = str(Path(args.secondary_adjacency).resolve())
        model = Station24DiffusionModel(
            recipe, static["station_features"], static["station_adjacency"],
            static["station_capacities"], graph,
        ).to(device)
        incompatible = model.load_state_dict(source["model_state_dict"], strict=False)
        check(
            f"raw_initialization_{'on' if candidate else 'off'}",
            set(incompatible.missing_keys) == set(model.joint_multiresidual_new_state_dict_keys)
            and not incompatible.unexpected_keys,
        )
        model.train()
        return model

    control, candidate = build(False), build(True)
    candidate.load_state_dict(control.state_dict(), strict=True)
    control_hash, candidate_hash = digest_state(control), digest_state(candidate)
    params_control = [p for n, p in control.named_parameters() if n.startswith(PREFIX)]
    params_candidate = [p for n, p in candidate.named_parameters() if n.startswith(PREFIX)]
    check("trainable_parameter_count", sum(p.numel() for p in params_candidate) == 20588, 20588)
    check("raw_frozen", all(not p.requires_grad for n, p in candidate.named_parameters() if not n.startswith(PREFIX)))
    check("checkpoint_state_compatible", set(control.state_dict()) == set(candidate.state_dict()))
    weight = candidate.diffusion.auxiliary_timestep_weight.detach().cpu()
    check("schedule_mean_one", abs(float(weight.mean()) - 1.0) < 1e-6, float(weight.mean()))

    schedule_rows = []
    alpha_hat = candidate.diffusion.alpha_hat.detach().cpu()
    for t in range(500):
        snr = alpha_hat[t] / (1.0 - alpha_hat[t])
        jacobian = torch.rsqrt(snr)
        schedule_rows.append({
            "t": t, "alpha_bar": float(alpha_hat[t]), "snr": float(snr),
            "sqrt_snr": float(torch.sqrt(snr)), "inverse_sqrt_snr": float(jacobian),
            "w_aux": float(weight[t]), "compensated_amplification": float(weight[t] * jacobian),
        })
    with (output / "compensation_schedule.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=schedule_rows[0]); writer.writeheader(); writer.writerows(schedule_rows)
    with (output / "w_aux_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("t", "w_aux")); writer.writeheader(); writer.writerows({"t": r["t"], "w_aux": r["w_aux"]} for r in schedule_rows)

    thresholds = fit_station_jstd_event_thresholds(args.data_path, config["model"])
    targets = build_station_jstd_target_arrays(args.data_path, "train", thresholds, event_sampling_target_fraction=.60)
    loader, _ = get_station_dataloader(
        args.data_path, "train", source["residual_scale"], batch_size=8,
        seed=2027, condition_config=config["model"],
        state_thresholds=source["state_thresholds"], jstd_targets=targets,
    )
    batch = next(iter(loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    torch.manual_seed(2027)
    noise = torch.randn_like(batch["residual_target"])
    gradient_rows = []
    for timestep_value in (83, 250, 416):
        timestep = torch.full((noise.shape[0],), timestep_value, dtype=torch.long, device=device)
        records, vectors = {}, {}
        for label, model, params in (("control", control, params_control), ("candidate", candidate, params_candidate)):
            model.diffusion.capture_loss_graph_for_audit = True
            loss = model(batch, timestep=timestep, noise=noise)
            tensors = model.diffusion.audit_loss_tensors
            records[label] = {
                "epsilon": grad_norm(tensors["epsilon"], params),
                "ramp": grad_norm(tensors["ramp"], params),
                "shape": grad_norm(tensors["shape"], params),
                "slow": grad_norm(tensors["slow"], params),
                "auxiliary": grad_norm(.18*tensors["ramp"] + .14*tensors["shape"] + .10*tensors["slow"], params),
                "total_loss": float(loss.detach()),
            }
            vectors[label] = {
                key: grad_vector(value, params)
                for key, value in tensors.items()
                if key in {"epsilon", "ramp", "shape", "slow"}
            }
        for component in ("epsilon", "ramp", "shape", "slow", "auxiliary"):
            old, new = records["control"][component], records["candidate"][component]
            cosine = None
            if component in vectors["control"]:
                cosine = float(torch.nn.functional.cosine_similarity(
                    vectors["control"][component][None],
                    vectors["candidate"][component][None],
                ))
            gradient_rows.append({"t": timestep_value, "component": component, "control_norm": old, "candidate_norm": new, "ratio": new / old if old else None, "w_aux": float(weight[timestep_value]), "gradient_cosine": cosine})
        check(f"epsilon_unchanged_t{timestep_value}", abs(records["control"]["epsilon"] - records["candidate"]["epsilon"]) <= 1e-6 * max(1.0, records["control"]["epsilon"]), records)
    with (output / "preflight_gradient_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=gradient_rows[0]); writer.writeheader(); writer.writerows(gradient_rows)
    for row in gradient_rows:
        if row["component"] in {"ramp", "shape", "slow"}:
            check(f"scaled_{row['component']}_t{row['t']}", abs(row["ratio"] - row["w_aux"]) < 2e-4, row)
    check("high_t_reduced_not_zero", .1 < float(weight[416]) < 1.0, float(weight[416]))
    check("low_t_bounded", float(weight[83]) < 3.0, float(weight[83]))
    check("raw_and_tail_state_unchanged", digest_state(control) == control_hash and digest_state(candidate) == candidate_hash)
    if device.type == "cuda":
        timestep = torch.full((noise.shape[0],), 250, dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            amp_loss = candidate(batch, timestep=timestep, noise=noise)
        amp_gradients = torch.autograd.grad(amp_loss, params_candidate, allow_unused=True)
        check(
            "CUDA_AMP_forward_backward",
            bool(torch.isfinite(amp_loss)) and all(
                g is None or bool(torch.isfinite(g).all()) for g in amp_gradients
            ),
        )
    else:
        check("CPU_forward_backward", True)
        report["checks"]["CUDA_AMP_forward_backward"] = {"status": "NOT RUN", "evidence": "target server required"}
    report.update({
        "weight_summary": {"min": float(weight.min()), "median": float(weight.median()), "mean": float(weight.mean()), "max": float(weight.max())},
        "selected_timesteps": [schedule_rows[t] for t in (83, 250, 416)],
        "theoretical_low_high_ratio": {
            "original": schedule_rows[416]["inverse_sqrt_snr"] / schedule_rows[83]["inverse_sqrt_snr"],
            "compensated": schedule_rows[416]["compensated_amplification"] / schedule_rows[83]["compensated_amplification"],
        },
        "launch_eligible": device.type == "cuda",
    })
    required_uploads = [
        "src/models/station_conditioned_diffusion.py",
        "station_lightweight_tail.py",
        "configs/station24_lightweight_joint_tail_aux_timestep_compensation_168h.yaml",
        "tools/audit_station24_aux_timestep_compensation.py",
        "run_station24_lightweight_joint_tail_aux_timestep_compensation_formal.sh",
    ]
    hashes = {}
    for relative in required_uploads:
        data = (ROOT / relative).read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
    (output / "server_upload_manifest.json").write_text(json.dumps({
        "required": required_uploads,
        "prerequisite_already_on_server": ["run_station24_lightweight_joint_tail_finalize.sh"],
        "sha256": hashes,
    }, indent=2), encoding="utf-8")
    (output / "modified_file_manifest.json").write_text(json.dumps({
        "source_and_formal_files": required_uploads,
        "tests": ["tests/test_station24_aux_timestep_compensation.py"],
        "diagnostic_output": str(output),
    }, indent=2), encoding="utf-8")
    (output / "integrity_audit.json").write_text(json.dumps({
        "base_epsilon_changed": False,
        "auxiliary_coefficients": {"ramp": .18, "shape": .14, "slow": .10},
        "event_sampling_fraction": .60,
        "trainable_parameters": 20588,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "control_candidate_state_hash_equal": control_hash == candidate_hash,
        "cuda_amp": report["checks"]["CUDA_AMP_forward_backward"]["status"],
    }, indent=2), encoding="utf-8")
    implementation = subprocess.run(
        ["git", "diff", "--", "src/models/station_conditioned_diffusion.py", "station_lightweight_tail.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    (output / "implementation.diff").write_text(implementation, encoding="utf-8")
    control_yaml = yaml.safe_dump(expected_config(), sort_keys=False).splitlines()
    candidate_yaml = yaml.safe_dump(expected_config(auxiliary_timestep_compensation=True), sort_keys=False).splitlines()
    import difflib
    (output / "config.diff").write_text("\n".join(difflib.unified_diff(
        control_yaml, candidate_yaml, fromfile="control", tofile="candidate", lineterm=""
    )) + "\n", encoding="utf-8")
    (output / "server_launch_command.sh").write_text(
        "cd /root/autodl-tmp/DM\n"
        "bash -n run_station24_lightweight_joint_tail_aux_timestep_compensation_formal.sh\n"
        "nohup bash run_station24_lightweight_joint_tail_aux_timestep_compensation_formal.sh > /dev/null 2>&1 &\n",
        encoding="utf-8",
    )
    (output / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Auxiliary timestep compensation preflight", "",
        "Only ramp/shape/slow receive a fixed per-sample timestep multiplier before reduction; base epsilon is unchanged.",
        "The [0.25, 2.0] clipped sqrt(SNR) schedule is normalized to mean one and is not a search space.",
        f"Device: {device}; formal training: NOT RUN; generation: NOT RUN; optimizer step: NOT RUN.", "",
        "| Check | Status |", "|---|---|"
    ]
    lines += [f"| {name} | {value['status']} |" for name, value in report["checks"].items()]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"AUX_TIMESTEP_COMPENSATION_PREFLIGHT_COMPLETE output={output} NO FORMAL TRAINING")


if __name__ == "__main__":
    main()
