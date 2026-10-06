"""Short evidence-only preflight. Never launches train_station24 or formal generation."""
import argparse
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from station_lightweight_tail import validate_config
from station_dataset import (
    build_station_daylight_mask,
    load_station_static_data,
    get_station_dataloader,
)
from station_jstd_targets import fit_station_jstd_event_thresholds, build_station_jstd_target_arrays
from src.models.station_conditioned_diffusion import Station24DiffusionModel

RAW_RUN = "outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027"
PREFIX = "denoiser.joint_multiresidual_tail."


def audit_ramp_selection(dataset, station_features, fraction=.10):
    """Compare legacy and proposed selectors over the complete train split."""
    source_masks = {
        "wind": np.asarray(station_features[:, 0] > .5),
        "solar": np.asarray(station_features[:, 1] > .5),
    }
    result = {"legacy": {}, "proposed": {}, "physical_top_retention": {}}
    retained = {}
    for lag in (1, 3, 6):
        for source in source_masks:
            for direction in ("positive", "negative"):
                key = f"{source}_{direction}_{lag}h"
                result["legacy"][key] = {
                    "eligible": 0, "top_selected": 0, "focus_selected": 0,
                }
                result["proposed"][key] = {
                    "eligible": 0, "top_selected": 0, "focus_selected": 0,
                }
                retained[key] = {"value": [], "selected": []}
    daylight_excluded = {f"solar_{lag}h": 0 for lag in (1, 3, 6)}
    for sample_index in range(len(dataset)):
        row = dataset[sample_index]
        actual = row["actual"].numpy()
        valid = row["valid_mask"].numpy() > 0
        daylight = row["daylight_mask"].numpy() > 0
        support = row["jstd_event_station_support"].numpy()
        time_support = row["jstd_event_time_support"].numpy()[None, :]
        active = float(row["jstd_event_active"])
        event = np.maximum(support, .25 * time_support) * active
        event = F.max_pool1d(
            torch.from_numpy(event)[None], kernel_size=13, stride=1, padding=6
        )[0].numpy()
        for lag in (1, 3, 6):
            delta = actual[:, lag:] - actual[:, :-lag]
            pair_valid = valid[:, lag:] & valid[:, :-lag]
            local_event = np.maximum(event[:, lag:], event[:, :-lag]) > 0
            old_magnitude = np.where(pair_valid, np.abs(delta), -np.inf)
            old_k = max(1, int(np.ceil(delta.shape[-1] * fraction)))
            old_threshold = np.partition(
                old_magnitude, -old_k, axis=-1
            )[:, -old_k, None]
            old_top = (old_magnitude >= old_threshold) & pair_valid
            daylight_pair = daylight[:, lag:] & daylight[:, :-lag]
            daylight_excluded[f"solar_{lag}h"] += int(
                np.sum(
                    source_masks["solar"][:, None]
                    & pair_valid
                    & ~daylight_pair
                )
            )
            for source, stations in source_masks.items():
                source_valid = pair_valid & stations[:, None]
                if source == "solar":
                    source_valid &= daylight_pair
                for direction, sign_mask in (
                    ("positive", delta > 0), ("negative", delta < 0)
                ):
                    key = f"{source}_{direction}_{lag}h"
                    proposed_eligible = source_valid & sign_mask
                    legacy_eligible = pair_valid & stations[:, None] & sign_mask
                    proposed_top = np.zeros_like(proposed_eligible)
                    eligible_index = np.flatnonzero(proposed_eligible)
                    if eligible_index.size:
                        k = max(1, int(np.ceil(eligible_index.size * fraction)))
                        flat_values = np.abs(delta).reshape(-1)[eligible_index]
                        threshold = np.partition(flat_values, -k)[-k]
                        proposed_top.reshape(-1)[
                            eligible_index[flat_values >= threshold]
                        ] = True
                    directed_event = local_event & proposed_eligible
                    proposed_focus = proposed_top | directed_event
                    legacy_focus = old_top | (local_event & pair_valid)
                    legacy_selected = legacy_focus & legacy_eligible
                    result["legacy"][key]["eligible"] += int(legacy_eligible.sum())
                    result["legacy"][key]["top_selected"] += int(
                        (old_top & legacy_eligible).sum()
                    )
                    result["legacy"][key]["focus_selected"] += int(
                        legacy_selected.sum()
                    )
                    result["proposed"][key]["eligible"] += int(
                        proposed_eligible.sum()
                    )
                    result["proposed"][key]["top_selected"] += int(
                        proposed_top.sum()
                    )
                    result["proposed"][key]["focus_selected"] += int(
                        proposed_focus.sum()
                    )
                    # The loss is computed on physical normalized actual power,
                    # not standardized residuals. Audit that same target scale.
                    retained[key]["value"].append(
                        np.abs(delta)[proposed_eligible]
                    )
                    retained[key]["selected"].append(
                        proposed_top[proposed_eligible]
                    )
    for version in ("legacy", "proposed"):
        for values in result[version].values():
            values["top_fraction_of_eligible"] = (
                values["top_selected"] / values["eligible"]
                if values["eligible"] else None
            )
            values["focus_fraction_of_eligible"] = (
                values["focus_selected"] / values["eligible"]
                if values["eligible"] else None
            )
    for key, values in retained.items():
        magnitude = np.concatenate(values["value"])
        selected = np.concatenate(values["selected"])
        threshold = float(np.quantile(magnitude, .99))
        physical_top = magnitude >= threshold
        maximum = magnitude == magnitude.max()
        result["physical_top_retention"][key] = {
            "definition": (
                "global q99 of eligible absolute ramp on physical normalized "
                "actual-power scale"
            ),
            "threshold": threshold,
            "count": int(physical_top.sum()),
            "retained": int((selected & physical_top).sum()),
            "retention": float(selected[physical_top].mean()),
            "maximum_retained": bool(selected[maximum].all()),
        }
    result["solar_invalid_pair_exclusions"] = daylight_excluded
    return result


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if "joint_multiresidual_tail." not in name:
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/station24_lightweight_joint_tail_v2_fair_168h.yaml")
    p.add_argument("--checkpoint", default=RAW_RUN + "/checkpoints/model_best.pt")
    p.add_argument("--secondary-adjacency", default=RAW_RUN + "/graphs/secondary_adjacency.npy")
    p.add_argument("--data-path", default="diffusion_input_station")
    p.add_argument("--body-results", default="outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242")
    p.add_argument("--v2-mixture", default="outputs_shandong/station24/independent_joint_tail_v2_event_balanced_20260910_200653/mixture_body400_tail100_n500")
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = {"checks": {}, "launch_eligible": False, "formal_training_started": False,
              "device": args.device, "torch": torch.__version__, "formal_500_member_generation": "NOT RUN"}

    def check(name, value, evidence=None):
        report["checks"][name] = {"status": "PASS" if value else "FAIL", "evidence": evidence}
        if not value:
            raise AssertionError(name)

    try:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        validate_config(config)
        ramp_selection = (
            config["model"].get(
                "event_balanced_ramp_selection_version", "legacy_abs_topk_v1"
            )
            == "source_direction_daylight_pooled_v1"
        )
        check("V2_recipe_and_version", True)
        if Path(args.data_path).resolve() != Path(config["data"]["data_path"]).resolve():
            raise ValueError("data override differs from the V2 recipe")
        config["model"]["secondary_adjacency_path"] = str(Path(args.secondary_adjacency).resolve())
        (out / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        report["config_sha256"] = sha(args.config)
        report["config_used_sha256"] = sha(out / "config_used.yaml")
        wrapper_name = (
            "run_station24_auxiliary_alpha_stage1a.sh"
            if "auxiliary_strength_alpha" in config["model"]
            else
            "run_station24_lightweight_joint_tail_ramp_selection_preflight.sh"
            if config["model"].get("event_balanced_ramp_selection_version")
            == "source_direction_daylight_pooled_v1"
            else "run_station24_lightweight_joint_tail_preflight.sh"
        )
        report["wrapper"] = wrapper_name
        report["wrapper_sha256"] = sha(ROOT / wrapper_name)
        source_files = [
            "station_lightweight_tail.py", "src/models/station_conditioned_diffusion.py",
            "src/models/station_joint_multiresidual_tail.py", "train_station24.py", "generate_station24.py",
            "tools/audit_station24_lightweight_joint_tail.py",
        ]
        if ramp_selection:
            source_files += [
                "tools/diagnose_station24_ramp_distribution.py",
                "tools/diagnose_station24_ramp_dynamics.py",
                "tools/summarize_station24_ramp_mechanism.py",
                "run_station24_lightweight_joint_tail_ramp_selection_preflight.sh",
                "run_station24_lightweight_joint_tail_ramp_selection_formal.sh",
                "run_station24_lightweight_joint_tail_ramp_selection_finalize.sh",
            ]
        if "auxiliary_strength_alpha" in config["model"]:
            source_files += [
                "tools/materialize_station24_auxiliary_alpha_config.py",
                "tools/audit_station24_auxiliary_alpha_inputs.py",
                "tools/audit_station24_auxiliary_alpha_run.py",
                "tools/evaluate_station24_auxiliary_alpha_stage1a.py",
                "tools/check_station24_auxiliary_alpha_launch.py",
                "run_station24_auxiliary_alpha_candidate.sh",
                "run_station24_auxiliary_alpha_stage1a_finalize.sh",
                "run_station24_lightweight_joint_tail_finalize.sh",
                "tools/diagnose_station24_wind_direction_recovery.py",
                "tools/diagnose_station24_ramp_distribution.py",
            ]
        report["source_hashes"] = {str(f): sha(ROOT / f) for f in source_files}
        if args.device == "cuda" and not torch.cuda.is_available():
            report["checks"]["CUDA_AMP_forward_backward"] = {"status": "NOT RUN", "evidence": "CUDA unavailable"}
            raise RuntimeError("CUDA unavailable; CPU does not replace this gate")
        device = torch.device(args.device)
        torch.set_num_threads(4)
        torch.manual_seed(2027)
        static = load_station_static_data(args.data_path)
        graph = torch.as_tensor(np.load(args.secondary_adjacency), dtype=torch.float32)
        source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        report["checkpoint_sha256"] = sha(args.checkpoint)
        report["secondary_graph_sha256"] = sha(args.secondary_adjacency)
        from tools.merge_station24_independent_tail_members import MEMBER_ARRAYS
        body_hashes = {}
        for name in MEMBER_ARRAYS:
            body = np.load(Path(args.body_results) / name, mmap_mode="r")
            reference = np.load(Path(args.v2_mixture) / name, mmap_mode="r")
            check("same_V2_body400_" + name, body.shape == (23, 500, 168, 24)
                  and np.array_equal(body[:, :400], reference[:, :400]))
            body_hashes[name] = hashlib.sha256(np.ascontiguousarray(body[:, :400]).tobytes()).hexdigest()
        report["frozen_body400_hashes"] = body_hashes
        report["body_results"] = args.body_results

        def construct(c):
            return Station24DiffusionModel(c, static["station_features"], static["station_adjacency"],
                                           static["station_capacities"], graph).to(device)

        model = construct(config["model"])
        incompatible = model.load_state_dict(source["model_state_dict"], strict=False)
        check("Raw_initialization_complete", set(incompatible.missing_keys) == set(model.joint_multiresidual_new_state_dict_keys)
              and not incompatible.unexpected_keys)
        raw = construct(source["config"]["model"])
        raw.load_state_dict(source["model_state_dict"], strict=True)
        raw.eval()
        model.train()
        trainable = {n: v for n, v in model.named_parameters() if v.requires_grad}
        count = sum(v.numel() for v in trainable.values())
        frozen_count = sum(v.numel() for v in model.parameters() if not v.requires_grad)
        report["auxiliary_strength_alpha"] = float(
            config["model"].get("auxiliary_strength_alpha", 1.0)
        )
        report["actual_auxiliary_weights"] = {
            "ramp": float(config["model"]["event_balanced_ramp_loss_weight"]),
            "shape": float(config["model"]["event_balanced_shape_loss_weight"]),
            "slow": float(config["model"]["event_balanced_slow_loss_weight"]),
        }
        report["trainable_parameter_names"] = sorted(trainable)
        report["trainable_parameter_count"] = count
        report["frozen_parameter_count"] = frozen_count
        check("trainable_parameters", count == 20588 and all(n.startswith(PREFIX) for n in trainable), count)
        check("Raw_eval", all(not m.training for n, m in model.named_modules()
                              if "joint_multiresidual_tail" not in n))
        before_hash = frozen_hash(model)
        before_tail = {n: v.detach().clone() for n, v in trainable.items()}
        thresholds = fit_station_jstd_event_thresholds(args.data_path, config["model"])
        targets = build_station_jstd_target_arrays(args.data_path, "train", thresholds,
                                                   event_sampling_target_fraction=.60)
        loader, dataset = get_station_dataloader(args.data_path, "train", source["residual_scale"],
            batch_size=8, seed=2027, condition_config=config["model"],
            state_thresholds=source["state_thresholds"], jstd_targets=targets)
        if config["model"].get("event_balanced_ramp_selection_version") == "source_direction_daylight_pooled_v1":
            _, daylight_audit = build_station_daylight_mask(args.data_path, "train")
            report["daylight_mask"] = daylight_audit
            selection_audit = audit_ramp_selection(
                dataset,
                static["station_features"].numpy(),
                fraction=float(config["model"]["event_balanced_ramp_top_fraction"]),
            )
            report["ramp_selection"] = selection_audit
            for key, values in selection_audit["proposed"].items():
                check("ramp_selection_nonempty_" + key, values["top_selected"] > 0, values)
            for key, values in selection_audit["physical_top_retention"].items():
                check(
                    "physical_top_retained_" + key,
                    values["maximum_retained"] and values["retention"] >= .95,
                    values,
                )
        from torch.utils.data import WeightedRandomSampler
        check("actual_sampler_type", isinstance(loader.sampler, WeightedRandomSampler))
        draws = [np.asarray(list(loader.sampler)) for _ in range(20)]
        rates = [float(np.asarray(targets.event_active)[d].mean()) for d in draws]
        expected = float(np.dot(targets.sample_weights, targets.event_active) / targets.sample_weights.sum())
        check("actual_sampler_event_fraction", abs(np.mean(rates) - .60) < .03 and abs(expected - .60) < 1e-5,
              {"expected": expected, "first_epoch": rates[0], "20_epoch_mean": float(np.mean(rates)), "rates": rates})
        # All four physical strata plus an ordinary issue; training labels are
        # used only as supervision, never as denoiser inputs.
        indices = []
        for kind in ("wind", "solar"):
            for direction in ("negative", "positive"):
                choices = [int(r["sample_index"]) for r in targets.catalog if r["source"] == kind and r["direction"] == direction]
                check("stratum_" + kind + "_" + direction, bool(choices))
                indices.append(choices[0])
        indices.append(int(np.flatnonzero(np.asarray(targets.event_active) == 0)[0]))
        rows = [dataset[i] for i in indices]
        batch = {k: torch.stack([r[k] for r in rows]).to(device) for k in rows[0] if k != "sample_index"}
        report["train_issue_indices"] = indices
        noise = torch.randn_like(batch["residual_target"])
        timestep = torch.full((len(indices),), 100, dtype=torch.long, device=device)
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        amp = lambda: torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        model.diffusion.capture_loss_graph_for_audit = True
        history = []
        for step in range(6):
            optimizer.zero_grad(set_to_none=True)
            with amp():
                loss = model(batch, timestep=timestep, noise=noise)
            check("finite_forward_step_" + str(step), bool(torch.isfinite(loss)))
            components = model.diffusion.last_loss_components
            ramp_weight = float(config["model"]["event_balanced_ramp_loss_weight"])
            shape_weight = float(config["model"]["event_balanced_shape_loss_weight"])
            slow_weight = float(config["model"]["event_balanced_slow_loss_weight"])
            expected_loss = components["epsilon"] + ramp_weight * components["event_balanced_ramp"] + shape_weight * components["event_balanced_shape"] + slow_weight * components["event_balanced_slow"]
            check("exact_V2_loss_step_" + str(step), bool(torch.allclose(loss.detach(), expected_loss, atol=1e-7, rtol=1e-6)))
            check("old_losses_zero_step_" + str(step), all(float(components[k]) == 0 for k in components if k.startswith("jstd_")))
            history.append({k: float(v) for k, v in components.items()})
            print(f"PREFLIGHT_STEP {step + 1}/6 loss={float(loss.detach()):.8f}", flush=True)
            if step == 5:
                aux_gradients = {}
                for key in ("ramp", "shape", "slow"):
                    scalar = model.diffusion.audit_loss_tensors[key]
                    gradients = torch.autograd.grad(scalar, list(trainable.values()), retain_graph=True, allow_unused=True)
                    norm = sum(float(g.float().square().sum()) for g in gradients if g is not None) ** .5
                    aux_gradients[key] = {"loss": float(scalar.detach()), "gradient_norm": norm}
                    check("auxiliary_" + key + "_gradient", float(scalar.detach()) > 0 and np.isfinite(norm) and norm > 0, aux_gradients[key])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            check("Raw_zero_grad_step_" + str(step), all(v.grad is None or torch.count_nonzero(v.grad) == 0
                  for n, v in model.named_parameters() if not n.startswith(PREFIX)))
            check("finite_tail_grad_step_" + str(step), all(v.grad is None or torch.isfinite(v.grad).all() for v in trainable.values()))
            torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0)
            scaler.step(optimizer)
            scaler.update()
        report["six_step_loss_history"] = history
        report["parameter_gradient_norms"] = {n: float(v.grad.norm()) if v.grad is not None else None for n, v in trainable.items()}
        report["parameter_update_norms"] = {n: float((v.detach() - before_tail[n]).norm()) for n, v in trainable.items()}
        # Every submodule/head must learn. Record per-tensor details as well,
        # including possible projection-null constant biases (never hide them).
        groups = {n.split(".")[2] for n in trainable}
        check("all_tail_modules_updated", all(any(report["parameter_update_norms"][n] > 0 for n in trainable if n.split(".")[2] == g) for g in groups))
        report["unchanged_tail_tensors"] = [n for n, v in report["parameter_update_norms"].items() if v == 0]
        check("every_tail_parameter_updated", not report["unchanged_tail_tensors"], report["unchanged_tail_tensors"])
        check("every_tail_parameter_gradient_nonzero", all(v is not None and np.isfinite(v) and v > 0
              for v in report["parameter_gradient_norms"].values()))
        check("Raw_parameters_and_buffers_unchanged", before_hash == frozen_hash(model), {"before": before_hash, "after": frozen_hash(model)})
        check("CPU_forward_backward" if device.type == "cpu" else "CUDA_AMP_forward_backward", True)
        if device.type == "cpu":
            report["checks"]["CUDA_AMP_forward_backward"] = {"status": "NOT RUN", "evidence": "local CPU-only environment"}
        model.eval()
        def epsilon(net, route, b=batch):
            return net.denoiser(noise, timestep, b["forecast"], b["calendar"], b["lead"],
                recent_error=b["recent_error"], recent_error_mask=b["recent_error_mask"],
                node_state=b["node_state"], tail_expert_route=route)
        with torch.no_grad():
            check("disabled_tail_strict_Raw_epsilon", torch.equal(epsilon(model, 0.), epsilon(raw, 0.)))
            prediction = []
            hook = model.denoiser.register_forward_hook(lambda module, args, result: prediction.append(result[0] if isinstance(result, tuple) else result))
            plain = model(batch, timestep=timestep, noise=noise, include_auxiliary=False)
            hook.remove()
            valid = batch["valid_mask"]
            full_epsilon = ((prediction[0] - noise).square() * valid).sum() / valid.sum().clamp(min=1)
            check("epsilon_all_valid_not_event_masked", torch.equal(plain, full_epsilon))
            blob = io.BytesIO()
            torch.save({"config": config, "model_state_dict": model.state_dict()}, blob)
            blob.seek(0)
            saved = torch.load(blob, map_location=device, weights_only=False)
            reloaded = construct(saved["config"]["model"])
            reloaded.load_state_dict(saved["model_state_dict"], strict=True)
            reloaded.eval()
            check("save_reload_epsilon", torch.equal(epsilon(model, 1.), epsilon(reloaded, 1.)))
            # Run the real generate() routing for 100 members without pretending
            # this mock diffusion is a completed 500-step generation.
            def fake_sample(forecast, calendar, lead, n_samples, **kw):
                check("all_100_tail_routes_enabled", n_samples == 100 and torch.all(kw["tail_expert_route"] == 1).item())
                return forecast.new_zeros(forecast.shape[0], n_samples, forecast.shape[1], forecast.shape[2])
            with patch.object(model.diffusion, "sample", side_effect=fake_sample):
                model.generate(batch, n_samples=100)
            tiny = {k: v[:1] for k, v in batch.items()}
            z = torch.randn(1, 2, 24, 168, device=device)
            def sample(net, route):
                return net.diffusion.differentiable_ddim_sample(tiny["forecast"], tiny["calendar"], tiny["lead"],
                    n_samples=2, sampling_steps=8, backprop_steps=1, initial_noise=z,
                    recent_error=tiny["recent_error"], recent_error_mask=tiny["recent_error_mask"],
                    node_state=tiny["node_state"], tail_expert_route=torch.full((1, 2), route, device=device))
            check("disabled_tail_strict_Raw_sample", torch.equal(sample(model, 0.), sample(raw, 0.)))
            check("enabled_tail_short_sampling_finite", torch.isfinite(sample(model, 1.)).all().item())
            # Use the public generation interface with a short deterministic
            # sampler to check labels do not leak into generation conditions.
            def short_sample(forecast, calendar, lead, n_samples, **kw):
                return model.diffusion.differentiable_ddim_sample(forecast, calendar, lead,
                    n_samples=n_samples, sampling_steps=8, backprop_steps=1, initial_noise=z,
                    recent_error=kw["recent_error"], recent_error_mask=kw["recent_error_mask"],
                    node_state=kw["node_state"], tail_expert_route=kw["tail_expert_route"])
            changed = {k: (torch.randn_like(v) if v.is_floating_point() and
                       (k in ("actual", "residual", "residual_target") or k.startswith("jstd_")) else v)
                       for k, v in tiny.items()}
            with patch.object(model.diffusion, "sample", side_effect=short_sample):
                check("future_labels_do_not_change_generation", torch.equal(model.generate(tiny, n_samples=2), model.generate(changed, n_samples=2)))
            _, decomposition = model.denoiser(noise, timestep, batch["forecast"], batch["calendar"], batch["lead"],
                recent_error=batch["recent_error"], recent_error_mask=batch["recent_error_mask"],
                node_state=batch["node_state"], tail_expert_route=1., return_jstd_audit=True)
            slow, fast = decomposition.slow_correction.float(), decomposition.fast_correction.float()
            energy = float(slow.square().sum() + fast.square().sum())
            overlap = float((slow * fast).sum().abs()) / max(energy, 1e-30)
            check("slow_fast_active_and_Haar_complementary", slow.square().sum() > 0 and fast.square().sum() > 0 and overlap < 1e-5,
                  {"slow_rms": float(slow.square().mean().sqrt()), "fast_rms": float(fast.square().mean().sqrt()), "normalized_inner_product": overlap})
        report["preflight_execution"] = "PASS"
        report["launch_eligible"] = False
        report["reason"] = "Preflight only; formal training requires separate user confirmation and all target-device gates."
    except Exception as exc:
        report["preflight_execution"] = "FAIL"
        report["error"] = repr(exc)
        raise
    finally:
        (out / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        lines = ["# Lightweight Joint Tail preflight", "", f"Device: {args.device}; formal training: NOT RUN; launch eligible: false", "", "| Check | Status |", "|---|---|"]
        lines += [f"| {k} | {v['status']} |" for k, v in report["checks"].items()]
        if "daylight_mask" in report:
            daylight = report["daylight_mask"]
            lines += [
                "", "## Solar daylight-pair definition", "",
                "A solar ramp pair is eligible only when both endpoints have solar elevation "
                f"> {daylight['elevation_threshold_deg']} degrees at the lead-hour midpoint "
                f"(+{daylight['timestamp_offset_minutes']:.0f} minutes), using station latitude/longitude "
                f"and {daylight['timezone']}. No power or actual value is used to construct this mask.",
                "", "## Legacy versus proposed ramp selection", "",
                "| Stratum | Legacy eligible | Legacy top | Legacy focus | Proposed eligible | Proposed top | Proposed focus | Proposed top/eligible |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            legacy = report["ramp_selection"]["legacy"]
            proposed = report["ramp_selection"]["proposed"]
            for key in sorted(proposed):
                old, new = legacy[key], proposed[key]
                lines.append(
                    f"| {key} | {old['eligible']} | {old['top_selected']} | {old['focus_selected']} | "
                    f"{new['eligible']} | {new['top_selected']} | {new['focus_selected']} | "
                    f"{new['top_fraction_of_eligible']:.2%} |"
                )
            lines += [
                "", "## Physical top-ramp retention", "",
                "The check uses q99 absolute ramps on the physical normalized actual-power scale "
                "used by the auxiliary loss; this is not standardized residual space.", "",
                "| Stratum | q99 threshold | q99 retained | Maximum retained |",
                "|---|---:|---:|---|",
            ]
            for key, values in sorted(
                report["ramp_selection"]["physical_top_retention"].items()
            ):
                lines.append(
                    f"| {key} | {values['threshold']:.6f} | {values['retention']:.2%} | "
                    f"{values['maximum_retained']} |"
                )
        if "error" in report:
            lines += ["", report["error"]]
        (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("LIGHTWEIGHT_PREFLIGHT_COMPLETE", out, "NO FORMAL TRAINING", flush=True)


if __name__ == "__main__":
    main()
