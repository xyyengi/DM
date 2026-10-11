"""Generate isolated Shandong91 V3 low-rank Raw Body scenarios."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
import yaml

from datasets.shandong91_faithful24 import Shandong91Faithful24Dataset, threshold_sha256
from datasets.shandong91_low_rank import factor_sha256, fit_train_only_pca
from train_shandong91_v2 import data_contract_record, move_batch
from train_shandong91_v3 import MODEL_ID, SUPPORTED_MODEL_IDS, build_model


def atomic_json(document, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), "utf-8")
    os.replace(temporary, path)


def atomic_torch_save(document, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary)
    os.replace(temporary, path)


def open_array(path, *, shape, dtype, resume):
    if resume:
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            array._mmap.close()
            raise ValueError(f"partial generation array contract changed: {path}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/shandong91/raw_body_v3_low_rank_common_factor.yaml")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--n-samples", type=int, default=500); parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--member-chunk", type=int, default=64); parser.add_argument("--method", choices=("ddpm", "ddim"))
    parser.add_argument("--inference-steps", type=int); parser.add_argument("--checkpoint-state", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--resume-output", action="store_true")
    parser.add_argument("--stop-after-windows", type=int,
                        help="bounded resume test only; stop after persisting this many windows")
    args = parser.parse_args(); output = Path(args.output_dir)
    if output.exists() and not args.resume_output:
        raise FileExistsError(f"refusing to overwrite V3 generation output: {output}")
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    config = yaml.safe_load(Path(args.config).read_text("utf-8"))
    model_id = config["model"]["architecture"]
    if model_id not in SUPPORTED_MODEL_IDS: raise ValueError("wrong V3/V3.1 config")
    checkpoint = Path(args.run_dir) / "checkpoints/best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("model_identifier") != model_id: raise ValueError("checkpoint model mismatch")
    if saved["config_snapshot"]["model"] != config["model"]: raise ValueError("checkpoint/config model mismatch")
    if saved.get("data_contract") != data_contract_record(config["data"]): raise ValueError("data contract mismatch")
    current_factor = fit_train_only_pca(config["data"]["data_path"], config["model"]["common_factor_counts"])
    if (factor_sha256(saved["factor_document"]) != saved["factor_document"].get("sha256")
            or current_factor["sha256"] != saved["factor_document"]["sha256"]):
        raise ValueError("train-only PCA factors changed")
    dataset = Shandong91Faithful24Dataset(config["data"]["data_path"], args.split, saved["state_thresholds"])
    if threshold_sha256(dataset.thresholds) != saved["state_threshold_sha256"]: raise ValueError("state threshold mismatch")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model = build_model(config, dataset, device, saved["factor_document"])
    model.load_state_dict(saved["ema_state_dict" if args.checkpoint_state == "ema" else "model_state_dict"], strict=True)
    model.eval(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)
    method = args.method or config["sampling"]["method"]
    steps = int(args.inference_steps or config["sampling"]["inference_steps"])
    if method == "ddpm" and steps != int(config["diffusion"]["num_steps"]): raise ValueError("DDPM requires full schedule")
    windows = min(len(dataset), args.max_windows) if args.max_windows else len(dataset)
    shape = (windows, args.n_samples, 168, 91, 3)
    static_shape = (windows,168,91,3)
    contract = {
        "version": (
            "shandong91_v3_window_resume_v2" if model_id == MODEL_ID
            else "shandong91_v3_1_window_resume_v1"
        ),
        "model_identifier":model_id,
        "projection_backend":"precomputed_masked_least_squares_v1",
        "checkpoint_epoch":int(saved["epoch"]),"checkpoint_state":args.checkpoint_state,
        "factor_document_sha256":saved["factor_document"]["sha256"],
        "split":args.split,"windows":windows,"n_samples":args.n_samples,"seed":args.seed,
        "member_chunk":args.member_chunk,"shape":list(shape),"method":method,"inference_steps":steps,
    }
    progress_state_path=output/"generation_progress.pt"; progress_json_path=output/"generation_progress.json"
    resuming=output.exists()
    if resuming:
        if (output/"metadata.json").exists():
            print(json.dumps({"state":"already_complete","output":str(output)}),flush=True); return
        if not progress_state_path.exists():
            raise RuntimeError("existing generation directory has no resumable progress state")
        progress=torch.load(progress_state_path,map_location="cpu",weights_only=False)
        if progress.get("contract")!=contract: raise ValueError("partial generation contract changed; refusing mixed output")
        completed=int(progress["completed_windows"])
    else:
        output.mkdir(parents=True)
        completed=0; progress={"contract":contract,"completed_windows":0,"generator_state":None}
    scenarios=open_array(output/"actual_scenarios_mw.npy",shape=shape,dtype=np.float32,resume=resuming)
    actual=open_array(output/"actual_mw.npy",shape=static_shape,dtype=np.float32,resume=resuming)
    forecast=open_array(output/"forecast_mw.npy",shape=static_shape,dtype=np.float32,resume=resuming)
    publication_mask=open_array(output/"valid_mask.npy",shape=static_shape,dtype=bool,resume=resuming)
    effective_mask=open_array(output/"effective_mask.npy",shape=static_shape,dtype=bool,resume=resuming)
    node = dataset.node_type_mask.to(device).view(1,1,91,3); train = dataset.channel_train_mask.to(device).view(1,1,91,3)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if resuming:
        generator.set_state(progress["generator_state"])
        print(json.dumps({"state":"resuming","completed_windows":completed,"total_windows":windows}),flush=True)
    else:
        atomic_torch_save(progress,progress_state_path); atomic_json({"state":"running","completed_windows":0,"total_windows":windows,"contract":contract},progress_json_path)
    for issue in range(completed,windows):
        batch = move_batch({k:v.unsqueeze(0) for k,v in dataset[issue].items()}, device)
        publication = node & batch["forecast_valid_mask"].bool(); generation = publication & train
        actual[issue] = dataset.denormalize_mw(batch["actual"])[0].cpu().numpy()
        forecast[issue] = dataset.denormalize_mw(batch["forecast"])[0].cpu().numpy()
        publication_mask[issue] = publication[0].cpu().numpy(); effective_mask[issue] = batch["effective_mask"][0].cpu().numpy()
        for start in range(0, args.n_samples, args.member_chunk):
            members = min(args.member_chunk, args.n_samples-start)
            expanded = {k:v.expand(members,*v.shape[1:]) for k,v in batch.items()}
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                residual = model.sample(expanded, generation.expand(members,-1,-1,-1), method=method,
                                        inference_steps=steps, generator=generator).float()
            generated = dataset.denormalize_mw(expanded["forecast"] + residual)
            generated.masked_fill_(~publication.expand_as(generated), 0)
            scenarios[issue,start:start+members] = generated.cpu().numpy()
            print(json.dumps({"phase":"generation","window":issue+1,"total_windows":windows,
                              "members_completed":start+members,"members_total":args.n_samples}),flush=True)
        for array in (scenarios,actual,forecast,publication_mask,effective_mask): array.flush()
        progress={"contract":contract,"completed_windows":issue+1,"generator_state":generator.get_state().cpu()}
        atomic_torch_save(progress,progress_state_path)
        atomic_json({"state":"running" if issue+1<windows else "generated","completed_windows":issue+1,
                     "total_windows":windows,"contract":contract},progress_json_path)
        print(json.dumps({"window":issue+1,"total":windows,"persisted":True}),flush=True)
        if (args.stop_after_windows is not None and issue+1 >= args.stop_after_windows
                and issue+1 < windows):
            print(json.dumps({"state":"bounded_stop","completed_windows":issue+1}),flush=True)
            return
    if not np.isfinite(scenarios).all(): raise FloatingPointError("V3 scenarios contain NaN/Inf")
    metadata = {
        "model_identifier":model_id,"checkpoint":str(checkpoint),"checkpoint_epoch":int(saved["epoch"]),
        "checkpoint_state":args.checkpoint_state,"split":args.split,"n_samples":args.n_samples,"seed":args.seed,
        "member_chunk":args.member_chunk,"shape":list(shape),"sampler":method,"inference_steps":steps,
        "residual_sign":"generated_actual = forecast + generated_residual",
        "factor_document_sha256":saved["factor_document"]["sha256"],
        "load_local_branch":"disabled_strict_rank1","data_contract":data_contract_record(config["data"]),
        "projection_backend":"precomputed_masked_least_squares_v1",
        "formal_objective":config["loss"]["formal_objective"],
        "physical_clipping":False,"bounded_engineering_generation":args.max_windows is not None,
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "resumable_generation":True,"completed_windows":windows,
    }
    atomic_json(metadata,output/"metadata.json")
    atomic_json({"state":"complete","completed_windows":windows,"total_windows":windows,
                 "contract":contract},progress_json_path)


if __name__ == "__main__": main()
