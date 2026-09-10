"""Fail closed before paid training or formal generation; never infer approval."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
        return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--pretrain", action="store_true")
    args = parser.parse_args()
    root = Path(args.pipeline_root)
    preflight = json.loads((root / "cuda_preflight.json").read_text())
    if preflight.get("status") != "PASS" or not preflight.get("cuda_amp_executed"):
        raise ValueError("target CUDA/AMP preflight has not passed")
    if args.pretrain:
        if not preflight.get("launch_eligible", False):
            raise ValueError("mandatory learning/performance evidence is incomplete; paid training blocked")
        return
    run = Path((root / "train_run.txt").read_text().strip())
    decision = json.loads((root / "development_decision.json").read_text())
    if decision.get("status") != "PASS" or decision.get("formal_launch_eligible") is not True:
        raise ValueError("development gate has not passed")
    if decision.get("checkpoint_sha256") != digest(run / "checkpoints/model_best.pt"):
        raise ValueError("development decision does not match the trained checkpoint")
    for name in ("development_tail_n20", "development_mixture_body80_tail20_n100"):
        evidence = root / name / "generation_metadata.json"
        if decision.get(name + "_metadata_sha256") != digest(evidence):
            raise ValueError("development evidence changed or is missing")
    if not decision.get("evidence"):
        raise ValueError("development decision must explain measured criteria")


if __name__ == "__main__":
    main()
