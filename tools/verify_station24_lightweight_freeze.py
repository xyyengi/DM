"""Verify every tensor inherited from the Raw checkpoint stayed bit-identical."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-checkpoint", required=True)
    p.add_argument("--trained-checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    raw = torch.load(a.raw_checkpoint, map_location="cpu", weights_only=False)
    trained = torch.load(a.trained_checkpoint, map_location="cpu", weights_only=False)
    source = raw["model_state_dict"]
    result = trained["model_state_dict"]
    missing = sorted(set(source) - set(result))
    changed = sorted(k for k in source if k in result and not torch.equal(source[k], result[k]))
    tail_names = trained.get("joint_multiresidual_trainable_parameter_names", [])
    tail_count = sum(result[n].numel() for n in tail_names)
    digest = hashlib.sha256()
    for name in sorted(source):
        digest.update(name.encode()); digest.update(result[name].contiguous().numpy().tobytes())
    report = {
        "status": "PASS" if not missing and not changed and tail_count == 20588 else "FAIL",
        "raw_shared_tensor_count": len(source), "missing": missing, "changed": changed,
        "tail_trainable_parameter_count": tail_count, "raw_state_sha256": digest.hexdigest(),
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "PASS": raise SystemExit(json.dumps(report))
    print("LIGHTWEIGHT_RAW_FREEZE_PASS", out)


if __name__ == "__main__": main()
