"""Build an isolated Shandong91 solar-nonnegative *modeling* data version.

This is an explicit modeling projection, not a claim that negative REAL_POWER
measurements are erroneous.  Projection happens in 15-minute MW space, then
hourly and 168-hour products are rebuilt from the projected actual values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

CHANNELS = ("Wind", "Solar", "Load")
SOLAR = 1
SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scales(root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    document = json.loads((root / "normalization_params.json").read_text("utf-8"))
    scales = np.zeros((91, 3), dtype=np.float32)
    index = {name: value for value, name in enumerate(CHANNELS)}
    for record in document["records"]:
        if record.get("scale") is not None:
            scales[int(record["node_id"]) - 1, index[record["resource"]]] = float(record["scale"])
    return scales, document


def timestamps(path: Path, column: str | None = None) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    key = column or next(iter(rows[0]))
    return [row[key] for row in rows]


def save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def masked_max(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.max(np.abs(values[mask]))) if mask.any() else 0.0


def build(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite data version: {output}")
    if source.resolve() == output.resolve():
        raise ValueError("source and output must differ")
    shutil.copytree(source, output)

    node_type = np.load(source / "static/node_type_mask.npy").astype(bool)
    train_enabled = np.load(source / "static/channel_train_mask.npy").astype(bool)
    active_solar = node_type[:, SOLAR] & train_enabled[:, SOLAR]
    scales, normalization = load_scales(source)
    if float(scales[47, SOLAR]) != 172.0:
        raise ValueError("node 48 Solar scale must remain the existing 172 MW convention")

    actual_mw = np.load(source / "15min/actual_mw.npy")
    forecast_mw = np.load(source / "15min/forecast_mw.npy", mmap_mode="r")
    actual_valid = np.load(source / "15min/actual_valid_mask.npy").astype(bool)
    forecast_valid = np.load(source / "15min/forecast_valid_mask.npy").astype(bool)
    residual_valid = np.load(source / "15min/residual_valid_mask.npy").astype(bool)
    before_actual = actual_mw[..., SOLAR].copy()
    projection_mask = (
        actual_valid[..., SOLAR] & active_solar[None, :] &
        (actual_mw[..., SOLAR] < 0)
    )
    row_index, node_index = np.nonzero(projection_mask)
    original_values = actual_mw[row_index, node_index, SOLAR].copy()
    actual_mw[row_index, node_index, SOLAR] = 0.0
    save_array(output / "15min/actual_mw.npy", actual_mw)

    actual_normalized = np.load(source / "15min/actual_normalized.npy")
    solar_scale = scales[:, SOLAR]
    active_positions = actual_valid[..., SOLAR] & active_solar[None, :]
    actual_normalized[..., SOLAR][active_positions] = (
        actual_mw[..., SOLAR][active_positions] /
        np.broadcast_to(solar_scale, active_positions.shape)[active_positions]
    )
    save_array(output / "15min/actual_normalized.npy", actual_normalized)

    residual_mw = np.load(source / "15min/residual_mw.npy")
    update_residual = residual_valid[..., SOLAR] & active_solar[None, :]
    residual_mw[..., SOLAR][update_residual] = (
        actual_mw[..., SOLAR][update_residual] -
        np.asarray(forecast_mw[..., SOLAR])[update_residual]
    )
    save_array(output / "15min/residual_mw.npy", residual_mw)
    residual_normalized = np.load(source / "15min/residual_normalized.npy")
    residual_normalized[..., SOLAR][update_residual] = (
        residual_mw[..., SOLAR][update_residual] /
        np.broadcast_to(solar_scale, update_residual.shape)[update_residual]
    )
    save_array(output / "15min/residual_normalized.npy", residual_normalized)

    record_path = output / "reports/solar_nonnegative_projection_records.npz"
    np.savez_compressed(
        record_path,
        time_index=row_index.astype(np.int32), node_id=(node_index + 1).astype(np.uint8),
        original_actual_mw=original_values.astype(np.float32),
        projected_actual_mw=np.zeros(len(row_index), dtype=np.float32),
        correction_mw=(-original_values).astype(np.float32),
        reason_code=np.full(len(row_index), 1, dtype=np.uint8),
    )
    save_array(output / "15min/solar_actual_projection_mask.npy", projection_mask)

    # Reaggregate projected actuals in MW space using the published >=2/4 rule.
    quarter_actual = actual_mw.reshape(8760, 4, 91, 3)
    quarter_valid = actual_valid.reshape(8760, 4, 91, 3)
    counts = quarter_valid.sum(axis=1)
    hourly_actual = np.load(source / "hourly/actual_mw.npy")
    numerator = (quarter_actual[..., SOLAR] * quarter_valid[..., SOLAR]).sum(axis=1)
    valid_hour_solar = counts[..., SOLAR] >= 2
    rebuilt_solar = np.zeros((8760, 91), dtype=np.float32)
    np.divide(numerator, counts[..., SOLAR], out=rebuilt_solar, where=valid_hour_solar)
    hourly_update = np.broadcast_to(active_solar, rebuilt_solar.shape)
    hourly_actual[..., SOLAR][hourly_update] = rebuilt_solar[hourly_update]
    save_array(output / "hourly/actual_mw.npy", hourly_actual)

    hourly_actual_normalized = np.load(source / "hourly/actual_normalized.npy")
    hourly_actual_valid = np.load(source / "hourly/actual_valid_mask.npy").astype(bool)
    hourly_actual_positions = hourly_actual_valid[..., SOLAR] & active_solar[None, :]
    hourly_actual_normalized[..., SOLAR][hourly_actual_positions] = (
        hourly_actual[..., SOLAR][hourly_actual_positions] /
        np.broadcast_to(solar_scale, hourly_actual_positions.shape)[hourly_actual_positions]
    )
    save_array(output / "hourly/actual_normalized.npy", hourly_actual_normalized)

    hourly_forecast = np.load(source / "hourly/forecast_mw.npy", mmap_mode="r")
    hourly_residual_valid = np.load(source / "hourly/residual_valid_mask.npy").astype(bool)
    hourly_residual = np.load(source / "hourly/residual_mw.npy")
    hourly_residual_positions = hourly_residual_valid[..., SOLAR] & active_solar[None, :]
    hourly_residual[..., SOLAR][hourly_residual_positions] = (
        hourly_actual[..., SOLAR][hourly_residual_positions] -
        np.asarray(hourly_forecast[..., SOLAR])[hourly_residual_positions]
    )
    save_array(output / "hourly/residual_mw.npy", hourly_residual)
    hourly_residual_normalized = np.load(source / "hourly/residual_normalized.npy")
    hourly_residual_normalized[..., SOLAR][hourly_residual_positions] = (
        hourly_residual[..., SOLAR][hourly_residual_positions] /
        np.broadcast_to(solar_scale, hourly_residual_positions.shape)[hourly_residual_positions]
    )
    save_array(output / "hourly/residual_normalized.npy", hourly_residual_normalized)

    hour_times = timestamps(source / "hourly/timestamps.csv")
    hour_lookup = {value: index for index, value in enumerate(hour_times)}
    split_report = {}
    for split in SPLITS:
        starts = timestamps(source / f"windows168/{split}/window_starts.csv", "window_start")
        starts_index = np.asarray([hour_lookup[value] for value in starts])
        source_dir = source / "windows168" / split
        target_dir = output / "windows168" / split
        for name, hourly in (
            ("actual_mw", hourly_actual), ("actual", hourly_actual_normalized),
            ("residual_mw", hourly_residual), ("residual", hourly_residual_normalized),
        ):
            windows = np.load(source_dir / f"{name}.npy")
            for issue, start in enumerate(starts_index):
                windows[issue, ..., SOLAR][:, active_solar] = hourly[start:start + 168, :, SOLAR][:, active_solar]
            save_array(target_dir / f"{name}.npy", windows)
        window_actual = np.load(target_dir / "actual_mw.npy", mmap_mode="r")
        window_mask = np.load(target_dir / "residual_valid_mask.npy", mmap_mode="r").astype(bool)
        effective = window_mask[..., SOLAR] & active_solar[None, None, :]
        source_window_actual = np.load(source_dir / "actual_mw.npy", mmap_mode="r")
        corrected = effective & (np.asarray(source_window_actual[..., SOLAR]) < 0)
        split_report[split] = {
            "effective_elements": int(effective.sum()),
            "negative_before": int((effective & (np.asarray(source_window_actual[..., SOLAR]) < 0)).sum()),
            "negative_after": int((effective & (np.asarray(window_actual[..., SOLAR]) < 0)).sum()),
            "corrected_window_elements_including_overlap": int(corrected.sum()),
            "total_upward_correction_mw_including_overlap": float(
                np.maximum(-np.asarray(source_window_actual[..., SOLAR]), 0)[corrected].sum()
            ),
        }

    # Version metadata; scales remain unchanged, including node 48 = 172 MW.
    metadata = json.loads((source / "preprocessing_metadata.json").read_text("utf-8"))
    metadata.update({
        "version": "reliable_channel_training_v2_solar_nonnegative",
        "parent_version": metadata.get("version", source.name),
        "transformation": "solar_nonnegative_projection",
        "transformation_claim": "modeling convention; not a confirmed measurement-error correction",
        "projection_rule": "on valid train-enabled existing Solar actual in 15-minute MW space: max(actual_raw, 0)",
        "forecast_modified": False, "mask_modified": False,
        "wind_modified": False, "load_modified": False,
        "node48_solar_capacity_mw": 172.0,
        "node48_capacity_note": "unit-detail sum retained; alternate reported value 254.56 MW remains unresolved",
        "projection_record": "reports/solar_nonnegative_projection_records.npz",
    })
    (output / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    normalization.update({
        "data_version": "reliable_channel_training_v2_solar_nonnegative",
        "solar_actual_projection": "max(valid train-enabled existing Solar actual MW, 0) before normalization",
        "scales_refit": False,
        "scale_note": "Renewable capacity scales unchanged; node 48 remains 172 MW",
    })
    (output / "normalization_params.json").write_text(json.dumps(normalization, indent=2), "utf-8")

    source_hourly_actual = np.load(source / "hourly/actual_mw.npy", mmap_mode="r")
    hourly_effective = hourly_residual_valid[..., SOLAR] & active_solar[None, :]
    audit = {
        "data_version": "reliable_channel_training_v2_solar_nonnegative",
        "parent": str(source), "rule": "solar_nonnegative_projection",
        "semantic_claim": "explicit modeling convention only",
        "projection_domain": "15-minute actual_valid & node_type & channel_train Solar observations",
        "source_output_manifest_sha256": sha256(source / "output_manifest.json"),
        "modified_15min_observations": int(projection_mask.sum()),
        "total_upward_correction_mw_samples": float((-original_values).sum()),
        "total_upward_correction_mwh": float((-original_values).sum() * 0.25),
        "absolute_correction_mw": {
            key: float(value) for key, value in zip(
                ("q50", "q90", "q95", "q99", "max"),
                np.quantile(-original_values, [.5, .9, .95, .99, 1]),
            )
        },
        "unique_hourly": {
            "negative_before": int((hourly_effective & (np.asarray(source_hourly_actual[..., SOLAR]) < 0)).sum()),
            "negative_after": int((hourly_effective & (hourly_actual[..., SOLAR] < 0)).sum()),
        },
        "windows168": split_report,
        "node48_capacity": {"used_mw": 172.0, "alternate_reported_mw": 254.56, "changed": False},
        "masks_changed": False, "solar_forecast_changed": False,
        "wind_changed": False, "load_changed": False,
        "reason_codes": {"1": "valid train-enabled Solar actual was negative; projected to zero by modeling convention"},
    }
    (output / "projection_audit.json").write_text(json.dumps(audit, indent=2), "utf-8")
    readme = (source / "README.md").read_text("utf-8")
    readme += (
        "\n\n## V2 Solar nonnegative modeling projection\n\n"
        "This independent version applies `max(Solar actual MW, 0)` only to valid, "
        "train-enabled, existing Solar observations. It is a modeling convention, not "
        "a confirmed measurement-error repair. Solar forecast and every validity mask "
        "are unchanged. Wind and Load are unchanged. Node 48 keeps the 172 MW "
        "unit-detail-sum normalization scale; 254.56 MW remains an unresolved alternate.\n"
    )
    (output / "README.md").write_text(readme, "utf-8")

    return audit


def verify(source: Path, output: Path) -> dict[str, Any]:
    node_type = np.load(output / "static/node_type_mask.npy").astype(bool)
    train_enabled = np.load(output / "static/channel_train_mask.npy").astype(bool)
    active = node_type[:, SOLAR] & train_enabled[:, SOLAR]
    scales, _ = load_scales(output)
    checks: dict[str, dict[str, Any]] = {}
    def record(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"status": "PASS" if passed else "FAIL", "detail": detail}

    for resolution in ("15min", "hourly"):
        actual = np.load(output / f"{resolution}/actual_mw.npy", mmap_mode="r")
        forecast = np.load(output / f"{resolution}/forecast_mw.npy", mmap_mode="r")
        residual = np.load(output / f"{resolution}/residual_mw.npy", mmap_mode="r")
        actual_n = np.load(output / f"{resolution}/actual_normalized.npy", mmap_mode="r")
        forecast_n = np.load(output / f"{resolution}/forecast_normalized.npy", mmap_mode="r")
        residual_n = np.load(output / f"{resolution}/residual_normalized.npy", mmap_mode="r")
        av = np.load(output / f"{resolution}/actual_valid_mask.npy", mmap_mode="r").astype(bool)
        rv = np.load(output / f"{resolution}/residual_valid_mask.npy", mmap_mode="r").astype(bool)
        effective = rv[..., SOLAR] & active[None, :]
        record(f"{resolution}_solar_nonnegative", not bool((np.asarray(actual[..., SOLAR])[effective] < 0).any()), {"effective": int(effective.sum())})
        identity = np.asarray(residual) - (np.asarray(actual) - np.asarray(forecast))
        identity_error = masked_max(identity, rv)
        record(f"{resolution}_residual_identity", identity_error == 0.0, {"max_abs_mw": identity_error})
        roundtrip = np.asarray(actual_n) * scales[None] - np.asarray(actual)
        record(f"{resolution}_actual_roundtrip", masked_max(roundtrip, av) <= 1e-3, {"max_abs_mw": masked_max(roundtrip, av)})
        normalized_identity = np.asarray(residual_n) - (np.asarray(actual_n) - np.asarray(forecast_n))
        record(f"{resolution}_normalized_residual_identity", masked_max(normalized_identity, rv) <= 1e-6, {"max_abs": masked_max(normalized_identity, rv)})
        for channel, name in ((0, "wind"), (2, "load")):
            same = np.array_equal(np.asarray(actual[..., channel]), np.load(source / f"{resolution}/actual_mw.npy", mmap_mode="r")[..., channel])
            record(f"{resolution}_{name}_actual_unchanged", same, same)
        record(f"{resolution}_solar_forecast_unchanged", np.array_equal(np.asarray(forecast[..., SOLAR]), np.load(source / f"{resolution}/forecast_mw.npy", mmap_mode="r")[..., SOLAR]), True)
        for mask_name in ("actual_valid_mask", "forecast_valid_mask", "residual_valid_mask"):
            same = np.array_equal(np.load(output / f"{resolution}/{mask_name}.npy"), np.load(source / f"{resolution}/{mask_name}.npy"))
            record(f"{resolution}_{mask_name}_unchanged", same, same)

    for split in SPLITS:
        target_dir = output / "windows168" / split; source_dir = source / "windows168" / split
        actual = np.load(target_dir / "actual_mw.npy", mmap_mode="r")
        forecast = np.load(target_dir / "forecast_mw.npy", mmap_mode="r")
        residual = np.load(target_dir / "residual_mw.npy", mmap_mode="r")
        mask = np.load(target_dir / "residual_valid_mask.npy", mmap_mode="r").astype(bool)
        effective = mask[..., SOLAR] & active[None, None, :]
        record(f"{split}_solar_nonnegative", not bool((np.asarray(actual[..., SOLAR])[effective] < 0).any()), int(effective.sum()))
        record(f"{split}_residual_identity", masked_max(np.asarray(residual) - (np.asarray(actual) - np.asarray(forecast)), mask) == 0.0, {"max_abs_mw": masked_max(np.asarray(residual) - (np.asarray(actual) - np.asarray(forecast)), mask)})
        for channel, name in ((0, "wind"), (2, "load")):
            same = np.array_equal(np.asarray(actual[..., channel]), np.load(source_dir / "actual_mw.npy", mmap_mode="r")[..., channel])
            record(f"{split}_{name}_unchanged", same, same)
        record(f"{split}_solar_forecast_unchanged", np.array_equal(np.asarray(forecast[..., SOLAR]), np.load(source_dir / "forecast_mw.npy", mmap_mode="r")[..., SOLAR]), True)
        for mask_name in ("actual_valid_mask", "forecast_valid_mask", "residual_valid_mask"):
            same = np.array_equal(np.load(target_dir / f"{mask_name}.npy"), np.load(source_dir / f"{mask_name}.npy"))
            record(f"{split}_{mask_name}_unchanged", same, same)

    # Every modified observation is recorded and maps to an actual zero.
    records = np.load(output / "reports/solar_nonnegative_projection_records.npz")
    projected = np.load(output / "15min/actual_mw.npy", mmap_mode="r")
    indexed = np.asarray(projected)[records["time_index"], records["node_id"].astype(int) - 1, SOLAR]
    record("projection_records_complete", len(indexed) == int(np.load(output / "15min/solar_actual_projection_mask.npy").sum()) and bool((indexed == 0).all()), {"records": len(indexed)})
    record("node48_scale_172", float(scales[47, SOLAR]) == 172.0, float(scales[47, SOLAR]))

    status = "PASS" if all(value["status"] == "PASS" for value in checks.values()) else "FAIL"
    report = {"status": status, "data_version": "reliable_channel_training_v2_solar_nonnegative", "checks": checks}
    (output / "projection_verification.json").write_text(json.dumps(report, indent=2), "utf-8")
    if status != "PASS":
        raise RuntimeError("Solar nonnegative projection verification failed")
    return report


def write_quality_and_manifest(output: Path, verification: dict[str, Any]) -> None:
    quality = json.loads((output / "quality_report.json").read_text("utf-8"))
    quality.update({
        "data_version": "reliable_channel_training_v2_solar_nonnegative",
        "parent_quality_checks_retained": 20,
        "projection_verification": verification,
        "all_checks_passed": bool(quality.get("all_checks_passed")) and verification["status"] == "PASS",
    })
    for check in quality["checks"]:
        if check["name"] == "MW residual relation": check["detail"] = "recomputed after Solar MW projection; max error=0"
        if check["name"] == "normalized residual relation": check["detail"] = "recomputed after Solar MW projection; max error within float tolerance"
    node_type = np.load(output / "static/node_type_mask.npy").astype(bool)
    hourly_valid = np.load(output / "hourly/actual_valid_mask.npy", mmap_mode="r").astype(bool)
    for name, file_name in (("actual_mw", "actual_mw.npy"), ("actual_normalized", "actual_normalized.npy")):
        values = np.load(output / "hourly" / file_name, mmap_mode="r")[..., SOLAR]
        selected = np.asarray(values)[hourly_valid[..., SOLAR] & node_type[None, :, SOLAR]]
        quality["hourly_value_ranges"]["solar"][name] = {
            "min": float(selected.min()), "max": float(selected.max()), "mean": float(selected.mean())
        }
    (output / "quality_report.json").write_text(json.dumps(quality, indent=2), "utf-8")

    files = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "output_manifest.json": continue
        relative = path.relative_to(output).as_posix()
        entry: dict[str, Any] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r"); entry.update({"shape": list(array.shape), "dtype": str(array.dtype)})
        files[relative] = entry
    manifest = {
        "data_version": "reliable_channel_training_v2_solar_nonnegative",
        "transformation": "solar_nonnegative_projection",
        "files": files,
    }
    (output / "output_manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="reliable_channel_training_v1")
    parser.add_argument("--output", default="reliable_channel_training_v2_solar_nonnegative")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(); source = Path(args.source); output = Path(args.output)
    if args.verify_only:
        report = verify(source, output)
    else:
        audit = build(source, output)
        report = verify(source, output)
        write_quality_and_manifest(output, report)
        print(json.dumps({"audit": audit, "verification": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
