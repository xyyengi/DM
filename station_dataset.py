"""Dataset utilities for 24-station, 168-hour forecast-vintage experiments."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.eval.physical_projection import _solar_elevation_degrees


EXPECTED_STATIONS = 24
EXPECTED_HOURS = 168


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


def validate_station_data_dir(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    required = [
        "export_metadata.json",
        "station_order.csv",
        "station_features.npy",
        "station_adjacency.npy",
    ]
    for split in ("train", "val", "test"):
        required.extend(
            [
                f"{split}_forecast.npy",
                f"{split}_actual.npy",
                f"{split}_residual.npy",
                f"{split}_time_mark.npy",
                f"{split}_lead_mark.npy",
                f"{split}_fill_mask.npy",
                f"{split}_issue_dates.csv",
            ]
        )
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"station data artifacts missing: {missing}")
    return data_dir


def fit_station_residual_scale(
    data_dir: str | Path,
    epsilon: float = 1e-4,
) -> dict[str, object]:
    """Fit per-station residual standard deviations on valid train values only."""
    data_dir = validate_station_data_dir(data_dir)
    residual = np.load(data_dir / "train_residual.npy", mmap_mode="r")
    fill_mask = np.load(data_dir / "train_fill_mask.npy", mmap_mode="r")
    if residual.shape != fill_mask.shape:
        raise ValueError("train residual and fill mask shapes must match")
    scales = []
    counts = []
    for station_index in range(residual.shape[-1]):
        valid = fill_mask[:, :, station_index] == 0
        values = np.asarray(residual[:, :, station_index][valid], dtype=np.float64)
        scale = max(float(values.std()), float(epsilon))
        scales.append(scale)
        counts.append(int(values.size))
    return {
        "method": "per_station_std",
        "fit_split": "train",
        "center": False,
        "epsilon": float(epsilon),
        "scale": scales,
        "valid_value_count": counts,
    }


def validate_residual_scale(
    residual_scale: Mapping[str, object],
    station_count: int = EXPECTED_STATIONS,
) -> np.ndarray:
    if residual_scale.get("fit_split") != "train":
        raise ValueError("residual scale must be fitted on train")
    if bool(residual_scale.get("center", False)):
        raise ValueError("station experiment uses scale-only residual normalization")
    scale = np.asarray(residual_scale.get("scale"), dtype=np.float32)
    if scale.shape != (station_count,) or not np.isfinite(scale).all():
        raise ValueError(f"invalid residual scale shape/value: {scale.shape}")
    if np.any(scale <= 0):
        raise ValueError("all residual scales must be positive")
    return scale


class StationForecastDataset(Dataset):
    """One item is one forecast issuance with 24 stations and 168 lead hours."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        residual_scale: Mapping[str, object],
        condition_config: Mapping[str, object] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split={split!r}")
        self.data_dir = validate_station_data_dir(data_dir)
        self.split = split
        self.forecast = np.load(self.data_dir / f"{split}_forecast.npy")
        self.actual = np.load(self.data_dir / f"{split}_actual.npy")
        self.residual = np.load(self.data_dir / f"{split}_residual.npy")
        self.time_mark = np.load(self.data_dir / f"{split}_time_mark.npy")
        self.lead_mark = np.load(self.data_dir / f"{split}_lead_mark.npy")
        self.fill_mask = np.load(self.data_dir / f"{split}_fill_mask.npy")
        self.scale = validate_residual_scale(residual_scale)
        self.condition_config = dict(condition_config or {})
        self.ramp_lags = tuple(
            int(value)
            for value in self.condition_config.get("forecast_ramp_lags", [1, 3, 6])
        )
        if not self.ramp_lags or any(
            lag <= 0 or lag >= EXPECTED_HOURS for lag in self.ramp_lags
        ):
            raise ValueError(f"invalid forecast_ramp_lags={self.ramp_lags}")
        self.recent_error_hours = int(
            self.condition_config.get("recent_error_hours", 24)
        )
        if not 1 <= self.recent_error_hours <= EXPECTED_HOURS:
            raise ValueError("recent_error_hours must be between 1 and 168")
        issue_frame = pd.read_csv(self.data_dir / f"{split}_issue_dates.csv")
        if len(issue_frame) != len(self.forecast):
            raise ValueError("issue date count does not match forecast sample count")
        issue_days = pd.to_datetime(issue_frame["issue_date"]).dt.normalize()
        lookup = {timestamp: index for index, timestamp in enumerate(issue_days)}
        self.previous_issue_index = np.asarray(
            [lookup.get(timestamp - pd.Timedelta(days=1), -1) for timestamp in issue_days],
            dtype=np.int64,
        )
        self.condition_audit = {
            "split": split,
            "sample_count": int(len(self.forecast)),
            "previous_issue_available_count": int(
                np.sum(self.previous_issue_index >= 0)
            ),
            "revision_overlap_hours": 144,
            "recent_error_hours": self.recent_error_hours,
            "forecast_ramp_lags": list(self.ramp_lags),
            "future_actual_used_as_condition": False,
        }
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        expected = (len(self.forecast), EXPECTED_HOURS, EXPECTED_STATIONS)
        for name in ["forecast", "actual", "residual", "fill_mask"]:
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{self.split}_{name} expected {expected}, got {value.shape}")
        if self.time_mark.shape != (len(self.forecast), EXPECTED_HOURS, 8):
            raise ValueError(f"invalid {self.split}_time_mark shape {self.time_mark.shape}")
        if self.lead_mark.shape != (len(self.forecast), EXPECTED_HOURS, 2):
            raise ValueError(f"invalid {self.split}_lead_mark shape {self.lead_mark.shape}")
        residual_error = np.max(
            np.abs(
                np.asarray(self.residual, dtype=np.float32)
                - (
                    np.asarray(self.actual, dtype=np.float32)
                    - np.asarray(self.forecast, dtype=np.float32)
                )
            )
        )
        if residual_error > 1e-6:
            raise ValueError(
                "station residual must equal actual - forecast; "
                f"max error={residual_error}"
            )

    def __len__(self) -> int:
        return int(self.forecast.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        forecast = np.asarray(self.forecast[index], dtype=np.float32).T.copy()
        actual = np.asarray(self.actual[index], dtype=np.float32).T.copy()
        residual = np.asarray(self.residual[index], dtype=np.float32).T.copy()
        target = residual / self.scale[:, None]
        valid_mask = 1.0 - np.asarray(
            self.fill_mask[index], dtype=np.float32
        ).T.copy()
        forecast_ramps = np.zeros(
            (EXPECTED_STATIONS, len(self.ramp_lags), EXPECTED_HOURS),
            dtype=np.float32,
        )
        for channel, lag in enumerate(self.ramp_lags):
            forecast_ramps[:, channel, lag:] = (
                forecast[:, lag:] - forecast[:, :-lag]
            )

        forecast_revision = np.zeros_like(forecast)
        revision_mask = np.zeros_like(forecast)
        recent_error = np.zeros(
            (EXPECTED_STATIONS, self.recent_error_hours), dtype=np.float32
        )
        recent_error_mask = np.zeros((EXPECTED_STATIONS, 1), dtype=np.float32)
        previous_index = int(self.previous_issue_index[index])
        if previous_index >= 0:
            overlap = EXPECTED_HOURS - 24
            previous_forecast = np.asarray(
                self.forecast[previous_index], dtype=np.float32
            ).T
            forecast_revision[:, :overlap] = (
                forecast[:, :overlap] - previous_forecast[:, 24:]
            )
            revision_mask[:, :overlap] = 1.0
            previous_residual = np.asarray(
                self.residual[previous_index], dtype=np.float32
            ).T
            recent_error[:] = previous_residual[:, : self.recent_error_hours]
            recent_error_mask[:] = 1.0
        return {
            "sample_index": torch.tensor(index, dtype=torch.long),
            "forecast": torch.from_numpy(forecast),
            "actual": torch.from_numpy(actual),
            "residual": torch.from_numpy(residual),
            "residual_target": torch.from_numpy(target),
            "calendar": torch.from_numpy(
                np.asarray(self.time_mark[index], dtype=np.float32).T.copy()
            ),
            "lead": torch.from_numpy(
                np.asarray(self.lead_mark[index], dtype=np.float32).T.copy()
            ),
            "valid_mask": torch.from_numpy(valid_mask),
            "forecast_ramps": torch.from_numpy(forecast_ramps),
            "forecast_revision": torch.from_numpy(forecast_revision),
            "revision_mask": torch.from_numpy(revision_mask),
            "recent_error": torch.from_numpy(recent_error),
            "recent_error_mask": torch.from_numpy(recent_error_mask),
        }


def load_station_static_data(data_dir: str | Path) -> dict[str, torch.Tensor]:
    data_dir = validate_station_data_dir(data_dir)
    features = np.load(data_dir / "station_features.npy").astype(np.float32)
    adjacency = np.load(data_dir / "station_adjacency.npy").astype(np.float32)
    if features.shape != (EXPECTED_STATIONS, 5):
        raise ValueError(f"station_features expected (24,5), got {features.shape}")
    if adjacency.shape != (EXPECTED_STATIONS, EXPECTED_STATIONS):
        raise ValueError(f"station_adjacency expected (24,24), got {adjacency.shape}")
    if not np.allclose(adjacency, adjacency.T, atol=1e-6):
        raise ValueError("station adjacency must be symmetric")
    if not np.allclose(features[:, :2].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("station wind/solar features must be one-hot")
    return {
        "station_features": torch.from_numpy(features),
        "station_adjacency": torch.from_numpy(adjacency),
    }


def build_station_daylight_mask(
    data_dir: str | Path,
    split: str,
    elevation_threshold_deg: float = -0.833,
    timestamp_offset_minutes: float = 30.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build [N,168,24] station-specific daylight without using power values."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split={split!r}")
    data_dir = validate_station_data_dir(data_dir)
    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    issues = pd.read_csv(data_dir / f"{split}_issue_dates.csv")
    mask = np.ones((len(issues), EXPECTED_HOURS, EXPECTED_STATIONS), dtype=bool)
    china_timezone = timezone(timedelta(hours=8))
    solar_indices = stations.index[stations.data_type.eq("solar")].to_numpy()
    for issue_index, target_start in enumerate(issues.target_start):
        start = datetime.fromisoformat(str(target_start)).replace(tzinfo=china_timezone)
        timestamps = [
            start + timedelta(hours=lead, minutes=float(timestamp_offset_minutes))
            for lead in range(EXPECTED_HOURS)
        ]
        for station_index in solar_indices:
            station = stations.iloc[station_index]
            mask[issue_index, :, station_index] = [
                _solar_elevation_degrees(
                    timestamp,
                    float(station.latitude),
                    float(station.longitude),
                )
                > float(elevation_threshold_deg)
                for timestamp in timestamps
            ]
    audit = {
        "method": "station_specific_noaa_solar_elevation",
        "split": split,
        "timezone": "Asia/Shanghai (UTC+08:00)",
        "elevation_threshold_deg": float(elevation_threshold_deg),
        "timestamp_offset_minutes": float(timestamp_offset_minutes),
        "solar_station_count": int(len(solar_indices)),
        "solar_daylight_fraction": float(mask[:, :, solar_indices].mean()),
        "uses_power_or_actual": False,
    }
    return mask, audit


def get_station_dataloader(
    data_dir: str | Path,
    split: str,
    residual_scale: Mapping[str, object],
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    condition_config: Mapping[str, object] | None = None,
) -> tuple[DataLoader, StationForecastDataset]:
    dataset = StationForecastDataset(
        data_dir, split, residual_scale, condition_config=condition_config
    )
    generator = torch.Generator()
    split_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
    generator.manual_seed(int(seed) + split_offset)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=split == "train",
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=_seed_worker,
    )
    return loader, dataset


def write_residual_scale(path: str | Path, residual_scale: Mapping[str, object]) -> None:
    Path(path).write_text(
        json.dumps(dict(residual_scale), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
