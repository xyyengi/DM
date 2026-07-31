"""Stage-0 diagnostics for 24-station, 168-hour wind/solar forecast vintages.

The script intentionally uses train for discovery and validation only for
replication checks.  The sealed test split is never loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-8
SOLAR_ACTIVE_THRESHOLD = 0.02
LEAD_DAYS = np.arange(168) // 24 + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("diffusion_input_station"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/station_stage0_diagnostics"),
    )
    return parser.parse_args()


def load_split(data_dir: Path, split: str) -> dict[str, np.ndarray | pd.DataFrame]:
    return {
        "forecast": np.load(data_dir / f"{split}_forecast.npy"),
        "actual": np.load(data_dir / f"{split}_actual.npy"),
        "residual": np.load(data_dir / f"{split}_residual.npy"),
        "fill_mask": np.load(data_dir / f"{split}_fill_mask.npy"),
        "dates": pd.read_csv(data_dir / f"{split}_issue_dates.csv"),
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def station_metrics(
    splits: dict[str, dict[str, np.ndarray | pd.DataFrame]],
    stations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, data in splits.items():
        residual = np.asarray(data["residual"])
        for station_index, station in stations.iterrows():
            values = residual[:, :, station_index].reshape(-1)
            rows.append(
                {
                    "split": split,
                    "channel_index": int(station.channel_index),
                    "station_id": int(station.station_id),
                    "station_type": station.data_type,
                    "station_name": station.FARM_NAME,
                    "capacity_mw": float(station.capacity_mw),
                    "bias": float(values.mean()),
                    "mae": float(np.abs(values).mean()),
                    "rmse": float(np.sqrt(np.mean(values**2))),
                    "residual_std": float(values.std()),
                    "abs_error_p90": float(np.quantile(np.abs(values), 0.90)),
                }
            )
    return pd.DataFrame(rows)


def lead_metrics(
    splits: dict[str, dict[str, np.ndarray | pd.DataFrame]],
    stations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, data in splits.items():
        residual = np.asarray(data["residual"])
        for station_type in ["wind", "solar", "all"]:
            if station_type == "all":
                station_indices = np.arange(len(stations))
            else:
                station_indices = stations.index[
                    stations.data_type.eq(station_type)
                ].to_numpy()
            for lead_day in range(1, 8):
                hour_indices = np.flatnonzero(LEAD_DAYS == lead_day)
                values = residual[:, hour_indices, :][:, :, station_indices].reshape(-1)
                rows.append(
                    {
                        "split": split,
                        "station_type": station_type,
                        "lead_day": lead_day,
                        "lead_start_hour": 24 * (lead_day - 1) + 1,
                        "lead_end_hour": 24 * lead_day,
                        "n_values": int(values.size),
                        "bias": float(values.mean()),
                        "mae": float(np.abs(values).mean()),
                        "rmse": float(np.sqrt(np.mean(values**2))),
                        "residual_std": float(values.std()),
                        "abs_error_p90": float(np.quantile(np.abs(values), 0.90)),
                    }
                )
    return pd.DataFrame(rows)


def active_mask_for_pair(
    forecast: np.ndarray,
    actual: np.ndarray,
    stations: pd.DataFrame,
    i: int,
    j: int,
) -> np.ndarray:
    station_i_is_solar = stations.iloc[i].data_type == "solar"
    station_j_is_solar = stations.iloc[j].data_type == "solar"
    mask = np.ones(forecast.shape[:2], dtype=bool)
    if station_i_is_solar:
        mask &= np.maximum(forecast[:, :, i], actual[:, :, i]) > SOLAR_ACTIVE_THRESHOLD
    if station_j_is_solar:
        mask &= np.maximum(forecast[:, :, j], actual[:, :, j]) > SOLAR_ACTIVE_THRESHOLD
    return mask


def spatial_diagnostics(
    train: dict[str, np.ndarray | pd.DataFrame],
    stations: pd.DataFrame,
    distance: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    residual = np.asarray(train["residual"])
    forecast = np.asarray(train["forecast"])
    actual = np.asarray(train["actual"])
    station_count = residual.shape[-1]
    correlation = np.eye(station_count, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for i in range(station_count):
        for j in range(i + 1, station_count):
            mask = active_mask_for_pair(forecast, actual, stations, i, j)
            pair_corr = safe_corr(
                residual[:, :, i][mask],
                residual[:, :, j][mask],
            )
            correlation[i, j] = pair_corr
            correlation[j, i] = pair_corr
            type_i = stations.iloc[i].data_type
            type_j = stations.iloc[j].data_type
            if type_i == type_j:
                pair_type = f"{type_i}-{type_j}"
            else:
                pair_type = "wind-solar"
            rows.append(
                {
                    "station_i": int(stations.iloc[i].station_id),
                    "station_j": int(stations.iloc[j].station_id),
                    "type_i": type_i,
                    "type_j": type_j,
                    "pair_type": pair_type,
                    "distance_km": float(distance[i, j]),
                    "residual_correlation": pair_corr,
                    "n_observations": int(mask.sum()),
                    "adjacent": bool(adjacency[i, j] > 0),
                }
            )
    pair_frame = pd.DataFrame(rows)
    summary: dict[str, object] = {}
    for pair_type, group in pair_frame.groupby("pair_type"):
        rho, p_value = safe_spearman(
            group.distance_km.to_numpy(),
            group.residual_correlation.to_numpy(),
        )
        adjacent = group.loc[group.adjacent, "residual_correlation"]
        non_adjacent = group.loc[~group.adjacent, "residual_correlation"]
        summary[pair_type] = {
            "pair_count": int(len(group)),
            "mean_correlation": float(group.residual_correlation.mean()),
            "median_correlation": float(group.residual_correlation.median()),
            "distance_correlation_spearman": rho,
            "distance_correlation_p_value": p_value,
            "adjacent_mean_correlation": float(adjacent.mean()),
            "non_adjacent_mean_correlation": float(non_adjacent.mean()),
        }
    return correlation, pair_frame, summary


def pca_diagnostics(
    residual: np.ndarray,
    forecast: np.ndarray,
    actual: np.ndarray,
    stations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    groups = {
        "all": np.arange(len(stations)),
        "wind": stations.index[stations.data_type.eq("wind")].to_numpy(),
        "solar": stations.index[stations.data_type.eq("solar")].to_numpy(),
    }
    flattened = residual.reshape(-1, residual.shape[-1]).astype(np.float64)
    flattened_forecast = forecast.reshape(-1, forecast.shape[-1])
    flattened_actual = actual.reshape(-1, actual.shape[-1])
    for group_name, station_indices in groups.items():
        values = flattened[:, station_indices]
        if group_name == "solar":
            active_fraction = np.mean(
                np.maximum(
                    flattened_forecast[:, station_indices],
                    flattened_actual[:, station_indices],
                )
                > SOLAR_ACTIVE_THRESHOLD,
                axis=1,
            )
            values = values[active_fraction >= 0.5]
        values = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), EPS)
        covariance = np.cov(values, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))[::-1]
        explained = eigenvalues / eigenvalues.sum()
        cumulative = np.cumsum(explained)
        effective_rank = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
        n80 = int(np.searchsorted(cumulative, 0.80) + 1)
        n90 = int(np.searchsorted(cumulative, 0.90) + 1)
        summary[group_name] = {
            "station_count": int(len(station_indices)),
            "pc1_explained": float(explained[0]),
            "pc1_to_pc3_explained": float(explained[:3].sum()),
            "components_for_80_percent": n80,
            "components_for_90_percent": n90,
            "effective_rank": effective_rank,
        }
        for component, (ratio, cumulative_ratio) in enumerate(
            zip(explained, cumulative, strict=True), start=1
        ):
            rows.append(
                {
                    "group": group_name,
                    "component": component,
                    "explained_variance_ratio": float(ratio),
                    "cumulative_explained_variance": float(cumulative_ratio),
                }
            )
    return pd.DataFrame(rows), summary


def temporal_acf(
    train: dict[str, np.ndarray | pd.DataFrame],
    stations: pd.DataFrame,
    max_lag: int = 48,
) -> pd.DataFrame:
    residual = np.asarray(train["residual"])
    forecast = np.asarray(train["forecast"])
    actual = np.asarray(train["actual"])
    rows: list[dict[str, object]] = []
    for station_type in ["wind", "solar"]:
        station_indices = stations.index[
            stations.data_type.eq(station_type)
        ].to_numpy()
        for lag in range(1, max_lag + 1):
            left = residual[:, :-lag, :][:, :, station_indices]
            right = residual[:, lag:, :][:, :, station_indices]
            if station_type == "solar":
                active_left = np.maximum(
                    forecast[:, :-lag, :][:, :, station_indices],
                    actual[:, :-lag, :][:, :, station_indices],
                ) > SOLAR_ACTIVE_THRESHOLD
                active_right = np.maximum(
                    forecast[:, lag:, :][:, :, station_indices],
                    actual[:, lag:, :][:, :, station_indices],
                ) > SOLAR_ACTIVE_THRESHOLD
                active = active_left & active_right
                correlation = safe_corr(left[active], right[active])
                count = int(active.sum())
            else:
                correlation = safe_corr(left.reshape(-1), right.reshape(-1))
                count = int(left.size)
            rows.append(
                {
                    "station_type": station_type,
                    "lag_hour": lag,
                    "autocorrelation": correlation,
                    "n_pairs": count,
                }
            )
    return pd.DataFrame(rows)


def consecutive_pairs(data: dict[str, np.ndarray | pd.DataFrame]) -> list[tuple[int, int]]:
    dates = pd.to_datetime(pd.DataFrame(data["dates"]).issue_date)
    return [
        (i - 1, i)
        for i in range(1, len(dates))
        if (dates.iloc[i] - dates.iloc[i - 1]).days == 1
    ]


def collect_revision_values(
    data: dict[str, np.ndarray | pd.DataFrame],
    stations: pd.DataFrame,
    station_type: str,
) -> dict[str, np.ndarray]:
    forecast = np.asarray(data["forecast"])
    actual = np.asarray(data["actual"])
    residual = np.asarray(data["residual"])
    station_indices = stations.index[
        stations.data_type.eq(station_type)
    ].to_numpy()
    revisions: list[np.ndarray] = []
    current_errors: list[np.ndarray] = []
    previous_errors: list[np.ndarray] = []
    for previous_index, current_index in consecutive_pairs(data):
        previous_forecast = forecast[previous_index, 24:, :][:, station_indices]
        current_forecast = forecast[current_index, :144, :][:, station_indices]
        current_actual = actual[current_index, :144, :][:, station_indices]
        current_error = residual[current_index, :144, :][:, station_indices]
        revision = current_forecast - previous_forecast
        previous_error = current_actual - previous_forecast
        if station_type == "solar":
            active = (
                np.maximum.reduce(
                    [current_forecast, previous_forecast, current_actual]
                )
                > SOLAR_ACTIVE_THRESHOLD
            )
        else:
            active = np.ones_like(revision, dtype=bool)
        revisions.append(revision[active])
        current_errors.append(current_error[active])
        previous_errors.append(previous_error[active])
    return {
        "revision": np.concatenate(revisions),
        "current_error": np.concatenate(current_errors),
        "previous_error": np.concatenate(previous_errors),
    }


def revision_diagnostics(
    splits: dict[str, dict[str, np.ndarray | pd.DataFrame]],
    stations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    values = {
        split: {
            station_type: collect_revision_values(data, stations, station_type)
            for station_type in ["wind", "solar"]
        }
        for split, data in splits.items()
    }
    train_thresholds: dict[str, np.ndarray] = {}
    train_extreme_thresholds: dict[str, float] = {}
    for station_type in ["wind", "solar"]:
        train_abs_revision = np.abs(values["train"][station_type]["revision"])
        boundaries = np.quantile(train_abs_revision, [0.2, 0.4, 0.6, 0.8])
        train_thresholds[station_type] = np.unique(boundaries)
        train_extreme_thresholds[station_type] = float(
            np.quantile(
                np.abs(values["train"][station_type]["current_error"]), 0.90
            )
        )

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for split in ["train", "val"]:
        summary[split] = {}
        for station_type in ["wind", "solar"]:
            item = values[split][station_type]
            revision = item["revision"]
            current_error = item["current_error"]
            previous_error = item["previous_error"]
            abs_revision = np.abs(revision)
            abs_current_error = np.abs(current_error)
            abs_previous_error = np.abs(previous_error)
            rho, p_value = safe_spearman(abs_revision, abs_current_error)
            correction_direction = np.sign(revision) == np.sign(previous_error)
            nonzero = (np.abs(revision) > EPS) & (np.abs(previous_error) > EPS)
            summary[split][station_type] = {
                "n_values": int(len(revision)),
                "abs_revision_abs_error_spearman": rho,
                "spearman_p_value": p_value,
                "mean_abs_revision": float(abs_revision.mean()),
                "current_forecast_mae": float(abs_current_error.mean()),
                "previous_forecast_mae_on_same_hours": float(abs_previous_error.mean()),
                "revision_improves_fraction": float(
                    np.mean(abs_current_error < abs_previous_error)
                ),
                "revision_direction_matches_previous_error_fraction": float(
                    correction_direction[nonzero].mean()
                ),
            }
            bin_indices = np.searchsorted(
                train_thresholds[station_type], abs_revision, side="right"
            )
            for bin_index in range(len(train_thresholds[station_type]) + 1):
                mask = bin_indices == bin_index
                if not np.any(mask):
                    continue
                rows.append(
                    {
                        "split": split,
                        "station_type": station_type,
                        "revision_bin": bin_index + 1,
                        "n_values": int(mask.sum()),
                        "mean_abs_revision": float(abs_revision[mask].mean()),
                        "mean_abs_error": float(abs_current_error[mask].mean()),
                        "residual_std": float(current_error[mask].std()),
                        "extreme_error_rate": float(
                            np.mean(
                                abs_current_error[mask]
                                >= train_extreme_thresholds[station_type]
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows), summary


def overlap_dependence(
    splits: dict[str, dict[str, np.ndarray | pd.DataFrame]],
    stations: pd.DataFrame,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for split, data in splits.items():
        residual = np.asarray(data["residual"])
        result[split] = {}
        for station_type in ["wind", "solar"]:
            station_indices = stations.index[
                stations.data_type.eq(station_type)
            ].to_numpy()
            correlations = []
            for previous_index, current_index in consecutive_pairs(data):
                previous = residual[previous_index, 24:, :][:, station_indices]
                current = residual[current_index, :144, :][:, station_indices]
                correlations.append(safe_corr(previous.reshape(-1), current.reshape(-1)))
            result[split][station_type] = {
                "consecutive_pair_count": int(len(correlations)),
                "mean_aligned_overlap_correlation": float(np.nanmean(correlations)),
                "median_aligned_overlap_correlation": float(np.nanmedian(correlations)),
            }
    return result


def plot_station_metrics(frame: pd.DataFrame, output: Path) -> None:
    train = frame.loc[frame.split.eq("train")].sort_values("channel_index")
    colors = train.station_type.map({"wind": "#277da1", "solar": "#f4a261"})
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].bar(np.arange(len(train)), train.mae, color=colors)
    axes[0].set_ylabel("MAE (p.u.)")
    axes[0].set_title("Training forecast error by station")
    axes[1].bar(np.arange(len(train)), train.residual_std, color=colors)
    axes[1].set_ylabel("Residual std (p.u.)")
    axes[1].set_xticks(np.arange(len(train)))
    axes[1].set_xticklabels(train.station_id.astype(str), rotation=45, ha="right")
    axes[1].set_xlabel("Station ID (blue=wind, orange=solar)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lead_metrics(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    styles = {
        ("train", "wind"): ("#277da1", "-", "Train wind"),
        ("val", "wind"): ("#277da1", "--", "Val wind"),
        ("train", "solar"): ("#f4a261", "-", "Train solar"),
        ("val", "solar"): ("#f4a261", "--", "Val solar"),
    }
    for (split, station_type), (color, linestyle, label) in styles.items():
        subset = frame.loc[
            frame.split.eq(split) & frame.station_type.eq(station_type)
        ].sort_values("lead_day")
        axes[0].plot(
            subset.lead_day,
            subset.mae,
            marker="o",
            color=color,
            linestyle=linestyle,
            label=label,
        )
        axes[1].plot(
            subset.lead_day,
            subset.residual_std,
            marker="o",
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axes[0].set_title("MAE by forecast lead day")
    axes[0].set_ylabel("MAE (p.u.)")
    axes[1].set_title("Residual spread by forecast lead day")
    axes[1].set_ylabel("Residual std (p.u.)")
    for axis in axes:
        axis.set_xlabel("Lead day")
        axis.set_xticks(range(1, 8))
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(
    correlation: np.ndarray,
    stations: pd.DataFrame,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(np.arange(len(stations)))
    axis.set_yticks(np.arange(len(stations)))
    labels = stations.station_id.astype(str).tolist()
    axis.set_xticklabels(labels, rotation=90)
    axis.set_yticklabels(labels)
    wind_count = int(stations.data_type.eq("wind").sum())
    axis.axhline(wind_count - 0.5, color="black", linewidth=1)
    axis.axvline(wind_count - 0.5, color="black", linewidth=1)
    axis.set_title("Active-aware training residual correlation")
    fig.colorbar(image, ax=axis, label="Pearson correlation", shrink=0.82)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distance_correlation(frame: pd.DataFrame, output: Path) -> None:
    colors = {
        "wind-wind": "#277da1",
        "solar-solar": "#f4a261",
        "wind-solar": "#7b2cbf",
    }
    fig, axis = plt.subplots(figsize=(9, 5.6))
    for pair_type, group in frame.groupby("pair_type"):
        axis.scatter(
            group.distance_km,
            group.residual_correlation,
            s=24,
            alpha=0.65,
            color=colors[pair_type],
            label=pair_type,
        )
        valid = group[["distance_km", "residual_correlation"]].dropna()
        if len(valid) >= 3:
            coefficients = np.polyfit(
                valid.distance_km, valid.residual_correlation, deg=1
            )
            xs = np.linspace(valid.distance_km.min(), valid.distance_km.max(), 100)
            axis.plot(xs, np.polyval(coefficients, xs), color=colors[pair_type])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Station distance (km)")
    axis.set_ylabel("Residual correlation")
    axis.set_title("Spatial error dependence versus distance")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pca(frame: pd.DataFrame, output: Path) -> None:
    colors = {"all": "#577590", "wind": "#277da1", "solar": "#f4a261"}
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for group, subset in frame.groupby("group"):
        axis.plot(
            subset.component,
            subset.cumulative_explained_variance,
            marker="o",
            markersize=3,
            label=group,
            color=colors[group],
        )
    axis.axhline(0.8, color="gray", linestyle="--", linewidth=1)
    axis.axhline(0.9, color="gray", linestyle=":", linewidth=1)
    axis.set_ylim(0, 1.02)
    axis.set_xlim(1, frame.component.max())
    axis.set_xlabel("Number of principal components")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_title("Spatial residual dimensionality")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_acf(frame: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for station_type, color in [("wind", "#277da1"), ("solar", "#f4a261")]:
        subset = frame.loc[frame.station_type.eq(station_type)]
        axis.plot(
            subset.lag_hour,
            subset.autocorrelation,
            color=color,
            label=station_type,
        )
    for lag in [6, 24, 48]:
        axis.axvline(lag, color="gray", linewidth=0.7, alpha=0.5)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Lag (hour)")
    axis.set_ylabel("Residual autocorrelation")
    axis.set_title("Within-window temporal residual dependence")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_revision(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for axis, station_type, color in [
        (axes[0], "wind", "#277da1"),
        (axes[1], "solar", "#f4a261"),
    ]:
        for split, linestyle in [("train", "-"), ("val", "--")]:
            subset = frame.loc[
                frame.split.eq(split) & frame.station_type.eq(station_type)
            ].sort_values("revision_bin")
            axis.plot(
                subset.mean_abs_revision,
                subset.mean_abs_error,
                marker="o",
                linestyle=linestyle,
                color=color,
                label=split,
            )
            for _, row in subset.iterrows():
                axis.annotate(
                    f"Q{int(row.revision_bin)}",
                    (row.mean_abs_revision, row.mean_abs_error),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
        axis.set_title(f"{station_type.capitalize()}: revision vs current error")
        axis.set_xlabel("Mean |forecast revision| (p.u.)")
        axis.set_ylabel("Current forecast MAE (p.u.)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_report(
    output_dir: Path,
    station_frame: pd.DataFrame,
    lead_frame: pd.DataFrame,
    spatial_summary: dict[str, object],
    pca_summary: dict[str, object],
    acf_frame: pd.DataFrame,
    revision_summary: dict[str, object],
    overlap_summary: dict[str, object],
) -> None:
    train_station = station_frame.loc[station_frame.split.eq("train")]
    train_lead = lead_frame.loc[lead_frame.split.eq("train")]
    val_lead = lead_frame.loc[lead_frame.split.eq("val")]
    wind_train = train_station.loc[train_station.station_type.eq("wind")]
    solar_train = train_station.loc[train_station.station_type.eq("solar")]
    acf_lookup = acf_frame.set_index(["station_type", "lag_hour"])
    wind_d1 = train_lead.query("station_type == 'wind' and lead_day == 1").iloc[0]
    wind_d7 = train_lead.query("station_type == 'wind' and lead_day == 7").iloc[0]
    solar_d1 = train_lead.query("station_type == 'solar' and lead_day == 1").iloc[0]
    solar_d7 = train_lead.query("station_type == 'solar' and lead_day == 7").iloc[0]
    val_wind_d1 = val_lead.query("station_type == 'wind' and lead_day == 1").iloc[0]
    val_wind_d7 = val_lead.query("station_type == 'wind' and lead_day == 7").iloc[0]
    val_solar_d1 = val_lead.query("station_type == 'solar' and lead_day == 1").iloc[0]
    val_solar_d7 = val_lead.query("station_type == 'solar' and lead_day == 7").iloc[0]

    train_revision_wind = revision_summary["train"]["wind"]
    train_revision_solar = revision_summary["train"]["solar"]
    val_revision_wind = revision_summary["val"]["wind"]
    val_revision_solar = revision_summary["val"]["solar"]

    text = f"""# 24场站风光联合场景生成：实验0数据诊断

## 1. 诊断口径

- 结构发现只使用训练集290个发布批次；验证集23个批次只用于复核结论。
- 测试集未加载、未计算、未参与任何判断。
- 残差定义为 `actual - forecast`，单位为场站容量标幺值（p.u.）。
- 光伏相关性与时间自相关只在光伏活跃点上计算：预测或实测大于{SOLAR_ACTIVE_THRESHOLD:.2f} p.u.，避免共同夜间零值人为抬高相关性。
- 空间相关性是同一发布批次、同一提前时刻下两个场站残差的Pearson相关系数。

## 2. 逐站误差与尺度差异

训练集13个风电场站的平均逐站MAE为 `{wind_train.mae.mean():.4f}` p.u.，逐站残差标准差平均为 `{wind_train.residual_std.mean():.4f}` p.u.；11个光伏场站分别为 `{solar_train.mae.mean():.4f}` 和 `{solar_train.residual_std.mean():.4f}` p.u.。

逐站MAE范围：

- 风电：`{wind_train.mae.min():.4f}`–`{wind_train.mae.max():.4f}` p.u.
- 光伏：`{solar_train.mae.min():.4f}`–`{solar_train.mae.max():.4f}` p.u.

这说明即使所有场站都已经按容量归一化，场站间误差尺度仍不完全一致。训练24站联合模型时，应使用训练集统计量做轻量的逐站损失平衡或残差尺度归一化，避免高误差场站主导损失；这不是把残差标准差当作运行时条件。

![逐站误差](figures/station_error_metrics.png)

## 3. 提前时距效应

训练集从第1天到第7天：

- 风电MAE：`{wind_d1.mae:.4f}` → `{wind_d7.mae:.4f}` p.u.，残差标准差：`{wind_d1.residual_std:.4f}` → `{wind_d7.residual_std:.4f}`。
- 光伏MAE：`{solar_d1.mae:.4f}` → `{solar_d7.mae:.4f}` p.u.，残差标准差：`{solar_d1.residual_std:.4f}` → `{solar_d7.residual_std:.4f}`。

验证集对应变化：

- 风电MAE：`{val_wind_d1.mae:.4f}` → `{val_wind_d7.mae:.4f}` p.u.
- 光伏MAE：`{val_solar_d1.mae:.4f}` → `{val_solar_d7.mae:.4f}` p.u.

这部分直接判断 `lead_mark` 是否是有用条件。若误差尺度随时距变化明显，则模型必须保留真实提前时距编码，不能只把168小时当作无差别的序列位置。

![分时距误差](figures/lead_error_profile.png)

## 4. 空间相关性与距离

训练残差的成对统计如下：

| 场站对 | 对数 | 平均相关系数 | 距离与相关性的Spearman系数 | p值 | 邻接场站平均相关 | 非邻接平均相关 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for pair_type in ["wind-wind", "solar-solar", "wind-solar"]:
        item = spatial_summary[pair_type]
        text += (
            f"| {pair_type} | {item['pair_count']} | "
            f"{item['mean_correlation']:.3f} | "
            f"{item['distance_correlation_spearman']:.3f} | "
            f"{item['distance_correlation_p_value']:.3g} | "
            f"{item['adjacent_mean_correlation']:.3f} | "
            f"{item['non_adjacent_mean_correlation']:.3f} |\n"
        )

    text += f"""

距离—相关性系数为负表示距离越远，误差相关性总体越弱。邻接场站相关性高于非邻接场站，则说明现有地理邻接矩阵具有可用信息；反之则不应仅凭经纬度强行加入图网络，而应进一步考虑训练残差相关图或预测天气共同因子。

![残差相关矩阵](figures/residual_correlation_heatmap.png)

![距离与相关性](figures/correlation_vs_distance.png)

## 5. 空间维度是否可以压缩

| 分组 | 场站数 | PC1解释率 | 前3个主成分解释率 | 达到80%所需主成分 | 达到90%所需主成分 | 有效秩 |
|---|---:|---:|---:|---:|---:|---:|
| 全部 | {pca_summary['all']['station_count']} | {percentage(pca_summary['all']['pc1_explained'])} | {percentage(pca_summary['all']['pc1_to_pc3_explained'])} | {pca_summary['all']['components_for_80_percent']} | {pca_summary['all']['components_for_90_percent']} | {pca_summary['all']['effective_rank']:.2f} |
| 风电 | {pca_summary['wind']['station_count']} | {percentage(pca_summary['wind']['pc1_explained'])} | {percentage(pca_summary['wind']['pc1_to_pc3_explained'])} | {pca_summary['wind']['components_for_80_percent']} | {pca_summary['wind']['components_for_90_percent']} | {pca_summary['wind']['effective_rank']:.2f} |
| 光伏 | {pca_summary['solar']['station_count']} | {percentage(pca_summary['solar']['pc1_explained'])} | {percentage(pca_summary['solar']['pc1_to_pc3_explained'])} | {pca_summary['solar']['components_for_80_percent']} | {pca_summary['solar']['components_for_90_percent']} | {pca_summary['solar']['effective_rank']:.2f} |

PCA使用每个场站标准化后的训练残差；光伏组只使用至少一半光伏场站处于活跃状态的时刻，避免共同夜间零值人为压低维度。有效秩越低，说明场站误差更多由少量公共天气因子驱动，适合采用共享编码器或低维空间隐变量；有效秩接近场站数则表示局部差异较多，不应过度压缩。

![PCA累计解释率](figures/pca_explained_variance.png)

## 6. 时间相关性

训练残差的典型自相关为：

| 类型 | 1 h | 6 h | 24 h | 48 h |
|---|---:|---:|---:|---:|
| 风电 | {acf_lookup.loc[('wind', 1), 'autocorrelation']:.3f} | {acf_lookup.loc[('wind', 6), 'autocorrelation']:.3f} | {acf_lookup.loc[('wind', 24), 'autocorrelation']:.3f} | {acf_lookup.loc[('wind', 48), 'autocorrelation']:.3f} |
| 光伏活跃时段 | {acf_lookup.loc[('solar', 1), 'autocorrelation']:.3f} | {acf_lookup.loc[('solar', 6), 'autocorrelation']:.3f} | {acf_lookup.loc[('solar', 24), 'autocorrelation']:.3f} | {acf_lookup.loc[('solar', 48), 'autocorrelation']:.3f} |

这说明生成目标不是彼此独立的4032个数，而是同时具有时间连续性与跨日结构。保留168小时整体生成是必要的。

![时间自相关](figures/temporal_autocorrelation.png)

## 7. “发布预测修正”是否包含信息

对相邻两个发布日，把同一个有效时刻对齐。例如：

```text
昨天发布：周五12:00预测 = 0.70
今天发布：周五12:00预测 = 0.45
发布预测修正量 ΔF = 0.45 - 0.70 = -0.25
```

这里的修正量在今天生成场景时已经可知，不使用未来实测，因此不存在标签泄漏。本实验只检查 `|ΔF|` 与今天预测的 `|actual-forecast|` 是否存在稳定关系。

| 数据 | 类型 | `|ΔF|`与`|误差|` Spearman | 当前预测MAE | 上一版预测在相同时刻的MAE | 修正后改善比例 |
|---|---|---:|---:|---:|---:|
| train | wind | {train_revision_wind['abs_revision_abs_error_spearman']:.3f} | {train_revision_wind['current_forecast_mae']:.4f} | {train_revision_wind['previous_forecast_mae_on_same_hours']:.4f} | {percentage(train_revision_wind['revision_improves_fraction'])} |
| val | wind | {val_revision_wind['abs_revision_abs_error_spearman']:.3f} | {val_revision_wind['current_forecast_mae']:.4f} | {val_revision_wind['previous_forecast_mae_on_same_hours']:.4f} | {percentage(val_revision_wind['revision_improves_fraction'])} |
| train | solar | {train_revision_solar['abs_revision_abs_error_spearman']:.3f} | {train_revision_solar['current_forecast_mae']:.4f} | {train_revision_solar['previous_forecast_mae_on_same_hours']:.4f} | {percentage(train_revision_solar['revision_improves_fraction'])} |
| val | solar | {val_revision_solar['abs_revision_abs_error_spearman']:.3f} | {val_revision_solar['current_forecast_mae']:.4f} | {val_revision_solar['previous_forecast_mae_on_same_hours']:.4f} | {percentage(val_revision_solar['revision_improves_fraction'])} |

只有当训练集与验证集方向一致，且大修正分组的后续误差确实更大，才值得在后续消融中把 `ΔF` 作为额外条件。否则不加入。

![预测修正诊断](figures/forecast_revision_diagnostic.png)

## 8. 样本独立性提醒

相邻发布批次共享144个有效小时。将相同有效时刻对齐后，残差相关性为：

| 数据 | 类型 | 相邻发布批次数 | 平均重叠相关系数 | 中位数 |
|---|---|---:|---:|---:|
| train | wind | {overlap_summary['train']['wind']['consecutive_pair_count']} | {overlap_summary['train']['wind']['mean_aligned_overlap_correlation']:.3f} | {overlap_summary['train']['wind']['median_aligned_overlap_correlation']:.3f} |
| train | solar | {overlap_summary['train']['solar']['consecutive_pair_count']} | {overlap_summary['train']['solar']['mean_aligned_overlap_correlation']:.3f} | {overlap_summary['train']['solar']['median_aligned_overlap_correlation']:.3f} |
| val | wind | {overlap_summary['val']['wind']['consecutive_pair_count']} | {overlap_summary['val']['wind']['mean_aligned_overlap_correlation']:.3f} | {overlap_summary['val']['wind']['median_aligned_overlap_correlation']:.3f} |
| val | solar | {overlap_summary['val']['solar']['consecutive_pair_count']} | {overlap_summary['val']['solar']['mean_aligned_overlap_correlation']:.3f} | {overlap_summary['val']['solar']['median_aligned_overlap_correlation']:.3f} |

因此290个训练批次不能视为290个完全独立样本。模型应保持小参数量，验证不应把每个“场站×小时”误认为独立样本；后续置信区间应按发布日进行块bootstrap。

## 9. 轻量空间模块与复杂空间模块

轻量模块只做固定图上的共享传播，例如：

```math
H' = σ(\\tilde A H W)
```

其中邻接矩阵 `A` 已由经纬度预先确定，只学习一个较小的特征变换矩阵 `W`。通常放置1–2层，参数量增长较小。

复杂模块则会同时学习边，例如多头全连接空间注意力、每个时刻不同的动态图、风—风/光—光/风—光多套边参数，甚至空间Transformer。它更灵活，但290个发布批次下过拟合风险明显更高。

更具体地说：固定图卷积不学习24×24条边，只学习共享特征变换；若增加风—风、光—光、风—光三个标量门控，也只有3个关系参数。完全可学习静态图至少引入576个自由边权，动态多头注意力还会在每个样本、每个时刻重新计算24×24的连接强度，属于复杂方案。

## 10. 实验0结论与下一步决策

1. **真实提前时距是必要条件。** 风电从第1天到第7天的MAE和残差标准差持续上升，验证集也重复了这个趋势。第一版模型必须加入 `lead_mark`。
2. **空间信息确实存在，尤其是风电。** 风—风相关性随距离显著下降，邻接风电场相关性明显高于非邻接场，因此地理图具有明确数据依据。
3. **光伏更像“强公共因子+局部修正”。** 光伏场站间相关性高，但距离效应弱于风电；因此不能假设风、光共用完全相同的空间传播规律。
4. **风—光联系存在但较弱。** 第一版不应强迫风电和光伏进行强耦合传播，应保留联合输出，同时用类型信息控制空间混合强度。
5. **发布预测修正量有信息，但放到后续消融。** 训练集和验证集均显示修正越大，当前预测误差越大；五分位结果基本单调。这支持未来把 `ΔF` 作为不确定性条件，但不应在首个24站基线中同时加入。
6. **先训练无显式空间基线，再增加轻量空间层。** 实验1先复用当前联合扩散主体，建立24站基线；实验2只加入固定地理图和极少量类型门控。这样才能把改进归因于空间建模。

## 11. 产物说明

- `station_metrics.csv`：逐站误差。
- `lead_metrics.csv`：逐类型、逐提前日误差。
- `spatial_pair_metrics.csv`：每一对场站的距离与残差相关性。
- `pca_metrics.csv`：空间主成分解释率。
- `temporal_acf.csv`：1–48小时残差自相关。
- `forecast_revision_bins.csv`：预测修正幅度分组诊断。
- `summary.json`：上述主要统计量的机器可读汇总。
"""
    (output_dir / "stage0_diagnostics.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    distance = np.load(data_dir / "station_distance.npy")
    adjacency = np.load(data_dir / "station_adjacency.npy")
    splits = {split: load_split(data_dir, split) for split in ["train", "val"]}

    station_frame = station_metrics(splits, stations)
    lead_frame = lead_metrics(splits, stations)
    correlation, spatial_frame, spatial_summary = spatial_diagnostics(
        splits["train"], stations, distance, adjacency
    )
    pca_frame, pca_summary = pca_diagnostics(
        np.asarray(splits["train"]["residual"]),
        np.asarray(splits["train"]["forecast"]),
        np.asarray(splits["train"]["actual"]),
        stations,
    )
    acf_frame = temporal_acf(splits["train"], stations)
    revision_frame, revision_summary = revision_diagnostics(splits, stations)
    overlap_summary = overlap_dependence(splits, stations)

    station_frame.to_csv(output_dir / "station_metrics.csv", index=False)
    lead_frame.to_csv(output_dir / "lead_metrics.csv", index=False)
    spatial_frame.to_csv(output_dir / "spatial_pair_metrics.csv", index=False)
    pca_frame.to_csv(output_dir / "pca_metrics.csv", index=False)
    acf_frame.to_csv(output_dir / "temporal_acf.csv", index=False)
    revision_frame.to_csv(output_dir / "forecast_revision_bins.csv", index=False)
    np.save(output_dir / "training_residual_correlation.npy", correlation)

    summary = {
        "data_policy": {
            "discovery_split": "train",
            "replication_split": "val",
            "test_loaded": False,
            "solar_active_threshold_pu": SOLAR_ACTIVE_THRESHOLD,
        },
        "spatial": spatial_summary,
        "pca": pca_summary,
        "forecast_revision": revision_summary,
        "overlap_dependence": overlap_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_station_metrics(station_frame, figure_dir / "station_error_metrics.png")
    plot_lead_metrics(lead_frame, figure_dir / "lead_error_profile.png")
    plot_correlation_heatmap(
        correlation, stations, figure_dir / "residual_correlation_heatmap.png"
    )
    plot_distance_correlation(
        spatial_frame, figure_dir / "correlation_vs_distance.png"
    )
    plot_pca(pca_frame, figure_dir / "pca_explained_variance.png")
    plot_temporal_acf(acf_frame, figure_dir / "temporal_autocorrelation.png")
    plot_revision(
        revision_frame, figure_dir / "forecast_revision_diagnostic.png"
    )

    write_report(
        output_dir,
        station_frame,
        lead_frame,
        spatial_summary,
        pca_summary,
        acf_frame,
        revision_summary,
        overlap_summary,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
