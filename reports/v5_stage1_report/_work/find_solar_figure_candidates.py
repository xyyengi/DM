from pathlib import Path
import json
import numpy as np

ROOT = Path(r"D:\DM_local\outputs_shandong\v5_stage1")
RUNS = {
    "V4-RS": ROOT / "20260722_143437_v4rs_repro_stage1_seed2026_20260722_143431_val_rank1_epoch11_posterior_n20_seed424242",
    "V5-T": ROOT / "20260722_151755_v5_t_stage1_seed2026_20260722_151749_val_rank1_epoch29_posterior_n20_seed424242",
    "V5-TF": ROOT / "20260722_155013_v5_tf_stage1_seed2026_20260722_155007_val_rank1_epoch8_posterior_n20_seed424242",
}

base = RUNS["V5-TF"]
actual_all = np.load(base / "actual_data.npy")
forecast_all = np.load(base / "forecast_data.npy")
solar_actual = actual_all[:, 1]
solar_forecast = forecast_all[:, 1]
day = solar_forecast > 1.0

solar_mae = np.array([
    np.mean(np.abs(solar_forecast[i, day[i]] - solar_actual[i, day[i]]))
    for i in range(len(solar_actual))
])
solar_energy = solar_actual.sum(axis=1)
solar_ramp = np.mean(np.abs(np.diff(solar_actual, axis=1)), axis=1)
net_actual = actual_all[:, 2] - actual_all[:, 0] - actual_all[:, 1]
net_forecast = forecast_all[:, 2] - forecast_all[:, 0] - forecast_all[:, 1]
net_mae = np.mean(np.abs(net_forecast - net_actual), axis=1)


def robust_z(values):
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return (values - med) / max(mad, 1e-9)


candidate_rules = {
    "solar_mae_median": np.abs(solar_mae - np.median(solar_mae)),
    "solar_profile_typical": (
        robust_z(solar_mae) ** 2
        + robust_z(solar_energy) ** 2
        + robust_z(solar_ramp) ** 2
    ),
    "joint_typical": (
        robust_z(net_mae) ** 2
        + robust_z(solar_mae) ** 2
        + robust_z(solar_energy) ** 2
    ),
}


def coverage_stats(idx):
    result = {}
    for name, path in RUNS.items():
        scenarios = np.load(path / "actual_scenarios_constrained.npy", mmap_mode="r")
        member = np.asarray(scenarios[idx, :, 1, :])
        lo, med, hi = np.quantile(member, [0.05, 0.5, 0.95], axis=0)
        mask = day[idx]
        strong = solar_forecast[idx] > 0.10 * solar_forecast.max()
        inside = (solar_actual[idx] >= lo) & (solar_actual[idx] <= hi)
        result[name] = {
            "daylight_coverage_pct": float(inside[mask].mean() * 100),
            "strong_daylight_coverage_pct": float(inside[strong].mean() * 100),
            "median_mae_daylight_mw": float(
                np.mean(np.abs(med[mask] - solar_actual[idx, mask]))
            ),
            "mean_width_daylight_mw": float(np.mean((hi - lo)[mask])),
        }
    return result


records = {}
for rule, score in candidate_rules.items():
    order = np.argsort(score)
    # Keep the first three model-independent candidates for visual choice.
    records[rule] = []
    for idx in order[:3]:
        records[rule].append({
            "index_zero_based": int(idx),
            "window_one_based": int(idx + 1),
            "solar_forecast_mae_daylight_mw": float(solar_mae[idx]),
            "solar_energy_mwh_window": float(solar_energy[idx]),
            "solar_ramp_mw": float(solar_ramp[idx]),
            "netload_forecast_mae_mw": float(net_mae[idx]),
            "models": coverage_stats(int(idx)),
        })

print(json.dumps(records, ensure_ascii=False, indent=2))
