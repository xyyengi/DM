from pathlib import Path
import json
import numpy as np

ROOT = Path(r"D:\DM_local\outputs_shandong\v5_stage1")
RUNS = {
    "V4-RS": ROOT / "20260722_143437_v4rs_repro_stage1_seed2026_20260722_143431_val_rank1_epoch11_posterior_n20_seed424242",
    "V5-T": ROOT / "20260722_151755_v5_t_stage1_seed2026_20260722_151749_val_rank1_epoch29_posterior_n20_seed424242",
    "V5-TF": ROOT / "20260722_155013_v5_tf_stage1_seed2026_20260722_155007_val_rank1_epoch8_posterior_n20_seed424242",
}
REP = 435


def coverage(actual, scenarios):
    lo, hi = np.quantile(scenarios, [0.05, 0.95], axis=1)
    inside = (actual >= lo) & (actual <= hi)
    return inside, lo, hi


def crps_points(actual, scenarios):
    members = scenarios.shape[1]
    term1 = np.mean(np.abs(scenarios - actual[:, None, :]), axis=1)
    xs = np.sort(scenarios, axis=1)
    weights = (2 * np.arange(1, members + 1) - members - 1).reshape(1, -1, 1)
    term2 = np.sum(xs * weights, axis=1) / (members**2)
    return term1 - term2


out = {}
for name, path in RUNS.items():
    actual = np.load(path / "actual_data.npy")[:, 1, :]
    forecast = np.load(path / "forecast_data.npy")[:, 1, :]
    scenarios = np.load(path / "actual_scenarios_constrained.npy")[:, :, 1, :]
    inside, lo, hi = coverage(actual, scenarios)
    crps = crps_points(actual, scenarios)
    daylight = forecast > 1.0
    strong_daylight = forecast > 0.10 * float(forecast.max())
    median = np.median(scenarios, axis=1)
    rep_day = daylight[REP]
    rep_strong = strong_daylight[REP]
    rep_outside = ~inside[REP]

    out[name] = {
        "all_points_coverage_pct": float(inside.mean() * 100),
        "forecast_daylight_coverage_pct": float(inside[daylight].mean() * 100),
        "strong_daylight_coverage_pct": float(inside[strong_daylight].mean() * 100),
        "all_points_crps_mw": float(crps.mean()),
        "forecast_daylight_crps_mw": float(crps[daylight].mean()),
        "strong_daylight_crps_mw": float(crps[strong_daylight].mean()),
        "daylight_median_mae_mw": float(np.mean(np.abs(median[daylight] - actual[daylight]))),
        "daylight_forecast_mae_mw": float(np.mean(np.abs(forecast[daylight] - actual[daylight]))),
        "representative_all_coverage_pct": float(inside[REP].mean() * 100),
        "representative_daylight_coverage_pct": float(inside[REP][rep_day].mean() * 100),
        "representative_strong_daylight_coverage_pct": float(
            inside[REP][rep_strong].mean() * 100
        ),
        "representative_daylight_points": int(rep_day.sum()),
        "representative_daylight_outside_points": int((rep_outside & rep_day).sum()),
        "representative_median_mae_daylight_mw": float(
            np.mean(np.abs(median[REP][rep_day] - actual[REP][rep_day]))
        ),
        "representative_forecast_mae_daylight_mw": float(
            np.mean(np.abs(forecast[REP][rep_day] - actual[REP][rep_day]))
        ),
        "representative_mean_band_width_daylight_mw": float(
            np.mean((hi[REP] - lo[REP])[rep_day])
        ),
        "representative_peak_actual_mw": float(actual[REP].max()),
        "representative_peak_median_mw": float(median[REP].max()),
        "representative_peak_upper_mw": float(hi[REP].max()),
    }

print(json.dumps(out, ensure_ascii=False, indent=2))
