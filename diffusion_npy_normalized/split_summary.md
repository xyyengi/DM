# Normalized Shandong Diffusion NPY Export Summary

Input CSV: `data_preprocess_shandong/cleaned_normalized/Shandong_WSL_hourly_normalized.csv`
Output dir: `data_preprocess_shandong/cleaned_normalized/diffusion_npy_normalized`
Window length: 168 hours
Data channel order: Wind, Solar, Load
Time mark order: month_sin, month_cos, day_sin, day_cos, weekday_sin, weekday_cos, hour_sin, hour_cos

## Normalized Data Meaning
- actual/forecast arrays contain only 3 normalized W/S/L channels.
- time_mark arrays contain 8 cyclical time features.
- *_true.npy, *_pred.npy, *_res.npy are compatibility files with 11 channels: 3 data/residual channels + 8 time channels.
- residual = actual_norm - forecast_norm.
- All actual/forecast values are already normalized. Do not apply another MinMaxScaler before diffusion unless the next project explicitly requires it.

## Denormalization Parameters
- wind_total_capacity: 31047.950000000004
- solar_total_capacity: 33796.746999999996
- load_denominator: 129681.02012968103
- load_denominator_scope: train_only
- Full bus-level parameters are saved in `normalization_params.json` inside this output dir.

## train
- local range: 2025-01-01 00:00:00+08:00 -> 2025-10-31 23:00:00+08:00
- hours: 7296
- 168h windows: 7129
- train_actual.npy: shape=(7129, 168, 3)
- train_forecast.npy: shape=(7129, 168, 3)
- train_residual.npy: shape=(7129, 168, 3)
- train_time_mark.npy: shape=(7129, 168, 8)
- train_true.npy: shape=(7129, 168, 11)
- train_pred.npy: shape=(7129, 168, 11)
- train_res.npy: shape=(7129, 168, 11)

## val
- local range: 2025-11-01 00:00:00+08:00 -> 2025-11-30 23:00:00+08:00
- hours: 720
- 168h windows: 553
- val_actual.npy: shape=(553, 168, 3)
- val_forecast.npy: shape=(553, 168, 3)
- val_residual.npy: shape=(553, 168, 3)
- val_time_mark.npy: shape=(553, 168, 8)
- val_true.npy: shape=(553, 168, 11)
- val_pred.npy: shape=(553, 168, 11)
- val_res.npy: shape=(553, 168, 11)

## test
- local range: 2025-12-01 00:00:00+08:00 -> 2025-12-31 23:00:00+08:00
- hours: 744
- 168h windows: 577
- test_actual.npy: shape=(577, 168, 3)
- test_forecast.npy: shape=(577, 168, 3)
- test_residual.npy: shape=(577, 168, 3)
- test_time_mark.npy: shape=(577, 168, 8)
- test_true.npy: shape=(577, 168, 11)
- test_pred.npy: shape=(577, 168, 11)
- test_res.npy: shape=(577, 168, 11)
