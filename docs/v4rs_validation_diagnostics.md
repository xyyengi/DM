# V4-RS validation 与 checkpoint 诊断

本轮工具只读取已有 checkpoint，不训练或更新模型。

## validation 场景生成

`generate.py` 现在显式支持 `--split val` 和 `--split test`。validation 在未指定 `--output_dir` 时会自动写入独立目录，避免覆盖训练目录。

```bash
python generate.py \
  --save_path outputs_shandong \
  --exp_name 20260720_233611_v4rs_residual_standardized_no_guidance_168h \
  --data_path diffusion_npy_normalized \
  --split val \
  --n_samples 20 \
  --batch_size 4 \
  --reverse_variance_type posterior \
  --seed 2026
```

所有生成配置、`metrics.json` 和 `denormalization_used.json` 都会记录 `data_split=val`。

## 无训练 checkpoint 诊断

```bash
python tools/audit_checkpoint_denoising.py \
  --train-dir outputs_shandong/20260720_233611_v4rs_residual_standardized_no_guidance_168h \
  --data-path diffusion_npy_normalized \
  --output-dir outputs_shandong/event_evaluation/v4rs_checkpoint_diagnostic_val_full \
  --batch-size 8 \
  --max-windows 553
```

诊断固定同一组前向噪声，并比较：

- 正常条件；
- 跨样本打乱 forecast；
- forecast 置零；
- 跨样本打乱8通道日历编码；
- 日历编码置零；
- forecast 与日历编码同时置零。

输出：

- `timestep_condition_metrics.csv`：逐扩散时刻、逐通道结果；
- `timestep_bin_condition_summary.csv`：五个时刻区间汇总；
- `timestep_denoising_quality.png`：噪声预测与单步 x0 重建；
- `condition_ablation.png`：条件消融；
- `reverse_trajectory_model_space.png`：完整500步反向轨迹；
- `reverse_trajectory_actual_space.png`：映射到实际功率空间的轨迹；
- `diagnostic_report.json`：checkpoint、validation索引和全部口径。

注意：高噪声时刻的“单步 x0 重建误差”会被 `1/sqrt(alpha_hat)` 数学放大，不能单独据此认定500步失败；必须结合逐步反向轨迹和最终概率指标判断。
