# V4 posterior variance 与 V4-RS 服务器实验

本分支包含两个严格分开的实验：

1. 使用现有 V4-s checkpoint，对比 `beta` 与 `posterior` 反向方差；不重新训练。
2. 训练分通道残差标准化版本 V4-RS；不使用 guidance、事件采样或事件标签。

分支名：

```bash
experiment/v4-posterior-residual-standardization
```

## 0. 更新代码

```bash
git fetch origin
git switch experiment/v4-posterior-residual-standardization
git pull
conda activate dm_env
```

## 1. 先做现有 checkpoint 的采样方差对照

默认使用现有 V4-s checkpoint，并分别生成50个 beta 场景和50个 posterior场景。两组使用相同随机种子，不覆盖原结果。

```bash
bash run_v4_posterior_ablation.sh
```

如果服务器上的 V4-s 文件夹名称不同：

```bash
SOURCE_RUN=你的V4s文件夹名 bash run_v4_posterior_ablation.sh
```

显存不足时只降低生成 batch，不改变模型或场景数量：

```bash
GEN_BATCH=2 NSAMPLES=50 bash run_v4_posterior_ablation.sh
```

输出为两个新目录：

```text
outputs_shandong/<source>_regen_beta_n50_seed2026/
outputs_shandong/<source>_regen_posterior_n50_seed2026/
```

快速比较：

```bash
python tools/compare_reverse_variance.py \
  outputs_shandong/<beta目录> \
  outputs_shandong/<posterior目录> \
  --output outputs_shandong/reverse_variance_comparison.json
```

主要检查：总CRPS、分通道90%覆盖率、90%宽度，以及负荷场景内部RMS离散度。posterior并不预设一定更好；若区间缩窄但CRPS或极端事件明显恶化，则不采用。

## 2. 训练 V4-RS

V4-RS只改变扩散残差目标：使用训练集唯一小时序列拟合风、光、荷各自的均值和标准差。validation/test严格复用训练统计量。

```bash
bash run_v4rs.sh
```

默认参数：

```text
epochs=150
patience=15
train batch=64
n_samples=50
generation batch=4
seed=2026
```

显存不足时：

```bash
BATCH=32 GEN_BATCH=2 bash run_v4rs.sh
```

训练完成后，同一个 V4-RS checkpoint 会分别生成 beta 与 posterior 两套50成员结果，避免把残差标准化效果和采样方差效果混在一起。

## 3. 后台运行与日志

先运行方差对照：

```bash
mkdir -p logs
LOG=logs/v4_posterior_$(date +%Y%m%d_%H%M%S).log
nohup env PYTHONUNBUFFERED=1 bash run_v4_posterior_ablation.sh > "$LOG" 2>&1 < /dev/null &
echo $! | tee "${LOG%.log}.pid"
echo "$LOG"
```

查看日志：

```bash
tail -f "$LOG"
```

方差对照确认后，再后台训练 V4-RS：

```bash
LOG=logs/v4rs_$(date +%Y%m%d_%H%M%S).log
nohup env PYTHONUNBUFFERED=1 bash run_v4rs.sh > "$LOG" 2>&1 < /dev/null &
echo $! | tee "${LOG%.log}.pid"
echo "$LOG"
```

## 4. 必须保留的文件

V4-RS训练目录：

- `config_used.yaml`
- `residual_standardization.json`
- `checkpoints/model_best.pt`
- `logs/train_log.txt`

每个生成结果目录：

- `generation_config_used.yaml`
- `denormalization_used.json`
- `actual_scenarios.npy`
- `actual_data.npy`
- `forecast_data.npy`
- `generated_samples_normalized.npy`
- `generated_samples_standardized.npy`（仅V4-RS）
- `metrics.json`
- `samples/scenarios.npz`

`generated_samples_normalized.npy`始终保存反标准化后的容量归一化残差；V4-RS的模型空间输出单独保存在 `generated_samples_standardized.npy`，避免坐标含义混淆。
