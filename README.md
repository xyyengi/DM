# 多变量协同条件扩散模型

基于论文 "Conditional Diffusion Model" 的风电、光伏、负荷**场景生成**模型。

## 项目结构

```
DM_local/
├── train.py                    # 训练脚本
├── generate.py                 # 场景生成脚本
├── dataset_multivariate.py     # 数据集处理
├── diff_models_multivariate.py # 扩散模型定义
├── config/
│   └── wind_scenario.yaml      # 配置文件
├── input_4.27/                 # 数据目录
│   ├── train_pred.npy          # 训练集预测值 (18917, 168, 11)
│   ├── train_res.npy           # 训练集残差
│   ├── val_pred.npy            # 验证集预测值
│   ├── val_res.npy             # 验证集残差
│   ├── test_pred.npy           # 测试集预测值
│   └── test_res.npy            # 测试集残差
├── save/                       # 模型保存目录
├── 2023-Conditional_Diffusion_Model.pdf  # 论文
├── README.md
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 基础训练
python train.py --exp_name my_experiment

# 自定义参数
python train.py --exp_name test --epochs 100 --patience 5 --batch_size 32
```

**训练参数：**
- `--config`: 配置文件路径 (默认: config/wind_scenario.yaml)
- `--data_path`: 数据目录 (默认: ./input_4.27/)
- `--save_path`: 模型保存目录 (默认: ./save/)
- `--exp_name`: 实验名称
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--batch_size`: 批大小
- `--patience`: 早停耐心值 (默认: 5)
- `--save_every`: checkpoint保存间隔 (默认: 50)

### 3. 生成场景

```bash
# 使用最佳模型生成场景
python generate.py --exp_name my_experiment

# 指定epoch和生成样本数
python generate.py --exp_name my_experiment --ckpt_epoch 100 --n_samples 50
```

**生成参数：**
- `--exp_name`: 实验名称或文件夹名
- `--ckpt_epoch`: 指定epoch (默认使用最佳模型)
- `--n_samples`: 生成场景样本数 (默认: 10)

## 模型架构

### 输入结构 (14通道)

| 通道 | 内容 | 说明 |
|------|------|------|
| 0-2 | Target Residuals | 风、光、负荷残差 (扩散目标) |
| 3-5 | Base Prediction | FEDformer预测趋势 |
| 6-13 | Time Encoding | 8维时间周期特征 |

### 条件构建 (论文公式9)

```
c_up = min(1, f + K_h(f))
c_down = max(0, f - K_h(f))
```

其中 K_h(f) 为核密度估计的误差期望。

### 输出

- **生成场景**: (N, n_samples, 3, 168) - 多个可能的残差场景
- **评估指标**: Energy Score, Coverage

## 场景生成 vs 预测

| 预测 | 场景生成 |
|------|----------|
| 输出单一确定值 | 输出多个可能场景 |
| 无法量化不确定性 | 可量化不确定性范围 |
| 传统方法 | 扩散模型方法 |

本项目使用扩散模型生成**多个可能的残差场景**，用于量化风电、光伏、负荷的不确定性。

## 实验结果

训练完成后，结果保存在 `save/run_{exp_name}_{timestamp}/`:
- `checkpoints/model_best.pt`: 最佳模型
- `checkpoints/model_epoch_X.pt`: 定期checkpoint
- `logs/train_log.txt`: 训练日志
- `results/generate_{timestamp}/`: 生成的场景

## 参考文献

- 论文: 2023-Conditional_Diffusion_Model.pdf
- 基于CSDI (Conditional Score-based Diffusion Model) 改进

**查询和选择模型的方法：**

**1. 列出所有可用实验：**
```bash
python generate.py --list
```
输出示例：
```
======================================================================
可用实验列表
======================================================================
  run_wind_scenario_20260508_0812
    最佳epoch: 150, Val Loss: 0.0234
  run_test_20260507_1534
    最佳epoch: 89, Val Loss: 0.0456
======================================================================
```

**2. 使用最佳模型生成：**
```bash
# 默认使用最佳模型
python generate.py --exp_name wind_scenario --n_samples 50

# 或指定完整文件夹名
python generate.py --exp_name run_wind_scenario_20260508_0812 --n_samples 50
```

**3. 使用指定epoch的模型：**
```bash
python generate.py --exp_name wind_scenario --ckpt_epoch 100 --n_samples 50
```

**选择逻辑：**
- 不指定`--ckpt_epoch`：自动使用`model_best.pt`（验证损失最低的模型）
- 指定`--ckpt_epoch`：使用`model_epoch_{epoch}.pt`

## Experiment Runbook: Train -> Generate -> Summary

Each run writes to `outputs/{run_id}/`, where `run_id` is generated as:

```text
{timestamp}_{experiment_name}
```

For each version, the recommended sequence is:

```text
train.py -> generate.py -> src/eval/collect_experiments.py
```

`generate.py --exp_name` should use the full `run_id` directory name, not only the short experiment name. The command blocks below find the newest matching `run_id` automatically.

### PowerShell: one version

Replace the config and experiment name as needed:

```powershell
$EXP = "v0_uncond_ddpm_actual_168h"
$CONFIG = "configs/v0_uncond_ddpm_actual_168h.yaml"
$DATA = "input_4.27"

python train.py --config $CONFIG --data_path $DATA --save_path outputs --epochs 1 --batch_size 8 --exp_name $EXP
$RUN_ID = (Get-ChildItem -Directory outputs | Where-Object { $_.Name -like "*_$EXP" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
python generate.py --save_path outputs --exp_name $RUN_ID --data_path $DATA --n_samples 2 --max_batches 1
python src/eval/collect_experiments.py --outputs_dir outputs
```

### PowerShell: four-version smoke test

This runs V0/V1/V2/Vmix with 1 epoch and one generate batch. It is for CPU/local smoke tests, not final results.

```powershell
$DATA = "input_4.27"
$EPOCHS = 1
$BATCH = 8
$NSAMPLES = 2
$MAX_BATCHES = 1

$RUNS = @(
  @{ Exp = "v0_uncond_ddpm_actual_168h"; Config = "configs/v0_uncond_ddpm_actual_168h.yaml" },
  @{ Exp = "v1_2023_guidance_actual_168h"; Config = "configs/v1_2023_guidance_actual_168h.yaml" },
  @{ Exp = "v2_csdi_cond_actual_given_forecast_168h"; Config = "configs/v2_csdi_cond_actual_given_forecast_168h.yaml" },
  @{ Exp = "v_mix_residual_forecast_concat_guidance"; Config = "configs/v_mix_residual_forecast_concat_guidance.yaml" }
)

foreach ($R in $RUNS) {
  python train.py --config $R.Config --data_path $DATA --save_path outputs --epochs $EPOCHS --batch_size $BATCH --exp_name $R.Exp
  $RUN_ID = (Get-ChildItem -Directory outputs | Where-Object { $_.Name -like "*_$($R.Exp)" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
  python generate.py --save_path outputs --exp_name $RUN_ID --data_path $DATA --n_samples $NSAMPLES --max_batches $MAX_BATCHES
  python src/eval/collect_experiments.py --outputs_dir outputs
}
```

### Bash: server sequential run

Adjust `EPOCHS`, `PATIENCE`, `BATCH`, `NSAMPLES`, and `DATA` before running on the server. Remove `--max_batches` for full test-set generation.

For a first overnight run, `NSAMPLES=20` is usually a safer default than `50`: training time is unchanged, but generation/evaluation time and output size are lower. Use `NSAMPLES=50` later for the final comparison if the first pass looks stable.

```bash
DATA=input_4.27
EPOCHS=150
PATIENCE=15
BATCH=64
NSAMPLES=20

run_one () {
  EXP="$1"
  CONFIG="$2"

  python train.py --config "$CONFIG" --data_path "$DATA" --save_path outputs --epochs "$EPOCHS" --patience "$PATIENCE" --batch_size "$BATCH" --exp_name "$EXP"
  RUN_ID=$(ls -td outputs/*_"$EXP" | head -n 1 | xargs basename)
  python generate.py --save_path outputs --exp_name "$RUN_ID" --data_path "$DATA" --n_samples "$NSAMPLES"
  python src/eval/collect_experiments.py --outputs_dir outputs
}

run_one v0_uncond_ddpm_actual_168h configs/v0_uncond_ddpm_actual_168h.yaml
run_one v1_2023_guidance_actual_168h configs/v1_2023_guidance_actual_168h.yaml
run_one v2_csdi_cond_actual_given_forecast_168h configs/v2_csdi_cond_actual_given_forecast_168h.yaml
run_one v_mix_residual_forecast_concat_guidance configs/v_mix_residual_forecast_concat_guidance.yaml
```

To run the four versions unattended overnight, put the block above into a script, for example `run_all_versions.sh`, then use one of:

```bash
bash run_all_versions.sh 2>&1 | tee overnight_run.log
```

or:

```bash
nohup bash run_all_versions.sh > overnight_run.log 2>&1 &
tail -f overnight_run.log
```

After all runs:

```bash
python src/eval/collect_experiments.py --outputs_dir outputs
```

The consolidated table is:

```text
outputs/experiment_summary.csv
```
