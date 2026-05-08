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