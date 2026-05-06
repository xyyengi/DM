# CSDI
This is the github repository for the NeurIPS 2021 paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)".

## 项目结构

本项目包含两个主要模块：

1. **原始CSDI模块** - 用于时间序列填补任务（PM25, Physio, Forecasting）
2. **多变量协同条件扩散模块** - 用于风电场景生成（论文复现）

---

## 多变量协同条件扩散模型 (Multi-Channel CDM)

### 论文复现说明

当前代码用于复现论文 `2023-Conditional_Diffusion_Model.pdf`（风电场景生成）。

**核心改进：将单变量条件扩展为风、光、负荷三通道协同条件**

### 论文公式对应

| 公式 | 内容 | 代码实现位置 |
|------|------|-------------|
| 公式7 | 区间划分与条件概率 P(e|f) | `dataset_multivariate.py` - `MultiChannelKDE.fit()` |
| 公式8 | 核密度估计 K_h(x) | `dataset_multivariate.py` - `MultiChannelKDE.get_conditional_interval()` |
| 公式9 | 条件区间构造 c = [c_down, c_up] | `dataset_multivariate.py` - `MultiChannelKDE.get_conditional_interval()` |
| 公式10 | 反向去噪梯度引导 ∇_{x_t} ||γ·x_t - c||²_F | `diff_models_multivariate.py` - `GaussianDiffusionMultivariate.compute_conditional_gradient()` |

### 数据结构

**数据文件结构 (input_4.27):**

| 文件 | 形状 | 说明 |
|------|------|------|
| train_pred.npy | (18917, 168, 11) | 训练集预测值 |
| train_res.npy | (18917, 168, 11) | 训练集残差 |
| val_pred.npy | (2608, 168, 11) | 验证集预测值 |
| val_res.npy | (2608, 168, 11) | 验证集残差 |
| test_pred.npy | (5381, 168, 11) | 测试集预测值 |
| test_res.npy | (5381, 168, 11) | 测试集残差 |

**11维特征定义:**
- Channel [0:3]: 风、光、负荷残差 (Residuals) - 生成核心主体
- Channel [3:11]: 8维时间周期编码 (Sin/Cos) - 环境背景条件

**张量结构**: `(Batch, Channels=11, Length=168)`
**残差计算**: `Residual = Forecast (FEDformer) - Actual`

### 模型架构

1. **Res-UNet + 空洞卷积**: Bottleneck层使用空洞率[1,2,4,8]，感受野覆盖168点
2. **时间特征注入**: 小时、周几、月份三个尺度的Embedding
3. **多通道条件引导**: Frobenius范数梯度修正

### 场景生成原理

**整体流程：**
```
预测值(forecast) → KDE构建条件区间c → 扩散模型生成残差 → 最终场景 = forecast + 残差
```

**详细步骤：**

1. **条件构建 (论文公式7-9)**
   - 基于训练数据的预测值和残差，使用核密度估计(KDE)拟合误差分布
   - 对于每个预测值f，构建条件区间 c = [c_down, c_up]
   - c_up = min(1, f + K_h(f)), c_down = max(0, f - K_h(f))

2. **扩散模型训练**
   - 输入: residual_3ch (风、光、负荷残差, 3×168)
   - 条件: cond_matrix (条件区间, 3×168×2)
   - 目标: 学习从噪声到残差的去噪过程

3. **场景生成 (论文公式10)**
   - 从纯噪声开始，逐步去噪(50步)
   - 每步去噪时，使用条件梯度引导:
     ∇_{x_t} ||γ·x_t - c||²_F
   - 生成多个残差场景(如10个)

4. **最终场景**
   - 生成的残差 + 预测值 = 最终风电/光伏/负荷场景

**mode test 模型加载逻辑：**
- 模型保存路径: `./save/wind_scenario/model_multivariate.pth`
- 如果文件存在: 直接加载预训练模型
- 如果文件不存在: 自动开始训练并保存

### 使用方法

**环境激活:**
```bash
conda activate torch_env
```

**训练模型 (实验留痕):**
```bash
# 训练多变量协同条件扩散模型
# 自动创建时间戳文件夹: run_wind_scenario_[YYYYMMDD_HHMM]/
python exe_wind_scenario.py --mode train --exp_name wind_scenario --n_samples 10
```

**生成场景 (精确加载指定实验):**
```bash
# 方式1: 使用完整实验文件夹名
python exe_wind_scenario.py --mode predict --exp_name run_wind_scenario_20260506_1515 --ckpt_epoch 200

# 方式2: 使用关键字搜索（自动匹配第一个）
python exe_wind_scenario.py --mode predict --exp_name wind_scenario --ckpt_epoch 100
```

**参数说明:**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| --config | config/wind_scenario.yaml | 配置文件路径 |
| --data_path | ./input_4.27/ | 数据目录路径 |
| --save_path | ./save/ | 实验保存基础路径 |
| --exp_name | wind_scenario | 实验名称（用于文件夹命名） |
| --mode | train | 运行模式: train(训练) 或 predict(生成) |
| --ckpt_epoch | 200 | 加载的checkpoint epoch编号 |
| --n_samples | 10 | 每个条件生成的场景数量 |
| --save_every | 50 | checkpoint保存间隔(epoch) |

**实验文件夹结构:**
```
save/
  run_wind_scenario_20260506_1515/       # 时间戳隔离文件夹
    checkpoints/
      model_epoch_50.pt                  # 中间checkpoint
      model_epoch_100.pt
      model_epoch_200.pt                 # 最终checkpoint
    results/
      generated_samples.npy              # 生成结果
      forecast_data.npy
      metrics.txt
      predict_20260506_1600/             # predict模式结果子文件夹
    logs/
      train_log.txt                      # 训练日志
    config_used.yaml                     # 本次实验配置副本
```

**checkpoint内容:**
```python
{
    'epoch': epoch,                      # 当前epoch
    'model_state_dict': model.state_dict(),  # 模型权重
    'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
    'loss': avg_loss,                    # 当前loss
    'config': config                     # 配置字典
}
```

**数据集分配:**
| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| 训练集 | 18,917 | 模型训练 + KDE拟合 |
| 验证集 | 2,608 | 超参数调优 |
| 测试集 | 5,381 | 场景生成与评估 |

### 文件说明

| 文件 | 功能 |
|------|------|
| `dataset_multivariate.py` | 多通道数据集、KDE条件构造 |
| `diff_models_multivariate.py` | Res-UNet、扩散模型、条件梯度引导 |
| `exe_wind_scenario.py` | 训练、生成、评估脚本 |
| `config/wind_scenario.yaml` | 模型配置文件 |

---

## 原始CSDI模块

### 条件c构建方式的差异

| 论文要求 | 当前代码实现 |
|---------|-------------|
| c = [c_down, c_up] 基于预测误差分布的区间 | c = time_embed + feature_embed + cond_mask |
| c_up = min(1, f + K_h(f)) | 时间嵌入 + 特征嵌入 + 条件掩码 |
| c_down = max(0, f - K_h(f)) | CSDI NeurIPS 2021的填补任务实现 |

如需正确复现论文，需要修改 `main_model.py` 中的 `get_side_info` 方法，按照论文公式9构建条件c。


## Requirement

Please install the packages in requirements.txt

## Preparation
### Download the healthcare dataset 
```shell
python download.py physio
```
### Download the air quality dataset 
```shell
python download.py pm25
```

### Download the elecricity dataset 
Please put files in [GoogleDrive](https://drive.google.com/drive/folders/1krZQofLdeQrzunuKkLXy8L_kMzQrVFI_?usp=drive_link) to the "data" folder.

## Experiments 

### training and imputation for the healthcare dataset
```shell
python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### imputation for the healthcare dataset with pretrained model
```shell
python exe_physio.py --modelfolder pretrained --testmissingratio [missing ratio] --nsample [number of samples]
```

### training and imputation for the healthcare dataset
```shell
python exe_pm25.py --nsample [number of samples]
```

### training and forecasting for the electricity dataset
```shell
python exe_forecasting.py --datatype electricity --nsample [number of samples]
```

### Visualize results
'visualize_examples.ipynb' is a notebook for visualizing results.

## Acknowledgements

A part of the codes is based on [BRITS](https://github.com/caow13/BRITS) and [DiffWave](https://github.com/lmnt-com/diffwave)

## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
# 标准训练（200 epochs，默认参数）
python exe_wind_scenario.py --mode train --exp_name wind_scenario

# 自定义实验名称
python exe_wind_scenario.py --mode train --exp_name wind_baseline_v1

# 自定义参数
python exe_wind_scenario.py --mode train --exp_name wind_scenario --n_samples 100 --save_every 20
# 预测
# 使用完整文件夹名
python exe_wind_scenario.py --mode predict --exp_name run_wind_scenario_20260506_XXXX --ckpt_epoch 200 --n_samples 100

# 使用关键字搜索
python exe_wind_scenario.py --mode predict --exp_name wind_scenario --ckpt_epoch 200 --n_samples 100
# 使用nohup后台运行
nohup python exe_wind_scenario.py --mode train --exp_name wind_scenario > train.log 2>&1 &

# 查看训练进度
tail -f train.log
服务器训练命令：

**训练：**
```bash
python exe_wind_scenario.py --mode train --exp_name wind_scenario
```

**预测：**
```bash
python exe_wind_scenario.py --mode predict --exp_name run_wind_scenario_20260506_XXXX --ckpt_epoch 200 --n_samples 100
```

**后台运行：**
```bash
nohup python exe_wind_scenario.py --mode train --exp_name wind_scenario > train.log 2>&1 &
```

GPU会自动检测使用，无需手动指定。