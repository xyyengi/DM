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

- **张量结构**: `(Batch, Channels=3, Length=168)`
- **通道映射**: Channel 0=风电, 1=光伏, 2=负荷
- **残差计算**: `Residual = Forecast (FEDformer) - Actual`

### 模型架构

1. **Res-UNet + 空洞卷积**: Bottleneck层使用空洞率[1,2,4,8]，感受野覆盖168点
2. **时间特征注入**: 小时、周几、月份三个尺度的Embedding
3. **多通道条件引导**: Frobenius范数梯度修正

### 使用方法

```bash
# 训练模型并生成场景
python exe_wind_scenario.py --config config/wind_scenario.yaml --data_path ./wind_solar_load_168_FEDformer/ --mode train --n_samples 10

# 使用预训练模型生成场景
python exe_wind_scenario.py --config config/wind_scenario.yaml --mode test --n_samples 10
```

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
