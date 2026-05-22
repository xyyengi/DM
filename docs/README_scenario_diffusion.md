# 风光负荷场景生成扩散模型复现与改造说明

## 1. 项目目标

本项目面向电力系统周尺度机组组合与电力电量平衡计算，目标是基于历史风电、光伏、负荷实际出力及其预测信息，生成能够反映不确定性的风光负荷场景集合。

最终输出不是单条预测曲线，而是一组可能发生的场景：

```text
输入：未来一周风电/光伏/负荷预测曲线、历史实际曲线、可选天气/日期特征
输出：N 个未来一周风光负荷联合场景
形状：N × 3 × 168
```

其中：

```text
N：场景数量，例如 50、100、200
3：风电、光伏、负荷三类变量
168：周尺度 168 小时
```

如果原始数据是 15 分钟分辨率，应先明确是否重采样为小时级。如果目标是周尺度机组组合，建议主版本使用小时级 168 点；15 分钟数据可以用于扩展实验或更细粒度场景生成。

---

## 2. 核心建模对象

### 2.1 样本定义

单个训练样本建议定义为：

```text
x0 = actual_power_window
shape = [C, L]
```

其中：

```text
C = 3，表示 wind / pv / load
L = 168，表示一周 168 小时
```

即：

```text
x0[0, :] = 风电实际出力
x0[1, :] = 光伏实际出力
x0[2, :] = 负荷实际功率
```

如果做单变量实验：

```text
C = 1
L = 168
```

### 2.2 条件定义

条件不应和生成目标混淆。建议把条件记为：

```text
cond = forecast_or_auxiliary_information
```

常见条件包括：

```text
forecast_wind : 风电预测曲线，shape [1, 168]
forecast_pv   : 光伏预测曲线，shape [1, 168]
forecast_load : 负荷预测曲线，shape [1, 168]
calendar_feat : 小时、星期、是否周末、节假日等
weather_feat  : 温度、辐照、风速、天气型、寒潮/高温标签等
```

主版本建议先只使用预测曲线作为条件：

```text
cond = forecast_power_window
shape = [3, 168]
```

扩展版本再加入天气、日期、极端天气标签。

---

## 3. 需要重点澄清：扩散模型输入到底是预测值还是真实值？

在标准 DDPM 训练中，被加噪的对象是目标数据 x0。对于场景生成任务，x0 应该是历史真实出力曲线，而不是预测曲线。

也就是说：

```text
训练扩散模型时：
x0 = 历史实际风光负荷曲线
xt = 对 x0 加噪后的曲线
模型学习：从 xt 预测噪声 epsilon
```

预测值的作用取决于条件方案：

### 方案 A：2023 条件引导法

```text
x0       = 历史实际出力曲线
forecast = 历史预测出力曲线
error    = actual - forecast
cond_c   = 由 forecast 与误差分布构造出的上下界区间
```

训练扩散模型本体时，主要对 actual 加噪；forecast 用于估计预测误差分布，并在采样阶段修正反向去噪均值。

### 方案 B：2021 CSDI 式条件输入法

```text
x0       = 历史实际出力曲线
cond     = 历史预测出力曲线
模型学习：p(actual | forecast)
```

此时预测曲线作为神经网络输入，参与噪声估计：

```text
epsilon_theta = model(xt, t, cond)
```

因此，在本项目中应明确区分：

```text
actual：扩散模型要生成的目标
forecast：用于约束或条件输入的信息
residual = actual - forecast：可作为一种建模目标或辅助分析对象，但不是默认必须生成的对象
```

---

## 4. 三类模型版本

建议项目先拆成三个可复现实验版本，避免一开始混在一起。

### V0：无条件 DDPM baseline

目标：

```text
学习 p(actual)
```

训练：

```text
actual x0
  -> 加噪得到 xt
  -> model(xt, t) 预测 epsilon
  -> MSE(epsilon, epsilon_theta)
```

生成：

```text
随机噪声 xT
  -> 逐步去噪
  -> 生成 actual-like scenario
```

用途：

```text
检验扩散模型本体是否能学到风光负荷历史场景分布
```

### V1：2023 forecast-error guidance 条件引导法

目标：

```text
学习 p(actual)，采样时用 forecast-error interval 约束生成结果
```

训练：

```text
仍然用 actual 训练 DDPM 噪声预测器
```

额外统计：

```text
error = actual - forecast
按 forecast 分箱，估计 error distribution
得到每个 forecast 水平下的误差范围 K_h(f)
构造 c_down, c_up
```

生成：

```text
xT
  -> model 预测 epsilon_theta
  -> 计算 mu_theta
  -> 用条件区间 c 修正 mu_theta
  -> 得到 x_{t-1}
```

用途：

```text
复现 2023 论文条件引导思想
检验区间约束是否改善覆盖率、区间宽度、能量评分、ACF 等指标
```

### V2：2021 CSDI-inspired 条件输入法

目标：

```text
学习 p(actual | forecast)
```

训练：

```text
actual x0 加噪得到 xt
cond = forecast
epsilon_theta = model(xt, t, cond)
loss = MSE(epsilon, epsilon_theta)
```

生成：

```text
给定未来一周 forecast
xT ~ N(0, I)
反复调用 model(xt, t, forecast)
生成 actual scenarios
```

用途：

```text
检验“预测曲线作为网络条件输入”是否比“2023 区间引导”更有效
```

---

## 5. 数据处理建议

### 5.1 原始数据字段

建议统一整理成如下字段：

```text
timestamp
actual_wind
actual_pv
actual_load
forecast_wind
forecast_pv
forecast_load
```

可选字段：

```text
temperature
wind_speed
irradiance
weather_type
holiday_flag
extreme_weather_flag
```

### 5.2 时间频率

目标是周尺度机组组合，因此主版本建议使用小时级：

```text
freq = 1H
window_length = 168
```

如果原始数据是 15min：

```text
wind/pv: 可用均值重采样到小时
load: 可用均值或整点值，需保持与调度模型一致
```

### 5.3 滑动窗口

用连续周样本构造训练集：

```text
sample_0 = 第 1~168 小时
sample_1 = 第 169~336 小时
...
```

如果数据量不足，可以使用滑动窗口：

```text
stride = 24 小时 或 168 小时
```

建议先用非重叠周窗口保证样本独立性，再尝试 stride=24 扩充训练样本。

### 5.4 归一化

建议每个变量单独归一化：

```text
wind_actual / wind_capacity
pv_actual   / pv_capacity
load_actual / load_base 或 max_load
```

为了让生成结果便于约束在合理范围：

```text
wind, pv 建议归一化到 [0, 1]
load 可以归一化到 [0, 1] 或标准化后再反归一
```

如果模型输出的是 actual 场景，最终必须反归一化回 MW。

### 5.5 数据泄漏检查

训练预测模型和训练扩散模型时必须避免未来信息泄漏：

```text
forecast[t] 只能是当时可获得的预测值
actual[t] 只能作为训练标签，不能作为未来条件
归一化参数只能从训练集拟合，然后应用到验证/测试
```

---

## 6. 推荐项目结构

```text
project/
├── README.md
├── configs/
│   ├── base.yaml
│   ├── ddpm_uncond.yaml
│   ├── guidance_2023.yaml
│   └── csdi_cond.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   ├── dataset.py
│   │   └── normalization.py
│   ├── models/
│   │   ├── res_unet_1d.py
│   │   ├── diffusion.py
│   │   ├── condition_encoder.py
│   │   └── guidance_2023.py
│   ├── train/
│   │   ├── train_ddpm.py
│   │   ├── train_cond.py
│   │   └── train_forecast.py
│   ├── sample/
│   │   ├── sample_uncond.py
│   │   ├── sample_guidance.py
│   │   └── sample_cond.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── plot_scenarios.py
│   │   └── uc_export.py
│   └── utils/
│       ├── seed.py
│       └── io.py
├── scripts/
│   ├── 01_preprocess.sh
│   ├── 02_train_forecast.sh
│   ├── 03_train_ddpm.sh
│   ├── 04_sample.sh
│   └── 05_eval.sh
└── outputs/
    ├── checkpoints/
    ├── scenarios/
    ├── figures/
    └── logs/
```

---

## 7. 模块职责

### data/preprocess.py

负责：

```text
读取原始数据
统一时间戳
重采样到小时
处理缺失值
构造 actual 与 forecast 对齐表
保存 processed parquet/csv
```

### data/dataset.py

负责：

```text
按窗口切分为 [C, 168]
返回 actual, forecast, optional_features
```

返回格式建议：

```python
{
    "actual": Tensor[C, L],
    "forecast": Tensor[C, L],
    "time_feat": Tensor[F_time, L],
    "weather_feat": Tensor[F_weather, L],
}
```

### models/diffusion.py

负责：

```text
beta schedule
q_sample(x0, t, noise)
p_mean_variance(...)
p_sample(...)
sample_loop(...)
training_loss(...)
```

### models/res_unet_1d.py

负责：

```text
输入 xt, t, optional cond
输出 epsilon_theta
```

建议接口：

```python
epsilon_theta = model(x_t, t, cond=None)
```

### models/guidance_2023.py

负责：

```text
根据 forecast 与历史 error 构造 c_down/c_up
采样时修正 mu_theta
```

### models/condition_encoder.py

负责：

```text
把 forecast / weather / calendar 编码后传给 Res-UNet
用于 CSDI-inspired 条件输入版本
```

### eval/metrics.py

建议至少实现：

```text
MAE/RMSE of mean or median scenario
coverage rate
PINAW/PIAW interval width
energy score
ACF similarity
ramp distribution similarity
correlation matrix error
extreme event coverage
```

---

## 8. 实验设计

### 实验 0：确定预测模型 baseline

```text
Fedformer forecast:
输入历史 actual
输出未来一周 forecast
```

如果已有业务预测值，优先使用业务预测值；如果没有，则使用 Fedformer 生成预测曲线。

注意：扩散模型不是替代预测模型，而是在预测基础上生成不确定性场景。预测越差，条件扩散模型越难学到稳定的 p(actual | forecast)。

### 实验 1：无条件 DDPM

```text
model = DDPM(actual)
condition = None
```

目的：

```text
确认模型能生成合理的周尺度风光负荷曲线
```

### 实验 2：2023 条件引导

```text
model = DDPM(actual)
condition = forecast-error interval
```

目的：

```text
确认 forecast-error guidance 是否提升覆盖率和贴近预测背景
```

### 实验 3：CSDI-inspired 条件输入

```text
model = DDPM(actual | forecast)
condition = forecast curve
```

目的：

```text
确认直接把 forecast 输入网络是否优于区间引导
```

### 实验 4：残差建模可选对照

可以尝试：

```text
target = residual = actual - forecast
diffusion 生成 residual scenarios
scenario = forecast + residual_scenario
```

该版本不是 2023 论文默认逻辑，但在工程上常见且可能有效。需要作为单独实验，不要和论文复现混在一起。

---

## 9. 输出给机组组合的场景格式

建议保存为：

```text
outputs/scenarios/scenarios_test_week_YYYYMMDD.npz
```

包含：

```python
{
    "wind": shape [N, 168],
    "pv": shape [N, 168],
    "load": shape [N, 168],
    "forecast_wind": shape [168],
    "forecast_pv": shape [168],
    "forecast_load": shape [168],
    "prob": shape [N],
}
```

初始可设等概率：

```text
prob = 1 / N
```

后续可加入场景削减与概率重分配。

---

## 10. 当前最重要的工程原则

1. 先保证数据管线正确，再改模型。
2. 先实现无条件 DDPM，再加 2023 guidance，再加 CSDI-style condition。
3. 不要同时改目标变量、条件输入、网络结构和评价指标。
4. 每个实验必须保存 config、随机种子、checkpoint、生成场景和评价结果。
5. 明确 actual、forecast、residual 三者的角色，不要混用。
6. 最终服务对象是机组组合与电力电量平衡，因此评价指标不能只看曲线好不好看，还要看场景覆盖、极端场景、爬坡和相关性。