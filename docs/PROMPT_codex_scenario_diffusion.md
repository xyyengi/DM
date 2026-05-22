# Prompt.md：给 Codex / 编程助手的项目改造提示词

你正在协助我改造一个风光负荷场景生成项目。请严格遵守以下建模逻辑和工程约束。

## 1. 项目背景

我的研究任务是电力系统风光负荷功率场景生成。生成的场景将作为周尺度机组组合与电力电量平衡计算的输入。

场景长度为 168 小时，变量包括：

```text
wind, pv, load
```

输出场景形状为：

```text
[num_scenarios, 3, 168]
```

## 2. 关键概念

请在代码中严格区分：

```text
actual   = 历史真实出力或负荷，是扩散模型要生成的目标
forecast = 预测出力或预测负荷，是条件信息
residual = forecast - actual，是可选实验目标，不是默认目标
actual = forecast - residual
```

默认情况下，扩散模型训练目标是 actual，而不是 forecast，也不是 residual。

## 3. 需要实现的三个版本

### V0：无条件 DDPM

训练：

```text
x0 = actual
xt = q_sample(x0, t, noise)
epsilon_theta = model(xt, t)
loss = MSE(noise, epsilon_theta)
```

采样：

```text
xT ~ N(0, I)
逐步 denoise 得到 scenario
```

### V1：2023 forecast-error guidance

训练：

```text
仍然用 actual 训练 DDPM
```

额外模块：

```text
根据历史 forecast 和 actual 计算 error = forecast - actual
按照 forecast 分箱或 KDE 估计误差范围
构造 c_down, c_up
采样时对 mu_theta 做 guidance 修正
```

接口建议：

```python
mu_guided = apply_forecast_error_guidance(mu, x_t, forecast, t, guidance_config)
```

注意：不要把 2023 guidance 写成普通 concat condition，二者是不同方法。

### V2：CSDI-inspired conditional DDPM

训练：

```text
x0 = actual
cond = forecast
xt = q_sample(x0, t, noise)
epsilon_theta = model(xt, t, cond=forecast)
loss = MSE(noise, epsilon_theta)
```

采样：

```text
给定未来 forecast
xT ~ N(0, I)
epsilon_theta = model(xt, t, cond=forecast)
逐步去噪生成 actual scenarios
```

接口建议：

```python
epsilon_theta = model(x_t, t, cond=forecast)
```

## 4. 数据处理要求

请把数据处理单独模块化，不要写在训练脚本里。

建议实现：

```text
src/data/preprocess.py
src/data/dataset.py
src/data/normalization.py
```

处理步骤：

```text
读取原始 csv/parquet
统一 timestamp
重采样到 1H
检查缺失与异常
对 actual 和 forecast 对齐
构造长度 168 的窗口
划分 train/val/test
保存处理后的数据
```

Dataset 返回：

```python
{
    "actual": Tensor[3, 168],
    "forecast": Tensor[3, 168],
    "time_feat": optional Tensor[F, 168],
    "meta": optional dict
}
```

## 5. 模型接口要求

所有模型尽量使用统一接口：

```python
epsilon_theta = model(x_t, t, cond=None)
```

其中：

```text
x_t: Tensor[B, C, L]
t: Tensor[B]
cond: None 或 Tensor[B, C_cond, L]
```

V0 中 cond=None。

V2 中 cond=forecast。

V1 中 model 仍可 cond=None，guidance 在 p_sample 阶段修改 mu。

## 6. Diffusion 模块要求

请把扩散过程封装为类：

```python
class GaussianDiffusion1D:
    def q_sample(self, x0, t, noise=None): ...
    def predict_start_from_noise(self, x_t, t, noise): ...
    def p_mean_variance(self, model, x_t, t, cond=None): ...
    def p_sample(self, model, x_t, t, cond=None, guidance=None): ...
    def sample_loop(self, model, shape, cond=None, guidance=None): ...
    def training_loss(self, model, x0, cond=None): ...
```

不要把采样逻辑散落在训练脚本里。

## 7. 配置管理

使用 yaml 配置：

```text
configs/ddpm_uncond.yaml
configs/guidance_2023.yaml
configs/csdi_cond.yaml
```

配置至少包含：

```yaml
data:
  freq: "1H"
  length: 168
  variables: ["wind", "pv", "load"]

model:
  channels: 3
  hidden_dim: 64
  num_res_blocks: 4
  cond_channels: 0

diffusion:
  timesteps: 1000
  beta_schedule: "linear"

train:
  batch_size: 32
  lr: 1e-4
  epochs: 200
  seed: 2026
```

## 8. 评价指标

请实现：

```text
MAE/RMSE of scenario mean or median
coverage rate
interval width
energy score
ACF similarity
ramp distribution
wind/pv/load correlation matrix error
extreme event coverage
```

最终不是只画图，而是输出 csv/json 评价结果。

## 9. 输出格式

生成场景保存为：

```text
outputs/scenarios/{experiment_name}_{test_week}.npz
```

内容包括：

```python
wind: [N, 168]
pv: [N, 168]
load: [N, 168]
prob: [N]
forecast_wind: [168]
forecast_pv: [168]
forecast_load: [168]
```

## 10. 编码原则

- 先跑通 V0 无条件 DDPM。
- 再实现 V1 guidance。
- 最后实现 V2 conditional DDPM。
- 每次只改一个模块。
- 所有 shape 都写 assert。
- 所有保存结果都带 config 备份。
- 所有随机性都设置 seed。
- 不要把 forecast 当作 x0 加噪，除非是在单独的预测模型或残差建模实验中明确这样做。
- 不要把 residual 逻辑混入主线模型，除非配置中 target_type="residual"。
