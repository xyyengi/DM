# AGENT_DECOUPLE_VERSIONS.md

# 给 VSCode / Codex Agent 的指令：新分支解耦 V0 / V1 / V2 / Vmix

## 0. 先新建分支

请基于当前代码新开一个 git 分支，专门做“版本解耦重构”。

```bash
git status
git checkout -b refactor_decouple_v0_v1_v2_vmix
```

如果当前工作区有未提交修改，请先停止并告诉我，不要强行切分支或覆盖代码。

---

## 1. 当前项目状态

当前代码已经确认不是纯 V0、V1 或 V2，而是混合版本：

```text
当前训练目标：
  x0 = residual_3ch

当前网络条件：
  input_14ch = [x_t, forecast_3ch, time_encoding]
  forecast 进入了网络，类似 V2-like concat condition

当前采样引导：
  denoise_step() 中使用 KDE forecast-error interval 对 mean 做 guidance 修正
  类似 V1-like guidance
```

所以当前版本定义为：

```text
Vmix = residual target + forecast concat condition + forecast-error guidance
```

当前数据 residual 的定义已经确认是：

```text
residual = forecast - actual
actual = forecast - residual
```

如果模型生成的是 residual，最终实际场景必须是：

```text
actual_scenario = forecast - generated_residual
```

绝对不能写成：

```text
actual_scenario = forecast + generated_residual
```

---

## 2. 本次重构目标

请不要删除当前混合逻辑，而是把它保留为一个明确版本：

```text
v_mix_residual_forecast_concat_guidance
```

同时拆出三个干净版本：

```text
V0 = v0_uncond_ddpm_actual_168h
V1 = v1_2023_guidance_actual_168h
V2 = v2_csdi_cond_actual_given_forecast_168h
Vmix = v_mix_residual_forecast_concat_guidance
```

本次目标不是提升指标，也不是大改网络结构，而是把版本逻辑拆清楚，让每个版本可以通过 config 独立运行。

---

## 3. 四个版本的严格定义

### 3.1 V0：无条件 DDPM baseline

```text
experiment.name = v0_uncond_ddpm_actual_168h
target.type = actual
condition.mode = none
use_forecast = false
use_network_condition = false
use_guidance = false
```

训练逻辑必须是：

```text
x0 = actual
x_t = q_sample(actual, t, noise)
epsilon_theta = model(x_t, t, cond=None)
loss = MSE(noise, epsilon_theta)
```

采样逻辑必须是：

```text
x_T ~ N(0, I)
不使用 forecast
不使用 residual
不使用 KDE
不使用 cond_matrix
不使用 guidance
最终 sample 直接是 actual scenario
```

V0 中 forecast 不能进入网络。如果当前模型默认构造：

```text
input_14ch = [x_t, forecast_3ch, time_encoding]
```

那么 V0 必须绕开 forecast 通道，或重构模型输入，使 V0 输入只有：

```text
[x_t, time_encoding]
```

如果临时只能把 forecast 通道置零，请明确命名为 pseudo-V0，并告诉我这不是真正纯 V0。优先实现真正 V0。

---

### 3.2 V1：2023-style forecast-error guidance

```text
experiment.name = v1_2023_guidance_actual_168h
target.type = actual
condition.mode = guidance_2023
use_forecast = true
use_network_condition = false
use_guidance = true
```

训练逻辑必须和 V0 一样：

```text
x0 = actual
x_t = q_sample(actual, t, noise)
epsilon_theta = model(x_t, t, cond=None)
loss = MSE(noise, epsilon_theta)
```

注意：

```text
V1 训练时 forecast 不进入网络。
V1 不是 concat forecast。
V1 不是 residual diffusion。
```

V1 的 forecast 只用于采样阶段：

```text
1. 根据历史 forecast 和 actual 计算 forecast-error interval
2. 采样时先由 model 得到 epsilon_theta
3. 根据 epsilon_theta 计算原始 mu_theta
4. 用 forecast-error interval 修正 mu_theta
5. 从修正后的 mean 采样 x_{t-1}
```

V1 必须支持分变量 guidance scale：

```yaml
guidance:
  wind_scale: 0.1
  pv_scale: 0.5
  load_scale: 0.5
```

并且必须支持：

```text
wind_scale = pv_scale = load_scale = 0
```

此时 V1 应退化为 V0 采样逻辑。

---

### 3.3 V2：CSDI-style forecast conditional DDPM

```text
experiment.name = v2_csdi_cond_actual_given_forecast_168h
target.type = actual
condition.mode = csdi_forecast
use_forecast = true
use_network_condition = true
use_guidance = false
```

训练逻辑：

```text
x0 = actual
cond = forecast
x_t = q_sample(actual, t, noise)
epsilon_theta = model(x_t, t, cond=forecast)
loss = MSE(noise, epsilon_theta)
```

采样逻辑：

```text
给定未来 forecast
x_T ~ N(0, I)
epsilon_theta = model(x_t, t, cond=forecast)
逐步去噪
最终 sample 是 actual scenario
```

V2 中：

```text
forecast 可以进入网络
但不能使用 KDE guidance
不能修正 mu_theta
不能把 target 改成 residual
```

V2 需要支持变量级条件 mask：

```yaml
condition:
  cond_mask: [1, 1, 1]
```

以及：

```yaml
condition:
  cond_mask: [0, 1, 1]
```

含义是：

```text
[1,1,1] = wind/pv/load forecast 全部作为条件
[0,1,1] = wind forecast 不作为条件，pv/load forecast 作为条件
[0,0,0] = 不使用 forecast 条件
```

实现时可以：

```python
cond = forecast * cond_mask.view(1, 3, 1)
```

---

### 3.4 Vmix：保留当前混合版本

```text
experiment.name = v_mix_residual_forecast_concat_guidance
target.type = residual
condition.mode = mix
use_forecast = true
use_network_condition = true
use_guidance = true
```

训练逻辑保持当前混合版思想：

```text
x0 = residual = forecast - actual
forecast 进入网络
采样时使用 KDE / forecast-error interval guidance
```

但必须修正最终输出：

```text
generated_residual = model sample
actual_scenario = forecast - generated_residual
```

请在代码注释中明确：

```text
本项目 residual 定义为 forecast - actual
所以 actual = forecast - residual
```

---

## 4. 新增统一配置字段

请整理或新增 config，使四个版本都可以通过配置控制。

建议字段如下：

```yaml
experiment:
  name: "v0_uncond_ddpm_actual_168h"

data:
  variables: ["wind", "pv", "load"]
  length: 168
  freq: "1H"

target:
  type: "actual"   # actual / residual

condition:
  mode: "none"     # none / guidance_2023 / csdi_forecast / mix
  use_forecast: false
  use_network_condition: false
  use_guidance: false
  cond_mask: [0, 0, 0]

guidance:
  enable: false
  method: "forecast_error_interval"
  wind_scale: 0.0
  pv_scale: 0.0
  load_scale: 0.0
  num_bins: 20
  quantile_low: 0.05
  quantile_high: 0.95

model:
  input_channels: 3
  cond_channels: 0

diffusion:
  timesteps: 1000
  beta_schedule: "linear"

train:
  batch_size: 32
  lr: 0.0001
  epochs: 200
  seed: 2026
```

请至少创建四个配置文件：

```text
configs/v0_uncond_ddpm_actual_168h.yaml
configs/v1_2023_guidance_actual_168h.yaml
configs/v2_csdi_cond_actual_given_forecast_168h.yaml
configs/v_mix_residual_forecast_concat_guidance.yaml
```

如果项目原来不是 configs 结构，请尽量适配，但要保持四个版本配置独立。

---

## 5. 数据与 residual 符号修正

当前文件关系：

```text
pred.npy / train_pred.npy / val_pred.npy / test_pred.npy = forecast
true.npy = actual
train_res.npy / val_res.npy / test_res.npy = residual = forecast - actual
```

请全局搜索以下关键词：

```text
residual
res
actual_3ch
forecast_3ch
pred
true
x0
target
```

重点检查所有地方是否存在：

```python
actual = forecast + residual
actual_3ch = forecast_3ch + residual_3ch
scenario = forecast + generated_residual
```

如果存在，必须改为：

```python
actual = forecast - residual
actual_3ch = forecast_3ch - residual_3ch
scenario = forecast - generated_residual
```

注意：

```text
不要覆盖原始 .npy 文件
不要改变 train_res.npy / val_res.npy / test_res.npy 的值
只在加载和输出换算时正确解释 residual 符号
```

请在数据加载处加入注释：

```python
# In this project, residual is defined as:
# residual = forecast - actual
# Therefore:
# actual = forecast - residual
```

---

## 6. 模型接口要求

请尽量统一模型接口：

```python
epsilon_theta = model(x_t, t, cond=None)
```

其中：

```text
x_t.shape = [B, 3, 168]
t.shape = [B]
cond = None 或 [B, 3, 168]
```

四个版本对应：

```text
V0:
  model(x_t, t, cond=None)

V1:
  model(x_t, t, cond=None)
  guidance 只在采样 mean 阶段启用

V2:
  model(x_t, t, cond=forecast)

Vmix:
  model(x_t, t, cond=forecast)
  target = residual
  guidance 启用
```

如果当前模型内部仍然使用 input_14ch，请把构造逻辑封装成一个函数，例如：

```python
build_model_input(x_t, t, cond, config)
```

并根据配置决定是否拼接 forecast。

不要在多个地方手写 input concat，避免后续失控。

---

## 7. 采样输出统一

所有版本的最终输出都应该保存 actual scenario，而不是残差。

保存路径：

```text
outputs/{experiment.name}/samples/scenarios.npz
```

必须包含：

```python
wind: [N, 168]
pv: [N, 168]
load: [N, 168]
prob: [N]
forecast_wind: [168]
forecast_pv: [168]
forecast_load: [168]
```

不同版本的输出换算：

```text
V0:
  generated_sample = actual_scenario

V1:
  generated_sample = actual_scenario

V2:
  generated_sample = actual_scenario

Vmix:
  generated_sample = generated_residual
  actual_scenario = forecast - generated_residual
```

---

## 8. 必须添加的日志与 assert

所有版本训练开始时必须打印：

```text
experiment.name
target.type
condition.mode
use_forecast
use_network_condition
use_guidance
residual definition
```

训练 batch 中必须打印或 assert：

```text
actual shape
forecast shape
residual shape
x0 shape
x_t shape
noise shape
epsilon_theta shape
```

要求：

```python
assert x0.ndim == 3
assert x0.shape[1] == 3
assert x0.shape[2] == 168
```

如果使用 forecast：

```python
assert forecast.shape == x0.shape
```

如果 target.type == residual：

```python
# residual = forecast - actual
reconstructed_actual = forecast - residual
```

如果 true 对齐可用，请计算：

```text
MAE(reconstructed_actual, true_slice)
```

---

## 9. 本次重构完成后的报告要求

请本次不要直接训练模型。先完成版本解耦和配置整理，然后输出报告。

报告必须包括：

```text
1. 当前 git 分支名；
2. 修改了哪些文件；
3. 新增了哪些文件；
4. 四个版本的 config 路径；
5. 每个版本的 target 是 actual 还是 residual；
6. 每个版本 forecast 是否进入网络；
7. 每个版本 guidance 是否启用；
8. 每个版本最终 scenario 如何从模型输出得到；
9. residual 符号在哪里修正；
10. 是否还有 forecast + residual 的旧逻辑残留；
11. 每个版本对应的运行命令；
12. 当前还不能确定的问题。
```

请用表格总结：

```text
version | target | forecast_in_network | guidance | model_output | final_scenario
V0      | actual | no                  | no       | actual       | output
V1      | actual | no                  | yes      | actual       | output
V2      | actual | yes                 | no       | actual       | output
Vmix    | residual | yes               | yes      | residual     | forecast - output
```

---

## 10. 完成后等待确认

请完成上述“版本解耦重构”后停止，不要自动开始训练，不要继续实现新模型细节。

等我确认四个版本逻辑正确后，再进入下一步：

```text
V0/V1 最小训练与采样对比
```
