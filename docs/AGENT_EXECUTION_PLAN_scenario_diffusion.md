# AGENT_EXECUTION_PLAN.md
# 风光负荷 168h 场景生成扩散模型：给 Codex / VSCode Agent 的执行指令

## 0. 你的角色

你是我的代码工程助手。请帮助我整理和改造一个用于“风电、光伏、负荷周尺度场景生成”的扩散模型项目。

我目前深度学习基础较弱，所以你的任务不是炫技，而是帮我把工程结构整理清楚、让实验能稳定复现、让每个版本的逻辑可以解释。

请严格遵守：

1. 不要一上来大规模重写项目。
2. 不要删除我已有代码。
3. 每次只做一个小目标。
4. 每次修改前先说明将修改哪些文件、为什么修改、输入输出 shape 是什么。
5. 所有训练目标、条件输入、采样逻辑必须明确区分。
6. 每个阶段完成后必须给出运行命令和验证方式。

---

## 1. 项目研究目标

本项目研究电力系统风光负荷功率场景生成。

最终目标：

```text
给定未来一周风电、光伏、负荷预测曲线，
生成 N 条未来一周可能发生的风光负荷联合场景，
作为周尺度机组组合 / 电力电量平衡计算的输入。
```

目标输出：

```text
scenarios.shape = [N, 3, 168]
```

其中：

```text
N   = 场景数量，例如 50 / 100 / 200
3   = wind, pv, load
168 = 一周 168 小时
```

当前需要支持三类模型版本：

```text
V0：无条件 DDPM baseline
V1：2023 论文风格 forecast-error guidance
V2：2021 CSDI-inspired forecast 条件输入
```

---

## 2. 必须严格区分的三个概念

请在阅读和修改代码时，先查清楚当前代码中这三个量分别叫什么。

### 2.1 actual

真实历史曲线，是扩散模型要生成的目标。

```text
actual = 历史真实风电 / 光伏 / 负荷曲线
shape  = [B, 3, 168]
```

在扩散模型训练中，被加噪的 x0 默认应该是 actual。

### 2.2 forecast

预测曲线，是条件信息。

```text
forecast = 风电 / 光伏 / 负荷预测曲线
shape    = [B, 3, 168]
```

forecast 不应默认作为 x0 加噪。

forecast 的作用取决于模型版本：

```text
V0：不使用 forecast
V1：用 forecast 和历史误差构造 guidance interval
V2：作为 cond 输入神经网络
```

### 2.3 residual

残差，是实际值和预测值的差。

```text
residual = forecast - actual
actual = forecast - residual
```

residual 可以作为额外实验，但不是默认主线。

只有当配置明确写：

```yaml
target_type: "residual"
```

才允许把 residual 作为扩散模型生成目标。

否则默认：

```yaml
target_type: "actual"
```

---

## 3. 最高优先级原则

请优先帮我做到：

```text
先把项目跑清楚，而不是先追求最高指标。
```

工程顺序必须是：

```text
Step 1：诊断当前项目
Step 2：增加 forecast 质量评估脚本
Step 3：整理 config，使当前代码能关闭条件，退化为 V0
Step 4：跑通 V0 无条件 DDPM
Step 5：保留并整理当前 V1-like 逻辑
Step 6：实现 V2 CSDI-style 条件输入
Step 7：做三者统一评价
```

---

## 4. 当前最重要的第一轮任务

请你先执行“第一轮工程诊断与最小重构”，不要直接改模型结构。

### 4.1 阅读文档

请先阅读：

```text
docs/README_scenario_diffusion.md
docs/PROMPT_codex_scenario_diffusion.md
docs/AGENT_scenario_diffusion.md
```

如果这些文件不存在，请先提醒我，而不是继续乱改。

### 4.2 扫描项目结构

请输出当前项目结构摘要，包括：

```text
1. 数据处理文件有哪些
2. Dataset 类在哪里
3. 模型文件有哪些
4. diffusion / scheduler / sampler 逻辑在哪里
5. train 脚本在哪里
6. sample / generate 脚本在哪里
7. evaluation 脚本在哪里
8. config 文件在哪里
```

### 4.3 判断当前代码属于哪个版本

请判断当前代码更像：

```text
V0：无条件 DDPM
V1：2023 guidance-like
V2：CSDI-style conditional diffusion
混合版本：actual / forecast / residual 混在一起
```

并说明理由。

### 4.4 查清变量命名

请在代码中查找并列出以下变量对应关系：

```text
actual 对应哪些变量名？
forecast 对应哪些变量名？
residual 对应哪些变量名？
x0 对应 actual / forecast / residual 中的哪一个？
xt 是在哪里生成的？
epsilon / noise 是在哪里生成的？
epsilon_theta 是哪个模型输出？
mu_theta 是在哪里计算的？
condition / cond / mask / observed_data 分别代表什么？
```

请特别检查：

```text
是否把 forecast 当作 x0 加噪了？
是否把 residual 当作 x0 加噪了？
是否训练时用 actual，采样时却用 forecast 造成逻辑不一致？
是否既使用 cond 输入网络，又额外 guidance，导致条件重复？
```

---

## 5. 第一轮必须新增的功能：forecast 质量诊断

我目前担心 Fedformer 对风电预测不准，而光伏和负荷预测还可以。请新增一个脚本：

```text
src/eval/eval_forecast_quality.py
```

如果项目没有 src/eval 结构，请按现有结构放在合适位置，但不要混入训练脚本。

### 5.1 脚本目标

评估 forecast 相对于 actual 的质量，分别对 wind / pv / load 输出：

```text
MAE
RMSE
MAPE 或 sMAPE
Pearson correlation
Bias = mean(forecast - actual)
Max error
P90 absolute error
```

### 5.2 输入

脚本应支持读取 processed 数据，至少包含：

```text
actual_wind
actual_pv
actual_load
forecast_wind
forecast_pv
forecast_load
timestamp
```

如果当前数据不是这个字段名，请自动适配或在代码顶部写清楚字段映射。

### 5.3 输出

保存：

```text
outputs/forecast_quality/forecast_quality.csv
outputs/forecast_quality/forecast_quality.json
```

并在终端打印类似：

```text
Forecast Quality:
wind: MAE=..., RMSE=..., Corr=...
pv:   MAE=..., RMSE=..., Corr=...
load: MAE=..., RMSE=..., Corr=...
```

### 5.4 特殊要求

请注意：

```text
wind / pv 可能有实际值接近 0，MAPE 容易爆炸；
因此 MAPE 要加 eps，或者同时输出 sMAPE；
负荷可以正常计算相对误差。
```

---

## 6. 第二轮任务：增加 V0 开关，而不是重写项目

我现在可能已经做了 V1-like 版本。请不要删除它，而是通过 config 增加开关，使当前代码可以退化为 V0。

### 6.1 新增配置字段

请在 config 中增加或整理：

```yaml
experiment:
  name: "v0_uncond_ddpm_actual_168h"

data:
  variables: ["wind", "pv", "load"]
  length: 168
  freq: "1H"

target:
  type: "actual"       # actual / residual

condition:
  mode: "none"         # none / guidance_2023 / csdi_forecast
  use_forecast: false
  use_guidance: false
  cond_mask: [0, 0, 0]

guidance:
  enable: false
  wind_scale: 0.0
  pv_scale: 0.0
  load_scale: 0.0

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

### 6.2 V0 行为

当：

```yaml
condition:
  mode: "none"
  use_forecast: false
  use_guidance: false
target:
  type: "actual"
```

模型必须满足：

```text
x0 = actual
cond = None
epsilon_theta = model(x_t, t, cond=None)
loss = MSE(noise, epsilon_theta)
```

采样时：

```text
x_T ~ N(0, I)
不使用 forecast
不使用 guidance
生成 scenarios.shape = [N, 3, 168]
```

### 6.3 V0 验证

请添加 shape assert：

```python
assert actual.ndim == 3
assert actual.shape[1] == 3
assert actual.shape[2] == 168
```

训练时打印：

```text
x0 shape
xt shape
noise shape
epsilon_theta shape
loss
```

采样时打印：

```text
sample shape = [N, 3, 168]
```

---

## 7. 第三轮任务：整理 V1 2023 guidance-like 版本

当前代码如果已经类似 V1，请把它整理成可开关版本。

### 7.1 V1 的定义

V1 不是普通 concat 条件输入。

V1 的逻辑是：

```text
训练 DDPM 时，仍然用 actual 作为 x0；
forecast 用于历史误差分布建模；
采样时，用 forecast-error interval 修正 mu_theta。
```

训练：

```text
x0 = actual
epsilon_theta = model(x_t, t, cond=None)
loss = MSE(noise, epsilon_theta)
```

采样：

```text
epsilon_theta = model(x_t, t, cond=None)
mu_theta = compute_mu(x_t, epsilon_theta, t)
mu_guided = mu_theta + guidance_term
x_{t-1} ~ N(mu_guided, sigma_t)
```

### 7.2 V1 配置

```yaml
experiment:
  name: "v1_2023_guidance_actual_168h"

target:
  type: "actual"

condition:
  mode: "guidance_2023"
  use_forecast: true
  use_guidance: true

guidance:
  enable: true
  method: "forecast_error_interval"
  wind_scale: 0.1
  pv_scale: 0.5
  load_scale: 0.5
  num_bins: 20
  quantile_low: 0.05
  quantile_high: 0.95
```

### 7.3 误差区间构造

请实现或整理：

```python
error = forecast - actual
```

按变量分别处理：

```text
wind error distribution conditioned on forecast_wind
pv error distribution conditioned on forecast_pv
load error distribution conditioned on forecast_load
```

最小可行版本先用分箱 quantile，不要一开始就上 KDE。

例如：

```text
按 forecast 值分成 num_bins 个区间；
每个区间计算 error 的 q05 和 q95；
给定未来 forecast，查表得到 error_low / error_high；
c_down = clip(forecast + error_low, lower_bound, upper_bound)
c_up   = clip(forecast + error_high, lower_bound, upper_bound)
```

对于 wind / pv：

```text
lower_bound = 0
upper_bound = 1
```

对于 load：

```text
如果归一化到 [0,1]，同样 clip 到 [0,1]
否则按归一化方式处理
```

### 7.4 guidance_scale=0 测试

必须实现测试：

```text
当 wind_scale=pv_scale=load_scale=0 时，
V1 采样逻辑应退化为 V0。
```

请至少保证同一随机种子下逻辑路径一致，或说明随机采样导致数值不能完全一致但代码路径等价。

---

## 8. 第四轮任务：实现 V2 CSDI-inspired 条件输入

V2 是我接下来重点想和 V1 对比的方法。

### 8.1 V2 的定义

V2 学的是：

```text
p(actual | forecast)
```

训练时：

```text
x0 = actual
cond = forecast
xt = q_sample(x0, t, noise)
epsilon_theta = model(xt, t, cond=forecast)
loss = MSE(noise, epsilon_theta)
```

采样时：

```text
给定未来 forecast
xT ~ N(0, I)
epsilon_theta = model(xt, t, cond=forecast)
逐步去噪得到 actual scenarios
```

### 8.2 V2 配置

```yaml
experiment:
  name: "v2_csdi_cond_actual_given_forecast_168h"

target:
  type: "actual"

condition:
  mode: "csdi_forecast"
  use_forecast: true
  use_guidance: false
  cond_mask: [1, 1, 1]

model:
  input_channels: 3
  cond_channels: 3
```

### 8.3 模型接口统一

请统一模型接口为：

```python
epsilon_theta = model(x_t, t, cond=None)
```

其中：

```text
x_t.shape = [B, 3, 168]
t.shape   = [B]
cond      = None 或 [B, 3, 168]
```

V0：

```python
model(x_t, t, cond=None)
```

V2：

```python
model(x_t, t, cond=forecast)
```

### 8.4 cond_mask

为了处理 Fedformer 风电预测不准的问题，请支持变量级条件开关：

```yaml
condition:
  cond_mask: [0, 1, 1]
```

含义：

```text
wind forecast 不作为强条件
pv forecast 使用
load forecast 使用
```

实现方式：

```python
cond = forecast * cond_mask.view(1, 3, 1)
```

后续可比较：

```text
[1,1,1]：风光负荷全部使用 forecast
[0,1,1]：风电不使用 forecast，光伏负荷使用 forecast
[0,0,0]：退化为无条件
```

---

## 9. 第五轮任务：评价指标统一

请新增或整理：

```text
src/eval/metrics.py
```

至少支持：

### 9.1 场景基本指标

```text
scenario_mean MAE
scenario_median MAE
RMSE
```

### 9.2 概率场景指标

```text
Coverage Rate
Interval Width
Energy Score
```

### 9.3 时间序列结构指标

```text
ACF similarity
Ramp distribution error
```

ramp 定义：

```python
ramp[t] = x[t] - x[t-1]
```

分别评估 wind / pv / load。

### 9.4 多变量相关性

评估生成场景中：

```text
wind-pv
wind-load
pv-load
```

相关系数矩阵与真实数据相关系数矩阵的差异。

### 9.5 极端场景覆盖

至少统计：

```text
低风电
低光伏
高负荷
高净负荷 = load - wind - pv
```

---

## 10. 输出给 UC / 电力电量平衡的格式

最终生成场景请保存为：

```text
outputs/{experiment_name}/samples/scenarios.npz
```

内容：

```python
wind: [N, 168]
pv: [N, 168]
load: [N, 168]
prob: [N]
forecast_wind: [168]
forecast_pv: [168]
forecast_load: [168]
```

初始概率：

```python
prob = np.ones(N) / N
```

同时保存：

```text
outputs/{experiment_name}/config.yaml
outputs/{experiment_name}/metrics.json
outputs/{experiment_name}/figures/
```

---

## 11. 今日最小可执行任务

请你今天只完成以下任务，不要继续往后扩展：

```text
任务 A：扫描项目结构，输出诊断报告
任务 B：查清 actual / forecast / residual 的变量名和流向
任务 C：新增 eval_forecast_quality.py
任务 D：新增 config 开关，使当前代码能关闭 condition/guidance 跑 V0
任务 E：给出 V0 与当前 V1-like 的运行命令
```

完成后请输出：

```text
1. 修改了哪些文件
2. 新增了哪些文件
3. 如何运行 forecast quality 诊断
4. 如何运行 V0
5. 如何运行 V1
6. 当前还不能确定的问题
7. 下一步建议
```

---

## 12. 给 Agent 的第一条可直接执行指令

请从这里开始执行：

```text
请先阅读 docs/README_scenario_diffusion.md、docs/PROMPT_codex_scenario_diffusion.md、docs/AGENT_scenario_diffusion.md，然后不要立即大改模型。先完成第一轮工程诊断：

1. 扫描当前项目结构，说明数据、模型、扩散过程、训练、采样、评价分别在哪些文件；
2. 查清 actual、forecast、residual、x0、xt、noise、epsilon_theta、mu_theta 在当前代码中的变量名和流向；
3. 判断当前项目更像 V0、V1、V2 还是混合版本；
4. 新增 src/eval/eval_forecast_quality.py，分别输出 wind/pv/load 的 MAE、RMSE、sMAPE、Pearson correlation、Bias、P90 absolute error；
5. 增加 config 开关，使当前模型可以设置 condition.mode='none'、target.type='actual'，从而退化为 V0 无条件 DDPM；
6. 不要删除已有 V1-like 逻辑；
7. 所有关键 tensor 加 shape assert，统一使用 [B, 3, 168]；
8. 完成后告诉我修改了哪些文件、如何运行 forecast quality、如何运行 V0、如何运行当前 V1-like。
```

---

## 13. 重要提醒

当前 Fedformer 风电预测可能不准，因此不要默认“forecast 条件越强越好”。

后续实验必须支持：

```text
V0：完全不用 forecast
V1：wind/pv/load 分变量 guidance scale
V2：cond_mask = [1,1,1] 和 [0,1,1] 对比
```

重点比较：

```text
1. 无条件模型是否更稳健
2. forecast 条件是否在 pv/load 上有效
3. wind forecast 差时，弱化 wind 条件是否更好
4. 强 guidance 是否破坏风电波动和极端场景
```

这将直接服务于论文中的方法分析与实验讨论。
