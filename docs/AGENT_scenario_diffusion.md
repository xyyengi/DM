# Agent.md：项目协作规范与任务分解

## 1. Agent 角色

你是一个负责帮助我整理、复现和改造扩散模型代码的工程助手。我的深度学习基础较弱，因此你在修改代码时必须做到：

```text
解释清楚为什么改
只改必要模块
保持接口统一
保留可运行 baseline
不要一次性大改
```

## 2. 当前研究路线

本项目有三个核心实验分支：

```text
V0_unconditional_DDPM
V1_2023_forecast_error_guidance
V2_CSDI_inspired_conditional_DDPM
```

优先级：

```text
1. 数据处理正确
2. V0 无条件 DDPM 跑通
3. V1 复现 2023 条件引导
4. V2 实现 forecast 作为网络条件输入
5. 对比三者效果
6. 输出周尺度 UC 所需场景
```

## 3. 不允许混淆的概念

### actual

真实历史曲线，是生成目标。

```text
actual.shape = [B, 3, 168]
```

### forecast

预测曲线，是条件，不是默认生成目标。

```text
forecast.shape = [B, 3, 168]
```

### residual

实际与预测的差值。

```text
residual = forecast - actual
actual = forecast - residual
```

只有当配置明确：

```yaml
target_type: residual
```

才允许把 residual 作为扩散目标。

## 4. 每次修改代码前必须回答的问题

在动代码前，先说明：

```text
1. 本次改哪个文件？
2. 本次属于 V0 / V1 / V2 哪个分支？
3. 输入 shape 是什么？
4. 输出 shape 是什么？
5. 是否改变训练目标？
6. 是否影响已有 baseline？
7. 如何验证改动正确？
```

## 5. 推荐开发顺序

### Step 1：整理数据

目标文件：

```text
src/data/preprocess.py
src/data/dataset.py
src/data/normalization.py
```

验收标准：

```text
能输出 train/val/test dataset
每个 batch 包含 actual 和 forecast
actual.shape == [B, 3, 168]
forecast.shape == [B, 3, 168]
没有 NaN
归一化可反归一
```

### Step 2：实现 V0 无条件 DDPM

目标文件：

```text
src/models/diffusion.py
src/models/res_unet_1d.py
src/train/train_ddpm.py
src/sample/sample_uncond.py
```

验收标准：

```text
loss 能下降
能生成 [N, 3, 168] 场景
风光负荷数值范围合理
可以反归一化并画图
```

### Step 3：实现 V1 2023 guidance

目标文件：

```text
src/models/guidance_2023.py
src/sample/sample_guidance.py
```

验收标准：

```text
能根据 forecast 和历史 error 生成 c_down/c_up
能在 p_sample 中修改 mu_theta
guidance_scale=0 时结果与 V0 一致
guidance_scale>0 时场景更贴近 forecast interval
```

### Step 4：实现 V2 条件输入

目标文件：

```text
src/models/condition_encoder.py
src/models/res_unet_1d.py
src/train/train_cond.py
src/sample/sample_cond.py
```

验收标准：

```text
model(x_t, t, cond=forecast) 能运行
训练 loss 能下降
同一个 forecast 下能采样多个不同 scenario
打乱 forecast 后评价指标应变差，证明模型确实使用条件
```

### Step 5：评价与导出

目标文件：

```text
src/eval/metrics.py
src/eval/plot_scenarios.py
src/eval/uc_export.py
```

验收标准：

```text
输出 metrics.json
输出 scenarios.npz
输出可视化图片
能被周尺度机组组合程序读取
```

## 6. 关键测试

### Shape test

所有模型入口必须加断言：

```python
assert x_t.ndim == 3
assert x_t.shape[1] == 3
assert x_t.shape[2] == 168
```

条件模型：

```python
assert cond is not None
assert cond.shape[-1] == x_t.shape[-1]
```

### Guidance zero test

对 V1：

```text
当 guidance_scale = 0 时，sample_guidance 输出应与 sample_uncond 逻辑一致。
```

### Condition shuffle test

对 V2：

```text
训练/测试时把 forecast 随机打乱。
如果指标几乎不变，说明模型没有真正使用条件。
如果指标明显变差，说明条件有效。
```

### Residual isolation test

如果做 residual 版本，必须保证：

```text
target_type=residual
scenario = forecast - generated_residual
```

并与 actual target 版本分开保存，不允许覆盖主实验。

## 7. 代码风格

- 使用 PyTorch。
- 数据 shape 统一为 `[B, C, L]`。
- 所有函数写 docstring。
- 所有重要 tensor 写注释说明含义。
- 训练脚本只负责调度，不写复杂模型逻辑。
- 模型、扩散过程、数据、评价分离。
- 任何修改必须保持 V0 baseline 可运行。

## 8. 实验命名规范

```text
v0_uncond_ddpm_actual_168h
v1_2023_guidance_actual_168h
v2_csdi_cond_actual_given_forecast_168h
v3_residual_given_forecast_168h
```

输出目录：

```text
outputs/{experiment_name}/
├── config.yaml
├── checkpoints/
├── samples/
├── figures/
└── metrics.json
```

## 9. 给工程助手的注意事项

这个项目不是图像生成任务，而是电力系统功率时间序列场景生成。评价时必须关注：

```text
时序相关性
爬坡分布
风光负荷相关性
极端天气/极端出力覆盖
区间覆盖率
场景宽度
最终对 UC 结果的影响
```

不要只用训练 loss 判断模型好坏。

## 10. 最终交付目标

最终应能执行类似流程：

```bash
python src/data/preprocess.py --config configs/base.yaml
python src/train/train_ddpm.py --config configs/ddpm_uncond.yaml
python src/train/train_cond.py --config configs/csdi_cond.yaml
python src/sample/sample_guidance.py --config configs/guidance_2023.yaml
python src/eval/metrics.py --scenario outputs/.../samples/scenarios.npz
python src/eval/uc_export.py --scenario outputs/.../samples/scenarios.npz
```

交付结果：

```text
1. 可复现实验代码
2. 三类扩散模型对比
3. 周尺度风光负荷场景集合
4. 面向机组组合的输入文件
5. 评价指标与图表
```
