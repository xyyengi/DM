# Coverage过低问题诊断与改进方案

## 问题诊断

### 评估结果解读

| 指标 | Wind | Solar | Load | 理想值 | 问题 |
|------|------|-------|------|--------|------|
| Coverage (80%区间) | 29.6% | 25.5% | 25.3% | 80% | **严重不足** |
| Coverage (90%区间) | 35.6% | 31.2% | 30.6% | 90% | **严重不足** |
| Coverage (95%区间) | 38.6% | 34.3% | 33.2% | 95% | **严重不足** |

**关键澄清**：用户描述"结果过于保守"是误解。实际问题是**模型过于自信**：
- **保守** = 预测区间宽，覆盖率高（宁可多预测也不漏）
- **过于自信** = 预测区间窄，覆盖率低（漏掉大量真实值）

当前Coverage远低于理想值，说明生成的场景**多样性不足**，分布过于集中。

---

## 根因分析

### 1. KDE条件区间过窄（核心问题）

**位置**：`dataset_multivariate.py` 第104-105行

```python
c_down_idx = max(0, min(np.searchsorted(cdf_values, 0.10), n_points - 1))
c_up_idx = max(0, min(np.searchsorted(cdf_values, 0.90), n_points - 1))
```

**问题**：
- 使用 **10%和90%分位数**，对应 **80%置信区间**
- 即使模型完美遵循KDE条件，理论覆盖率上限也只有80%
- 实际上由于扩散模型的噪声压缩，覆盖率会更低

**数学分析**：
- 80%置信区间理论上应覆盖80%的真实值
- 但当前只有25-30%覆盖率，说明：
  1. KDE区间本身可能比实际残差分布窄
  2. 扩散模型进一步压缩了输出分布

### 2. 条件梯度引导过强

**位置**：`diff_models_multivariate.py` 第501-511行

```python
t_decay = 1.0 - t / self.num_steps  # t=0时为1.0，t=499时为0.002
cond_gradient_clamped = torch.clamp(cond_gradient, min=-1.0, max=1.0)
mean = mean + self.guidance_scale * t_decay * cond_gradient_clamped * gamma_mask
```

**问题**：
- `guidance_scale=1.0` + `t_decay≈1.0`（去噪后期）= 强约束
- 强条件引导会将样本强制拉向KDE区间**中心**
- 导致所有样本趋于相似，多样性丧失

### 3. 扩散步数过多导致过度收敛

**位置**：`config/wind_scenario.yaml` 第48行

```yaml
num_steps: 500
```

**问题**：
- 500步去噪可能导致样本过度收敛
- 每一步的条件引导累积效应强
- 最终输出分布比训练数据分布更窄

### 4. 残差空间vs归一化空间的混淆

**位置**：`dataset_multivariate.py` 第293-294行

```python
self.forecast_norm = self.forecast_data / self.max_values
self.residual_norm = self.residual_data / self.max_values
```

**问题**：
- KDE在**归一化空间**拟合残差分布
- 但残差本身是 `预测-真实`，可能存在负值
- 归一化后的残差范围可能与原始残差分布特性不一致

---

## 改进方案

### 方案A：扩大KDE条件区间（推荐首选）

**修改**：`dataset_multivariate.py` 第104-105行

```python
# 原来：80%置信区间
c_down_idx = max(0, min(np.searchsorted(cdf_values, 0.10), n_points - 1))
c_up_idx = max(0, min(np.searchsorted(cdf_values, 0.90), n_points - 1))

# 改为：99%置信区间（覆盖几乎所有真实值）
c_down_idx = max(0, min(np.searchsorted(cdf_values, 0.005), n_points - 1))  # 0.5%分位
c_up_idx = max(0, min(np.searchsorted(cdf_values, 0.995), n_points - 1))    # 99.5%分位
```

**效果预期**：
- Coverage (95%区间) 应提升至接近95%
- 场景宽度增加，更符合实际不确定性

### 方案B：降低条件引导强度

**修改**：`config/wind_scenario.yaml` 第60行

```yaml
# 原来
guidance_scale: 1.0

# 改为（温和引导）
guidance_scale: 0.3  # 或 0.1
```

**效果预期**：
- 减少对生成样本的约束
- 保留更多扩散模型的自然多样性

### 方案C：减少扩散步数

**修改**：`config/wind_scenario.yaml` 第48行

```yaml
# 原来
num_steps: 500

# 改为
num_steps: 100  # 或 50
```

**效果预期**：
- 减少条件引导的累积效应
- 保留更多初始噪声的随机性

### 方案D：增加采样数量

**修改**：`config/wind_scenario.yaml` 第73行

```yaml
# 原来
n_samples: 10

# 改为
n_samples: 50  # 或 100
```

**效果预期**：
- 更稳定的统计评估
- 更好地反映生成分布的真实特性

---

## 推荐实施顺序

1. **首先实施方案A**（扩大KDE区间）- 这是根本解决方案
2. **清除缓存**：删除 `input_4.27/cond_matrix_*.npy` 和 `kde_multivariate.pkl`
3. **重新生成**：运行 `python generate.py`
4. **如果Coverage仍不足**，叠加方案B（降低guidance_scale）
5. **最后考虑方案C**（减少扩散步数）

---

## 验证指标

改进后的目标值：

| 指标 | 目标范围 |
|------|----------|
| Coverage (80%区间) | 75-85% |
| Coverage (90%区间) | 85-95% |
| Coverage (95%区间) | 93-97% |
| Width (100%) | 15-25% (Wind), 10-15% (Solar/Load) |

---

## 代码修改清单

1. `dataset_multivariate.py`：修改KDE分位数计算
2. `config/wind_scenario.yaml`：调整guidance_scale和num_steps
3. 清除缓存文件后重新运行