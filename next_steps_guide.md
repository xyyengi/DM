# 后续实验建议与流程说明

## 一、为什么论文的条件引导效果差？

### 可能的原因分析

**1. 论文 vs 实现的关键差异**

| 方面 | 论文假设 | 实际实现 |
|------|---------|---------|
| 数据分布 | 理想分布 | 实际风电/光伏/负荷分布复杂 |
| 区间定义 | 残差区间 | 功率值区间（我们修改后） |
| 梯度计算 | 理论公式 | 代码实现可能有bug |
| 引导强度 | 未明确 | guidance_scale=0.5可能过强 |

**2. 最可能的问题：梯度方向或强度**

论文公式10：
```
x_{t-1} = x_{t-1} + γ · ∇_x log p(c|x_t)
```

可能的问题：
- **梯度方向反了**：应该是"拉回"区间，但实现成了"推离"
- **引导强度过大**：即使0.5也过强，导致过度约束
- **硬截断问题**：论文可能用软约束，我们用硬截断

**3. 数据差异**

论文可能使用：
- 更简单的合成数据
- 更规范的分布（如高斯分布）
- 更小的残差范围

实际数据：
- 风电波动大，非高斯分布
- 光伏有零值截断（夜间）
- 负荷有周期性但受天气影响

---

## 二、关闭条件引导后的完整流程

### 训练阶段（与之前相同）

```
输入：历史数据 + FEDformer预测值
      ↓
计算残差：residual = forecast - actual
重构实际值：actual = forecast - residual
      ↓
归一化残差 → 输入扩散模型
      ↓
训练：预测噪声 ε_θ(x_t, t, condition)
      ↓
损失：MSE(ε_θ, ε)  # 只有噪声预测损失，没有条件损失
      ↓
保存模型
```

**注意**：训练时条件（预测值+时间特征）仍然输入模型，只是**不用于梯度引导**。

### 采样/生成阶段（关键区别）

**有条件引导时（guidance=0.5）：**
```
从纯噪声 x_T ~ N(0, I) 开始
      ↓
对于 t = T, T-1, ..., 1:
    1. 模型预测噪声：ε_θ(x_t, t, condition)
    2. 计算去噪：x_{t-1} = denoise(x_t, ε_θ)
    3. 【条件引导】计算梯度：∇ = compute_gradient(x_{t-1}, cond_interval)
    4. 【条件引导】更新：x_{t-1} = x_{t-1} + guidance_scale * ∇
    5. 如果 x_{t-1} 超出区间，截断到边界
      ↓
最终输出：x_0（残差）
      ↓
功率值 = forecast + x_0
```

**无条件引导时（guidance=0）：**
```
从纯噪声 x_T ~ N(0, I) 开始
      ↓
对于 t = T, T-1, ..., 1:
    1. 模型预测噪声：ε_θ(x_t, t, condition)
    2. 计算去噪：x_{t-1} = denoise(x_t, ε_θ)
    3. 【无】没有梯度计算
    4. 【无】没有引导更新
    5. 【无】没有截断
      ↓
最终输出：x_0（残差）
      ↓
功率值 = forecast + x_0
```

### 关键区别总结

| 步骤 | guidance=0.5 | guidance=0 |
|------|--------------|------------|
| 条件输入 | ✅ 有 | ✅ 有 |
| 梯度计算 | ✅ 有 | ❌ 无 |
| 引导更新 | ✅ 有 | ❌ 无 |
| 硬截断 | ✅ 有 | ❌ 无 |
| 结果 | 约束严格，coverage低 | 自由生成，coverage高 |

**核心发现**：条件输入本身（预测值+时间特征）已经提供了足够信息，额外的梯度引导反而有害！

---

## 三、后续实验建议

### 实验1：验证条件输入的作用（推荐先做）

**目的**：确认是条件输入有用，还是纯扩散就好

**方法**：
```bash
# 测试1：有条件输入，无梯度引导（当前最佳）
python generate.py --config config/wind_scenario.yaml --guidance_scale 0.0

# 测试2：无条件输入（把cond_channels设为0或随机）
# 需要修改代码，暂时跳过
```

**预期**：测试1 > 测试2，说明条件输入本身有用

### 实验2：软约束替代硬截断

**目的**：用论文的软约束方式

**修改 diff_models_multivariate.py**：
```python
# 当前：硬截断
if power_t < c_down:
    gradient = c_down - power_t
    x_new = x_new + gradient

# 改为：软约束（高斯加权）
def soft_constraint(x, c_down, c_up, sigma=0.1):
    """高斯软约束"""
    if x < c_down:
        return np.exp(-((x - c_down)**2) / (2*sigma**2))
    elif x > c_up:
        return np.exp(-((x - c_up)**2) / (2*sigma**2))
    else:
        return 1.0

# 在采样时加权
weight = soft_constraint(power_t, c_down, c_up)
x_new = x_new * weight + x_new * (1 - weight) * guidance_scale
```

**测试**：guidance_scale = 0.1, 0.5, 1.0

### 实验3：检查梯度方向

**目的**：确认梯度方向是否正确

**方法**：在 `compute_conditional_gradient` 中加日志：
```python
print(f"power_t={power_t:.3f}, c_down={c_down:.3f}, c_up={c_up:.3f}")
print(f"gradient={gradient:.3f}, direction={'拉回' if gradient > 0 else '推离'}")
```

**判断**：
- 如果 power_t < c_down，gradient 应该 > 0（拉回）
- 如果 power_t > c_up，gradient 应该 < 0（拉回）

### 实验4：降低引导强度

**目的**：找到最优 guidance_scale

**测试**：
```bash
for scale in 0.01 0.05 0.1 0.2 0.5 1.0; do
    python generate.py --config config/wind_scenario.yaml --guidance_scale $scale --output_dir save/test_scale_$scale
done
```

**分析**：绘制 coverage vs guidance_scale 曲线，找到最佳点

### 实验5：后处理筛选（实用方案）

**目的**：在guidance=0的基础上，通过后处理缩小区间

**方法**：
```python
# 1. 用 guidance=0 生成100个场景
scenarios = generate(n_samples=100, guidance_scale=0.0)

# 2. 计算每个场景的条件区间
c_down, c_up = get_conditional_interval(forecast)

# 3. 筛选符合条件的场景
valid_scenarios = []
for s in scenarios:
    if c_down <= s <= c_up:
        valid_scenarios.append(s)

# 4. 如果不够，重新生成
while len(valid_scenarios) < 10:
    new_scenarios = generate(n_samples=10, guidance_scale=0.0)
    for s in new_scenarios:
        if c_down <= s <= c_up:
            valid_scenarios.append(s)
```

**优点**：
- 保留高coverage
- 通过筛选缩小区间
- 实现简单

---

## 四、推荐优先级

### 短期（立即做）
1. **实验5：后处理筛选** - 最实用，立即改善
2. **实验4：降低引导强度** - 找到最优参数

### 中期（本周内）
3. **实验2：软约束** - 更接近论文方法
4. **实验3：检查梯度方向** - 确认实现正确性

### 长期（可选）
5. **实验1：验证条件输入** - 理论验证
6. **对比论文代码** - 如果有开源代码

---

## 五、当前最佳配置（推荐直接使用）

```yaml
# config/wind_scenario_best.yaml
model:
  guidance_scale: 0.0  # 关闭梯度引导
  
  # 其他参数保持不变
  num_steps: 500
  beta_start: 0.0001
  beta_end: 0.04
  
evaluation:
  n_samples: 100  # 增加样本数，用于后处理筛选
  quantiles: [0.8, 0.9, 1.0]
```

**使用**：
```bash
# 生成大量场景
python generate.py --config config/wind_scenario_best.yaml --n_samples 100

# 后处理筛选（在evaluation.py中添加筛选逻辑）
# 保留符合条件的10个场景
```

---

## 六、总结

**为什么论文方法效果差？**
1. 可能梯度方向或强度有问题
2. 实际数据分布比论文复杂
3. 硬截断不如软约束

**关闭条件引导后的流程？**
- 训练：不变，条件输入仍用于模型
- 采样：只用模型去噪，不额外约束
- 结果：coverage大幅提升，width增加

**下一步做什么？**
1. 立即：用guidance=0 + 后处理筛选
2. 短期：测试不同guidance_scale
3. 中期：实现软约束

**核心结论**：条件输入本身足够，额外梯度引导有害！
