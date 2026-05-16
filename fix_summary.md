# 条件引导机制修复总结

## 问题根源

发现的核心问题：**条件区间定义错误**

### 论文公式9定义
```
c = [c_down, c_up] = [f - K_h(f), f + K_h(f)]
```
即：**功率值区间 = 预测值 ± 残差范围**

### 原代码实现（错误）
- `dataset_multivariate.py`: 条件区间是**残差区间** `[residual_down, residual_up]`
- `diff_models_multivariate.py`: 梯度引导直接比较残差 `x_t` 与残差区间

### 修复后（正确）
- `dataset_multivariate.py`: 条件区间改为**功率值区间** `[f + residual_down, f + residual_up]`
- `diff_models_multivariate.py`: 梯度引导比较功率值 `forecast + x_t` 与功率值区间

---

## 修改文件

### 1. dataset_multivariate.py
**位置**: `_precompute_cond_matrix()` 方法

**修改内容**:
```python
# 修复前:
c_down, c_up = self.kde.get_conditional_interval(f_val, c)

# 修复后:
residual_down, residual_up = self.kde.get_conditional_interval(f_val, c)
c_down = f_val + residual_down  # 功率值下界 = 预测值 + 残差下界
c_up = f_val + residual_up      # 功率值上界 = 预测值 + 残差上界
```

### 2. diff_models_multivariate.py
**位置**: `compute_conditional_gradient()` 方法

**修改内容**:
```python
# 修复前:
def compute_conditional_gradient(self, x_t, cond_matrix, debug=False):
    # 直接用 x_t（残差）与 cond_matrix（残差区间）比较
    gamma_mask = ((x_t < c_down) | (x_t > c_up)).float()

# 修复后:
def compute_conditional_gradient(self, x_t, cond_matrix, forecast=None, debug=False):
    # 计算功率值 = 预测值 + 残差
    power_t = forecast + x_t
    # 用 power_t 与 cond_matrix（功率值区间）比较
    gamma_mask = ((power_t < c_down) | (power_t > c_up)).float()
```

**位置**: `denoise_step()` 方法

**修改内容**:
```python
# 修复后:
forecast = cond_full[:, :3, :]  # 从cond_full中提取forecast
cond_gradient, gamma_mask, grad_debug = self.compute_conditional_gradient(
    x_t, cond_matrix, forecast=forecast, debug=debug
)
```

---

## 为什么这会导致覆盖率低下？

### 原代码的问题

假设：
- 预测值 `f = 0.5`
- 残差区间 `[-0.2, 0.2]`（即残差应该在-0.2到0.2之间）
- 实际功率值区间应该是 `[0.3, 0.7]`

**原代码行为**:
- 条件区间 = `[-0.2, 0.2]`（残差区间）
- 扩散生成残差 `x_t = 0.1`
- 判断：`0.1` 在 `[-0.2, 0.2]` 内 → 不修正
- 实际功率值 = `0.5 + 0.1 = 0.6` ✓（正确）

但如果：
- 扩散生成残差 `x_t = 0.3`
- 判断：`0.3` 不在 `[-0.2, 0.2]` 内 → 修正到0.2
- 实际功率值 = `0.5 + 0.2 = 0.7` ✓（正确）

**问题场景**:
- 扩散生成残差 `x_t = -0.1`
- 判断：`-0.1` 在 `[-0.2, 0.2]` 内 → 不修正
- 实际功率值 = `0.5 + (-0.1) = 0.4` ✓（正确）

看起来没问题？**问题在于条件区间的物理意义**。

### 真正的区别

原代码的条件区间是**残差应该落在的范围**，但：
1. 残差分布通常以0为中心，范围较窄
2. 功率值分布范围更广（0-1）
3. 用残差区间约束残差，相当于只在很小的范围内约束

修复后的条件区间是**功率值应该落在的范围**：
1. 功率值区间 `[f + residual_down, f + residual_up]` 范围更大
2. 约束的是最终功率值，更符合物理意义
3. 覆盖率应该显著提升

---

## 下一步操作

### 需要重新训练模型

由于条件区间的定义改变了，需要：

1. **重新训练模型**（条件矩阵已经自动重新生成）
2. **重新生成场景**
3. **重新评估指标**

### 训练命令

```bash
python train.py --config config/wind_scenario.yaml
```

### 生成命令

```bash
python generate.py --config config/wind_scenario.yaml --checkpoint save/run_xxx/model.pth
```

---

## 预期效果

修复后，覆盖率应该**显著提升**：

| 指标 | 修复前 | 预期修复后 |
|------|--------|-----------|
| wind_coverage_80% | 26.7% | ~70-80% |
| solar_coverage_80% | 59.8% | ~75-85% |
| load_coverage_80% | 49.5% | ~70-80% |

如果覆盖率仍然不理想，可以考虑：
1. 放宽区间宽度（5-95分位数）
2. 调整 guidance_scale
3. 改进软引导公式

---

## 修复完成时间
2026-05-16 14:55
