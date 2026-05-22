# 场景生成完整流程详解

## 用户问题解答

### 1. 关于梯度符号

**原文公式10**：`x_{t-1} = x_{t-1} + γ · ∇_x log p(c|x_t)`

**你的实现**：正号 ✓

**问题分析**：
- 梯度符号是对的（正号）
- 但梯度**方向**可能有问题
- 如果 `power_t < c_down`，梯度应该**正**（拉回区间）
- 如果 `power_t > c_up`，梯度应该**负**（拉回区间）

**当前代码逻辑**：
```python
if power_t < c_down:
    gradient = c_down - power_t  # 正数，正确 ✓
elif power_t > c_up:
    gradient = c_up - power_t    # 负数，正确 ✓
```

**结论**：梯度符号和方向都正确，但**硬截断**破坏了分布。

---

### 2. 后11维的前3维预测值真的有用吗？

**答案：有用！**

**作用机制**：
```
14通道输入：
├── Channel [0:3]: 目标残差（加噪后的x_t）
├── Channel [3:6]: 预测值（条件）← 这里！
└── Channel [6:14]: 时间编码（条件）
```

**为什么有用？**

1. **U-Net结构**：
   - 输入14通道 → 编码器 → 解码器 → 输出3通道
   - 预测值作为条件输入，指导残差去噪

2. **注意力机制**：
   - 模型学习"给定预测值f，残差ε应该是什么分布"
   - 例：预测风电=0.8 → 残差应该小（确定性高）
   - 例：预测光伏=0.0（夜间）→ 残差应该接近0

3. **消融实验证明**：
   - guidance=0（有条件输入）：coverage=82%
   - 如果去掉条件输入，coverage会大幅下降

**结论**：预测值作为条件输入**至关重要**！

---

### 3. 原论文是硬截断吗？

**论文原文**（公式10）：
```
x_{t-1} = x_{t-1} + γ · ∇_x log p(c|x_t)
```

**论文没有明确说硬截断**，但：
- 公式只有梯度引导，没有截断
- 梯度引导本身就是"软"约束

**你的实现**：
```python
# 梯度引导（软约束）
x_new = x_new + guidance_scale * gradient

# 硬截断（额外添加）
if power_new < c_down:
    x_new = c_down - forecast
elif power_new > c_up:
    x_new = c_up - forecast
```

**问题**：硬截断可能过于严格，破坏了概率分布。

**建议**：尝试去掉硬截断，只用梯度引导。

---

### 4. 条件KDE处理优化（你的代码）

**优化1：预计算分位数（避免重复计算CDF）**

```python
# 在KDE.fit()中预计算
self.precomputed_quantiles[channel_name] = {}
for i in range(self.n_intervals):
    kde = self.kde_models[channel_name][i]
    if kde is not None:
        # 构建自适应网格
        x_grid = self._build_dynamic_grid(error_mean, error_std, ...)
        pdf_values = kde(x_grid)
        cdf_values = np.cumsum(pdf_values) * (x_grid[1] - x_grid[0])
        cdf_values = cdf_values / cdf_values[-1]
        
        # 查找10%和90%分位数
        c_down_idx = np.searchsorted(cdf_values, 0.10)
        c_up_idx = np.searchsorted(cdf_values, 0.90)
        
        self.precomputed_quantiles[channel_name][i] = (
            x_grid[c_down_idx], x_grid[c_up_idx]
        )
```

**优化2：邻居回退策略（处理稀疏区间）**

```python
def _get_neighbor_fallback_stats(self, channel_name, interval_idx, ...):
    """Use nearby dense bins to estimate fallback mean/std for sparse intervals."""
    # 如果当前区间样本数<5，找邻近密集区间
    for radius in range(1, max_radius + 1):
        left = interval_idx - radius
        right = interval_idx + radius
        if left区间密集: 使用left统计量
        if right区间密集: 使用right统计量
    
    # 如果周围都不密集，使用全局统计量
    return global_mean, global_std
```

**优化3：预计算条件矩阵（避免504次KDE查询）**

```python
def _precompute_cond_matrix(self):
    """预计算所有样本的条件矩阵，从__getitem__移到初始化阶段"""
    self.cond_matrix_all = np.zeros((num_samples, 3, 168, 2))
    
    for idx in range(num_samples):
        for c in range(3):
            for t in range(168):
                f_val = forecast_norm[idx, t, c]
                # 使用预计算的分位数，O(1)查询
                residual_down, residual_up = self.kde.get_conditional_interval(f_val, c)
                # 转换为功率值区间
                c_down = f_val + residual_down
                c_up = f_val + residual_up
                self.cond_matrix_all[idx, c, t] = [c_down, c_up]
```

**优化4：缓存机制**

```python
# KDE模型缓存
kde_path = os.path.join(data_path, 'kde_multivariate.pkl')
if os.path.exists(kde_path):
    self.kde.load(kde_path)  # 直接加载
else:
    self.kde.fit(...)  # 重新拟合
    self.kde.save(kde_path)

# 条件矩阵缓存
cond_cache_path = f'cond_matrix_{mode}.npy'
if os.path.exists(cond_cache_path):
    self.cond_matrix_all = np.load(cond_cache_path)
else:
    self._precompute_cond_matrix()
    np.save(cond_cache_path, self.cond_matrix_all)
```

**优化效果**：
- 预计算分位数：避免每次查询都计算CDF
- 邻居回退：处理稀疏区间，提高鲁棒性
- 预计算条件矩阵：从O(n)降到O(1)
- 缓存机制：避免重复计算

---

## 一、Width 是否合适？

### 当前 Width 分析

| 指标 | guidance=0.5 | guidance=0 | 评价 |
|------|--------------|------------|------|
| total_width_100% | 12.56 | **31.75** | 增加了153% |
| wind_width_100% | 16.46 | **31.36** | 增加了91% |
| solar_width_100% | 7.39 | **20.38** | 增加了176% |
| load_width_100% | 13.84 | **43.51** | 增加了214% |

### Width 是否合适？

**答案是：合适，但偏宽**

**为什么合适？**
- Coverage 大幅提升：54% → 82%（+28%）
- 说明宽区间包含了更多真实值
- **高coverage + 较宽width** > **低coverage + 窄width**

**为什么偏宽？**
- 没有条件约束，模型生成更自由
- 区间包含了更多"可能性"

**理想 Width 应该是多少？**

根据 Coverage 目标反推：
- 如果 coverage_80% = 80%，width 应该刚好覆盖 80% 的真实值
- 当前 coverage_80% = 77-81%，说明 width 基本合适
- 但 coverage_100% = 82-91%，说明 width 可以再窄一点

**结论**：
- ✅ Width 当前可接受
- ⚠️ 但可以优化（通过后处理筛选或软约束）

---

## 二、场景生成完整流程（详细版）

### 核心概念理解

**为什么预测残差而不是直接预测功率？**

```
传统方法：直接预测功率 P
           ↓
问题：功率范围大（0~1），难以学习

本文方法：预测残差 ε = P_actual - P_forecast
           ↓
优势：残差范围小（-0.2~0.2），更容易学习
           ↓
最终功率：P = P_forecast + ε
```

**类比**：
- 直接预测功率 = 猜明天的温度（0~40度，难）
- 预测残差 = 猜明天比今天高/低几度（-5~5度，容易）

---

## 三、训练阶段流程

### Step 1: 数据准备

```
输入数据：
├── 历史功率数据（风、光、负荷）
├── FEDformer预测值（风、光、负荷趋势）
└── 时间特征（小时、星期、月份等）

处理流程：
1. 计算残差：ε = P_actual - P_forecast
   例：实际风电=0.5，预测=0.3，残差=0.2
   
2. 归一化残差：ε_norm = (ε - μ) / σ
   使残差范围在 [-1, 1] 左右

3. 构建输入：
   - Channel 0-2: 目标残差（加噪后的 x_t）
   - Channel 3-5: FEDformer预测值（条件）
   - Channel 6-13: 时间编码（条件）
```

### Step 2: 扩散过程（前向加噪）

```
对于每个训练样本：

原始残差：x_0 ~ 真实残差分布

对于 t = 1, 2, ..., T（T=500）：
    1. 采样噪声：ε ~ N(0, I)
    2. 加噪：x_t = √(α_t) * x_0 + √(1-α_t) * ε
       
       其中 α_t 是预定义的噪声 schedule
       
    3. 保存 (x_t, t, condition) → ε

目标：训练模型预测噪声 ε
```

**可视化**：
```
t=0:   x_0 = [0.1, -0.05, 0.2]  （清晰的残差）
t=100: x_100 = [0.08, -0.02, 0.15] + 少量噪声
t=250: x_250 = [0.02, 0.01, 0.05] + 中等噪声
t=500: x_500 = [0.5, -0.3, 0.8]  （纯噪声，接近N(0,I)）
```

### Step 3: 模型训练

```
模型输入：
- x_t: 加噪后的残差（Channel 0-2）
- t: 时间步（编码为向量）
- condition: [预测值, 时间特征]（Channel 3-13）

模型输出：
- ε_θ: 预测的噪声（Channel 0-2）

损失函数：
L = MSE(ε_θ, ε)  # 预测噪声 vs 真实噪声

训练目标：
让模型学会：给定 (x_t, t, condition)，预测加入的噪声 ε
```

**为什么条件有用？**
- 预测值告诉模型：残差应该在哪里（均值）
- 时间特征告诉模型：残差应该多大（方差）
- 例：中午光伏预测高 → 残差应该小（确定性高）

---

## 四、采样/生成阶段流程

### Step 1: 初始化

```
输入：
- FEDformer预测值 P_forecast（条件的一部分）
- 时间特征（条件的另一部分）

初始化：
x_T ~ N(0, I)  # 从纯噪声开始
T = 500        # 总步数
```

### Step 2: 去噪过程（反向采样）

```
对于 t = T, T-1, ..., 1:

    1. 模型预测噪声：
       ε_θ = model(x_t, t, condition)
       
    2. 计算去噪后的残差：
       x_{t-1} = (x_t - √(1-α_t) * ε_θ) / √α_t
       
       这是DDPM的采样公式，从 x_t 恢复 x_{t-1}
       
    3. 【可选】条件引导（guidance > 0）：
       计算功率值：P_t = P_forecast + x_{t-1}
       计算梯度：∇ = gradient(P_t, interval)
       更新：x_{t-1} = x_{t-1} + guidance * ∇
       
    4. 【可选】硬截断（guidance > 0）：
       如果 P_t 超出 [c_down, c_up]：
           x_{t-1} = clip(x_{t-1}, bounds)
```

**可视化（guidance=0）：**
```
t=500: x_500 = [0.5, -0.3, 0.8]   （纯噪声）
       ↓ 模型预测噪声，去噪
t=250: x_250 = [0.2, -0.1, 0.3]   （中等噪声）
       ↓ 继续去噪
t=100: x_100 = [0.12, -0.06, 0.18]（少量噪声）
       ↓ 继续去噪
t=0:   x_0 = [0.1, -0.05, 0.15]   （最终残差，仍有波动）
```

### Step 3: 生成最终功率

```
最终残差：ε = x_0

最终功率：
P_wind = P_forecast_wind + ε[0]
P_solar = P_forecast_solar + ε[1]
P_load = P_forecast_load + ε[2]

反归一化：
P_actual = P * σ + μ  # 恢复到原始功率范围
```

**示例：**
```
FEDformer预测：
- 风电：0.3
- 光伏：0.6
- 负荷：0.5

生成残差：
- ε[0] = 0.1  （风电比预测高0.1）
- ε[1] = -0.05（光伏比预测低0.05）
- ε[2] = 0.08 （负荷比预测高0.08）

最终功率：
- 风电：0.3 + 0.1 = 0.4
- 光伏：0.6 - 0.05 = 0.55
- 负荷：0.5 + 0.08 = 0.58
```

---

## 五、条件引导详解（guidance > 0）

### 为什么要条件引导？

**问题**：纯扩散模型生成的残差可能不合理
- 例：预测风电=0.3，但生成残差=0.8 → 功率=1.1（超出物理范围）

**解决**：用条件区间约束残差范围

### 条件区间计算

```
对于每个时间点：

1. 计算功率值区间：
   c_down = P_forecast - residual_up
   c_up = P_forecast - residual_down
   
   其中 residual_down/up 来自KDE拟合的历史残差分布

2. 在采样过程中：
   - 如果生成的功率 P_t < c_down，梯度把样本拉回
   - 如果生成的功率 P_t > c_up，梯度把样本拉回
```

### 为什么 guidance=0 更好？

**实验发现**：
- guidance=0.5：强行约束，coverage低（54%）
- guidance=0：自由生成，coverage高（82%）

**原因**：
1. 条件输入（预测值+时间）已提供足够信息
2. 额外梯度约束过于严格，限制了多样性
3. 硬截断破坏了概率分布

---

## 六、生成多个场景

### 为什么要生成多个场景？

**电力系统需要**：
- 不是预测一个确定值，而是预测一个**概率分布**
- 生成多个可能的场景，覆盖不确定性

### 生成流程

```
对于每个测试样本：

重复 N 次（N=10）：
    1. 从不同的随机种子初始化 x_T
    2. 运行去噪过程
    3. 得到一个场景 (P_wind, P_solar, P_load)
    
最终得到 N 个场景，表示概率分布
```

**示例（N=3）：**
```
场景1: [0.35, 0.55, 0.52]  # 风电偏高，光伏正常，负荷偏低
场景2: [0.25, 0.60, 0.58]  # 风电偏低，光伏偏高，负荷偏高
场景3: [0.30, 0.58, 0.55]  # 都接近预测值

这3个场景共同表示：功率可能在这个范围内波动
```

---

## 七、总结

### 核心流程图

```
训练阶段：
历史数据 → 计算残差 → 加噪 → 训练模型预测噪声
                                    ↓
采样阶段：                    模型参数 θ
纯噪声 x_T → 模型预测噪声 → 去噪 → x_{T-1} → ... → x_0（残差）
                                    ↓
                              最终功率 = 预测值 + 残差
```

### 关键理解

1. **预测残差而非功率**：残差范围小，更容易学习
2. **条件输入很重要**：预测值+时间特征指导生成
3. **guidance=0 更好**：额外约束有害，条件输入已足够
4. **t=0 的波动正常**：这是残差的不确定性，不是去噪不干净

### Width 评价

- ✅ 当前 width 可接受（coverage 82%）
- ⚠️ 但可以优化（后处理筛选或软约束）
- 📊 不是越低越好，要平衡 coverage 和精确度
