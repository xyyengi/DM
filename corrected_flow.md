# 修正后的场景生成流程

## 你的数据输入（input_4.27）

```
train_pred.npy: (18917, 168, 11) - 训练集预测值（FEDformer输出）
train_res.npy: (18917, 168, 11) - 训练集残差（实际-预测）
val_pred.npy: (2608, 168, 11) - 验证集预测值
val_res.npy: (2608, 168, 11) - 验证集残差
test_pred.npy: (5381, 168, 11) - 测试集预测值
test_res.npy: (5381, 168, 11) - 测试集残差
```

**11维特征定义：**
- Channel [0:3]: 风、光、负荷残差（Residuals）
- Channel [3:11]: 8维时间周期编码（Sin/Cos）

**注意**：你的数据里**没有直接的功率值**，只有**预测值**和**残差**！

---

## 修正后的完整流程

### 训练阶段

**Step 1: 数据加载（无需计算残差，直接读取）**

```python
# 从input_4.27直接读取
forecast_data = train_pred.npy  # (18917, 168, 11)
residual_data = train_res.npy   # (18917, 168, 11)

# 11维分解：
# - Channel [0:3]: 风、光、负荷残差
# - Channel [3:11]: 8维时间编码
```

**Step 2: 归一化**

```python
# 使用预测值的最大值进行归一化
max_values = np.max(np.abs(forecast_data), axis=(0, 1))

forecast_norm = forecast_data / max_values  # 预测值归一化
residual_norm = residual_data / max_values  # 残差用同样的值归一化
```

**Step 3: 构建14通道输入**

```python
# 对于每个样本：
input_14ch = np.concatenate([
    residual_norm[:3, :],    # Channel [0:3]:  目标残差（扩散目标）
    forecast_norm[:3, :],    # Channel [3:6]:  预测值（条件）
    forecast_norm[3:11, :]  # Channel [6:14]: 时间编码（条件）
], axis=0)  # 形状: (14, 168)
```

**Step 4: 训练扩散模型**

```python
# 输入: x_t (14, 168) - 加噪后的14通道数据
# 输出: ε_θ (3, 168)   - 只预测前3维的噪声

# 损失: MSE(ε_θ, ε)  # 预测噪声 vs 真实噪声
```

---

### 采样/生成阶段（guidance=0）

**Step 1: 初始化**

```python
# 从测试集读取条件
forecast_test = test_pred.npy  # (5381, 168, 11)

# 归一化（用训练集的max_values）
forecast_norm = forecast_test / max_values

# 初始化纯噪声
x_T = torch.randn(14, 168)  # 14通道纯噪声
```

**Step 2: 去噪过程**

```python
for t in range(T, 0, -1):
    # 1. 模型预测噪声（只预测前3维）
    ε_θ = model(x_t, t, condition)  # ε_θ形状: (3, 168)
    
    # 2. 去噪：恢复x_{t-1}
    # 前3维（残差）：用模型预测去噪
    x_t[:3] = (x_t[:3] - sqrt(1-α_t) * ε_θ) / sqrt(α_t)
    
    # 后11维（条件）：保持不变（因为是条件，不是目标）
    x_t[3:14] = forecast_norm  # 保持预测值和时间编码
    
    # 3. 添加随机噪声（DDPM采样）
    if t > 1:
        noise = torch.randn_like(x_t[:3])
        x_t[:3] = x_t[:3] + sqrt(1-α_{t-1}) * noise
```

**Step 3: 获取最终残差**

```python
# 去噪完成后
x_0 = x_t  # 形状: (14, 168)

# 提取生成的残差（前3维）
generated_residual = x_0[:3, :]  # (3, 168) - 风、光、负荷残差

# 提取预测值（Channel 3-6）
forecast_3ch = x_0[3:6, :]  # (3, 168) - 风、光、负荷预测
```

**Step 4: 计算最终功率**

```python
# 反归一化
# 注意：残差和预测值用的是同一个max_values

# 生成的残差（反归一化）
generated_residual_denorm = generated_residual * max_values[:3].reshape(3, 1)

# 预测值（反归一化）
forecast_denorm = forecast_3ch * max_values[:3].reshape(3, 1)

# 最终功率 = 预测值 + 残差
final_power = forecast_denorm + generated_residual_denorm

# final_power[0]: 风电功率
# final_power[1]: 光伏功率
# final_power[2]: 负荷功率
```

---

## 关键修正点

### 1. 输入数据

**我之前说的（错误）：**
```
输入：历史功率数据
计算残差：ε = actual - forecast
```

**实际（正确）：**
```
输入：train_pred.npy 和 train_res.npy（已经计算好了）
直接使用：无需再计算残差
```

### 2. 14通道结构

**我之前说的（模糊）：**
```
Channel [0:3]: 目标残差
Channel [3:6]: 预测值
Channel [6:14]: 时间编码
```

**实际（正确）：**
```
Channel [0:3]:  目标残差（来自residual_data[0:3]）
Channel [3:6]:  预测值（来自forecast_data[0:3]）
Channel [6:14]: 时间编码（来自forecast_data[3:11]）
```

### 3. 生成过程

**我之前说的（模糊）：**
```
生成残差 → 加预测值 → 得到功率
```

**实际（正确）：**
```
1. 从纯噪声x_T开始（14通道）
2. 去噪过程中：
   - 前3维（残差）：用模型预测去噪
   - 后11维（条件）：保持输入的预测值和时间编码
3. 最终x_0的前3维就是生成的残差
4. 功率 = 输入的预测值 + 生成的残差
```

---

## 为什么 guidance=0 更好？

### 有条件引导时（guidance=0.5）

```python
# 在采样过程中，每一步都检查：
generated_power = forecast + x_t[:3]
if generated_power < c_down or generated_power > c_up:
    # 计算梯度，强行拉回区间
    x_t[:3] = x_t[:3] + guidance * gradient
```

**问题**：
- 硬截断破坏了残差的自然分布
- 梯度可能方向错误
- 过度约束导致 coverage 低（54%）

### 无条件引导时（guidance=0）

```python
# 纯扩散过程，不额外约束
# 模型只根据条件（预测值+时间）生成残差
# 残差自然分布在合理范围内
```

**优势**：
- 保留残差的自然分布
- coverage 高（82%）
- 模型更自由，生成更多样化的场景

---

## 总结

### 核心流程图（修正版）

```
训练阶段：
input_4.27/
├── train_pred.npy (预测值+时间编码)
├── train_res.npy (残差)
        ↓
构建14通道输入:
├── Channel [0:3]: 残差（目标）
├── Channel [3:6]: 预测值（条件）
├── Channel [6:14]: 时间编码（条件）
        ↓
训练扩散模型: 预测噪声 ε_θ

采样阶段：
test_pred.npy (测试集预测值)
        ↓
纯噪声 x_T ~ N(0,I)
        ↓
去噪过程:
├── 前3维: 模型预测去噪
├── 后11维: 保持预测值和时间编码
        ↓
x_0 (14通道)
├── x_0[0:3]: 生成的残差
├── x_0[3:6]: 预测值
        ↓
最终功率 = 预测值 + 生成的残差
```

### 关键理解

1. **数据已经准备好了**：input_4.27里有预测值和残差，无需再计算
2. **14通道输入**：前3维是目标（残差），后11维是条件（预测值+时间）
3. **模型只预测前3维的噪声**：后11维保持不变
4. **guidance=0 更好**：额外约束有害，条件输入已足够
5. **最终功率 = 输入的预测值 + 生成的残差**
