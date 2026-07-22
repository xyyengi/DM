# V5 风险条件扩散模型 Codex 交接指令

## 1. 总体要求

请先完整读取并遵守：

- `D:\Extreme_Scenario\AGENTS.md`
- 当前仓库中的相关说明、配置、测试和已有实验结果

代码项目位于：

```text
D:\DM_local
```

本任务需要先进行只读检查并给出实施计划。在计划得到用户确认前，不要修改模型代码，不要启动新的完整训练。不得移动、复制、重命名或重构整个 `DM_local` 项目，也不得覆盖已有 V4-RS、V4-C、V4-S 实验结果和 checkpoint。

论文目标是：

> 面向 168h 风电-光伏-负荷联合序列，构建“三通道联合扩散状态 + 条件向量化/编码 + 多层 AdaGN/FiLM 调制 + 显式跨变量耦合 + 复合极端风险条件 + partial classifier-free guidance”的风险条件扩散模型。

不要继续通过增加卷积输入通道来堆叠条件。扩散状态 `x_t` 必须始终只有风、光、负荷三个通道。

## 2. 当前基线与训练暂停要求

当前 V4 基线分支为：

```text
experiment/v4-posterior-residual-standardization
```

当前应保留的基线提交为：

```text
e4ef644  train: make validation and replicate selection reproducible
```

该提交包含以下应当保留的基础设施：

- 固定 Python、NumPy、PyTorch 和 DataLoader 随机种子；
- 每个 epoch 的 validation 使用相同扩散时刻和噪声；
- validation 不改变后续训练随机状态；
- 保存 validation loss 最好的 top-3 checkpoint；
- 可复现训练相关测试；
- 已有 residual standardization、posterior reverse variance 和评估流程。

不要回退这些修改。

暂时暂停“使用 `seed=2026` 和 `seed=2027` 完整重训 V4-RS 两次”：

- 如果 `seed=2026` 尚未启动或刚启动，先停止；
- 如果 `seed=2026` 已接近完成，可以完成这一轮，作为可复现 V4-RS 基线；
- 暂时不要启动 `seed=2027`；
- 如果 `seed=2026` 已完成而 `seed=2027` 正在排队，取消第二轮；
- 如果两轮已经完成，保留结果，不删除、不覆盖。

现在没有必要先花费两次完整训练成本验证旧结构。最终模型结构基本确定后，再对 V4-RS 和 V5 使用相同流程统一运行 2 至 3 个随机种子。

## 3. V5 必须使用新分支

V5 不允许直接在 V4 基线分支上开发。

创建分支前必须确认：

1. 当前基线提交确实为 `e4ef644`，或者明确记录实际起点；
2. `git status` 工作区干净；
3. 用户现有修改已经提交或得到妥善保留；
4. 没有仍会动态读取源码的训练或生成进程；
5. 不执行 `git reset --hard`、`git checkout --` 等破坏性操作。

从基线提交创建新分支：

```bash
git switch -c experiment/v5-risk-conditioned-film
```

如果该分支已经存在，应先只读检查，不要强制覆盖或删除。

V5 必须使用新的模型类、模型入口和独立配置文件。保留 V4 配置、旧模型入口及旧 checkpoint 加载兼容性，不允许直接把旧 `ResUNet` 改到无法复现实验或加载 V4 checkpoint。

## 4. 必须先解释和修复的核心结构问题

### 4.1 缺少 diffusion timestep 条件

必须区分三种完全不同的时间信息：

1. `diffusion timestep t`：当前处于第几个加噪/去噪阶段，表示噪声强度；
2. 168h 相对序列位置：窗口内第 1 至第 168 个小时；
3. 真实 calendar：小时、日、星期、月份的 sin/cos 编码。

当前代码虽然随机采样扩散步 `t` 并据此构造 `x_t`，但 denoiser 调用没有接收 `t`。标准形式应为：

```text
epsilon_pred = epsilon_theta(x_t, t, condition)
```

必须实现独立的 diffusion timestep sinusoidal embedding 和 MLP，并将其注入每一个残差块。禁止把 calendar embedding 当作 diffusion timestep embedding。

必须增加测试：固定 `x_t` 和其他条件，仅改变 `t` 时，网络输出应发生变化；训练和采样两条路径都必须把正确的 `t` 传入 denoiser。

### 4.2 真实日历与固定 `0..167` 错位

当前 dataset 给每个窗口返回固定的 `timepoints = 0..167`，旧模型又根据它推导小时、周几和月份。这会把所有窗口错误地当成相同月份和相同星期起点，月份甚至通常始终为 0。

需要执行以下调整：

- 不再根据固定 `0..167` 推导真实月份、星期和小时；
- 保留数据中已有的 8 维真实 calendar sin/cos；
- 将 168h 相对位置编码与真实 calendar 编码分开；
- 连续 sin/cos calendar 应通过独立 condition encoder 编码，不再作为 noisy-state 卷积通道；
- 不要改成只能记忆训练月份的离散月份 embedding，尤其当前 validation/test 为 11 月和 12 月。

### 4.3 条件与扩散状态没有解耦

当前模型将以下张量直接拼接为 14 或 16 个卷积输入通道：

```text
x_t(3) + forecast(3) + calendar(8) + optional dynamic features
```

V5 必须改为：

```text
扩散状态：x_t                    [B, 3, 168]
序列条件：forecast               [B, 3, 168]
日历条件：calendar sin/cos       [B, 8, 168]
全局条件：risk/state/intensity   [B, D_risk]
扩散条件：diffusion timestep t   [B]
```

forecast 和 calendar 通过独立 sequence condition encoder 形成多尺度条件特征图；risk 和 diffusion timestep 形成全局条件向量。它们通过多层条件调制进入 UNet，而不是不断增加 `x_t` 的输入通道。

## 5. V5 目标架构

建议的结构为：

```text
x_t [B,3,168]
  -> variable-specific stem / joint three-variable representation
  -> temporal ResUNet

forecast [B,3,168] + calendar [B,8,168]
  -> sequence condition encoder
  -> conditions at 168 / 84 / 42 resolutions

diffusion timestep t
  -> sinusoidal embedding -> MLP

risk vector
  -> risk MLP

每个 ResBlock：
  GroupNorm -> SiLU -> Conv
  AdaGN/FiLM using diffusion-t + risk embedding
  add/project sequence condition at the matching resolution
  residual connection
```

具体要求：

- 用 `GroupNorm + SiLU` 替换 V5 中的 `BatchNorm + ReLU`；
- 使用 AdaGN/FiLM 产生每层的 `gamma` 和 `beta`，例如 `h * (1 + gamma) + beta`；
- sequence condition 在 UNet 各分辨率下分别投影和注入；
- `model.num_layers` 等配置必须与真实结构一致，不能保留无效配置；
- 新模型必须具有明确、可测试的张量形状和条件开关；
- V4 与 V5 使用模型工厂或明确入口区分，禁止依靠模糊的通道数推断版本。

## 6. 风光荷显式耦合

当前 V4 主要依靠普通卷积隐式混合风、光、荷。V5 后续阶段需要加入轻量 `Cross-Variable Attention` 或等价的显式变量交互模块：

- 三个变量保留可区分的表示；
- 在每个时刻或选定的 1 至 2 个 UNet 分辨率上，对 wind/PV/load 三个变量做注意力或门控交互；
- 不要称为地理空间注意力，应称为跨变量注意力；
- 不要一开始在所有层都加入注意力，先做轻量版本并消融。

除架构耦合外，后续可以增加：

```text
L = L_epsilon + lambda_grad * L_gradient + lambda_net * L_netload
```

- `L_epsilon`：噪声预测损失，可使用合理的 SNR/min-SNR 加权；
- `L_gradient`：基于可微的 `x0_hat`，约束 1h 和 6h 爬坡；
- `L_netload`：在统一物理尺度下约束 `load - wind - PV`；
- 高噪声时 `x0_hat` 会被放大，梯度/净负荷辅助损失必须进行 SNR 或时刻权重控制；
- 不要未经验证一次加入复杂相关矩阵损失、梯度损失、风险损失和物理损失。

## 7. 复合极端风险条件

风险条件至少考虑：

- 高净负荷峰值或强度；
- 低风低光/低总新能源强度；
- 连续新能源不足持续时间；
- 1h 或 6h 净负荷快速爬坡；
- 可选事件状态、阶段或提前期。

必须先定义风险条件的数据语义：

1. 风险阈值只用 train 的唯一小时拟合；
2. 事件标签和风险强度在训练时由 train actual 构建；
3. 风、光、负荷先转换到一致物理尺度，再计算净负荷；
4. validation/test 只能复用 train 阈值；
5. 推理时支持显式给定目标风险向量；
6. 检查目标风险与给定 forecast 是否明显冲突；
7. 离散标签之外，优先保留连续风险强度，便于控制性实验。

风险条件稳定后，加入 partial classifier-free guidance：

- forecast 和 calendar 是基础条件，通常始终保留；
- 训练时以一定概率仅丢弃 risk condition；
- 生成时比较 `forecast+calendar` 与 `forecast+calendar+risk` 两个预测；
- 使用 risk guidance scale 控制极端风险强度；
- 加入 condition-dropout 和 CFG 的单元测试及消融。

## 8. 稀有事件与大偏差部分

现有 event-aware sampler 可以保留，但它只代表“增加极端事件训练曝光”，不等于“风险可控生成”。必须与风险条件和 CFG 分开做消融。

在风险条件模型稳定后，才考虑：

- 按事件严重度进行样本重加权；
- 风险能量引导；
- 基于可微风险函数的采样梯度；
- 候选生成后的风险筛选或重要性重采样。

论文中只能表述为“大偏差/稀有事件启发的引导、重加权或筛选”，除非真正推导并验证了多变量 168h 序列的 rate function。禁止把普通条件扩散包装成严格的大偏差理论方法。

旧 KDE forecast-error interval guidance 不作为 V5 主方法。原因包括：

- 它约束的是逐通道常规预测区间，不是复合极端风险；
- 它缺少跨变量联合风险语义；
- 中间噪声状态 `x_t` 不能简单视为实际功率；
- 可保留为旧方法基线，但不要与 V5 风险引导混为一谈。

## 9. 数据、滑窗和物理尺度审计

必须审计高度重叠的 168h 滑窗。当前 7296 个 train 唯一小时生成了 7129 个 stride-one 窗口，相邻窗口共享 167 小时。因此：

- 不能把 7129 个窗口表述为 7129 个独立周样本；
- 标准化、风险阈值和事件计数必须基于 train 唯一小时或唯一事件；
- event-aware sampler 应继续限制单一事件的重复抽取；
- 最终极端评价应按唯一事件汇总；
- 如做统计显著性分析，应按事件或时间块处理，不能把重叠窗口视为独立观测。

必须审计物理尺度与边界：

- wind、PV、load 的归一化分母分别是什么；
- 净负荷是否在统一 MW 尺度计算；
- residual 的符号始终保持 `forecast - actual`；
- residual standardization 的逆变换是否可微且在训练/生成一致；
- 生成后 wind/PV 是否小于 0 或超过装机容量；
- load 是否小于 0；
- 必须报告未经裁剪的物理越界率；
- 不要只靠生成后硬裁剪来掩盖模型问题，可研究软边界损失或有界参数化。

## 10. Validation、checkpoint 与最终评价

保留固定 validation 扩散时刻/噪声和 top-3 checkpoint，但最终选模不能只依赖 epsilon MSE。

训练后应对 top-3 checkpoint 使用：

- 同一个固定 validation 子集；
- 同一组 forecast/calendar/risk 条件；
- 同一组生成随机种子；
- 同样数量的 ensemble members；
- 相同的 reverse variance 和采样设置。

选模至少综合：

- CRPS；
- multivariate Energy Score；
- 80%/90%/95% coverage deviation；
- interval width；
- ACF 和 168h 周期结构；
- 1h/6h ramp error；
- 跨变量相关性和净负荷误差；
- 高净负荷、低新能源、持续不足、快速爬坡命中率；
- 目标风险等级与生成风险等级的一致性；
- 唯一事件级持续时间和峰值误差；
- 物理越界率。

test 只能在模型、checkpoint 和超参数确定后使用，不能反复依据 test 调参。不同 ensemble size 的结果不能直接作为严格配对消融。

## 11. 分阶段实施与消融顺序

禁止一次性把所有模块合并后只训练一个模型。按以下阶段推进：

### 阶段 0：只读审计与计划

输出：

- 当前训练/采样张量流；
- diffusion timestep 缺失证据；
- calendar 错位证据；
- 数据与物理尺度审计；
- 文件级修改清单；
- V4 checkpoint 兼容方案；
- 每个阶段的测试和实验预算。

等待用户确认后再修改。

### 阶段 1：V5-T/F 基础修复

只实现：

1. 正确的 diffusion timestep embedding；
2. 正确的 calendar sequence condition encoder；
3. 168h 相对位置与真实 calendar 分离；
4. `GroupNorm + SiLU + AdaGN/FiLM`；
5. V4/V5 模型入口和配置解耦；
6. 张量形状、timestep 敏感性、训练/采样一致性测试；
7. 与 V4-RS 相同数据、seed 和选模流程的配对实验。

在阶段 1 验证完成前，不加入 risk CFG、cross-variable attention、梯度/净负荷损失或稀有事件引导。

### 阶段 2：显式跨变量耦合

- 加入轻量 Cross-Variable Attention；
- 先只放在 1 至 2 个分辨率；
- 对比无 attention 与有 attention；
- 检查 CRPS、Energy Score、跨变量相关性、净负荷和计算成本。

### 阶段 3：结构敏感损失

- 增加 SNR 加权的 1h/6h gradient loss；
- 增加统一 MW 尺度的 net-load consistency loss；
- 分别消融，不要同时加入后只报告总效果。

### 阶段 4：显式风险条件与 partial CFG

- 构建风险向量和训练标签；
- 加入 risk embedding；
- 加入 risk-only condition dropout；
- 加入 partial CFG；
- 评价风险命中率、强度控制单调性和普通场景质量。

### 阶段 5：稀有事件增强

- 比较无增强、event-aware sampler、严重度重加权和风险能量引导；
- 评价极端覆盖、校准、持续时间、跨变量一致性和物理合理性；
- 此阶段才讨论大偏差/稀有事件启发方法。

推荐的论文消融链：

```text
V4-RS baseline
-> V5 + correct diffusion timestep
-> V5 + condition encoder/AdaGN-FiLM
-> V5 + cross-variable attention
-> V5 + gradient/net-load loss
-> V5 + risk condition/partial CFG
-> V5 + rare-event guidance or reweighting
```

## 12. 第一轮需要 Codex 返回的内容

现在请先停止在分析和计划阶段，不要直接修改。第一轮回复必须包括：

1. 当前 git 分支、HEAD、工作区状态和运行中训练任务检查结果；
2. 是否安全创建 `experiment/v5-risk-conditioned-film`；
3. diffusion timestep、calendar、condition concat 的实际代码路径；
4. 数据字段、张量形状、残差符号和物理尺度；
5. V5 第一阶段准备新增和修改的文件；
6. 如何保持 V4 checkpoint 与旧配置兼容；
7. 第一阶段单元测试清单；
8. V4-RS 与 V5-T/F 的最小配对实验设计；
9. 预计训练成本和停止条件；
10. 需要用户确认的关键选择。

不要在第一轮中启动训练，不要修改已有模型文件，不要删除或覆盖任何实验结果。
