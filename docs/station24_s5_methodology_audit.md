# Station-24 Full Independent Tail V2 + Original S5 方法论审查

**审查日期：2026-10-13**  
**范围：24 场站、168 h 风光联合条件扩散；Full Independent Joint Tail V2；Original S5**  
**性质：只读方法论审查。未修改模型、checkpoint、成员或正式结果；未训练、未生成。**

## 执行摘要

本审查的核心结论不是“S5 完全无效”，而是：**Original S5 把三个不同问题混成了一个操作**——Raw–Tail 成员耦合、事件位置识别、局部概率分布修正。现有按序号配对只定义了一个任意的独立样本耦合，没有足够依据把 `Tail_i - Raw_i` 解释为第 `i` 条轨迹的“模型修正”。因此，现有最终集合的 CRPS、Energy Score 等经验结果仍然是真实结果，但对“为什么有效”的成员级因果解释并不成立，而且所得局部化分布依赖成员排列。

Original S5 的最大修正能量窗口确实很好地限制了作用域：窗口外严格恢复 Raw，普通概率指标没有被 Full V2 的全时域修正明显拖累。然而，现有成员级归因也给出了明确反例：Full V2 Tail 的 100 个事件命中中，局部化后仅保留 28 个，丢失 72 个；其中 68 个丢失成员的选中窗口覆盖真实事件不足 50%。因此“Raw–Tail 差异越大，Tail 修正越有事件价值”没有被数据支持。S5 更像一个有效的**主体保护器**，而不是已经验证的**事件定位器**。

没有一种成熟文献方法可以直接替换 S5。ECC/Schaake Shuffle 解决边缘分布校准后的依赖结构重排；Matrix Profile/DTW 解决子序列发现或时间对齐；最优传输解决给定代价下的耦合；它们均不能同时证明事件位置、极端结构保留和修正后联合分布合理。最值得先做的不是 S5-V3，而是一次**全随机流共同随机数（full-path common-random-number, CRN）配对归因实验**：保持 Original S5 规则不变，只让 Raw 与 Tail 共享初始噪声及每个反向步的噪声，从而先回答当前最根本的可识别性问题。

## 1. 审查对象与代码事实

### 1.1 Original S5 的实际算法

实现位于 `tools/localize_station24_independent_joint_tail.py`：

1. 从 Raw 500 条成员中取第 401–500 条，按序号与 Full V2 Tail 第 1–100 条配对；元数据记录 `background_member_range=[400,500]`。
2. 对每对成员计算 `delta = Tail - Raw_background`。
3. 风、光分别按容量聚合，对绝对修正幅度求和，得到每小时联合修正能量。
4. 从训练集事件持续时间经验分位数构造 100 个持续时间，并在每个 issue 内按固定种子打乱分配给成员。
5. 对给定持续时间，选择滚动修正能量和最大的单个连续窗口。
6. 窗口边缘使用最多 3 h 的半余弦掩码；窗口外掩码严格为零。
7. 输出

   \[
   X_{S5}=X_{Raw}+M\odot(X_{Tail}-X_{Raw}),
   \]

   最终组成 400 Raw + 100 局部化 Tail。

### 1.2 正式扩散采样的随机性

`generate_station24.py:set_seed()` 同时设置 Python、NumPy、PyTorch 和 CUDA seed，但正式 `sample()` 并不接收显式的 `initial_noise`、`torch.Generator` 或完整反向噪声计划。`src/models/station_conditioned_diffusion.py:denoise_step()` 在每个随机反向步调用 `torch.randn_like(noisy)`，而 `sample()` 还会生成初始高斯噪声。

项目另有 `differentiable_ddim_sample(..., initial_noise=...)`，可以从指定初始噪声走确定性 DDIM 类路径；但它不是当前正式 ancestral DDPM 采样协议。直接用它重算会改变采样器，不能被描述成对现有正式成员的等价配对。

## 2. 按序号配对是否有充分依据

### 2.1 统计含义

若 Raw 成员与 Tail 成员在各自集合内可交换，按序号配对在数学上可以看成从两个边缘分布构造的一种**独立乘积耦合的有限样本实现**。所以它不是无法计算，也不是自动产生错误指标；但成员序号没有共同潜变量、共同噪声或物理身份，不能据此称为“对应轨迹”。

设 Raw 与 Tail 的生成映射分别为 \(F_R\) 与 \(F_T\)，随机流为 \(z_R,z_T\)。现有差值为

\[
\Delta_{ind}=F_T(z_T)-F_R(z_R).
\]

若人为选定同一随机流作为反事实参照，可写成

\[
\Delta_{ind}=
\underbrace{F_T(z_T)-F_R(z_T)}_{\text{给定耦合下的模型/检查点差异}}
+
\underbrace{F_R(z_T)-F_R(z_R)}_{\text{Raw 自身的采样差异}}.
\]

第二项不是 Tail 学到的修正。更重要的是，这个分解依赖所选择的耦合；不存在无需假设的唯一“真实成员对应关系”。

### 2.2 当前结果能否区分两类差异

不能。现有数组只保留最终样本，没有保存每个成员完整的初始噪声和逐步反向噪声，也没有使 Raw 与 Tail 共享同一随机流。即便二者元数据都写 `generation_seed=424242`，Raw 生成 500 条、Tail 生成 100 条，成员切片、batch/chunk 以及模型调用图不同，都会改变全局 RNG 的消费顺序。

已有固定种子随机置换配对是有价值的敏感性证据：不同排列的 primary member-hit 约为 0.115、0.1175、0.1225，any-hit 在 3/4 与 4/4 之间变化；原顺序为 0.110、4/4。它说明 S5 结果对任意耦合并非完全不敏感，但不能告诉我们哪一种排列更接近模型差异。

### 2.3 应如何解读现有正式结果

- 可以说：在**该固定成员池、固定序号耦合和固定 duration permutation** 下，Original S5 的最终 500 成员集合取得了所报告的经验指标。
- 不可以说：第 `i` 条 Tail 对第 `i` 条 Raw 的差值就是 Tail 对该基础轨迹的修正。
- 不可以把排列依赖的局部化分布视作仅由 Raw、Tail 两个边缘分布唯一决定的结果。
- 这一缺陷削弱的是成员级归因和方法不变性；它不会反向抹掉已经计算出的最终集合指标。

## 3. 能否构造更可比较的 Raw–Tail 成员

### 3.1 可行方案

代码上可行，但仅“设置同一个 seed”不够。对当前正式 ancestral sampler，稳健的共同随机数实现应显式控制：

1. 每个 `issue × member` 的相同初始 \(x_T\)；
2. 每个反向 timestep 的相同高斯噪声；
3. 相同的成员映射、张量形状及 timestep 顺序；
4. 最好用 `issue/member/timestep/seed` 生成无状态 keyed noise，避免 batch/chunk 改变 RNG 消费顺序；
5. Raw 与 Tail 各自仍使用原模型、原条件和原采样方程，只共享外生随机数。

共同随机数是比较随机系统的成熟方差缩减设计，但并非任何系统中都必然降低方差；其效果依赖两个响应对共同随机输入的相关性，需要实测而不是假定。[Glasserman 与 Yao, 1992, *Some Guidelines and Guarantees for Common Random Numbers*](https://doi.org/10.1287/mnsc.38.6.884)

### 3.2 局限

- 同噪声是一种有明确实验依据的耦合，但仍不是唯一的自然耦合，也不等同于真实世界中的成员身份。
- 如果 Raw 与 Tail 的逆过程对噪声响应非常不同，共同随机数未必产生“语义相似”的最终轨迹。
- 必须验证共享随机流没有改变各模型各自的边缘采样律；只能改变二者的联合耦合。
- 现有正式成员无法事后恢复为 CRN 配对，必须另做生成审计；但不需要重新训练。
- 保存所有 timestep 噪声会很大，推荐无状态即时生成；显存可接近现有生成，运行量约为每个 seed 一次 Raw + 一次 Tail 生成及评价。

### 3.3 其他耦合选择

- **只共享初始噪声**：不足。当前每个反向步仍注入独立随机噪声。
- **重置全局 RNG 并保持完全相同调用顺序**：理论可行但工程脆弱，batch/chunk 或分支变化即可错位。
- **确定性 DDIM + 相同初始噪声**：可形成清晰配对。DDIM 允许非马尔可夫、可确定的生成路径，[Song、Meng、Ermon, 2021](https://openreview.net/forum?id=St1giarCHLP)。但它改变正式采样协议，应作为单独机制审计，不能用于重新解释现有 DDPM 结果。
- **输出后最近邻/最优传输匹配**：可以作为敏感性对照，但匹配代价会决定答案。全轨迹 L2 最近邻可能主动抹掉 Tail 的极端差异，不能默认比 CRN 更正确。

## 4. 最大修正能量窗口的隐含假设

Original S5 实际假设：在一个给定持续时间内，\(|Tail-Raw|\) 容量加权和最大的地方，是最值得保留的 Tail 修正。这个假设把“改动幅度”当成“事件价值”的代理，但没有区分：

- 随机采样导致的普通轨迹错位；
- 对主体水平或相位的大幅无益偏移；
- 真正的持续低出力、强 ramp 或下降—恢复结构；
- 风、光之间方向相反但绝对值相加的变化。

### 4.1 已有成员级证据

| 证据 | 观察 | 对假设的含义 |
|---|---:|---|
| Full V2 Tail 事件命中 | 100 | Tail 池中确有可保留结构 |
| S5 局部化后命中 | 44 | 局部化后还产生 16 个新命中，不能只看净数 |
| 原命中被保留 | 28/100 | 微平均保留率仅 28% |
| 原命中丢失 | 72/100 | 大量有价值结构没有被完整留下 |
| 丢失且选窗覆盖真实事件 <50% | 68/72 | 主要表现为位置/作用域失配 |
| 最强修正窗与生成场景最强事件结构窗平均重叠 | 1.49% | 能量最大与结构最强通常不是同一位置 |
| 二者完全不重叠 | 97.17% | 给出强反例 |

逐事件保留率差异也很大：两个 wind-negative 事件约为 17.1% 和 0%，wind-positive 为 50%，solar-positive 为 33.9%。这不足以建立稳定事件规律，却足以否定“最大能量天然等于最有价值事件修正”的默认解释。

另一方面，Original S5 的 ordinary/body 指标优于 Full V2 全时域混合：Renewable CRPS 约 151.215（Raw 151.482，Full mix 152.333），Energy Score 约 6.588（Raw 6.593，Full mix 6.598），90% width 约 794.43（Raw 794.36，Full mix 816.99）。所以它作为**局部作用域约束**有经验价值；这并不能验证其窗口具有事件语义。

## 5. 三个 S5 组件分别解决什么、限制什么

| 组件 | 解决的问题 | 引入的限制 |
|---|---|---|
| 单个连续窗口 | 将 Tail 影响压缩为紧支撑；窗口外严格恢复 Raw；避免零散全周改写 | 假设每个成员只有一个主要连续作用域；无法同时覆盖多事件、风光错时事件或相隔的下降与恢复；边界截断时更严重 |
| 训练持续时间分位数 | 不读取验证真实持续时间；使窗口长度边缘分布接近训练事件 | duration 通过随机排列分给成员，与该成员轨迹、事件源、方向、issue 条件无关；只匹配边缘分布，不匹配条件分布；168 h 边界会截断 |
| 最多 3 h 半余弦融合 | 缓和硬拼接的边界跳变，保证窗外精确 Raw | 会衰减 onset、depth、短 ramp 和恢复端；线性插值轨迹未必来自 Raw 或 Tail 任一预测分布；不保证时空 copula 或物理联合分布仍合理 |

半余弦只是数值平滑器，不是概率校准器。单窗只是作用域先验，不是事件检测证明。训练 duration 只提供无验证泄漏的长度先验，不提供成员级正确长度。

## 6. 三个必须分开的研究问题

### 6.1 事件位置识别是否准确

研究对象是窗口选择器。应报告选窗与预先定义事件窗口的 intersection-over-union、precision/recall、onset 距离，以及与**生成场景自身事件结构**的重叠。真实事件只能用于选完后的离线评价，不能用于挑窗。

### 6.2 局部修正是否保留有价值结构

研究对象是同一个 Tail 成员局部化前后：原命中保留率、丢失率、onset/duration/depth、1/3/6 h ramp、下降后恢复幅度与时间。任何“400 Raw 稀释”必须和局部化损失分开：Full Tail-only macro member-hit 0.25 在 500 成员中的配额贡献为 0.05；S5 Tail-only 为 0.11，对应 0.022。配额造成共同的 80% 稀释，局部化又额外减少约 0.028 的总成员命中贡献。

### 6.3 修正后的联合场景分布是否合理

研究对象是完整 500 成员分布。CRPS、coverage、width、Interval Score 检查边缘质量；Energy Score 检查多变量整体，但可能对相关结构不够敏感。Variogram Score 专门利用分量差异，对错误相关结构更敏感，适合作为当前时空/风光联合评价的补充。[Scheuerer 与 Hamill, 2015](https://doi.org/10.1175/MWR-D-14-00269.1)

因此还应保留 temporal correlation、station spatial、wind–wind、solar–solar、wind–solar correlation RMSE 及边界 ramp。事件命中改善不能替代分布合理性验证。

## 7. 相关成熟方法：能借鉴什么，不能迁移什么

### 7.1 集合依赖结构保持

#### Ensemble Copula Coupling（ECC）

- 文献：Schefzik、Thorarinsdottir、Gneiting，2013，*Uncertainty Quantification in Complex Simulation Models Using Ensemble Copula Coupling*，DOI [10.1214/13-STS443](https://doi.org/10.1214/13-STS443)。
- 原任务：对多个地点、时刻或变量分别做单变量统计后处理后，恢复原始集合中的经验 copula/秩依赖。
- 核心：先从校准后的边缘分布取样，再按原始 ensemble 的 ranks 重排。
- 可借鉴：把“边缘改善”和“依赖模板”明确分开；评价局部修正是否破坏时空秩结构。
- 不能直接迁移：S5 不是边缘校准后重排；Raw 与 Tail 也不是同一个可交换原始 ensemble 的校准前后版本。ECC 不提供 Raw–Tail 成员反事实配对，也不识别事件位置。

#### Schaake Shuffle 与相似性版本

- 文献：Clark 等，2004，*The Schaake Shuffle: A Method for Reconstructing Space–Time Variability in Forecasted Precipitation and Temperature Fields*，DOI [10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2](https://pubs.usgs.gov/publication/70027101)。
- 文献：Schefzik，2016，*A Similarity-Based Implementation of the Schaake Shuffle*，DOI [10.1175/MWR-D-15-0227.1](https://doi.org/10.1175/MWR-D-15-0227.1)。
- 原任务：使用历史观测场或与当前预报相似的历史日期作为空间—时间依赖模板。
- 可借鉴：依赖模板必须有来源；可在将来设计独立的“结构重排基线”。
- 不能直接迁移：Station-24 的 168 h × 24 站 × 风光维度很高，历史独立周和极端事件有限；模板选择本身可能泄漏季节/事件信息。它仍不决定哪个 Tail 成员应与哪个 Raw 成员局部相减。

Wilks 的 multivariate EMOS 实验也说明 ECC 与 Schaake 类方法的差别在于所选 empirical copula 模板，而非一个普遍最优的重排规则。[Wilks, 2015](https://doi.org/10.1002/qj.2414)

### 7.2 局部时间序列模式识别

#### Matrix Profile

- 文献：Yeh 等，2017，*Time Series Joins, Motifs, Discords and Shapelets: A Unifying View that Exploits the Matrix Profile*，DOI [10.1007/s10618-017-0519-9](https://doi.org/10.1007/s10618-017-0519-9)。
- 原任务：高效计算子序列最近邻距离，用于 motif、discord、join 和 shapelet 发现。
- 可借鉴：在完全不读取验证 truth 的前提下，诊断 Tail 自身是否出现训练事件 motif，或定位异常子序列；比“绝对差最大”更直接描述局部形态。
- 不能直接迁移：窗口长度、归一化和距离度量仍需预注册；常见 z-normalization 会消去幅度，而 depth 恰是本任务关键。它不定义概率耦合，也不保证拼接后分布合理。

#### DTW / Soft-DTW

- 文献：Sakoe 与 Chiba，1978，*Dynamic Programming Algorithm Optimization for Spoken Word Recognition*，DOI [10.1109/TASSP.1978.1163055](https://doi.org/10.1109/TASSP.1978.1163055)。
- 文献：Cuturi 与 Blondel，2017，*Soft-DTW: a Differentiable Loss Function for Time-Series*，[PMLR 70](https://proceedings.mlr.press/v70/cuturi17a.html)。
- 原任务：DTW 对齐局部加速/减速的时间序列；Soft-DTW 提供可微的软最小对齐代价。
- 可借鉴：诊断“形态已生成但时间平移/伸缩”的事件，区分 amplitude deficit 与 timing delay。
- 不能直接迁移：时间扭曲可能把 onset、duration、恢复滞后这些实际错误“对齐掉”；Soft-DTW 是新训练目标，不是零训练 S5 替代；二者都不解决成员耦合或联合概率校准。

### 7.3 样本耦合与最优传输

#### 共同随机数（CRN）

- 原任务：比较两个随机系统时，以共享外生随机数降低差值估计方差。
- 对本任务：最适合检验 `Tail(z)-Raw(z)` 的路径级模型差异，是当前首选实验设计，而不是最终生产算法。
- 限制：耦合仍是研究者选择；必须证明差值方差确实下降、跨 seed 稳定，并保持各自边缘采样律。

#### 最优传输（OT）

- 文献：Peyré 与 Cuturi，2019，*Computational Optimal Transport*，DOI [10.1561/2200000073](https://doi.org/10.1561/2200000073)。
- 核心：在保持两个边缘分布的条件下，求使给定 ground cost 最小的联合耦合。
- 可借鉴：把“配对”显式表述成 coupling optimization；作为全轨迹 L2、低频主体距离或多尺度距离下的敏感性臂。
- 不能直接迁移：ground cost 没有中立选择。以全轨迹相似度配对可能系统性消除 Tail 的极端差异；用真实事件命中定义 cost 又会泄漏评价目标。OT 最优只相对于选定 cost，不等于物理正确。

#### 场景缩减

- 文献：Dupačová、Gröwe-Kuska、Römisch，2003，*Scenario Reduction in Stochastic Programming*，DOI [10.1007/s10107-002-0331-0](https://doi.org/10.1007/s10107-002-0331-0)。
- 可借鉴：用概率度量讨论有限场景集与原分布的距离。
- 不能直接迁移：其目标是减少场景数并重分配概率，不是给两个独立生成器配对后局部拼接。

### 7.4 概率场景局部修正的相邻思路

#### RePaint 类扩散 inpainting

- 文献：Lugmayr 等，2022，*RePaint: Inpainting using Denoising Diffusion Probabilistic Models*，[CVPR/OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html)。
- 原任务：在已知图像区域约束下，用预训练无条件扩散模型补全未知区域，并在反向过程中反复协调已知与未知区域。
- 可借鉴：局部修正若要具有分布一致性，原则上应在生成过程中条件化，而不是生成后直接线性拼接。
- 不能直接迁移：Raw 背景不是观测真值，时间窗口也不是缺失区；将其硬当已知条件会低估背景不确定性。采用该思路需要新的条件采样机制和完整验证，超出 S5 的零训练后处理范围。

#### 风电统计场景生成

- 文献：Pinson、Madsen、Nielsen、Papaefthymiou、Klöckl，2009，*From Probabilistic Forecasts to Statistical Scenarios of Short-Term Wind Power Production*，DOI [10.1002/we.284](https://doi.org/10.1002/we.284)。
- 可借鉴：从概率预测生成场景时，时间依赖本身是建模对象，不能只校准逐时边缘。
- 不能直接迁移：该类统计场景构造与本项目的双模型、24 站风光、168 h 扩散成员局部拼接不同，不提供 S5 的现成替代。

## 8. 四个独立验证事件是否足够

不够支撑 S5 方法优劣的稳定比较。

1. any-hit 只有 0%、25%、50%、75%、100% 五档，一个事件即可改变 25 个百分点。
2. 若把四个事件视为独立配对 Bernoulli 试验，即使新方法对 4/4 全胜，双侧精确 sign test 的最小 p 值也为 0.125，达不到常用 0.05；这只是说明分辨率不足，不主张 p 值是唯一标准。
3. 100 个生成成员共享同一组 issue、条件与模型，不是 100 个独立真实事件，不能替代事件样本量。
4. 相邻 168 h issue 可能覆盖同一物理事件，应按物理事件块分组，不能把重叠窗口当独立样本。

### 8.1 不依赖反复验证集调参的设计

1. **先冻结算法与阈值**：训练统计确定 event definition、window length、mask 和所有阈值；现有验证只用于一次方法比较。
2. **事件块作为统计单位**：把同一物理事件覆盖到的所有 issue 放在同一 fold；按不重叠时间块做 rolling-origin 或 grouped validation。
3. **预注册三层终点**：位置准确性、结构保留率、联合分布质量分别判定，不构造事后总分。
4. **paired generation seeds**：采用 CRN，使同一事件、同一 seed 下 Raw/Tail/S5 可配对；报告逐 seed 与均值/区间。
5. **分层不确定性**：事件指标对“物理事件块 × generation seed”做分层/层级 bootstrap；普通概率指标使用 paired 7-day block bootstrap。
6. **逐事件与 leave-one-event-out**：任何总体结论必须同时展示每个事件，检查是否由单个 solar 或单个方向主导。
7. **前瞻验证或最终 test**：算法完全冻结后才使用新增时间段或一次性 test；test 结果不得反向触发 S5 权重或阈值搜索。
8. **先做精度规划**：根据希望得到的置信区间宽度与可接受差异，估算所需独立事件数；当前数据不足时报告不确定，而非把窗口/成员数量当事件数量。

## 9. 分类结论

### A. 已经证实的设计问题

1. **序号配对没有成员身份依据。** 当前正式采样没有共享完整随机流；同 seed 不等于同路径。
2. **模型差异与采样差异不可识别。** 现有最终数组无法拆分二者。
3. **S5 结果依赖任意成员排列。** random permutation 已导致 member-hit 与 any-hit 变化，因此方法不是两个边缘集合的 permutation-invariant 函数。
4. **最大修正能量不是可靠事件定位器。** 100 个 Full V2 Tail 命中仅保留 28 个；72 个丢失中 68 个的窗口覆盖事件不足一半；修正能量窗与生成事件结构窗绝大多数不重合。
5. **单窗和半余弦确实牺牲事件结构。** 它们保护主体，但会截断或衰减 onset、depth、ramp 和恢复。
6. **四个独立验证事件不足以稳定排序多个局部化方法。** 现有 4/4 只能是描述性结果。

### B. 有较强依据但尚未证实的问题

1. Raw–Tail 独立采样差异可能是最大能量窗失配的重要来源，但没有 CRN 对照，不能量化其占比。
2. 训练 duration 的随机分配可能造成成员条件不匹配，但尚未隔离于配对和窗口评分问题。
3. 半余弦拼接可能破坏高阶时空 copula；现有 ordinary/相关 RMSE 未覆盖全部联合分布性质。
4. CRN 很可能提高路径差值的可解释性和稳定性，但共同随机数不保证对所有非线性模型都降低方差。
5. Full V2 与 Raw 的路径差异可能包含真正的局部极端能力，也可能包含全周主体漂移；现有 S5 无法可靠拆开。

### C. 值得借鉴的成熟方法

1. **全路径 CRN**：作为模型差异归因和配对实验设计，优先级最高。
2. **ECC/Schaake 的“边缘—依赖模板分离”思想**：用于建立依赖结构保持基线和审计，不用于直接做事件定位。
3. **Variogram Score 与多变量秩/相关诊断**：补足 Energy Score 对错误相关结构不够敏感的问题。
4. **Matrix Profile/受限 DTW**：仅作为预注册、truth-free 的生成事件结构诊断工具。
5. **OT**：作为不同 ground cost 下的配对敏感性分析，而非默认主配对。

### D. 不适合当前任务直接采用的方法

1. 把按序号配对解释成自然对应或因果修正。
2. 只用相同 seed 或只共享初始噪声，宣称完成 DDPM 路径耦合。
3. 把 Matrix Profile、DTW 或 Soft-DTW 当成完整 S5 替代；它们不保证概率分布与时空依赖。
4. 把 ECC/Schaake Shuffle 当成事件检测器或 Raw–Tail 反事实配对器。
5. 使用验证真实事件、CRPS 或命中率定义 OT cost 或逐成员窗口，造成目标泄漏。
6. 直接切换 DDIM 后把结果与原 DDPM 正式成员视为同一协议。
7. 直接套用 RePaint，把随机 Raw 背景当作已知观测区域。
8. 用场景缩减方法解决局部修正；其问题定义不同。

### E. 最值得开展的一个后续验证实验

**全路径共同随机数 Raw–Tail 配对归因实验（不训练，Original S5 规则完全冻结）。**

实验只回答“成员配对是否是当前根本瓶颈”，不开发 S5-V3：

1. 冻结 Raw、Full V2 checkpoint、条件、正式 ancestral DDPM schedule、500 成员组成、Original S5 duration 与半余弦规则。
2. 为 Raw100 与 Tail100 显式提供相同的 `issue/member/timestep` keyed 初始及逐步噪声；做至少 3 个 generation seeds。
3. 保留现有独立/序号成员结果为参照；不要用验证 truth 做配对或选窗。
4. 首先验证 Raw 与 Tail 各自边缘统计在 Monte Carlo 波动内未因实现改变。
5. 比较独立配对与 CRN 配对的：
   - 配对差值方差、低频主体差异、跨 seed 稳定性；
   - Full Tail → Original S5 的逐成员事件保留/丢失；
   - 窗口与生成事件结构的重叠；
   - onset/duration/depth/ramp/recovery；
   - CRPS、coverage-width、Interval Score、Energy Score、Variogram Score、时空及风光相关误差。
6. 若 CRN 显著降低普通主体差值并提高归因/选窗跨 seed 稳定性，说明独立配对是主要设计缺陷，之后才值得讨论局部事件定位。
7. 若 CRN 不降低噪声，或事件保留仍然很低，则问题更可能在“最大能量单窗”或 Tail 事件结构本身；应停止把配对当主要原因。

这是当前信息价值最高的实验，因为它直接检验 S5 的基础对象 `Tail - Raw` 是否可解释；在该问题未回答前继续设计更复杂窗口评分，会把定位算法建立在不可识别的差值上。

## 10. 最终判断：是否选错了问题建模方式

部分是。

- 若目标是**解释 Tail 相对 Raw 的模型作用**，当前“独立成员按序号相减”建模不充分，应先使用显式耦合的随机流。
- 若目标是**识别极端事件位置**，应对生成轨迹自身做事件结构识别；Raw–Tail 差值能量不是已经验证的事件分数。
- 若目标是**组合两个预测分布并保持联合结构**，应把问题表述为分布耦合、重排或条件生成，而不是默认逐成员线性拼接仍有正确 copula。

Original S5 的真正已证实价值是：以简单、可复现的紧支撑掩码保护 168 h 主体概率质量；其事件语义、成员级归因和联合分布理论依据尚未建立。因此当前应保留 Original S5 作为经验正式方法，但论文或专利表述必须避免把“最大修正能量”描述成已证实的最优事件定位或把序号差值描述成可识别的 Tail 因果修正。
