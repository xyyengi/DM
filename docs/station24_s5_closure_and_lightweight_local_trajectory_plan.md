# Station-24 S5 正式收束与 Lightweight 局部轨迹学习计划

> 日期：2026-10-13  
> 范围：24 场站、168 h 风光联合场景生成。未访问 test，未训练、未生成新成员、未修改 checkpoint。  
> 最终决策：**停止 S5-V3 及 S5 参数搜索；原始 S5 继续作为正式局部化方法。下一项研究仅保留一个 Lightweight drop–recovery 局部轨迹监督实验。**

## 0. 五点结论

1. Full V2 Tail 在 4 个独立事件上共有 100 次 primary 成员命中；原始 S5 局部化后为 44 次。逐成员追踪显示仅 28 次原命中被保留，72 次丢失，同时局部混合新产生 16 次命中。
2. 固定 `400 Raw + 100 Tail` 会把 Tail-only 命中率按定义缩为其五分之一；四事件宏平均中，Full Tail 的 `.25` 只贡献混合总体 `.05`，这是配额稀释。局部化再把该贡献降至 `.022`，额外损失 `.028`。
3. ESPC 不应替换原始 S5；事件感知+自适应长度只保留为微小、未证实稳定的消融结果。S5 研究至此封存。
4. c16/c24/c48 均只完成 `1,520/3,040` updates，不是完整预算。初筛显示容量增大提高持续事件命中，但普通 CRPS 变差，drop–recovery 几乎不变；目前不值得为局部恢复问题补齐三点完整容量曲线。
5. 训练集有 207 个经 6 h 非极大抑制后的独立物理强下降锚点，足以支持一个最小 drop–recovery 监督实验；但验证集严格事件子集只有 2 个锚点，因此只能把恢复结果作为有限证据，不能宣称普遍有效。

## 1. 证据边界与成员身份

### 1.1 Primary 命中定义

本报告沿用既有正式评价，不修改事件定义：

- onset 绝对误差不超过 6 h；
- 真实事件区间 recall 不低于 0.50；
- depth ratio 不低于 0.75；
- 三项同时成立才算该成员命中。

事件只有 4 个独立物理事件：`wind_negative_000`、`wind_negative_001`、`wind_positive_002`、`solar_positive_003`。

### 1.2 身份追踪门禁

本轮逐元素核对通过：

- Full V2 mixture 的成员 `400:500` 与 `independent_tail_n100` 的 `0:100` 完全相同；
- Full V2 mixture 与原始 S5 的成员 `0:400` 完全相同；
- Tail 成员 `j` 在 Full V2 与原始 S5 中均对应 ensemble member `400+j`；
- 原始 S5 以 Raw member `400+j` 作为该 Tail member 的局部化背景。

因此逐成员归因是可追踪的，不是由总体比例倒推。审计文件为：

- `closure_audit/s5_member_identity_audit.json`；
- `closure_audit/s5_member_transitions.csv`；
- `closure_audit/s5_event_member_attribution.csv`；
- `closure_audit/s5_quota_vs_localization_attribution.csv`。

## 2. S5 成员级归因

### 2.1 同一 100 条 Tail 局部化前后

四事件共 `4×100=400` 个 event-member 判断：

| 数量 | Full V2 → Original S5 状态 |
|---:|---|
| 28 | 命中 → 命中（真正保留） |
| 72 | 命中 → 未命中（局部化丢失） |
| 16 | 未命中 → 命中（与 Raw 背景混合后新形成） |
| 284 | 未命中 → 未命中 |

Full V2 Tail 的总命中为 100，Original S5 Tail 的总命中为 `28+16=44`。所以：

- 不能把 `.44` 解释成“原 Tail 命中保留率”；
- 原命中的微平均保留率实际为 `28/100=.28`；
- 四事件逐事件保留率的宏平均为 `.2527`；
- 局部化既会破坏事件，也可能通过 Raw 背景与 Tail 片段重组产生新命中。

### 2.2 逐事件保留与丢失

| 事件 | Full hits | S5 hits | 保留 | 丢失 | 新增 | 原命中保留率 |
|---|---:|---:|---:|---:|---:|---:|
| wind negative 000 | 35 | 15 | 6 | 29 | 9 | 17.1% |
| wind negative 001 | 3 | 4 | 0 | 3 | 4 | 0% |
| wind positive 002 | 6 | 3 | 3 | 3 | 0 | 50.0% |
| solar positive 003 | 56 | 22 | 19 | 37 | 3 | 33.9% |

`wind_negative_001` 是最清楚的反例：S5 的 4 个命中全部不是 Full V2 原来的 3 个命中。因此“总体命中略增/保持”不等于保留了同一批有效 Tail 轨迹。

### 2.3 失效表现

以下标签可重叠，只是事后诊断，不是互斥因果分类，更没有用于窗口选择：

| 事件 | 丢失 | 窗口覆盖真实事件不足50% | onset失败 | interval失败 | depth失败 | 可能截掉事后6h恢复段 |
|---|---:|---:|---:|---:|---:|---:|
| wind negative 000 | 29 | 26 | 13 | 18 | 18 | 3 |
| wind negative 001 | 3 | 3 | 1 | 1 | 3 | 2 |
| wind positive 002 | 3 | 2 | 1 | 2 | 3 | 0 |
| solar positive 003 | 37 | 37 | 19 | 21 | 31 | 0 |

证据支持的最强解释是：

- **窗口错位/截断是主因。** 72 个丢失命中中 68 个的 S5 窗口覆盖真实事件不足 50%；这与此前“最大能量窗口和结构最强窗口平均仅重叠 1.49%”一致。
- **修正幅度衰减普遍伴随失效。** 55 个丢失命中在局部化后 depth ratio 低于 primary 阈值；它通常和窗口错位同时发生，不能独立归因。
- **恢复段丢失只得到有限支持。** 两个负向事件中有 5 个丢失成员的窗口没有覆盖事件后 6 h；这是“可能截断恢复段”的结构证据，不足以声称全部负向事件失效都由恢复截断造成。

### 2.4 配额稀释与局部化损失分离

| 事件 | Full Tail-only率 | Tail对500混合的贡献 | S5 Tail-only率 | S5对500混合的贡献 | 局部化造成的500成员率变化 |
|---|---:|---:|---:|---:|---:|
| wind negative 000 | .35 | .070 | .15 | .030 | -.040 |
| wind negative 001 | .03 | .006 | .04 | .008 | +.002 |
| wind positive 002 | .06 | .012 | .03 | .006 | -.006 |
| solar positive 003 | .56 | .112 | .22 | .044 | -.068 |
| 四事件宏平均 | .25 | .050 | .11 | .022 | -.028 |

固定 20% Tail 配额本身把 Tail-only 信号缩小 80%，即 `.25 → .05`；这是混合设计的确定性稀释，不是 S5 的错误。S5 局部化再造成 `.05 → .022`，平均额外损失 2.8 个 500-member 命中率百分点。混合总体还包含 Raw 400 的命中，不能把 Tail 贡献直接等同于总体 member-hit。

## 3. S5 正式封存

### 3.1 最终状态

| 方法 | 状态 | 处理 |
|---|---|---|
| Original S5 | **正式保留** | 当前技术实施中的局部化方法 |
| S5 event-aware fixed | 机制消融 | 不作为正式方法 |
| S5 event-aware + adaptive length | 微小、未稳定收益的消融 | 保留负/中性证据，不升级 |
| Full S5-ESPC | **负结果** | 不替换 Original S5，不声称性能改进 |
| S5-V3、权重/长度/阈值搜索 | **停止** | 不继续开发 |

### 3.2 Original S5 成功与局限

已验证成功：

- 精确复现历史结果，mask 与场景逐元素误差均为 0；
- 窗口外严格恢复 Raw；
- 相比 Full V2 全时域混合，将 Renewable CRPS 从 `152.333` 降至 `151.215 MW`，Energy Score 从 `6.59835` 降至 `6.58817`，90% width 从 `816.99` 降至 `794.43 MW`；
- 仍保持 4/4 primary any-hit，但 Tail member-hit 从 Full V2 `.250` 降至 `.110`。

已验证局限：

- 最大差值能量不是事件结构的可靠代理；
- 对 Full Tail 原命中的微平均保留率只有 28%；
- 稀有事件 any-hit 对成员配对有一定敏感性；
- 局部化不能创造 Full V2 本身缺失的 wind 高频和恢复结构。

### 3.3 ESPC 为何失败

完整 ESPC 将边界诱发步变均值/最大值从 `133.50/924.66 MW` 降到 `95.83/498.65 MW`，但 Tail primary member-hit 降到 `.085`，并丢失 `wind_positive_002` 的 any-hit。它证明更严格的边界控制可以让轨迹更平滑，却没有证明这种平滑能保留更多有效极端事件；相反，当前约束会过滤有价值的正向/恢复变化。

因此论文中应把 ESPC 写为**负结果/消融边界**，不能写成已验证改进方法。

## 4. Lightweight 容量实验状态

### 4.1 实际 checkpoint/update 状态

`stage1_status.csv` 的 `complete` 只表示“1/3预算初筛、30 Tail生成及轻量评价完成”，不是完整 3,040 updates。

| 配置 | Tail参数 | epoch | 实际updates | 完整目标 | 状态 | 还需updates |
|---|---:|---:|---:|---:|---|---:|
| c16_d2_h3 | 10,020 | 80 | 1,520 | 3,040 | pause checkpoint | 1,520 |
| c24_d2_h3 | 20,588 | 80 | 1,520 | 3,040 | pause checkpoint | 1,520 |
| c48_d2_h3 | 74,564 | 80 | 1,520 | 3,040 | pause checkpoint | 1,520 |

三者的 best/last validation 仍在第 80 epoch 附近，短预算不能替代完整容量曲线。

### 4.2 初筛结果能说明什么

| 配置 | Wind CRPS | Renewable CRPS MW | Energy | Tail primary hit | event positive 1h q99/truth | 6h half-recovery |
|---|---:|---:|---:|---:|---:|---:|
| c16 | .091256 | 153.462 | 6.63574 | .2167 | .3969 | .2269 |
| c24 | .091328 | 153.573 | 6.64017 | .2333 | .3961 | .2260 |
| c48 | .091375 | 154.064 | 6.64463 | .2750 | .3987 | .2282 |

初筛成员数为 `120 Raw + 30 Tail`，只能作探索性比较，不能与正式 `400+100` 直接等价。现有趋势是：

- 加宽提高了持续事件 member-hit 和 depth ratio；
- 普通 CRPS/Energy 随宽度增加而变差；
- event positive 1 h q99/truth 几乎停在 `.40`；
- 6 h half-recovery 三者都约 `.226–.228`，没有容量收益。

所以容量可能影响“持续事件强度”，但没有证据表明它是“下降后恢复不足”的主要瓶颈。

### 4.3 补齐成本与决定

服务器实测每 epoch 中位训练时间为：c16 `10.99 s`、c24 `12.17 s`、c48 `14.83 s`。各续训 80 epoch 的纯训练时间约为 `14.6/16.2/19.8 min`，合计约 51 min；加上三组正式 100 Tail 生成、评价与打包，按已有吞吐保守估计总计约 1.5–2 h GPU/CPU流水线时间。

**当前决定：不补齐。** 原因不是成本不可接受，而是三点初筛已经显示恢复缺口对容量不敏感。若未来论文必须给出“完整预算容量—性能曲线”，再从 pause checkpoint 续训；本轮转向局部轨迹监督，不自动扩大结构搜索。

## 5. Drop–recovery 证据审计

### 5.1 训练支持量与时间分布

定义沿用既有诊断：聚合 wind 有效 1 h 负向 ramp 的 train q90，阈值为 `173.230 MW`，同一 6 h 内做非极大抑制。

| 统计口径 | 数量 | 3h half-recovery | 6h half-recovery | 命中后恢复时间中位数 |
|---|---:|---:|---:|---:|
| 重叠168h窗口中的训练暴露 | 1,365 | .145 | .377 | 4 h |
| 去重后的独立物理锚点 | **207** | **.145** | **.372** | **4 h** |

原始候选为 2,353 个窗口位置；重叠窗口会把同一物理过程重复约 6–7 次，因此证据强度应按 207 个独立锚点理解。独立锚点覆盖 1–10 月；数量最高的月份为 3月38、4月34、5月32。小时分布并不均匀，09时43、10时26最多，后续训练报告必须按物理时间块复核，不能把重叠窗口当独立样本。

### 5.2 当前生成缺口

- validation 全体强下降锚点：101 个；actual 3h/6h half-recovery 为 `.240/.410`，Original Lightweight 为 `.088/.218`；
- actual 的 half-recovery 时间中位数为 3 h，Lightweight 为 4 h；
- event±6 h 的 wind positive 1 h q99/truth 为 `.398`，non-event 为 `.562`；缺口明显集中在事件附近；
- 同一 event 区域的 negative 1 h q99/truth 为 `1.236`，说明模型不是“不会急跌”，而是“急跌后正向恢复幅度和时序不足”；
- ramp-selection 将 event positive 1 h q99/truth 仅从 `.398` 提到 `.407`，没有解决条件恢复；
- positive/negative 1 h 配置后梯度范数比为 `.834`，positive 梯度并未消失，其 standalone loss 还更大。

因此问题同时包含 **amplitude deficit + timing delay**，主要是“给定已经发生强下降，后续局部轨迹如何恢复”的条件关系不足，而不是单纯 positive 1 h 权重太小。

### 5.3 现有损失为何约束不到

- `ramp`：按 wind/solar、正/负、1/3/6 h 选择高幅变化对，但每个 ramp pair 独立；它不知道某个正向 ramp 是否发生在强下降之后，也不约束恢复顺序。
- `shape/depth`：约束持续 residual 事件的整体形态和深度，能把事件做深/做久，却不明确要求“先下降、后恢复”。
- `slow`：强调 12/24 h 低频/恢复尺度，容易约束长期均值，却不足以锁定 1–6 h 的返回幅度与首次恢复时刻。
- 全位置 epsilon：主体监督占绝大多数，局部强下降后的六小时条件轨迹在总体目标中权重很低。

## 6. 唯一建议的最小局部轨迹监督

### 6.1 训练目标

不增加网络，只在现有 c24 Lightweight Tail 上增加一个 **Drop–Recovery Trajectory Loss**。对训练 actual 中的强下降锚点 `t`：

\[
D_t^*=P_{t-1}^*-P_t^*\ge Q_{0.90}^{train}(-\Delta_1P^*),
\]

其中 `P` 是容量还原后的聚合 wind MW。对 `h=1,...,6` 定义相对谷底的有符号恢复轨迹：

\[
r_{t,h}^*=\frac{P_{t+h}^*-P_t^*}{D_t^*},\qquad
\hat r_{t,h}=\frac{\hat P_{t+h}-\hat P_t}{D_t^*}.
\]

采用一个统一的 Smooth-L1：

\[
L_{DR}=\frac{\sum_{t,h}m_{t,h}\,\operatorname{SmoothL1}(\hat r_{t,h}-r_{t,h}^*)}
{\sum_{t,h}m_{t,h}}.
\]

这里不把恢复强制截成正数：真实轨迹若继续下降，目标也保留负值，从而避免把所有强下降都错误地规定为立即反弹。

### 6.2 样本、mask与参数来源

- 只用 train actual 构造监督；不作为生成条件；validation actual 只用于最终离线评价；test 锁定。
- anchor 阈值固定为 train aggregate-wind negative 1 h q90=`173.230 MW`。
- 同一窗口内按下降幅度排序做 6 h NMS，防止一个过程被多个相邻 anchor 重复计权。
- `m_{t,h}` 要求 anchor 两端和未来 h 时刻均有效，且 `t+h<168`；只作用于 wind 聚合轨迹。
- `h=1...6` 等权 pooled mean，不再引入一套 1/3/6 h 权重。
- 网络、Raw冻结、采样、原 epsilon/ramp/shape/slow、优化器、预算、400+100与评价协议全部不变。
- 只增加一个系数 `lambda_DR=0.10`，来源是沿用现有 slow 系数，而不是验证集搜索。正式训练前必须在固定 batch/timestep/noise 上审计其配置后梯度；若其范数超过现有全部 auxiliary 总范数或出现明显 high-t 主导，门禁 FAIL，停止而不是临时改系数。

### 6.3 最小实验

| 项目 | Control | DR candidate |
|---|---|---|
| Tail | c24_d2_h3，20,588参数 | 完全相同 |
| Raw | 全冻结、eval | 完全相同 |
| 原损失 | epsilon + ramp .18 + shape .14 + slow .10 | 完全相同 |
| 新增 | 无 | `0.10 × L_DR` |
| train seed / generation seed | 与正式 control 配对 | 相同 |
| 正式成员 | 400 Raw + 100 Tail | 相同 |

训练前门禁：anchor计数、未来actual未进入条件、DR loss非零、Tail梯度非零、Raw参数/buffer hash不变、CUDA/AMP、保存重载、各 timestep 梯度比例。门禁全 PASS 后只跑一个候选，不展开多损失或新网络。

### 6.4 预期改善与失败判据

优先指标：

1. drop-conditioned 3h 与 6h half-recovery 同方向提高；
2. 6h命中成员的恢复时间中位数由 4 h 向 actual 的 3 h 靠近；
3. event positive 1 h q99/truth 高于当前约 `.40`，同时 negative 1 h 不被削弱；
4. 改善不能只来自区间变宽。

主体门槛沿用已冻结工程 guardrail：Renewable/Wind/Solar daylight CRPS 相对 Raw 退化不超过2%，Energy不超过1%，spatial与wind-solar correlation RMSE不超过5%；coverage、width和Interval Score联合检查。

下列任一成立即判失败并停止：

- 3h/6h recovery 无一致改善，或只改善一个偶然事件；
- positive q99提高但 negative drop、持续事件或主体指标明显退化；
- coverage改善仅由width扩大造成，Interval Score不改善；
- DR梯度压过原有目标，或重叠窗口/少数月份主导学习；
- 严格事件证据仍只有2个锚点，却据此宣称普遍有效。

## 7. 专利材料同步边界

### 7.1 当前可写入“已实施技术”的内容

以 **Raw + Full Independent Joint Tail V2 + Original S5** 为技术实施基础，可以开始整理：

1. 冻结主体扩散模型生成长期主体场景；
2. 独立 Tail 在不改动 Raw 参数的情况下生成极端适应成员；
3. 固定 `400 Raw + 100 Tail` 形成主体—尾部组合；
4. 对 Raw 背景与 Tail 成员逐条配对，计算风光容量加权联合修正能量；
5. 依据训练集事件持续时间统计确定局部窗口长度；
6. 在最大能量连续窗口内，以半余弦掩码保留 Tail 修正；
7. 窗口外严格恢复 Raw，保持 168 h 主体概率结构；
8. 输出完整 24 场站风光联合成员并统一评价。

可如实写入的实验效果：Original S5 在本次 validation 上把 Full V2 的区间宽度和普通概率指标拉回接近/略优于 Raw，同时保持 4/4 primary any-hit；但 Tail member-hit 从 `.25` 降到 `.11`，成员级原命中保留率只有 `.28`。

### 7.2 只能写作“待验证方案”的内容

- ESPC 不能写成已验证改进；完整版本是负结果。
- event-aware + adaptive length 只能列为未达到稳定收益的消融。
- drop–recovery loss 尚未训练，不能写成既有技术效果或实施例收益。
- 尚未完成专利查新，不能据此断言新颖性或创造性。

Lightweight Tail 已验证事实可以单列为备选实施：Raw 全冻结、外置 Joint Multiresolution Residual Tail、20,588 可训练参数、epsilon加性修正、400+100正式组合；其局限是 event positive 1 h 与下降后恢复不足。拟议 `L_DR` 应放在“后续可选实施/研究计划”，不放在已证实效果部分。

## 8. 最终研究路线

### 已完成技术事实

- Full V2、Original S5、Lightweight Tail 均有正式运行证据；Original S5 精确复现。
- S5成员身份可追踪，配额稀释和局部化损失已分离。
- ESPC 三个消融已完成零训练公平比较。
- c16/c24/c48 完成1,520-update初筛，而非完整预算。

### 已证实收益

- Full V2显著增加持续事件Tail命中；Original S5恢复主体概率质量并保留4/4 any-hit。
- Lightweight以20,588参数实现低成本独立修正，但恢复不足尚未解决。

### 已发现局限

- S5的最大能量窗口会大量错过原Tail事件结构；ESPC未解决且完整版本更差。
- 固定20% Tail配额显著稀释Tail-only事件率。
- Lightweight能产生强下降，但事件附近正向恢复幅度不足且约滞后1 h。
- 验证集独立持续事件和严格恢复锚点很少，结论必须克制。

### 下一项最值得开展的实验

只开展一个 `c24_d2_h3 + L_DR` 配对实验。它直接针对条件“下降→未来6 h轨迹”，比继续加宽、继续调统一ramp权重或开发S5-V3更可归因。

### 停止声明

本报告完成后：不开发S5-V3，不重新训练Full V2，不搜索S5参数，不自动补齐容量曲线，不自动启动drop–recovery训练。正式训练需另行确认并先通过完整预检。

