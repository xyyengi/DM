# 山东 91 节点可靠通道：现有模型谱系与第一版 Body baseline 审计

日期：2026-10-08  
性质：只读架构审计；不授权训练、生成、checkpoint 写入或远程提交。

## 0. 范围、状态与证据边界

- 数据侧已完成 `[B,168,91,3]`、mask、逐节点逐资源反归一化和最小 shape dry-run。
- 本报告没有运行训练、正式生成或 Gate 0。CUDA/AMP、梯度、optimizer step、保存重载均为 **NOT RUN**。
- 审计开始时实际 checkout 为 `experiment/24site-independent-joint-tail-v2`，而任务说明写的是 `feat/shandong91-reliable-channel`。本轮未切换分支，避免在脏工作树中改变用户状态。
- 仓库中“Raw”“Body”“Full V2”“Lightweight”分别可能指 architecture、checkpoint、训练 recipe、辅助 loss 或成员混合策略。以下严格拆开。

## 1. 明确结论

> **Recommended 91-node baseline = Route B：新版本的 91-node heterogeneous Raw Body，复用 `StationConditionalResUNet1D` 的扩散、时间 ResUNet、FiLM 和固定图传播思想，保留显式 `[B,T,N,C]`，以最小必要的三资源输入/输出、类型特征和 mask 适配从头训练；首版只用 91 节点物理图，不接 External Tail。**

它不是把 `[B,T,91,3]` 永久压成 273 个无结构特征，也不是直接加载 24 节点 checkpoint。24→91、单值 station→三资源 node 的参数形状已经变化，旧 checkpoint 只能作为架构与 recipe 证据，不能作为可直接复用权重。

首版建议保持：

- 节点轴 `N=91` 与资源轴 `C=3` 显式存在；
- 每个节点内部用共享 `3→hidden` 投影融合 Wind/Solar/Load；
- 图消息只沿 91 节点物理拓扑传播；
- `has_wind/has_solar/has_load` 进入静态 node feature projection，构成显式 type encoding；
- diffusion target 仍为 `residual = actual - forecast`；
- loss 使用发布数据定义的 `effective_mask`；
- 不启用 Tail、事件采样、ramp/shape/slow 辅助损失、self-localization 或 Protected Partial。

## 2. 模型谱系

| Candidate | 实际代码入口 | 输入 / 输出 | Graph | Condition | Extreme mechanism | 参数隔离 | 当前角色 | 91 节点适配难度 |
|---|---|---|---|---|---|---|---|---|
| 基础 Station24 ResUNet | `Station24DiffusionModel` → `StationConditionalResUNet1D`，`train_station24.py` | 当前为 `[B,S,L]` 标量/站，内部 `[B,S,H,L]`；输出同形 residual/noise | none/fixed/type-gated；可选早期并行融合 | forecast、calendar、lead；按配置加 recent error/state | 无 | 无 | 多个 24 站实验共享的主体 architecture | 中高 |
| **Raw historical-spatial Body** | 同上；配置 `station24_geo_history_actual_dual_168h.yaml` | 24×168 joint residual diffusion | 地理图 + train-only historical actual graph；encoder_0 与 bottleneck | forecast、calendar、lead、recent error、state；FiLM | 无 Tail；ramp auxiliary=0 | 全模型训练形成 Body checkpoint | 当前成熟主体和后续 Tail 初始化源 | **中高；推荐基础** |
| Raw body-tail MoE checkpoint | 同 architecture；配置 `station24_geo_history_actual_body_tail_moe_168h.yaml` | Body + legacy residual tail adapter | 继承 Raw 双图 | 继承 Raw 条件，另有 causal risk/router | legacy tail/event replay | Body 从 geo-history checkpoint 冻结，仅 tail/gate 更新 | 历史“Raw”正式比较基线；注意名称同时包含 Body 与旧 Tail | 高；不应作为首版 91 方案 |
| Independent Joint Tail V2 / Full V2 | 同 `Station24DiffusionModel`；V2 配置 | 完整 24站 diffusion expert，不是小 adapter | 与 Raw 相同 | 与 Raw 相同 | event-balanced sampling + ramp/shape/slow losses；生成后 400 Raw+100 Tail | 从 Raw 初始化后**整模型可训练** | 当前 Tail 能力参照/成员池 | 高；事件定义和 mask 均不适配 Load |
| Lightweight External Joint Tail / JMRT | `JointMultiresolutionResidualTail` 挂入同主体；`station_lightweight_tail.py` | frozen Body 上 additive epsilon correction | 读取同 graph/static | 继承 Body 条件 | 多分辨率 Tail + V2 auxiliaries | Raw 参数/缓冲冻结，仅约 20k Tail 参数 | 当前参数隔离、轻量化 Tail 路线 | 很高；不属于 Body baseline |
| JSTD / H1 / MSEP 系列 | `JointSpatioTemporalDecomposedTail` 等 | slow/fast 或事件段 Tail correction | primary/secondary graph | 条件、事件假设或 causal segment prior，依版本而异 | slow/fast、segment/event prior | 多数冻结 Raw、训练 Tail/prior | 专项机制研究；证据仍有限 | 很高；本轮排除 |
| Diffusion-TS joint v1 | `StationDiffusionTS` | 24维 trajectory token；`Linear(24,width)` | 无显式图 | forecast、recent history | 无 | 无 | 替代 renderer 候选 | 高；24 维线性层固定，且历史结果弱于 Raw |
| Diffusion-TS inherited v2 | `StationDiffusionTSInherited` | 仍以 24 stations 展平进入 projection | 双图条件 + FiLM，但实现显式检查 `(24,24)` | forecast/recent/calendar/lead/node_state | 无 Tail | 无 | 继承 Station24 priors 的研究候选 | 极高；`Linear(24*8,width)` 硬编码 |
| Protected Partial | 发现审计/预检工具与历史提法，但未发现一个应作为当前正式 Body 的独立稳定 architecture | UNKNOWN / NEEDS VERIFICATION | UNKNOWN | UNKNOWN | partial isolation protocol | 有隔离意图 | 已停止/非当前主线 | 不评估为候选；任务明确禁止重开 |

### 架构、checkpoint、recipe、loss、sampling 的边界

- **Architecture**：`StationConditionalResUNet1D` + `StationGaussianDiffusion`，由 `Station24DiffusionModel` 组装。
- **Raw Body checkpoint**：特定 24 节点权重和 buffers；节点数、embedding、图、静态特征形状都属于 checkpoint 语义。
- **Training recipe**：batch、LR、EMA、early stopping、初始化来源等；不是 architecture。
- **Auxiliary loss**：V2 的 ramp/shape/slow 等；不改变“使用同一个主体网络”这一事实，但改变学习目标。
- **Sampling strategy**：event-balanced training sampler、400/100 成员混合、self-localization；均不等于 backbone。
- **Tail isolation**：冻结 Body、只更新 adapter/JMRT/JSTD，或 Full V2 从 Raw 初始化后全模型更新；二者不可混称。

## 3. Raw / Body 到底是什么

### 3.1 唯一可操作定义

本报告将 **Body** 定义为配置 `configs/station24_geo_history_actual_dual_168h.yaml` 下的 historical-spatial conditional residual diffusion：

- 类：`Station24DiffusionModel`；
- denoiser：`StationConditionalResUNet1D`；
- diffusion：`StationGaussianDiffusion`；
- 训练入口：`train_station24.py`；
- 生成入口：`generate_station24.py`；
- 输出目标：forecast-centered residual/noise diffusion。

项目文档有时把 `geo_history_actual_body_tail_moe` 的 raw-state generation 称为 “Raw”。它包含冻结 historical-spatial Body 和 legacy Tail 模块的 checkpoint 容器。若讨论第一版 91 Body，应使用上面的纯 Body 定义，避免把旧 Tail 一并迁入。

### 3.2 Body 能力核实

| 能力 | 是否存在 | 证据/说明 |
|---|---|---|
| Diffusion backbone | 是 | `StationGaussianDiffusion`，500步 linear schedule 配置 |
| Temporal modeling | 是 | 1D ResUNet、下/上采样、共享 temporal Conv1d |
| Spatial graph | 是 | `StationSpatialBlock` 与 `StationParallelGraphFusion` |
| Forecast condition | 是 | `StationConditionEncoder.forecast_stem` |
| Calendar/time condition | 是 | 8维 calendar |
| Lead-time condition | 是 | 当前固定要求2维 lead；但 91 数据没有同语义 lead |
| Static node feature | 是 | Linear projection；24数据固定5维 |
| FiLM | 是 | 每个 ResBlock 的 diffusion-time affine、condition affine，可选 state affine |
| GCN | 是，轻量固定邻接消息传递 | `einsum` 后共享 1×1 Conv；不是通用 PyG GCN 类 |
| GAT | 否 | 当前 Raw 无 attention graph |
| 双图 | 是 | geographic + train-only historical actual graph，softmax mixing |
| Learnable station ID | 是 | `nn.Embedding(station_count, stem_channels)` |

### 3.3 已确认的 24 节点假设

1. `station_dataset.py` 全局 `EXPECTED_STATIONS=24`，大量 shape、state、事件数组依赖它。
2. 静态加载固定要求 `station_features=(24,5)`、adjacency=`(24,24)`。
3. 当前动态样本是每站一个标量 `[B,24,168]`，不是每节点三资源。
4. `StationConditionEncoder` 有大小为 station_count 的 node ID embedding；改为91必须新建，旧权重不兼容。
5. 输入 forecast stem 是 `Conv1d(1,hidden,...)`；输出 head也是每站一个标量。
6. static feature 前两列被解释为 Wind/Solar mask；没有 Load relation。
7. type-gated graph只定义 wind-wind、solar-solar、wind-solar。
8. state thresholds、daylight、capacity weighting、event replay和许多评价代码绑定 13 Wind + 11 Solar 语义。
9. Diffusion-TS inherited 额外硬编码 `(24,24)` 和 `Linear(24*8,width)`，扩展性更差。
10. 24 checkpoint 中包含图 buffers、station embedding 和 shape-sensitive projection，不能 strict-load 到91版本。

## 4. `[B,168,91,3]` 进入模型的正确方式

### A. 永久 flatten 成273 features为什么不推荐

单纯 `Linear(273,...)` 能 forward，但会把以下结构变成依赖固定列位置的隐式知识：

- node identity 与三资源身份；
- 91节点物理拓扑；
- node-wise/channel-wise mask；
- resource-absent 与真实0的区别；
- source/load 异构关系；
- 图消息传递的节点轴。

如果把273视为“273个站”并复制/扩张邻接，也会制造同节点资源间和跨节点资源间边的定义问题。这不是零成本 adapter，而是一个未说明的异构图设计。

### B. 推荐显式结构及最小模块改动

数据边界保持 `[B,T,N,C]`。进入 denoiser 时转为 PyTorch temporal layout `[B,N,C,T]`，不合并 N/C 语义：

| 模块 | 必须修改 |
|---|---|
| input projection | 从每站 `1→H` 改为共享的每节点 `3→H`；无效条件值清零并显式提供 condition mask |
| noisy target projection | diffusion noisy residual 同样从3资源投影到每节点 hidden |
| output head | 每节点 `H→3`，恢复 `[B,T,91,3]` |
| node embedding | 新建91节点 embedding；不能加载24节点值 |
| node type | 将 `has_wind/has_solar/has_load` 纳入 static projection；可选另加3类 resource embedding，但首版不必叠加两套表达 |
| graph convolution | graph轴保持91节点；共享 hidden 内已融合三资源，使用91×91物理 adjacency |
| condition encoder | forecast改为3通道；calendar广播到节点；lead输入必须重定义，不能把 `window_position` 叫 forecast lead |
| FiLM | affine机制不变；其 condition hidden 来自新 encoder |
| diffusion target | 仍为 normalized residual；逐元素加噪，shape保持 `[B,N,3,T]` |
| loss/metrics | elementwise 后乘 `effective_mask`；分母为有效元素数 |
| generation | inactive/absent channel生成值不用于物理输出，导出时由 node_type mask置中性值并随 mask一同保存；不能删除物理节点 |

时间 ResBlock、UNet层数、扩散时间 embedding 和固定图传播公式本身不要求 N=24，可保持。

## 5. Graph 专项审计

### 当前24图如何进入

- primary：固定地理 adjacency；在 early parallel fusion 与 bottleneck spatial block 中传播。
- secondary：只由 train actual 拟合的 historical graph；与 primary 通过两个可学习 logits softmax混合。
- correlation graph不是在线从 validation/test拟合。
- type-gated graph是其他配置能力，不是推荐 Raw dual-fixed recipe 的主路径。

### 91图是否可直接替换

- **数学上**：`StationSpatialBlock` 的 `einsum("ij,bjct->bict")` 与 N 无关，91×91方阵可工作；projection是共享的。
- **当前实现上**：不能直接替换。loader、static feature shape、模型配置、node embedding、输入输出和很多验证均固定24；Wind/Solar类型mask也没有Load。
- 91发布数据的 physical adjacency 可作为 primary graph。
- 当前数据包没有与 Raw secondary graph完全同定义的 train-only historical actual graph。首版不得用 validation/test拟合，也不应把 primary 图重复两次假装双图。因此建议首版 `use_dual_fixed_graph=false`，先隔离“节点/资源扩展是否能学习”。
- 后续若单独建立 train-only source/load dependence graph，应版本化为新数据资产和明确消融；这已经超出纯工程接入。

### type embedding判断

需要显式 Wind/Solar/Load 类型信息，但不一定要新增独立 `nn.Embedding(3,...)`。首版最小方案是利用现有 `node_features` 中 `has_wind/has_solar/has_load` 三列进入 static feature projection。若后续需要资源专属 hidden token 或 relation-specific message passing，独立 resource embedding 属于新方法设计，应消融。

## 6. Wind / Solar / Load 语义兼容性

### 必须修改

- 单标量 station 输入/输出 → 每节点3资源输入/输出。
- 全部 loss 与评价接入 `effective_mask`，不能把中性0当物理0。
- static type 从 Wind/Solar 扩到 Wind/Solar/Load。
- 反归一化使用发布的每节点每资源 scale；不得复用24站 residual scaler或再拟合 residual scaler。
- 物理后处理移除“所有输出 clip [0,1]”假设。91数据明确允许 normalized 超界。
- solar night规则只作用于实际存在的 Solar通道，不可按“solar station”覆盖整个节点的 Wind/Load。
- lead semantics：当前91 forecast是按目标timestamp拼接，`window_position`不是 forecast lead。模型字段和报告必须改名/版本化；不能静默复用旧checkpoint语义。

### 建议修改

- 以 `window_position` 构造明确命名的 horizon-position encoding，或首个对照禁用原 lead branch；两者都要在配置中显式记录。哪一种更好目前 **NEEDS VERIFICATION**。
- condition mask作为额外输入，使模型能区分缺测0与真实0；只清零而不提供mask风险较高。
- 分资源记录loss/gradient/metrics，防止Load有效点数量主导Wind/Solar。

### 暂时不需要修改

- residual定义：Load同样满足 `actual-forecast`。
- diffusion schedule、epsilon/x0 parameterization、UNet深度和时间卷积。
- 91数据已发布的normalization/inverse-normalization。

## 7. External Tail未来迁移分析（本轮不实现）

### 可复用的机制骨架

- Body/Tail checkpoint与输出目录隔离；
- event-balanced sampling作为一种训练recipe；
- 逐元素masked auxiliary loss实现模式；
- 400/100等成员配额机制的工程框架（具体比例不应预设）；
- self-localization“Tail改动只在连续时间窗生效”的后处理思想；
- slow/fast审计、参数冻结审计和paired generation协议。

### 必须重新定义

- 事件：24站低风/低光不能代表91节点复合风险。
- 聚合权重：新能源、负荷、净负荷的单位、符号与规模不同。
- ramp/shape/depth/slow：需区分低新能源、高负荷、高净负荷、source-load anti-peak与复合重叠。
- event-balanced标签和阈值：必须只由train拟合，且按独立事件而非密集位置报告证据。
- Tail输出mask：可能需资源/节点/时间三维，而现有self-localization主要是共享时间窗。
- member allocation：当前20%仅是24站经验，不可直接迁移。
- solar daylight、Load无夜间归零、net-load符号和容量/负荷scale均需独立处理。

因此现在直接 Route C 会同时混入节点扩展、Load语义、Body学习能力和新事件定义四类变量，失败时不可诊断。

## 8. 候选评分（1差—5好）

Implementation complexity 一列按“5=修改少”计分。

| Candidate | Maturity | 24 evidence | N scale | Heterogeneous | Graph | Mask | Condition | Impl. | Debug | Tail future | 总分/50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Route B: adapted Raw Body** | 4 | 5 | 4 | 4 | 4 | 4 | 4 | 3 | 5 | 5 | **42** |
| Route A: literal Raw Body，仅改24→91 | 5 | 5 | 4 | 1 | 4 | 1 | 2 | 4 | 4 | 5 | 35 |
| Basic single-fixed-graph ResUNet Body | 4 | 3 | 4 | 3 | 5 | 4 | 3 | 4 | 5 | 4 | 39 |
| Full V2直接扩展 | 4 | 4 | 4 | 1 | 4 | 1 | 4 | 2 | 2 | 5 | 31 |
| Lightweight JMRT Tail | 4 | 4 | 3 | 1 | 3 | 1 | 4 | 1 | 3 | 5 | 29 |
| Diffusion-TS inherited v2 | 2 | 2 | 1 | 1 | 2 | 1 | 4 | 1 | 2 | 2 | 18 |
| Route C: 91 External Tail now | 3 | 4 | 3 | 1 | 3 | 1 | 3 | 1 | 1 | 5 | 25 |

Route B高于基础single-graph一分，是因为它明确保留成熟Raw的condition/FiLM/temporal设计，同时把异构与mask适配列为基线必要组成；首版graph配置仍建议single physical graph。

## 9. 最小可执行接入清单

### Keep unchanged

- 168h horizon；
- forecast-conditioned residual生成原则；
- diffusion beta schedule、步数、reverse variance；
- epsilon/x0 target的当前正式选择（实际配置锁定前再核对）；
- temporal ResUNet层数、channel multipliers、GroupNorm、dropout；
- diffusion timestep embedding；
- FiLM数学形式；
- primary fixed-graph message passing公式；
- optimizer种类、EMA和checkpoint schema思想；
- 91发布normalization及inverse-normalization。

### Must change

- 新建独立91模型类/版本和训练入口，避免改写旧checkpoint语义；
- `station_count 24→node_count 91`；
- input/noisy-target/output projection `1↔3 resources`；
- static feature dim `5→10`并核对列语义；
- node embedding `24→91`（从头初始化）；
- primary adjacency换成91物理图；
- condition与target mask贯穿forward/loss/metrics/export；
- solar-only物理规则改为channel级；
- lead字段显式版本化为 horizon position 或禁用，绝不冒充forecast lead；
- checkpoint、log、result目录保持 `shandong91/` 隔离；
- validation objective和评价拆成Wind/Solar/Load/aggregate。

### Recommended change

- static projection显式使用 `has_wind/has_solar/has_load`；
- condition-valid mask编码；
- 分资源loss日志与gradient audit；
- 首版关闭dual historical graph，后续另做train-only第二图消融；
- batch size只在显存探针后决定，不照搬8；
- 评价增加新能源、负荷、净负荷及source-load dependence。

### Do NOT change yet

- 不改diffusion schedule；
- 不换UNet/backbone；
- 不加GAT/动态图/异构relation GNN；
- 不调auxiliary loss权重；
- 不启用event sampling；
- 不启用Tail、JMRT、JSTD、self-localization或Protected Partial；
- 不加载/覆盖24节点checkpoint；
- 不把273 flatten设为正式接口；
- 不重新拟合scale、不clip normalized值；
- 不预设24站batch size、Tail比例或事件阈值。

最少预计新增/修改（正式实施时）：

1. 新增 `src/models/shandong91_conditioned_diffusion.py`（版本化三资源wrapper/派生实现）；
2. 新增 `train_shandong91.py`；
3. 新增 `generate_shandong91.py`；
4. 扩展现有 `datasets/shandong91_reliable.py` 返回模型所需layout、condition mask和horizon encoding；
5. 新增 `configs/shandong91/raw_body_heterogeneous_168h.yaml`；
6. 新增 `tests/test_shandong91_raw_body.py`；
7. 后续新增独立91评价入口；不修改24默认路径。

## 10. Gate 0–4实验路线（计划，不执行）

### Gate 0：付费GPU前工程门禁

- CPU小batch forward/backward；
- masked epsilon loss手算对照，改变invalid fill不改变loss；
- 每个新projection、type/static路径的gradient finite且非零；
- `optimizer.step` 后每个新模块参数确实变化；
- 不应训练的buffers/参数保持不变；
- save/reload逐元素一致；
- generation smoke保持shape/mask/node order；
- target-server CUDA与AMP重复上述检查；
- 显存、step time、dataloader wait和可行batch探针。

当前纯CPU Gate 0有必要，因为会新增三资源projection、mask与checkpoint版本；但任务要求未经确认不扩展，所以状态为 **NOT RUN**。

### Gate 1：极短 smoke training

- 固定少量train batch；
- 记录初始/结束总loss及Wind/Solar/Load masked loss；
- 验证确有学习而非仅shape通过；
- 不据此声称科学收益。

### Gate 2：正式 Body baseline

- 只训练Route B；
- 固定数据、图、seed、recipe和version manifest；
- validation选模，test保持锁定；
- 不并行引入Tail或新graph方法。

### Gate 3：Body generation + evaluation

- Wind、Solar、Load分别报告CRPS/coverage/width及物理范围；
- renewable、load、net-load聚合；
- 1/3/6/12/24h temporal structure；
- 91节点spatial correlation/graph-distance diagnostics；
- Wind-Solar-Load与source-load dependence；
- 多seed或独立时间块不确定度。

### Gate 4：Extreme diagnosis

- 仅用train定义阈值；
- 分低新能源、高负荷、高净负荷、compound overlap；
- 报告独立事件数、onset/duration/depth、概率质量；
- 先排除实现缺陷和总体Body学习失败，再判断是否需要Tail。

## 11. 三个最大技术风险

1. **条件语义错配**：91 forecast不是一次发布的7天预测，window position也不是forecast lead；照搬lead/revision recipe会产生错误含义。
2. **稀疏异构监督失衡**：Load有效位置多，Wind/Solar/Load scale和动态不同；即使总masked loss下降，也可能只学会Load。
3. **图与checkpoint不可直接迁移**：图层公式支持N=91不等于实现已支持；24 node embedding、static features、双图buffers和input/output heads都不兼容。

## 12. 工程适配与可能创新的边界

工程适配：shape/layout、三资源projection、91 embedding、物理adjacency替换、mask loss、channel级反归一化、隔离目录、版本化checkpoint。

可能形成方法创新、必须另做消融：异构relation graph、独立resource tokens/embedding、source-load coupling block、train-only第二依赖图、net-load compound risk condition、资源自适应loss平衡、三维Tail localization。

## 13. 何时才值得进入Tail

必须同时看到以下证据：

1. Body已通过Gate 0–3，普通分布指标、时空相关和三资源联合关系基本可信；
2. 极端不足在多seed/独立时间块或足够独立事件中重复出现；
3. 不足不是mask、归一化、condition语义、图接线或Load主导loss造成；
4. 不足能被清楚定义为低新能源、高负荷、高净负荷或compound事件，并有train-only阈值；
5. 有可审计的Body-preservation门槛，能判断Tail是否以牺牲普通质量换取表面覆盖。

单次极端事件漏报、一个验证窗口或仅coverage变宽，不足以启动Tail。

## 14. 对任务十问的直接回答

1. 选Route B，因为它保留证据最强的Raw主体，同时只加入91异构系统必需的适配。
2. 现在不选External Tail，因为Body尚未建立，事件语义也从低风/低光变成source-load compound risk。
3. 最大风险是condition语义、异构masked学习失衡、24图/checkpoint硬绑定。
4. shape、projection、mask、node count、物理图替换是工程适配。
5. heterogeneous graph、compound risk、资源专属token和Tail定义可能是新方法。
6. 最少需要新增91模型、train/generate、配置、测试、评价，并小幅扩展91 dataset接口。
7. 正式接口保留 `[B,168,91,3]`；仅在局部张量layout转换，不永久flatten 273。
8. 需要显式类型信息；首版用static `has_*`列即可，独立type embedding不是必需。
9. graph数学模块可复用；当前代码不能直接复用，必须移除24假设并重建91 buffers。
10. 只有Body可靠、复合极端缺陷可重复且排除工程问题后，才进入Tail。

## 15. 建议下一步的唯一动作

> **只实现并运行 Route B 的 CPU-only Gate 0 预检包：新增版本化91 Body类与测试，验证 forward/backward、三资源梯度、optimizer update、mask不变性和save/reload；不做 smoke training，不生成正式场景。**

在执行前应由用户明确确认两个尚未锁定的接口选择：lead branch是禁用，还是改成明确命名的 horizon-position encoding。除此之外不应同时打开第二图或Tail。

