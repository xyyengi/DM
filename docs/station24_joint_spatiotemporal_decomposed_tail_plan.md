# 24场站风光联合扩散模型：时空分解尾场结构升级规划

## 1. 决策结论

本轮不再继续调整 tail 比例、固定掩码半径、EMA 或独立时间定位头。Raw body-tail 保留为正式 baseline；新结构推荐采用：

> **JSTD-Tail：联合时空分解尾场（Joint Spatio-Temporal Decomposed Tail Field）**

它不是在 Raw body-tail 后面继续拼接第三个专家，而是直接替换现有 tail 的生成方式：

1. 在每个扩散去噪步内，由共享特征联合生成场站×时间的软激活场与 tail 修正；
2. 把168 h残差的慢结构与快速变化分别建模，再重构为同一个24场站联合场景；
3. 以一个共享系统极端表示、带符号的场站载荷和局部图残差统一表达风电、光伏、同向和反向异常；
4. 不再使用“先预测一个168 h位置，再用固定半径硬切一段”的两阶段逻辑；
5. 生成时只使用发布时刻可得信息与当前扩散状态，不使用未来真实功率。

建议从代码上一次实现完整方案 F，并通过配置开关保留 B/C/D/E 消融；训练顺序上先跑 F 获取上限，再按 A–E 拆解贡献。这样是一次真正的 tail 机制重构，同时避免为了每个消融反复改代码。正式实现前必须先完成连续事件标签、命中评价解耦和slow/fast互补约束，具体审计见 `docs/station24_event_definition_and_hit_sensitivity_audit.md` 与 `docs/station24_jstd_event_condition_pretraining_audit.md`。

---

## 2. 现有实现与诊断证据

### 2.1 Raw body-tail 当前实际做了什么

当前正式基线为：

- 24个场站，13个风电、11个光伏；预测长度168 h；
- 目标为 `actual - current forecast` 残差；
- 主干为三层一维 ResUNet，时间尺度为168/84/42；
- 条件包括当前发布预测、日历、提前时距、场站静态信息和最近24 h已知误差；
- State V1由低出力、高出力、上爬坡和下爬坡四类预测状态构成；
- 空间部分使用地理图与仅由训练集真实功率拟合的历史相关图；
- 总参数约779,248；Raw body-tail 训练时冻结历史—空间主体，只训练风电 tail adapter 与发布级风险门控。

现有 tail 的核心形式近似为：

$$
\hat\epsilon
=
\hat\epsilon_{\mathrm{body}}
+g_m\,\Delta\epsilon_{\mathrm{tail,wind}},
$$

其中 $g_m$ 是成员级二值路由。没有 `tail_time_mask` 时，只要成员进入 tail，修正就作用于完整168 h；而且最终乘了 wind station mask，因此光伏没有 tail。

### 2.2 最近两轮诊断说明了什么

#### Tail比例诊断

把实际 tail 成员比例从10.23%提高到29.74%后：

- 严格命中成员总数由120增至162，但单个 tail 成员命中率由31.20%降至29.44%；
- 聚合风电 CRPS 变差3.08%，Energy Score变差2.15%；
- 聚合风电90%区间宽度增加6.75%；
- 三个独立持续深跌仍均可被至少一个成员命中，但 duration/depth 分布没有变得更合理。

因此结论属于情况 C：tail 有能力生成深跌，但不是 tail 数量不足，而是事件形态和位置分布不稳定。

#### 低频—高频诊断

使用12 h与24 h零相位离线分解后，三个独立事件中：

- low-frequency问题率为58.3%；
- high-frequency问题率为100%；
- 24 h尺度下，body的 onset MAE约6 h、duration MAE约8 h；
- 24 h尺度下，tail的 onset MAE约30 h、duration MAE约44.5 h；
- body与tail的1/3/6 h ramp相对中位误差均约39%–48%。

典型事件中，真实持续深跌约16–24 h，而 tail 会形成45–113 h的提前低偏。说明当前 tail 同时承担“持续低偏”和“局部突变”，结果是慢结构被拉得过长，快速变化又被卷积与MSE平均掉。

### 2.3 为什么不能重复以前的位置头

旧的独立时间定位头只增加833个参数，但其时间概率熵为5.020，对应约151.5个有效小时；完全均匀的168 h分布熵为5.124。它没有真正学会“第几小时”。

失败原因不是参数太少，而是结构割裂：位置头先根据确定性条件猜一个时刻，随后用固定半径掩码裁切已经学好的 tail 形态。位置错误时，tail 再强也只能被搬到错误位置；而位置头又看不到每个随机成员在当前去噪过程中逐渐形成的事件。

新方案必须让位置、持续时间、空间范围和修正形态共享同一组生成特征，并由最终场景重构误差共同反向传播。

---

## 3. Diffusion-TS 中真正值得迁移的思想

Diffusion-TS 并不是先用滑动平均生成两份固定数据，再训练两个互不相干的模型。其关键做法是：

1. 每个扩散步直接重建干净样本 $\hat x_0(x_t,t)$，而非只预测噪声；
2. 在深层解码器内部逐层提取不同语义分量；
3. 用低阶多项式基表示缓慢趋势；
4. 用Top-K Fourier基表示周期成分；
5. 最后一个解码块保留剩余周期与非周期误差；
6. 时域重构损失之外再加入频域重构损失。

原文形式为：

$$
\hat x_0=V_{\mathrm{tr}}+\sum_i S_i+R,
$$

以及：

$$
\mathcal L
=w_t\left[
\lambda_1\lVert x_0-\hat x_0\rVert_2^2
+\lambda_2\lVert \mathrm{FFT}(x_0)-\mathrm{FFT}(\hat x_0)\rVert_2^2
\right].
$$

本项目应迁移的是“在去噪器内部联合分解并直接约束最终重构”的思想，而不是机械复制其全部结构：

- 当前目标是预测残差，日周期大部分已被发布预测吸收，残差中的深跌也不是规则周期；
- 低阶多项式很难表达具有明确 onset、平台期与恢复段的持续深跌；
- Top-K Fourier基可能把突发事件扩散成整周振荡，并产生边界/Gibbs效应；
- 训练样本只有290个发布窗口，直接更换为完整 Transformer 主干的过拟合风险很高。

因此推荐保留 ResUNet、状态与图结构，把 Diffusion-TS 的直接 $x_0$ 重构、深层分量分解和频域约束迁入新的 tail/输出解码器。

---

## 4. 候选方案一：JSTD-Tail（推荐）

### 4.1 核心结构

#### 第一步：把现有 epsilon 输出转换为可解释的残差 $x_0$

现有主体仍先给出 $\hat\epsilon_{\mathrm{body}}$，再在每个扩散步内转换为：

$$
\hat r_{0,\mathrm{body}}
=
\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon_{\mathrm{body}}}
{\sqrt{\bar\alpha_t}}.
$$

这样可以加载 Raw body 权重，同时让后续 slow/fast 分量直接对应物理可解释的残差形态。

#### 第二步：只对新增tail修正做slow/fast分解

Raw body及其原有条件调制保持逐元素不变。新增tail利用现有 ResUNet 的42 h等效粗尺度特征 $H_c$ 与168 h细尺度特征 $H_f$，但只分解新增修正：

$$
\Delta \hat r_{\mathrm{tail}}
=\Delta \hat r_{\mathrm{slow}}
+\Delta \hat r_{\mathrm{fast}}.
$$

- slow 分支从粗尺度特征生成持续低偏、duration、depth和慢恢复；
- fast 分支从完整小时分辨率特征生成1/3/6 h ramp、突然下跌、快速恢复和局部波动；
- 新增可学习修正采用零初始化；关闭JSTD或令其路由为0时，模型逐元素退化为Raw body输出；
- slow可学习修正只能从粗尺度特征产生，并经过低频投影；fast修正必须经过互补高频投影，不能让两个全分辨率自由输出直接相加；
- 训练中不强迫唯一的12 h或24 h切分，而是同时约束12 h和24 h投影，并惩罚 fast 分支中的低频泄漏；
- 每个checkpoint必须审计slow-fast频带能量、余弦相关和抵消率，防止两个大项一正一负后得到虚假的良好重构。

#### 第三步：时空局部化tail与修正形态联合生成

新 tail 不输出一个168 h位置分类，而是在同一个解码器中同时产生：

$$
M_{\mathrm{slow}},M_{\mathrm{fast}}
\in[0,1]^{24\times168},
$$

以及：

$$
\Delta r_{\mathrm{slow}},\Delta r_{\mathrm{fast}}
\in\mathbb R^{24\times168}.
$$

最终输出为：

$$
\hat r_0
=
\hat r_{0,\mathrm{body}}
+g_m\left[
P_{12}\!\left(M_{\mathrm{slow}}\odot U_{\mathrm{slow}}\right)
+(I-P_{12})\!\left(M_{\mathrm{fast}}\odot U_{\mathrm{fast}}\right)
\right].
$$

$g_m$ 只表示“这个成员是否允许承载极端偏移”，不再决定改哪个小时。$M$ 与未投影修正 $U$ 由相同的多尺度隐藏特征、条件和当前随机扩散状态共同产生。必须先做 `mask × correction`，再投影到互补频带；若先投影再乘mask，mask边界会重新引入频带泄漏。由于不同成员的 $x_t$ 不同，最终 $M$ 也可以不同；模型可以在500个成员中把概率质量分配到多个可能时段。

生成结束前再把 $\hat r_0$ 转回 $\hat\epsilon$，沿用现有反向扩散代码。

#### 第四步：一个联合tail场表达风电、光伏和系统异常

不设置“风专家、光专家、系统专家”三个独立路由。使用一个共享极端时序表示与带符号的场站载荷：

$$
\Delta r_k(s,t)
=
\Delta r_{k,\mathrm{local}}(s,t)
+\sum_{j=1}^{K}a_{s,j}\,z_{k,j}(t),
\qquad k\in\{\mathrm{slow},\mathrm{fast}\}.
$$

- $z_{k,j}(t)$ 是系统级共享事件模式；
- $a_{s,j}$ 由场站类型、容量、经纬度嵌入和图特征产生，允许正负载荷；
- 同号载荷表达风光共同异常，异号载荷表达风光反向异常；
- local 项保留单站或局部区域异常；
- 固定从 $K=4$ 开始，避免再次引入大量离散专家。

### 4.2 分别解决什么问题

- **问题①定位**：$M(s,t)$ 是成员相关的连续时空场，直接参与最终场景生成；其起点、持续期和恢复段由重构共同学习。
- **问题②风光联合**：signed low-rank system modes + local graph residual 可以表达风、光、同向、反向和多站同步事件，同时始终输出一个 `[24,168]` 场景。
- **问题③slow/fast混合**：持续偏差和小时级变化使用不同时间感受野、不同输出约束，但在同一去噪步融合。

### 4.3 与Raw body-tail的本质差别

Raw 是“发布级二值路由 + 全周风电修正”；JSTD-Tail 是“发布级极端资格 + 成员级随机时空场 + 联合风光修正”。第一版Raw body及其所有条件调制完全冻结，不执行解冻阶段。只有在JSTD tail本身取得正收益、且消融证明剩余误差确实来自body后，才另立实验讨论解冻；不能把该变量混入首轮结论。

### 4.4 如何利用Diffusion-TS而非机械复现

- 使用其 $x_0$ 直接重构思想，但只把现有 epsilon 主体转换到 $x_0$ 域，不立刻抛弃已训练主干；
- 使用深层分解和分量重构思想，但以数据诊断支持的 slow/fast 代替固定“多项式趋势+周期”；
- 使用频域约束，但只作为多尺度重构的一部分，不让Top-K Fourier基主导突发事件；
- Transformer不是此方案成立的必要条件，先利用当前 ResUNet 已有的168/84/42多尺度表示。

### 4.5 训练监督与因果边界

训练时可使用 train actual 构造监督，但绝不作为生成条件：

| 监督 | 来源 | 生成时是否需要 |
|---|---|---:|
| 标准epsilon锚定 | train residual与随机噪声 | 否 |
| 12/24 h slow投影重构 | 仅train residual目标 | 否 |
| fast互补重构与1/3/6 h ramp | 仅train residual目标 | 否 |
| 软时空事件支持 | train residual的分位数、ramp与跨站同步 | 否 |
| 风光联合/反向关系 | train residual的风光聚合与场站同步关系 | 否 |
| 生成条件 | forecast、calendar、lead、station metadata、recent error、state、图 | 是，均已可得 |

中心滤波即使在训练标签中使用未来小时，也不构成部署泄露：模型本来就非自回归地生成完整168 h，滤波结果只进入目标/损失，不进入验证、测试或实际生成条件。

建议把损失归并为四组，避免堆叠十几个互相竞争的小项：

$$
\mathcal L
=\mathcal L_{\epsilon,\mathrm{anchor}}
+\lambda_D\mathcal L_{\mathrm{decomp}\text{-}x_0}
+\lambda_M\mathcal L_{\mathrm{joint\ mask\text{-}tail}}
+\lambda_J\mathcal L_{\mathrm{joint\ structure}}.
$$

- $\mathcal L_{\epsilon,\mathrm{anchor}}$：保护普通场景与原扩散训练锚点；
- $\mathcal L_{\mathrm{decomp}\text{-}x_0}$：slow的12/24 h投影、fast的互补与1/3/6 h增量、分频FFT约束；
- $\mathcal L_{\mathrm{joint\ mask\text{-}tail}}$：最终局部tail重构、软支持重叠、mask稀疏和边界平滑；mask不能脱离tail单独优化；
- $\mathcal L_{\mathrm{joint\ structure}}$：多站同步、风光同向/反向关系和聚合—单站一致性。

持续时间不再固定为6 h，也不把1/3/6/12/24 h定义成五类事件。持续事件只有一个可变区间标签，直接保存连续的 onset、stop、actual duration和depth；1/3/6 h只是fast的多尺度差分监督，12/24 h只是slow的低频投影与结构约束。持续事件同时报告宽松（±12 h/25%/50%）、主要（±6 h/50%/75%）和严格（±3 h/75%/100%）三档。

### 4.6 参数量与训练成本

- 预计总参数约1.05M–1.25M，为Raw的1.35–1.60倍；
- 单步训练计算约Raw的1.4–1.8倍；
- 生成仍是一个扩散过程，不是两个模型串行采样，预计500成员生成约增加20%–50%；
- 32 GB显存足够，成员块大小需重新自动探测，不能沿用旧模型的峰值结论。

### 4.7 最大风险

最大风险是 mask collapse：全部接近0会退化为body，全部接近1会退化为当前全周tail。必须在训练中审计每类事件的mask面积、熵、连通段数、事件内/外修正能量比，以及不同成员的mask多样性。若这些量塌缩，即使CRPS暂时改善也不能判为成功。

---

## 5. 候选方案二：可微事件原子尾场

### 5.1 核心结构

为每个tail成员生成少量事件原子，每个原子同时决定中心、宽度、深度、恢复速度和场站载荷：

$$
\Delta r_{\mathrm{tail}}(s,t)
=\sum_{q=1}^{Q}a_{s,q}d_q
\,\kappa_q\!\left(\frac{t-\mu_q}{w_q},\rho_q\right)
+\Delta r_{\mathrm{fast}}(s,t).
$$

$\kappa_q$ 不是硬mask，而是可学习的非对称软事件核；$\mu_q,w_q,d_q,\rho_q$ 直接参与合成，所以位置与形态通过最终重构联合训练。建议 $Q=3$，允许一个168 h窗口有多个事件。

### 5.2 解决范围

- 对①最直接：onset、duration、depth、恢复都成为生成参数；
- 对②可用带符号的场站载荷表达风光同向/反向与同步事件；
- 对③由事件核处理slow结构，额外fast残差处理1/3/6 h变化。

### 5.3 与Raw的差别

Raw的tail是自由卷积曲线；该方案把tail约束为“少数事件原子+局部残差”。它比独立位置头更强，因为位置不是分类结果，而是最终波形的坐标。

### 5.4 与Diffusion-TS的关系

同样采用“有语义分量直接合成 $x_0$”的思想，但将趋势/周期基替换为更符合极端功率事件的非对称事件基。

### 5.5 监督和因果性

事件参数只用train residual提取的软标签或最终重构监督；生成时参数由当前扩散状态与可得条件产生。不存在未来真实输入。

### 5.6 参数与成本

参数增量约15%–30%，采样成本约1.2–1.4倍；比完整JSTD-Tail更便宜。

### 5.7 最大风险

真实异常不一定能由少数参数化事件核表示；多个事件原子可能发生slot collapse，或者为降低损失而都生成宽而浅的事件。当前只有少量独立聚合深跌，显式事件参数可能比连续时空场更容易过拟合。

---

## 6. 候选方案三：完整Diffusion-TS式主干替换

### 6.1 核心结构

将当前ResUNet去噪器替换为Transformer encoder + deep decomposition decoder，直接预测：

$$
\hat r_0
=V_{\mathrm{trend}}
+S_{\mathrm{Fourier}}
+R_{\mathrm{error}},
$$

再在error分量中加入联合时空tail场和24场站图条件。

### 6.2 解决范围

- Transformer全局注意力有机会改善168 h长依赖和错位；
- 分解解码器处理慢趋势与周期；
- 联合tail场处理局部极端和多场站关系。

### 6.3 与Raw的差别

这是主干级替换，Raw只能作为外部baseline，无法自然保持bit-for-bit初始化。

### 6.4 与Diffusion-TS的关系

最接近原论文，包括直接 $x_0$、Transformer、趋势/Fourier/error三分量和FFT损失。

### 6.5 监督和因果性

训练仍只使用train residual；生成条件可保持因果。但原文的通用合成/插补设置与本项目的强发布预测条件、稀有极端和图结构不同，需要重新设计条件注入与图融合。

### 6.6 参数与成本

预计2M–5M参数；训练和生成约为Raw的2–4倍，具体取决于token定义、层数与注意力范围。

### 6.7 最大风险

290个训练窗口对完整Transformer主干过少；周期基可能把突发异常平滑成振荡；一旦整体指标变化，难以判断来自分解、注意力、参数量还是tail。它适合作为后续独立论文路线或外部baseline，不是当前第一实现。

---

## 7. 三个方案比较与推荐

| 方案 | ①时空定位 | ②风光联合 | ③slow/fast | 与Raw兼容 | 数据需求 | 风险 |
|---|---:|---:|---:|---:|---:|---:|
| JSTD-Tail | 强 | 强 | 强 | 高，可恒等初始化 | 中 | mask/分量塌缩 |
| 事件原子尾场 | 很强、可解释 | 中—强 | 中—强 | 中 | 中—高 | 事件形态受限、slot塌缩 |
| 完整Diffusion-TS主干 | 中—强 | 需额外设计 | 强 | 低 | 高 | 小数据过拟合、归因困难 |

**首选 JSTD-Tail。** 原因不是它改动最小，而是它同时命中三项已被数据证实的失败机制，同时仍能把Raw权重作为严格初始化点。相比事件原子，它不预设所有极端都只有固定形状；相比整套Transformer替换，它不会把“主干、分解、注意力、tail、条件注入”五个变量一次混在一起。

第一版应直接实现完整 F，而不是先只跑一个小定位头。B/C/D/E只作为同一实现的关闭开关，待F给出上限后做消融。

---

## 8. 完整实验路线

### 8.1 先锁定协议

1. Raw body-tail使用已归档的 raw `model_state_dict`、500成员、seed 424242作为正式A；
2. 数据划分不变，test继续封存；
3. 所有候选共享相同物理投影、验证23窗口、扩散步数和评价脚本；
4. 新增 A1：Raw架构在相同训练预算下继续训练，排除“只是多训练了若干epoch”的影响；
5. 每个结构至少3个训练seed；最终500成员评价使用同一组生成seed，并按独立物理事件bootstrap，而不是把重叠发布窗口当独立样本。

### 8.2 消融矩阵

| 编号 | 结构 | 目的 |
|---|---|---|
| A | Raw body-tail | 正式基线 |
| A1 | Raw + 相同训练预算 | 排除额外训练收益 |
| B | + 联合生成的场站×时间soft tail field | 单独验证时空局部化 |
| C | + body/tail slow-fast decomposition | 单独验证多尺度分解 |
| D | + 风光联合signed low-rank tail field | 单独验证联合极端结构 |
| E | + 时空定位 + slow-fast decomposition，仍只评风电tail | 验证①与③协同 |
| F | JSTD-Tail完整方案 | 时空定位 + 分解 + 风光联合 |

B/C/D不能各自另写一套模型。完整模块必须通过配置开关关闭，并审计关闭后参数与前向公式确实对应目标消融。

### 8.3 实现顺序

#### 阶段0：标签与评价冻结

- 从train residual建立可变duration连续事件目录、1/3/6 h fast监督、slow连续支持、风电/光伏日间支持、多站同步以及风光同向/反向软标签；
- 事件身份不包含1/3/6/12/24 h类别；持续时间直接取连续区间。1/3/6 h仅是fast观察尺度，12/24 h仅是slow投影尺度；
- 保存阈值、事件数、重叠规则和train-only审计；
- 扩展现有持续深跌与频率分解脚本，不改模型。

#### 阶段1：恒等初始化与单元测试

必须先证明：

- 新模块权重为零、$M=0$或route=0时，输出与Raw逐元素一致；
- fixed low-pass + complementary high-pass严格重构输入；
- 太阳夜间tail输出为零；
- 关闭三模块后A/B/C/D/E/F配置身份正确；
- val/test actual不能进入条件张量；
- 不同随机成员可产生不同 $M(s,t)$，同一成员的slow/fast和mask梯度均非零。

#### 阶段2：先训练完整F

为了既获得大胆结构的上限，又防止一开始毁掉主体，采用两阶段优化而不是永久冻结：

1. **新头成形期**：短暂冻结Raw主干，只训练slow/fast refiner、joint tail field和mask，使零初始化分支开始有意义；
2. **联合适配期**：解冻bottleneck、最后两级decoder、条件/状态融合门和相关图融合门；主干学习率约为新分支的0.1倍；
3. checkpoint选择不能只用epsilon MSE，使用“普通质量约束下的事件综合分数”；
4. 若mask塌缩或事件外低偏迅速增加，直接停止，不通过继续增大tail比例掩盖。

#### 阶段3：做消融

F有明确正收益后，再用同一初始化和优化预算训练B/C/D/E；如果F失败，先根据mask、slow/fast与风光载荷审计定位失败模块，再决定是否运行全部消融。

#### 阶段4：锁定后一次测试

仅在结构、损失权重、checkpoint规则与评价代码全部冻结后运行test。不能根据test事件再改阈值或结构。

---

## 9. 评价体系

### 9.1 普通概率质量

- wind station CRPS；
- aggregate wind CRPS；
- solar daylight CRPS；
- aggregate renewable CRPS；
- 80/90/95/99% coverage与interval width；
- Energy Score、Variogram Score；
- 全场站、相邻站、风—风、光—光、风—光空间相关RMSE；
- 第1日至第7日逐日指标。

### 9.2 持续深跌与定位

对独立物理事件与重叠窗口分开报告：

- any-hit、命中成员数与比例；
- onset误差与Hit@±3/±6/±12 h；
- duration MAE与相对误差；
- depth MAE与深度比；
- 区间IoU/主要时间区间覆盖率；
- 事件外额外负偏面积，用来识别“为了命中而整周拉低”；
- body、tail、all members分别评价。

主要覆盖标准固定为onset ±6 h、真实区间重叠≥50%、depth比例≥75%；同时报告宽松与严格标准。`any-hit`只作辅助指标，因为Raw在500成员下即使采用最严格标准也已对3个独立事件达到3/3，主排序应使用命中成员比例与tail-only单成员命中率。

### 9.3 高频变化

- 1/3/6 h ramp MAE、CRPS与90%覆盖；
- 错向ramp命中；
- 局部波动标准差误差；
- 事件下降沿与恢复沿分别统计；
- slow/fast频带内Energy与相关误差。

短时1/3 h ramp不套用持续事件的duration overlap；单独报告时刻Hit、方向和幅度覆盖。

### 9.4 多站同步与风光联合

- 同一事件中活跃场站集合的Jaccard；
- 跨站onset离散度误差；
- 聚合极端与单站极端的一致率；
- 风—光残差相关矩阵RMSE；
- 风光同向/反向事件命中率；
- 上下尾依赖系数误差；
- shared mode的场站载荷符号与空间连贯性。

### 9.5 成功判据

完整F至少应同时满足：

1. 普通wind/solar CRPS相对A不恶化超过2%，Energy Score不恶化超过3%；
2. 深跌命中增加不能只靠区间宽度增加，90%宽度增幅应控制在5%以内；
3. 可变duration持续事件的tail onset/duration错误明显低于当前水平，并在实际duration分层中方向一致；
4. 1/3/6 h ramp相对误差至少有一组稳定下降，且三个seed方向一致；
5. 事件外负偏面积下降，证明tail真正局部化；
6. 光伏与风光联合指标有可解释改善，不能只改善聚合风电。

若只出现“区间更宽、命中数更多、单成员质量不升”，仍判定失败。

---

## 10. 代码改造边界

建议新增独立模块文件，例如：

- `src/models/station_joint_decomposed_tail.py`：slow/fast refiner、joint tail field、soft mask与signed system modes；
- 在 `station_conditioned_diffusion.py` 中只增加必要的特征导出、$\epsilon\leftrightarrow x_0$ 转换和开关，不继续堆叠旧实验分支；
- 新建专用config、训练入口、审计脚本与测试，实验名称必须与Raw及旧tail实验隔离；
- 旧的time localizer、retrieval mismatch、discrete event memory、event transport Transformer默认全部关闭，不能暗中混入F。

正式实现前先记录当前分支与Raw checkpoint哈希。所有新增模块使用零初始化或恒等初始化，确保A能够被复现。

---

## 11. 暂不采用的做法

- 不继续提高tail比例；
- 不再用固定6 h事件窗口作为唯一结构；
- 不训练独立168 h位置分类头后再硬mask；
- 不继续让固定6 h窗口同时承担短时ramp、中时失配和长时低出力三种定义；
- 不允许两个无频带约束的slow/fast全分辨率头自由抵消；
- 不把风、光、系统拆成三个彼此独立的生成器；
- 不直接将中心滑动平均结果作为验证/生成条件；
- 不先替换为完整Diffusion-TS/CSDI主干；
- 不把Transformer本身当成定位成功的保证；
- 不在本规划阶段启动训练。

本计划的核心假设是：**当前主要瓶颈不是tail数量，而是同一条全周风电修正同时承担了事件位置、慢结构、快速变化和跨站关系；将这些语义在同一去噪器中进行可辨识的联合分解，才有机会同时改善深跌命中、错位和局部ramp。**
