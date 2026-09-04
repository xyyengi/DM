# JSTD-Tail V1 机制审计与无二分类门控升级路线

## 结论先行

JSTD-Tail V1 已证明共享 slow/fast tail 能生成一部分有效极端成员，但当前发布级二分类门控不能区分真实事件窗口：在23个验证窗口、4个连续事件的探索性统计中，AUROC约为0.434，且最高的几个 tail 概率出现在非事件窗口。下一版不再让 Bernoulli 门控决定一个发布窗口“有没有资格”生成极端。

在训练新模型前，先对当前 checkpoint 做强制 route=1 的机制审计，将三个问题分开：

1. 门控有没有选对发布窗口；
2. 给定 tail 必须工作时，slow/fast mask 有没有定位到正确场站和时间；
3. mask定位后，修正的方向、幅度以及 slow/fast 相消是否合理。

## 已补充的审计

运行 `tools/audit_station24_jstd_mechanism.py`，对4个真实验证事件和tail概率最高的3个非事件窗口，在固定扩散时刻50/150/300/450执行确定性噪声探针。

审计导出：

- `slow_mask.npy`、`fast_mask.npy`；
- slow/fast epsilon修正；
- slow/fast等效x0修正；
- mask在真实事件内外的强度比；
- 修正能量落在事件外的比例；
- slow/fast相消比例；
- 每个窗口的机制图与汇总报告。

真实actual/residual只用于离线构造探针和评价位置，不进入JSTD条件编码器。

## 审计后的决策规则

### 情况1：mask有定位能力，门控失败

参考判断：事件内/外mask强度比明显大于1，且tail强制打开后修正主要集中在事件附近。

处理：保留JSTD共享tail，移除发布级Bernoulli门控，改成下面的“连续事件假设条件化”。

### 情况2：mask基本均匀或修正大量泄漏

参考判断：事件内/外mask强度接近1，或大多数修正能量落在事件区间外。

处理：不能只换门控。需要将事件假设直接输入mask与slow/fast生成路径，使mask回答“这个假设应该放在哪里”，而不是仅从缺少信息的forecast中猜唯一答案。

### 情况3：mask位置正确但slow/fast明显相消

处理：保持一个共享tail，但加入频带一致性约束：slow只承担12/24h低频结构，fast只承担1/3/6h局部增量，并对同一频带中的反向抵消施加惩罚。不能简单禁止不同频带符号相反，因为快速恢复本身可能需要这种结构。

## 推荐主方案：连续事件假设条件化的共享tail

名称暂定为 Event-Hypothesis Conditioned JSTD，不增加第三专家，也不复制风电/光伏tail。

### 1. 不再预测一个“有/无事件”答案

每个tail成员显式携带一个连续事件假设：

\[
z=(\tau,\ d,\ a_w,\ a_s,\ u),
\]

其中：

- \(\tau\)：onset；
- \(d\)：连续duration；
- \(a_w\)：风电有符号depth；
- \(a_s\)：光伏有符号depth；
- \(u\)：少量系统/空间同步系数。

\((a_w,a_s)\) 是连续二维表示，因此自然覆盖风电独有、光伏独有、风光同向和反向异常，不需要四五个专家。

### 2. 分层采样，而不是门控后再决定是否生成

每个发布窗口均保留Raw body为主体，并从训练集事件先验中分层抽取一小部分事件假设成员。条件信息只调整假设权重和形态，不再把任何模式的概率压成零。

初始正式协议建议仍使用500成员：

- 约90% Raw body成员；
- 约10%事件假设成员，与Raw body-tail当前约10.2%的tail占比对齐；
- tail内部按风/光有符号depth、onset区间和duration分层采样，防止500次随机采样仍漏掉稀有模式。

这一比例不是最终超参数，后续用CRPS、事件命中率和区间宽度联合校准，但首版不再做15/20/30%的盲目比例扫描。

### 3. 一个共享JSTD tail接收事件假设

通过小型embedding/FiLM将 \(z\) 同时注入：

- slow mask与slow correction；
- fast mask与fast correction；
- 系统公共模式和场站loading。

这样，模型不再试图从同一个forecast中输出唯一的事件位置，而是学习：

> 给定一种可能的onset、duration和风光幅度组合，怎样生成物理连贯的24场站×168h联合场景。

### 4. 使用成熟的条件扩散引导思想

训练时随机丢弃事件假设条件，同时学习有条件和无事件假设条件的tail；生成时采用classifier-free guidance：

\[
\epsilon_{\mathrm{guided}}
=\epsilon_{\mathrm{null}}
+w\left(\epsilon_z-\epsilon_{\mathrm{null}}\right).
\]

它不需要额外训练一个决定“有/无”的分类器。该思想来自 Ho 与 Salimans 的 Classifier-Free Diffusion Guidance；本项目只把条件从图像类别替换为连续事件结构。

为控制成本，只对约10%的事件假设成员计算引导，主体成员仍走Raw路径。即使tail引导需要一次额外前向，总生成开销预计增加约10%，而不是翻倍。

## 与成熟工作的关系

- Diffusion-TS：保留其“把不同时间尺度结构分开建模”的原则，但不机械照搬Transformer；slow负责持续结构，fast负责局部变化。论文与代码入口：<https://openreview.net/forum?id=4h1apFjO99>。
- Classifier-Free Guidance：通过条件丢弃和条件/无条件score组合控制生成模式，不依赖外部二分类器：<https://arxiv.org/abs/2207.12598>。
- Classifier Guidance：证明在扩散反演过程中利用目标属性引导采样是成熟路径；本项目优先使用无需额外分类器的CFG版本：<https://arxiv.org/abs/2105.05233>。
- TSDiff自引导：说明时间序列扩散可以在不替换整个生成主干的情况下，通过采样阶段引导实现forecast/refinement任务：<https://arxiv.org/abs/2307.11494>。

这些工作提供的是条件化、引导和分解原则。连续的风光事件假设、24场站联合mask以及与Raw主体的概率配额组合仍是本项目需要验证的具体设计，不能在实验前宣称为已有结论或正式创新。

## 最少实验路线

审计完成后，不再一次堆很多模块：

1. **Raw baseline**：现有Raw body-tail；
2. **H1**：共享JSTD tail + 真实训练事件属性 \(z\) 条件化，验证tail是否学会“按指定假设生成”；
3. **H2**：H1 + 分层事件假设采样，完全取消Bernoulli发布级门控；
4. **H3**：H2 + classifier-free guidance，检验事件可控性和普通质量的权衡；
5. 只有H3确实优于Raw后，再消融slow、fast、空间同步和CFG。

首要成功条件不是区间更宽，而是：

- 4个独立事件及补充重叠窗口的主要/严格命中均增加；
- onset、duration、depth至少两项稳定改善；
- 第4～7天CRPS不再系统退化；
- 90%区间宽度增幅控制在5%以内；
- 非事件窗口的tail修正能量显著低于JSTD V1。

如果H1在使用训练真值事件属性作为条件时仍不能按指定位置和幅度生成，就应停止JSTD tail路线，而不是继续研究先验采样或增加网络模块。
