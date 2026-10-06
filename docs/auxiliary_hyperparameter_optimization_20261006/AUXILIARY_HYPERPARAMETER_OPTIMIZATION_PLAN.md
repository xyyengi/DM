# Extreme Tail Auxiliary-Loss Hyperparameter Optimization Protocol

日期：2026-10-06  
状态：**DESIGN ONLY — NOT RUN**  
范围：只设计与复用既有证据；未训练、未生成、未启动搜索。

## 1. 方法定型决策

本轮推荐只在 **Lightweight External Joint Tail** 上优化 `ramp/shape/slow` 权重。理由是：

- Raw 主体参数完全冻结，20,588 个外置 Tail 参数承受极端目标，最符合“隔离 Tail + 保持 168 h 主体 + 轻量化”的收口方向；
- 它已通过严格预检、完整训练、500成员生成、评价和 generation-seed 稳定性复核，具备继续做配方选择的最低证据；
- Full Independent Tail V2 的 772,290 个参数具有更强的极端与联合结构能力，但它是完整模型，不满足最终轻量化目标；应保留为 extreme-capability reference，而不是再复制一套超参搜索；
- 同一 `lambda` 在两个架构上的有效梯度、可训练参数空间和优化轨迹不同，**不可直接迁移**。如未来最终改回 Full V2，必须重新做其自身尺度审计，不能沿用本次最优值。

本协议锁定 `configs/station24_lightweight_joint_tail_v2_fair_168h.yaml` 的网络、初始化、60% event-balanced sampling、原有 ramp selector、优化器、训练预算、扩散配置和 `400 Raw + 100 Tail`。`source_direction_daylight_pooled_v1` ramp-selection run 只作外部机制参照；因为 selector 已改变，不能作为本搜索的同一设计空间样本。

## 2. 现有权重的定位

`0.18/0.14/0.10` 是**有梯度审计支持的合理 control**，不是已优化参数：

- mixed batch 中 weighted auxiliary combined gradient 为 epsilon 的 40.4%；event batch 为 54.1%；
- 没有单项长期压倒 epsilon，故不是病态设置；
- `epsilon–shape cosine ≈ -0.604`，shape 与主体目标存在明显冲突；
- slow 虽仅为 0.10，但 event batch 的实际梯度不弱；
- 辅助贡献从 `t=30` 到 `t=300` 显著增强，不能只按系数大小判断。

因此，本轮目标不是证明 control 错误，而是在窄范围内寻找更好的 Body–Extreme Pareto 折中。

## 3. 两阶段搜索

### Stage 1A：整体 auxiliary strength

保持 `0.18:0.14:0.10`，定义：

`L = L_epsilon + alpha * (0.18 L_ramp + 0.14 L_shape + 0.10 L_slow)`。

复用 `alpha=1.00`；只新增：

- `alpha=0.65`：按局部线性近似，mixed auxiliary/epsilon 从 0.404 降至约 0.263，测试较弱但仍非可忽略的辅助信号；
- `alpha=1.35`：对应约 0.545，接近 event batch 当前 0.541，但仍明显低于 epsilon；这是有边界的强档，而不是机械的 1.5 倍。

先做 overall strength 是必要的：否则同时移动三项权重时，无法区分“总辅助强度”与“相对组成”效应。

### Stage 1B：相对比例

在 Stage 1A 选出的 `alpha*` 下，采用 4 个预注册 structured points；控制未乘 alpha 的总和在 0.40–0.42，避免把组成试验重新变成总强度试验：

1. shape-relief：`0.20/0.10/0.11`；
2. ramp-emphasis：`0.22/0.09/0.10`；
3. slow-restraint：`0.20/0.13/0.08`；
4. balanced-low-shape：`0.17/0.12/0.12`。

实际训练权重为 `alpha* × 上述三元组`。当前比例点若 `alpha*=1` 已由 control 覆盖，不重复训练。

选择 small structured design，而不是全网格、无约束随机或 TPE：当前只有 23 个 validation issues 和 4 个主要持续事件，6 个新增首轮点已足以回答整体强度与三项组成的主要问题；TPE 在如此少的样本上没有可靠代理优势，反而容易制造伪精确最优。

## 4. 固定控制变量

- 架构：Lightweight External Joint Tail；Raw 全冻结并保持 eval；
- trainable params：20,588；
- Raw checkpoint、Tail 初始化、数据、图、State Encoder、条件输入不变；
- event-balanced sampling：60%；
- ramp selector：保持原 fair-control 语义，不切换到 ramp-selection ablation；
- optimizer/lr/batch/epoch/early-stop/diffusion steps 不变；
- train seed `2027`、validation seed `314159`、generation seed `424242`；
- 500成员固定为 `400 Raw + 100 Tail`；
- self-localization、risk gate、新事件表示、新 loss 均关闭；
- evaluation split 只用 validation；test 保持锁定。

## 5. 验证与选择

候选必须经过 `validation_objectives.md` 的 Body 和 Extreme 两组原始指标，并按 `pareto_selection_protocol.md` 做非支配排序。不得仅最大化 hit rate 或 coverage。

替换当前 control 的最低条件：

1. 通过 Body guardrails；
2. 在 Body–Extreme 图上不被 current control 支配；
3. 至少一个持续事件指标族和一个 ramp/trajectory 指标族产生方向一致的改善；
4. 改善不能仅来自区间变宽；
5. 时间块 bootstrap、leave-one-event-out 和最终多 seed 不显示结论反转。

若没有候选同时满足这些条件，保留 `0.18/0.14/0.10`，结论为“合理 control 未被有限预算搜索击败”，而不是宣称它全局最优。

## 6. 数据划分与过拟合控制

现有数据为 2025 年一年：train 290 issues（1–10月）、validation 23 issues（11月）、test 25 issues（12月），并有 train–validation embargo；正式配置均为 `evaluation.split=val` 且 `test_locked=true`。目前没有发现 Tail 权重选择直接使用 test 的证据。

风险在于 validation 已被多轮人工查看，且 168 h 相邻 issue 高度相关、主要持续事件只有 4 个。补救措施：

- 在运行前冻结本协议、候选点、指标、guardrails 和 tie-break；禁止看结果后扩范围；
- 以 issue-date 的 7天时间块 bootstrap 报告不确定度，不能把重叠窗口当独立样本；
- 对 4 个持续事件做 leave-one-event-out 稳定性检查；
- Body 指标按 lead day 1–7 检查，防止局部改善掩盖后段退化；
- Stage 1 全部候选使用同一 train/generation seed；只对最终 1–2 个 Pareto 候选追加 seed；
- test 只在唯一 recipe 冻结后运行一次。若 test 未确认，则不得使用 test 反馈继续改 lambda。

## 7. 训练次数与停止规则

- 已有 direct control：1 组（Lightweight fair，`alpha=1`），复用；
- 新增 alpha sensitivity：2 组；
- 新增 relative-weight structured design：4 组；
- Stage 1 新增合计：6 组；
- Stage 2：默认只复核 1 个最终候选，追加 2 个 train/generation seeds，总新增 **8 runs**；
- 若两个候选在 bootstrap 不可区分，则两者各追加 2 seeds，总新增 **10 runs（硬上限）**；
- 12 runs 只作为成本情景，不是推荐计划，不因结果不漂亮而启用。

每一 run 都必须使用完整 Station-24 formal pipeline；任何 CUDA/AMP、学习性、冻结性或产物合同 FAIL 均停止付费链路。成员比例固定，不进入搜索。

## 8. Q1–Q10

**Q1. 最终推荐在哪个 Tail 架构上优化？**  
Lightweight External Joint Tail。

**Q2. 为什么？**  
它直接实现 Raw 参数隔离、168 h 主体保持和轻量化；Full V2保留为能力参照。两者 lambda 不可直接迁移，也无必要双架构搜索。

**Q3. 0.18/0.14/0.10 如何定位？**  
是经过梯度审计的合理 control，不是已优化参数。

**Q4. 是否先做整体 alpha sensitivity？**  
是。复用 1.00，只新增 0.65 和 1.35，再研究相对比例。

**Q5. 三项 lambda 建议范围？**  
以未乘 alpha 的基准范围计：ramp `[0.14,0.22]`、shape `[0.08,0.14]`、slow `[0.08,0.12]`，并将三项和约束在 `[0.40,0.44]`。ramp保留适度上调空间；shape因与epsilon强负cosine只向下探索；slow因event梯度不弱保持窄范围。

**Q6. 推荐哪种搜索方法？**  
小型预注册 structured design；不使用暴力网格或仅6点条件下的TPE。

**Q7. 总共新增多少次训练？**  
推荐8次；出现两个不可区分Pareto候选时最多10次。

**Q8. 如何定义 Body–Extreme Pareto 最优？**  
先过Body guardrails，再保留非支配候选；候选须同时改善持续事件与ramp/trajectory至少两个独立指标族，且不是靠扩大区间获得。

**Q9. 如何防止过拟合？**  
预注册、validation-only、时间块bootstrap、leave-one-event-out、统一首轮seed、只复核最终候选，并将test锁到唯一recipe之后一次性确认。

**Q10. 优化后还剩哪些最终确认实验？**  
仅剩最终候选多seed确认、一次锁定test确认、与Raw/current Lightweight/Full V2的统一最终表，以及既有成员比例敏感性作为部署说明；不继续开发新机制。

