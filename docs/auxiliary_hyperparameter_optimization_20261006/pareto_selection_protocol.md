# Body–Extreme Pareto selection protocol

**FROZEN BEFORE HYPERPARAMETER TRAINING**  
状态：协议冻结，NOT RUN。

## 1. References与固定设计

- Body reference：Raw；
- Extreme-capability reference：Full Independent Tail V2；
- Current deployable control：Lightweight `0.18/0.14/0.10`；
- ramp-selection run：仅作selector机制参照，不进入同一lambda搜索；
- Stage 1A固定为`alpha=0.65/1.00/1.35`，其中1.00复用；
- Stage 1B固定为4个既定三元组，实际权重为`alpha* × 三元组`；不得增删搜索点。

## 2. 决策流程

```text
candidate
    ↓
integrity check
    ↓
Body guardrail
    ↓
calibration / sharpness check
    ↓
Extreme two-family check
    ↓
bootstrap / leave-one-event-out stability
    ↓
Pareto frontier
    ↓
pre-registered tie-break
    ↓
final candidate
```

任何前置门禁FAIL均停止该候选晋级，但保留其全部原始结果。

## 3. 操作化术语

所有CI、改善、退化和稳定性严格采用`validation_objectives.md`：配对7-day block bootstrap、10,000次、95% CI、固定seed `20261006`。

- “显著/明显恶化”：硬门槛突破，或loss型差值CI下界大于0；
- “方向一致/稳定改善”：至少一项primary CI完全改善，并通过相应跨事件或跨尺度门禁；
- “不可区分”：全部预注册primary差值CI均包含0，且双方稳定性门禁状态相同；
- “Pareto候选打平”：双方均在frontier，彼此在全部primary Body与两类Extreme判定上不可区分，且multi-seed一致性状态相同。

## 4. Pareto frontier定义

不构造性能加权总分。对通过所有门禁的候选，以三类结果比较：

1. Body primary指标向量；
2. Persistent-event family；
3. Ramp/local-trajectory family。

候选A支配B，当且仅当：

- A相对B没有任何Primary Body指标的paired 95% CI完全处于退化方向；
- A在Persistent和Ramp两族均不比B低一个稳定性等级（稳定改善／不可区分／稳定退化）；
- 并且A至少在一个Primary Body指标有CI完全改善，或在一个Extreme族高一个稳定性等级。

不满足支配关系的合格候选共同位于Pareto frontier。由于样本点过少，本协议**不使用数学knee point**。

## 5. Stage 1A选择唯一alpha*

对`0.65/1.00/1.35`依次执行完整流程，唯一alpha*按以下固定顺序选择：

1. 淘汰完整性、Body或calibration/sharpness门禁失败者；
2. 淘汰被其他alpha支配者；
3. 优先选择Persistent与Ramp两族均达到“稳定改善”的alpha；若只有一个，直接选定；
4. 若Extreme表现整体不可区分，按`renewable CRPS → Energy Score → 90% Interval Score`顺序比较；遇到第一个pairwise CI不包含0的指标，选择该指标更优者；
5. 若三项Body指标仍全部不可区分，选择`|alpha-1|`最小者；
6. 若0.65与1.35仍等距且alpha=1已被淘汰，选择0.65，以较低辅助强度作为保守tie-break。

alpha*一旦选定即冻结。Stage 1B只能围绕该alpha*运行，后续结果不得触发重选alpha。

## 6. Stage 1B与最终tie-break

两个或更多候选都通过门禁且位于frontier时，严格按以下顺序：

1. **主体优先**：先排除相对另一候选存在任何Primary Body指标CI完全退化者；若仍并列，按`renewable CRPS → Energy Score → 90% Interval Score`逐项进行pairwise比较，在第一个CI不包含0的指标上选择更优者。
2. **双Extreme族优先**：同时达到Persistent与Ramp“稳定改善”的候选优于只改善一个族或两族均不可区分的候选。
3. **稳定性优先**：依次比较通过的leave-one-event-out次数、满足改善规则的source-direction strata数、multi-seed一致seed数；在第一项不同处选择较高者。
4. **改动最小**：若仍为“Pareto候选打平”，先选择`|alpha-1|`更小者；若alpha相同，再选择相对current control的归一化L1距离
   `D = |lambda_r/0.18-1| + |lambda_shape/0.14-1| + |lambda_slow/0.10-1|`
   更小者；若D完全相同，按预注册candidate id字典序选择，禁止人工裁决。

这里的`D`仅作为最终完全打平后的“改动大小”定义，不是性能score，也不参与Pareto构造。

## 7. 替换current control的条件

候选只有同时满足以下条件才替换current control：

- 通过全部Body及calibration/sharpness门禁；
- 位于Pareto frontier且不被current control支配；
- Persistent与Ramp两族都达到“稳定改善”；
- primary any-hit保持4/4、strict any-hit不退化；
- 不属于“仅靠扩大区间提高coverage”；
- Stage 2满足multi-seed一致性。

若没有候选满足全部条件，保留`0.18/0.14/0.10`。这只表示有限搜索未找到可稳定替代者，不表示control是理论最优。

## 8. Test锁定

alpha选择、relative-weight选择、Pareto筛选和seed复核全部只用validation。只有唯一final recipe、全部超参数、`400 Raw + 100 Tail`成员比例及evaluation protocol全部冻结后，才允许对test运行一次最终确认。test结果不得反向触发lambda搜索、alpha重选或候选重排。

