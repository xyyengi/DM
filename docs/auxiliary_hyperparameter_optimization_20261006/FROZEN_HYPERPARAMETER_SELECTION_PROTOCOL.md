# Frozen Hyperparameter Selection Protocol

**FROZEN BEFORE HYPERPARAMETER TRAINING**  
冻结日期：2026-10-06  
状态：NOT RUN；禁止训练、生成、修改lambda或增加实验。

## 1. 固定候选

Stage 1A固定为`alpha=0.65/1.00/1.35`，其中1.00复用既有control。

Stage 1B固定为：

| candidate | ramp | shape | slow |
|---|---:|---:|---:|
| shape-relief | 0.20 | 0.10 | 0.11 |
| ramp-emphasis | 0.22 | 0.09 | 0.10 |
| slow-restraint | 0.20 | 0.13 | 0.08 |
| balanced-low-shape | 0.17 | 0.12 | 0.12 |

实际权重固定为`alpha* × 三元组`。不得新增、删除或事后移动搜索点。

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

## 3. 统计规则

- validation-only；配对7-day moving-block bootstrap；10,000次；seed=`20261006`；percentile 95% CI。
- loss型差值均为`candidate-reference`：CI下界大于0为明确退化，CI上界小于0为明确改善，CI包含0为统计不可区分。
- Body硬门槛保持：renewable CRPS `≤Raw+2%`，Energy `≤Raw+1%`，wind/solar-daylight CRPS各`≤Raw+2%`，overall spatial与wind–solar RMSE各`≤Raw+5%`。它们只适用于本次有限搜索。
- Interval判定联合使用`|coverage_90-0.90|`、width和Interval Score。coverage误差改善、width明确增加且Interval Score未明确改善，定义为“仅靠扩大区间”，不得计为收益。
- Persistent稳定改善必须覆盖至少3/4事件并通过至少3/4 leave-one-event-out；Ramp稳定改善必须覆盖至少3/4 source-direction strata、每个至少2/3 lag，并同时包含wind、solar、positive和negative证据。
- 单个事件、单个solar case、单个方向或单个lag不得决定晋级。

## 4. alpha*冻结规则

先过完整性、Body和calibration门禁，再去除被支配alpha。优先双Extreme族稳定改善者；Extreme不可区分时，按`renewable CRPS → Energy → Interval Score`顺序选择第一个有明确pairwise差异的更优者；仍不可区分时选择最接近1.00者，0.65与1.35等距时选择0.65。选定后不得回退重选。

## 5. Pareto与tie-break

不计算任意加权总分，也不使用不可靠的数学“膝点”。通过门禁后，在frontier上依次按以下顺序选择：

1. Primary Body无明确退化，随后按renewable CRPS、Energy、Interval Score依次比较；
2. Persistent与Ramp两族均稳定改善；
3. leave-one-event-out、跨strata及multi-seed更稳定；
4. 仍打平时选择更接近current control者：先比较`|alpha-1|`，再比较三项lambda相对control的归一化L1距离，最后按candidate id字典序。

候选只有在通过全部门禁、两类Extreme均稳定改善、primary any-hit保持4/4、strict any-hit不退化且3-seed方向一致时，才可替换current control。否则保留`0.18/0.14/0.10`。

## 6. Test封存

搜索、alpha*、relative-weight、Pareto和seed复核全部只使用validation。Test只能在唯一final recipe、全部超参数、成员比例和evaluation protocol冻结后运行一次；test结果不得触发任何lambda重搜或候选重排。

## 7. 本轮执行状态

- 协议文件更新：完成；
- lambda或候选修改：未发生；
- 训练：NOT RUN；
- 生成：NOT RUN；
- alpha=0.65/1.35：NOT RUN；
- 下一步：等待用户确认。

