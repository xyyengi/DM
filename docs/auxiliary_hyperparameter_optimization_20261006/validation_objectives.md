# Validation-only objectives

**FROZEN BEFORE HYPERPARAMETER TRAINING**  
状态：协议冻结，NOT RUN。选择阶段只允许使用 validation；test 锁定。

## 1. 配对统计口径

- 所有候选使用同一组 validation issues、相同 Raw 400成员、相同Tail成员数和配对generation seed。
- 普通指标以 issue date 为统计单位，使用**配对7-day moving-block bootstrap**：按发布日期排序，以所有连续7个issue构成候选块；有放回抽块直到覆盖原样本数并截断为原长度；候选与参照使用完全相同的重采样索引。
- bootstrap固定为10,000次，随机种子固定为`20261006`，报告paired difference的percentile 95% CI。
- 对“越低越好”的指标统一定义 `Δ = candidate - reference`；因此CI完全大于0表示候选退化，完全小于0表示候选改善。
- coverage使用 `E_cov = |coverage_90 - 0.90|`，同样越低越好。
- 持续事件以4个独立事件为统计单位；其CI使用4事件的配对event bootstrap（10,000次、seed=`20261006`），并另做4次leave-one-event-out；不得把重叠窗口或500个成员当成独立事件。
- ramp/local-trajectory的CI以issue date为单位，使用上述配对7-day moving-block bootstrap，并保留event/non-event、source、direction和lag分层。

以下术语固定解释：

- **显著/明显恶化**：硬门槛被突破，或相对指定参照的paired-difference 95% CI下界大于0。
- **显著/明确改善**：paired-difference 95% CI上界小于0。
- **统计不可区分**：paired-difference 95% CI包含0；两个候选只有在全部预注册primary指标上均不可区分，且没有一方通过另一方未通过的稳定性门禁时，才称整体不可区分。
- **方向一致改善/稳定改善**：除至少一项primary指标明确改善外，还必须满足本文第5节的跨事件或跨尺度规则；单个点估计变好不构成稳定改善。

## 2. Body quality（全部保留原始值）

Primary Body指标：

1. Wind station CRPS；
2. Solar daylight CRPS；
3. Renewable aggregate CRPS；
4. Energy Score；
5. `E_cov`、90% interval width、90% Interval Score；
6. overall spatial correlation RMSE、wind–solar correlation RMSE。

结构性诊断指标：

- wind–wind、solar–solar correlation RMSE；
- lead-day 1–7 CRPS/width；
- wind/solar ACF error at 24 h and 48 h。

## 3. 预注册Body guardrails

候选进入Pareto集合前必须同时满足：

- renewable CRPS相对Raw退化不超过2.0%；
- Energy Score相对Raw退化不超过1.0%；
- wind CRPS相对Raw退化不超过2.0%；
- solar daylight CRPS相对Raw退化不超过2.0%；
- overall spatial与wind–solar correlation RMSE相对Raw退化均不超过5.0%；
- 相对current control，任何Primary Body指标不得满足“paired 95% CI下界大于0”；
- lead day 5–7不得三个点估计全部恶化且其中至少2天的paired 95% CI下界大于0；
- 24/48 h ACF不得在wind和solar两类中各至少出现一项paired 95% CI下界大于0。

这些数值仅是**本次有限超参数搜索的预注册筛选标准**，不是理论、普适或最优阈值。

## 4. Calibration与sharpness自动判定

对`E_cov`、width和Interval Score分别计算相对current control的paired difference及95% CI。

- **真实calibration改善**：`E_cov`的CI上界小于0，且满足以下至少一项：width的CI下界不大于0（未确认变宽），或Interval Score的CI上界小于0（综合概率质量明确改善）。
- **仅靠扩大区间提高coverage**：`E_cov`的CI上界小于0，同时width的CI下界大于0，且Interval Score的CI上界不小于0。此时coverage改善不得计为Body或Extreme收益。
- **sharpness改善但calibration中性**：width的CI上界小于0、Interval Score的CI上界小于0，且`E_cov`的CI包含0。
- **calibration/sharpness退化**：Interval Score的CI下界大于0；或者width的CI下界大于0且`E_cov`未明确改善。两者任一成立即不能进入Pareto集合。

因此，hit rate提高若同时被判定为“仅靠扩大区间提高coverage”，不能支持候选替换control。

## 5. Extreme quality与稳定性

Persistent-event family和Ramp/local-trajectory family始终分别报告，不构造加权总分。

### 5.1 Persistent-event family

原始指标：三档any-hit、Tail/all-member hit rate、onset absolute error、duration error、depth error/depth ratio，以及逐事件结果。

候选被认定为该族“稳定改善”须全部满足：

1. primary any-hit保持4/4；strict any-hit不得低于current control；
2. 在`hit rate`、`onset error`、`duration error`、`depth error`四个primary维度中，至少2个点估计改善，且至少1个paired 95% CI完全位于改善方向；其余维度不得有CI完全位于退化方向；
3. 逐事件比较中至少3/4事件在上述四维中的多数维度不劣于control；
4. 做4次leave-one-event-out后，聚合改善方向至少3/4次保持，且没有一次触发primary/strict any-hit门禁。

这排除了由单个事件独自驱动的晋级。

### 5.2 Ramp/local-trajectory family

原始指标：wind/solar × positive/negative × event/non-event × 1/3/6 h的std、q90/q95/q99、truth ratio、median MAE、90% ramp coverage，以及drop-conditioned recovery probability/amplitude/delay。

对于std/q95/q99，比较量为其与truth的绝对偏差；对MAE和delay直接比较；均为越低越好。候选被认定为该族“稳定改善”须全部满足：

1. event窗口内，在wind+/wind-/solar+/solar-四个source-direction strata中，至少3个strata各自满足1/3/6 h中至少2个lag的q95或q99 truth-distance点估计改善；
2. 上述改善中至少一个wind stratum和一个solar stratum存在paired 95% CI完全位于改善方向；
3. non-event窗口中，不得有3个或以上strata各自在2个或以上lag出现paired 95% CI完全位于退化方向；
4. positive与negative两个方向都必须至少有一个stratum通过，不允许单一方向决定晋级；
5. recovery仅作为辅助确认：若宣称恢复改善，1/3/6 h中至少2个horizon的恢复概率或delay点估计改善，且至少一个horizon的paired CI完全位于改善方向。

这排除了单个solar case、单个方向或单个ramp尺度独自决定晋级。

## 6. Multi-seed一致性

Stage 2共有3个训练/生成seed（首轮seed加2个新增seed）。相对current control的主要方向只有满足以下条件才称为multi-seed consistent：

- 不同seed下使用相同Body与Extreme规则；
- 至少2/3个seed通过全部Body guardrails；
- Persistent与Ramp两族的关键差值方向分别至少在2/3个seed一致；
- 任一seed不得突破相对Raw的硬Body数值门槛。

不满足上述规则即不得称“稳定改善”。

