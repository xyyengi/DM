# JSTD-Tail训练前：事件、损失、评价与因果条件审计

## 1. 结论先行

本次审计采用一个明确原则：

> 1/3/6/12/24 h是模型观察和约束时间尺度，不是五种事件身份，也不对应五个专家。一个真实持续事件只保存一个连续区间及其真实属性。

据此得到以下结论。

1. Raw body-tail中没有一行显式写成“duration必须大于等于6 h”，但训练和评价都被6 h滑动窗口锚定，效果上形成了隐藏的6 h准入条件。
2. 后续mismatch/unified/event-memory路径仍存在固定12 h窗口、`[6,12,24]`离散duration和固定半径mask。这些旧实验路径不能直接作为JSTD标签实现。
3. 1/3/6 h可以且应该只作为fast分支的多尺度差分监督；12/24 h可以且应该只作为slow分支的低频投影与结构约束。
4. 持续事件目录应由连续支持区间产生，直接记录 `onset/stop/duration/depth`。17 h事件就是17 h事件，不映射到12 h或24 h类别。
5. slow与fast必须使用同一个规范分解边界和互补投影。不能让slow分别拟合12 h目标、24 h目标，又让fast重复拟合完整残差。
6. 生成条件中最值得优先加入的是：预测多尺度几何、因果recent-error状态、系统级风光状态和固定图邻域状态。forecast revision只应作为前144 h的定位/可信度证据，不应再全局修改body中心。

本轮只完成审计和设计，不训练模型，不修改Raw baseline。

---

## 2. 现有代码中的固定时间逻辑

### 2.1 Raw持续事件标签

`fit_station_event_replay()`当前执行：

1. 对每个train发布窗口计算容量加权风电 `forecast-actual`；
2. 搜索最大6 h滑动平均；
3. 使用每个发布窗口最大值的train q80/q90筛选事件；
4. 对重叠发布进行物理时间去重；
5. 只给代表样本构造固定6 h的 `event_window_mask`。

因此它不是字面上的“连续超阈值至少6 h”，而是：

> 只有6 h平均严重度足够大的样本才能成为事件，且进入训练后仍只监督固定6 h。

相关位置：

- `station_dataset.py:750`：`event_replay_window_hours`默认6；
- `station_dataset.py:783-800`：最大固定窗口严重度；
- `station_dataset.py:845-869`：`stop=start+window`；
- `station_dataset.py:1646-1650`：固定窗口mask；
- Raw配置：`event_replay_window_hours=6`与`event_x0_window_hours=6`。

### 2.2 Raw loss中的固定6 h

当前event x0 loss仍依赖同一个固定窗口：

- magnitude：固定窗口内聚合残差均值；
- timing：对固定宽度滚动均值做单一起点分类；
- sync：固定窗口内的场站均值；
- tail epsilon support：固定窗口再加固定context。

所以即使未来增加可变duration目录，如果不改loss，网络仍会被6 h目标拉回去。

### 2.3 当前持续深跌评价

当前评价比训练略好：先通过6 h平均找到峰值，再把峰值所在的连续超阈值区间向两侧扩展，因此已评价的真实duration可以是8、14、20 h。

但仍有两个6 h锚点：

- 事件能否进入目录由6 h平均超过train q80决定；
- 审计代码显式拒绝不是6 h的Raw replay定义。

因此短促但严重的1–3 h变化可能不进入持续事件目录；长事件虽然进入，但其训练监督仍只覆盖最严重6 h。

### 2.4 其他旧路径中的隐藏固定尺度

JSTD不能直接复用以下语义：

- mismatch replay：先识别1/3/6 h ramp，但随后放入固定12 h事件窗口；
- unified replay：要求持续事件窗口和mismatch窗口完全相同；
- tail time localizer：采样一个中心，再生成固定半径余弦mask；
- discrete event memory：默认duration集合为 `[6,12,24]`；
- retrieved-event诊断：部分持续事件仍要求 `duration>=6`；
- 多个绘图和困难事件选择脚本仍用最大6 h预测高估排序。

这些代码可以保留用于旧实验复现，但JSTD必须有独立的标签版本和评价入口。

---

## 3. JSTD事件表示：事件时长连续，观察尺度离散

### 3.1 唯一持续事件目录

JSTD不创建“6 h事件、12 h事件、24 h事件”。每个物理持续事件只保存一次：

```text
event_id
representative_sample_index
physical_onset
physical_stop
lead_onset
lead_stop
actual_duration_hours
max_depth
mean_depth
integrated_shortfall
direction
station_soft_support[24]
time_soft_support[168]
overlapping_issue_indices
```

其中：

$$
D_e=t_{\mathrm{stop}}-t_{\mathrm{onset}}
$$

直接是实际连续时长。若某事件持续17 h，则 `actual_duration_hours=17`。

### 3.2 连续区间如何识别

以train residual中的下侧预测失配为例：

$$
q(t)=\max\{-r_{\mathrm{agg}}(t),0\}
=\max\{f_{\mathrm{agg}}(t)-y_{\mathrm{agg}}(t),0\}.
$$

只在train拟合两个点态严重度阈值：进入阈值 $\tau_{\mathrm{enter}}$ 和保持阈值 $\tau_{\mathrm{keep}}$，其中后者较低。使用迟滞连通规则：

1. 当 $q(t)\ge\tau_{\mathrm{enter}}$ 时开启事件；
2. 相邻小时只要 $q(t)\ge\tau_{\mathrm{keep}}$ 就保持事件；
3. 允许桥接至多1 h的小缺口；
4. 不设6 h最小时长；
5. 重叠发布仍按物理时间去重，只保留一个真实事件。

阈值是事件严重度定义，不是事件duration类别。为避免单小时噪声被误叫“持续深跌”，报告时可以按实际duration做描述性分层，但不能改变标签身份或专家路由。

### 3.3 fast观察尺度

对同一个残差目标，构造：

$$
\Delta_h r(t)=r(t)-r(t-h),
\qquad h\in\{1,3,6\}.
$$

1/3/6 h只用于表示：

- 急跌和急升幅度；
- 下降沿与恢复沿；
- 局部波动强度；
- 错向或预测遗漏ramp；
- fast时空支持的软强度。

同一小时可以在1、3、6 h三个观察尺度上同时活跃；这不是三个事件，也不需要选择一个dominant class。

### 3.4 slow观察尺度

12/24 h只定义低频算子：

$$
P_{12}r,\qquad P_{24}r.
$$

它们用于检查和约束：

- 持续低偏平台；
- depth与积分短缺；
- duration；
- 慢恢复；
- fast分支是否泄漏低频。

12/24 h不是事件持续时间估计。一个17 h事件仍以17 h区间监督，只是使用 $P_{12}$ 和 $P_{24}$ 检查其慢结构。

---

## 4. slow/fast的规范分解与loss

### 4.1 只使用一个规范分解边界

为避免“slow同时被要求等于12 h和24 h目标”的冲突，采用12 h作为slow/fast的规范互补边界：

$$
r^*_{\mathrm{slow}}=P_{12}r,
\qquad
r^*_{\mathrm{fast}}=(I-P_{12})r.
$$

24 h不再产生第二套slow标签，而只对slow的更长期结构增加一致性检查：

$$
\mathcal L_{24}
=\left\|P_{24}\hat r_{\mathrm{slow}}-P_{24}r\right\|.
$$

这样12/24 h不会互相争抢同一个输出。

### 4.2 分解loss

建议将分解项写成：

$$
\mathcal L_{\mathrm{decomp}}
=\lambda_s\left\|\hat r_s-P_{12}r\right\|_1
+\lambda_{24}\left\|P_{24}\hat r_s-P_{24}r\right\|_1
+\lambda_f\left\|\hat r_f-(I-P_{12})r\right\|_1
+\lambda_\Delta\sum_{h\in\{1,3,6\}}w_h
\left\|\Delta_h\hat r_f-\Delta_h r_f^*\right\|_1.
$$

最终重构只约束一次：

$$
\mathcal L_{\mathrm{recon}}
=\left\|\hat r_s+\hat r_f-r\right\|.
$$

不能再让slow和fast分别拟合完整 $r$，否则标签必然重复。

### 4.3 mask监督不重复

两个mask的职责不同：

- $M_{\mathrm{slow}}^*$：来自连续事件的水平严重度与持续支持，只强调平台、duration和慢恢复；
- $M_{\mathrm{fast}}^*$：来自 $r_f^*$ 的1/3/6 h差分严重度，只强调下降沿、恢复沿和局部变化。

一个17 h事件中，slow mask覆盖持续平台，fast mask可以只在下降沿和恢复沿高。二者允许在边界少量重叠，但对应的修正已经经过互补频带投影，不是在重复拟合同一信号。

### 4.4 必须避免的冲突

1. 不把1/3/6 h ramp样本再扩成固定12 h tail窗口；
2. 不把12/24 h低通曲线当成两类事件标签；
3. 不对slow和fast都使用完整残差level loss；
4. 不以mask完全互斥作为目标；真实恢复沿允许fast与slow方向相反；
5. 不允许两个全分辨率自由头直接相加。

最终修正应保持：

$$
\Delta r_s=P_{12}u_s,
\qquad
\Delta r_f=(I-P_{12})u_f.
$$

并报告低频泄漏、频带能量、slow-fast相关和抵消率。

---

## 5. 三档评价如何具体实现

### 5.1 持续事件评价

真实事件和生成场景事件都使用同一组冻结的train阈值与迟滞连通规则，直接得到可变区间。对每个真实事件保存全部候选生成区间，不先使用某个onset容差挑选唯一候选。

同时计算：

- 宽松：onset ±12 h、真实区间覆盖≥25%、depth≥50%；
- 主要：onset ±6 h、真实区间覆盖≥50%、depth≥75%；
- 严格：onset ±3 h、真实区间覆盖≥75%、depth≥100%。

三档之外始终并列报告：

- duration MAE和相对误差；
- depth MAE、depth ratio和过深率；
- interval IoU；
- target-hour depth coverage；
- 事件外额外负偏面积；
- body/tail/all member hit rate；
- 独立物理事件与重叠发布视图。

实际duration可以在结果表中按区间分层展示，但只是事后分析列，不参与训练类别或模型路由。

### 5.2 fast评价

fast不使用持续事件duration overlap。对1/3/6 h分别报告：

- ramp CRPS/MAE/coverage；
- 极端ramp方向命中；
- 时刻Hit@±1/±3/±6 h；
- 幅度比和过冲率；
- 多站同步ramp集合Jaccard；
- 下降沿和恢复沿分开统计。

### 5.3 slow评价

slow使用真实可变duration事件，报告：

- onset、stop、duration、depth；
- $P_{12}$与$P_{24}$形状相关；
- 积分短缺误差；
- 慢恢复斜率误差；
- fast低频泄漏和slow高频泄漏。

---

## 6. 生成阶段严格因果可得的功率条件

### 6.0 现有条件与消融证据

Raw最佳配置当前实际为：

- `use_forecast_ramps=false`；
- `use_forecast_revision=false`；
- `use_recent_error=true`，历史长度24 h；
- `use_state_encoder=true`，状态中包含预测3/6 h ramp；
- fixed geographic graph + train-only historical actual graph。

早期条件消融中，三组候选都显式使用了1/3/6 h forecast ramp，因此该实验不能单独证明“forecast ramp一定有效”。但它能区分revision与recent error：

- recent error + ramps相对当时baseline，wind CRPS由0.09223改善到0.09163，聚合renewable CRPS由155.74改善到152.69，极端1/3/6 h风电ramp覆盖分别由28.06/34.77/39.88%提高到31.93/38.60/42.81%；
- revision + ramps的wind CRPS为0.09295，空间相关RMSE也更差；
- revision + recent error + ramps虽然提高部分coverage，但聚合renewable CRPS恶化到168.32。

因此recent error有继续深挖的经验依据；revision不能再作为全局body条件直接混入，只适合重新定义为tail定位/可信度证据。显式forecast几何则需要在JSTD中通过职责隔离和消融单独验证。

### 6.1 因果边界

允许输入：

- 当前发布时已经得到的未来168 h功率预测；
- 上一版已发布预测与当前预测的重叠部分；
- 当前发布时刻之前已经实现的实际功率及对应历史预测误差；
- train-only拟合后冻结的统计阈值、固定图和模型参数；
- calendar、lead、场站静态信息。

禁止输入：

- 当前168 h内任何未来actual；
- 当前168 h未来residual；
- 用val/test未来actual在线计算的风险分数、尺度、检索权重或mask。

### 6.2 条件一：当前预测的多尺度几何——最高优先级

当前模型虽然已经输入完整forecast，但对只有290个训练窗口的数据，显式提供与任务对齐的几何表示有助于定位头学习。

Fast定位使用：

$$
\Delta_hF(t)=F(t)-F(t-h),\quad h\in\{1,3,6\},
$$

以及局部二阶变化：

$$
\kappa_hF(t)=F(t+h)-2F(t)+F(t-h).
$$

Slow定位使用：

- $P_{12}F$、$P_{24}F$；
- 低频斜率；
- 谷底候选和恢复斜率；
- 预测低出力持续支持。

这些特征只送入JSTD mask/tail控制路径，不重新作为body的额外强锚点。

作用：直接回答“168 h中哪些时段具有急变、谷底或慢恢复结构”。

### 6.3 条件二：recent-error多尺度统计——高优先级

现有代码读取上一发布日首24 h已经实现的 `actual-forecast`，因果方向正确；但编码器把24 h序列压成“全局均值与最后一步”的平均，再广播到未来168 h，几乎丢失了误差状态的时间结构。

建议从同一段已知历史误差构造：

- 3/6/12/24 h signed mean；
- 3/6/12/24 h MAE或RMS；
- 最近误差斜率；
- 下侧误差比例与最长连续同号长度；
- wind/solar系统聚合误差；
- 图邻域聚合误差。

它主要回答“这个场站/系统最近是否处于持续高估、快速恶化或误差扩张状态”，用于风险强度和slow持续性门控。它不能单独决定未来第几小时发生事件，必须与当前forecast几何共同使用。

部署前还要增加 `observation_cutoff` 审计：只有在当前发布时间之前确实已落库的历史actual才可进入recent error。若业务存在1–2 h数据延迟，最后对应小时必须mask掉。

### 6.4 条件三：forecast revision——中高优先级、仅用于定位/可信度

定义：

$$
R_{s,t}=F^{(d)}_{s,t}-F^{(d-1)}_{s,t+24}.
$$

当前代码正确地只在前144 h设置revision，最后24 h通过mask标记缺失。已有消融表明：把revision作为全局条件注入主体没有稳定改善CRPS、Energy和空间相关，因此不能重复旧用法。

JSTD中的用途应限定为：

- tail mask的时间对齐不稳定度；
- forecast trust/route证据；
- 当前预测与上一版预测在谷底、ramp和恢复段上的分歧；
- revision跨站同步程度。

建议输入原始signed revision、绝对revision和3/6 h平滑revision，并保留availability mask。不要把revision直接加到预测中心或全周body特征上。

### 6.5 条件四：系统级风光状态——高优先级

现有State V1已经构造每站低/高出力、上/下ramp，并在状态编码器中容量加权聚合风、光；该基础可以直接复用。

JSTD需要把系统状态明确送入联合tail头：

- 聚合风电与聚合光伏的level、1/3/6 h ramp；
- wind/solar同时下降、反向变化和一方补偿另一方；
- 处于低出力或大ramp状态的场站容量占比；
- 系统revision与recent-error状态。

作用：区分单站异常、多风场同步异常、风光共同异常和风光反向异常，而不增加第三个专家。

### 6.6 条件五：固定图邻域状态——高优先级

不重新学习动态图。复用现有地理图与train-only历史实际功率图，对当前可得状态做消息聚合：

$$
N_X(s,t)=\sum_j A_{sj}X(j,t).
$$

优先聚合：

- 当前forecast level与1/3/6 h ramp；
- forecast revision及其mask；
- recent-error多尺度状态；
- 低出力/大ramp状态。

同时保留 `local-neighbor` 差值，帮助mask判断是局部异常还是区域同步过程。图只传播已知条件，不传播未来actual/residual。

### 6.7 条件接入方式

不把所有特征简单拼到U-Net输入。使用三个有职责的condition token/field：

```text
C_fast(s,t)   = forecast 1/3/6 h几何 + revision局部变化 + 邻域ramp
C_slow(s,t)   = forecast 12/24 h结构 + recent-error状态 + 邻域低频状态
C_system(t)   = 聚合风/光状态 + 同向/反向关系 + 系统可信度
```

然后：

- `C_fast`主要调制 $M_{fast}$ 与fast修正；
- `C_slow`主要调制 $M_{slow}$、depth和恢复；
- `C_system`调制共享风光模式与场站载荷；
- 原forecast仍保留在Raw body中；
- 所有新增条件必须输出门控利用率和置换/遮蔽消融，证明不是无效维度。

---

## 7. 正式训练前必须同步完成的三项

### 7.1 多尺度监督、单一连续事件标签

新增独立构建器，例如 `fit_station_jstd_event_targets()`：

```text
continuous_event_catalog.json
train_event_time_support.npy      [N,168]
train_event_station_support.npy   [N,24,168]
train_fast_ramp_target.npy        [N,24,3,168]
train_slow_target.npy             [N,24,168]
train_fast_target.npy             [N,24,168]
```

要求：

- `slow_target + fast_target == residual`在数值容差内成立；
- 事件duration来自连续区间，不来自滤波宽度；
- 1/3/6 h轴是监督尺度轴，不是event type轴；
- 12/24 h不出现在event type字段；
- 只在train附加监督，val/test dataset不能返回未来标签作为条件。

### 7.2 三档评价

新增JSTD评价器，不修改Raw旧脚本以免破坏复现：

1. 使用train冻结阈值提取真实和场景的全部可变duration区间；
2. 为每个真实事件保留所有候选匹配；
3. 同时计算宽松/主要/严格三档；
4. 输出body/tail/all、逐事件、按实际duration事后分层和逐提前日结果；duration分层只用于报告，不参与事件身份或路由；
5. 另行输出1/3/6 h fast评价及12/24 h slow结构评价；
6. 测试候选选择不依赖三档中的任何阈值。

### 7.3 slow/fast互补约束

在模型前向和单元测试中固定：

1. `P12`与 `I-P12`严格互补；
2. slow修正必须经过 `P12`；
3. fast修正必须经过 `I-P12`；
4. `P24`只做slow一致性，不生成第二个slow标签；
5. 关闭JSTD时逐元素复现Raw输出；
6. 保存并检查低频泄漏率、频带能量比、分量相关和抵消率。

只有这三项全部通过单元测试和train-only审计后，才启动完整JSTD训练。

### 7.4 具体代码实施顺序

#### 第一步：标签构建，不接模型

在 `station_dataset.py` 中新增独立的 `fit_station_jstd_event_targets()` 与 `validate_station_jstd_event_targets()`，不修改 `fit_station_event_replay()`。先只输出标签文件和审计JSON，并完成以下测试：

- 人工17 h连续事件输出duration必须为17；
- 人工2 h急跌不因小于6 h而消失；
- 一个17 h平台加两个边沿只产生一个持续event ID；
- val/test构造器不能拟合阈值；
- 同一物理事件的重叠发布不能重复计数。

#### 第二步：评价器先行

新增 `tools/evaluate_station24_jstd_events.py`，先对Raw现有500成员运行，形成新的固定基线。必须在训练JSTD前得到：

- 三档持续事件结果；
- 1/3/6 h fast结果；
- 12/24 h slow结果；
- body/tail/all拆分；
- 每个实际duration事件的候选列表。

这样新模型失败时可以确认是模型问题，而不是评价口径同时变化。

#### 第三步：互补分解算子

在新文件 `src/models/station_joint_decomposed_tail.py` 中先实现无参数的 `P12`、`P24`和互补高通。单元测试要求：

```text
max_abs(P12(x) + (x-P12(x)) - x) < 1e-6
fast_low_frequency_leakage在预设容差内
constant/linear/single-pulse输入的分解方向正确
边界长度仍为168
```

#### 第四步：因果条件构建器

在dataset侧只计算原始、可审计的因果特征；在新模块中按 `C_fast/C_slow/C_system` 编码。每批保存availability审计：

- revision有效小时必须为0或144；
- recent error的最大时间戳必须早于当前issue time；
- forecast几何只由当前forecast重算可复现；
- 邻域状态只能由固定图乘以已知条件得到；
- 把未来actual随机改写后，所有生成条件逐元素不变。

#### 第五步：JSTD前向恒等测试

先接入模型但不训练：

- 新分支零初始化、route为0或mask为0时，输出与Raw checkpoint逐元素一致；
- slow/fast mask形状为 `[B,24,168]`；
- fast/slow修正分别经过互补投影；
- 太阳夜间物理投影仍生效；
- future actual/residual只进入loss target，不能进入condition encoder。

#### 第六步：训练前审计门

只有以下文件全部存在且测试通过，服务器训练脚本才允许启动：

```text
jstd_event_targets.json
jstd_condition_audit.json
jstd_raw_baseline_evaluation.json
jstd_identity_test.json
jstd_frequency_projection_test.json
```

这不是额外实验，而是防止再次花费数小时后才发现标签仍固定6 h、条件泄漏或新结构没有保持Raw起点。

---

## 8. 最终推荐

JSTD第一版条件不应无限扩张。建议正式完整方案使用：

1. 当前forecast多尺度几何；
2. recent-error多尺度状态；
3. 由上述两类信号经现有固定双图和风/光容量聚合得到的邻域/系统状态。

其中forecast几何负责“何时”，recent error负责“当前误差制度与持续风险”，系统/邻域聚合负责“哪些场站以及是否同步”。`forecast revision`保留为后续独立消融，第一版不加入：已有实验显示它作为全局条件不稳定，而同步加入会使首轮结果难以归因。三组信息都来自现有功率信息并满足发布时因果可得，且只进入JSTD tail控制路径，不改变Raw body条件调制。

另固定四个训练安全约束：先做 `mask × correction` 再做互补频带投影；$x_0$辅助项按信噪比加权并设上限；普通样本必须参与mask负监督；checkpoint选择同时记录普通质量和事件目标，不允许仅凭旧的tail epsilon MSE淘汰有效的局部事件模型。
