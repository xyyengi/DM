# Station-24 Original S5 全路径 CRN 配对归因协议

**FROZEN BEFORE CRN GENERATION — 2026-10-14**

## 唯一研究问题

在模型、条件、正式 ancestral DDPM、Original S5 规则和成员预算全部冻结时，Raw–Tail 的独立成员耦合是否是修正差值不稳定及事件窗口失配的主要来源？本实验不训练，不开发 S5-V3，不改变任何窗口评分、持续时间或掩码。

## 冻结输入

- Raw run：`body_tail_moe_20260824_191036/.../20260824_191044_station24_body_tail_moe_20260824_191036_seed2027`
- Full Independent Tail V2 run：`independent_joint_tail_v2_event_balanced_20260910_200653/.../20260910_200730_independent_joint_tail_v2_event_balanced_seed2027_seed2027`
- checkpoint state：两者均使用 `raw/model_state_dict`，与 Original S5 来源一致。
- split：仅 `val`；Test 锁定。
- 条件、图、归一化、projection、500-step linear ancestral DDPM 均来自各自冻结 run。
- 最终成员预算：每个 seed 都是 400 Raw + 100 localized Tail。
- Original S5：训练事件持续时间分位数、最大联合修正能量单窗、最多 3 h 半余弦、窗口外严格 Raw。
- duration assignment seed：固定为 `424242`，两 coupling 臂完全相同。
- independent permutation seed：固定为 `424242`，每个 issue 独立置换 Raw 背景 401–500，Tail 成员顺序不变。

## 预注册 generation seeds

`271828, 314159, 161803`

看到结果后不得替换、增删 seed。

## 全路径噪声耦合

每个正态噪声块由以下 key 唯一决定：

`(generation_seed, validation_issue_index, global_member_block_start, diffusion_stage)`

- `diffusion_stage=500`：初始终端噪声；
- `diffusion_stage=499...0`：每个 ancestral reverse step 的噪声；
- Raw500 使用五个固定 100-member blocks：`0,100,200,300,400`；
- Tail100 使用 block `400`，因此 Tail 1–100 与 Raw 401–500 共享初始噪声和全部反向噪声；
- 噪声仍由目标设备上的 `torch.randn` 产生，只改变联合 coupling，不改变单个模型的标准正态边缘机制；
- 禁止 auto-tune member chunk；正式 `issue_batch_size=1`、`member_chunk_size=100`。

## 两个公平对照臂

每个 generation seed 只生成一套 Raw500 和一套 Tail100 边缘成员：

1. `crn_identity`：Tail `j` 与共享全随机流的 Raw `400+j` 配对。
2. `independent_permutation`：保持同一 Tail100 和 Raw500 数组不变，仅按 issue 使用固定随机置换选择 Raw 401–500 背景。

因此两臂的 Raw 主体池、Tail 边缘成员、条件、duration assignment 和总预算完全相同；唯一核心差异是 coupling。置换臂是有限样本 product-coupling 对照，不声称恢复历史独立生成的逐成员随机流。

## 预检门禁

正式生成前必须全部 PASS：

- keyed noise 对全局 RNG 扰动不敏感并可逐 bit 重放；
- Raw block 400 与 Tail block 400 的初始和所有反向 stage noise 一致；
- 不同 issue、stage、block 的 noise 不相同；
- 大样本 noise mean/std 与 N(0,1) 一致到预注册工程容差 `|mean|<=0.02, |std-1|<=0.02`；
- 默认模式仍使用历史 global RNG 语义；
- checkpoint/config/hash 和 val issue 顺序核对；
- CUDA/正式 dtype 上完成 replay；CPU 结果不能替代该门禁；
- CLI、路径、磁盘空间和 Test lock 检查；
- `training=NOT APPLICABLE`，无 optimizer、无 checkpoint 写回。

## 冻结评价

### Coupling/差值随机性

- `Tail-Raw background` 的全时域方差、均方差和容量加权能量；
- 12 h、24 h moving-average 低频差异；
- 跨 seed 均值、标准差及范围。

### 窗口与事件结构

- Original S5 最大能量窗与 Tail 自身结构参考窗的 overlap/IoU；
- 结构参考窗只用于事后诊断，使用既有 ESPC 的 train-only level/ramp/duration 统计，不进入 S5 选择；
- Full Tail 原命中到局部化后的 preserved/lost/gained；
- 逐事件 onset、duration、depth 和 member-hit。

### 最终概率分布

- Wind/Solar daylight/Renewable CRPS；
- Energy Score；
- 90% coverage、width、Interval Score；
- adjacency Variogram Score；
- overall spatial、wind–solar correlation RMSE；
- 时间相关结构和 1/3/6 h ramp 诊断。

## 解释门槛

三个结论分别判断，不相互替代：

1. `CRN降低差值随机性`：配对差值和低频主体差异跨三个 seed 方向一致下降。
2. `CRN提高事件保留`：四个事件逐事件展示；只允许说描述性方向，不以 4/4 宣称泛化。
3. `CRN改善最终概率质量`：ordinary/interval/multivariate 指标必须单独通过；事件改善不能替代。

只有当差值随机性明显下降、事件保留多数事件方向一致改善，且普通质量未出现系统性退化时，才把成员 pairing 视为 S5 的主要瓶颈。否则结论为“证据不足”或“主要瓶颈更可能是最大能量单窗/尾部轨迹本身”。

## 停止规则

完成三个 seed 后停止。不追加 seed、不改 S5、不调窗口或阈值、不训练、不访问 Test。最终只给一个下一步建议。
