# Station-24 JSTD 连续事件假设 H1 实验

## 实验目的

H1 不是可部署模型，也不是用验证集真实值“作弊”获得正式结果。它是一次结构可控性上界实验，用来回答：

> 当 Tail 已知事件大致何时开始、持续多久、风光修正方向与幅度以及同步程度时，现有共享 slow/fast Tail 能否在指定时空区域生成合理的联合极端场景？

若答案是否定的，应停止当前 JSTD Tail 路线；若答案是肯定的，才值得在下一阶段研究如何仅凭 train-only 先验和当前可得功率条件采样这些事件假设。

当前验证集按 train 阈值仅识别出 4 个事件窗口，因此这次结果只能用于淘汰或保留机制，不能单独证明泛化能力、论文结论或工程收益。

## 与 JSTD-Tail V1 的结构差别

Raw body、图结构和原条件调制全部保持冻结。JSTD V1 的发布级 Bernoulli 门控不再训练，也不参与生成决策。新增的不是第三专家，而是共享 Tail 内的连续事件条件：

新训练运行直接复制并冻结 JSTD V1 已保存的
`graphs/secondary_adjacency.npy`，不会重新拟合历史功率图，也不会退回单地理图。

\[
z=(e,\tau,d,a_w,a_s,u),
\]

其中：

- \(e\)：是否为事件假设；
- \(\tau\)：归一化 onset；
- \(d\)：归一化实际 duration；
- \(a_w,a_s\)：风电、光伏有符号深度；
- \(u\)：事件源场站的容量加权同步程度。

模型内部将六维向量展开为连续的事件时间包络、onset/offset 边界和风光有符号时空场，再分别注入 slow 与 fast 路径。没有把真实的 24×168 residual 或真实 mask 直接作为输入。

## 训练与生成协议

- 初始化：JSTD-Tail V1 的 Raw `model_state_dict`；
- 冻结：Raw body、原条件编码、图结构和失败的 issue gate；
- 更新：共享 JSTD slow/fast Tail 与新事件假设编码器；
- 训练事件：只由 train residual 构造；每个发布窗口选择一个最强的连续物理事件；
- 普通训练窗口：使用零事件假设，约束 Tail 回到零修正；
- 验证生成：只允许 `val`，并必须显式传入 `--allow-oracle-event-hypothesis`；
- 验证混合：每个真实事件窗口 500 个成员中固定约 10% 走假设 Tail，其余保持 body；非事件窗口全部走 body；
- test：禁止运行 H1 oracle。

H1 结果元数据必须标记：

```text
future_actual_used_as_generation_condition=true
reportable_as_causal_forecast=false
```

## 新增的定位约束

除原 slow/fast 分解、mask 和系统级结构损失外，H1 对事件区间外的 Tail 等效 \(x_0\) 修正施加零约束，使事件结束后恢复到 body。该约束只抑制指定事件范围外的 Tail 泄漏，不改变 Raw body。

## 判定规则

重点读取连续事件评价中的 `independent_physical + tail + primary`：

1. 4 个验证事件的 `events_with_any_hit` 是否高于 Raw；
2. tail 成员的 onset、duration、depth 是否同步改善；
3. issue 16 和 issue 22 的严重错位是否显著缩小；
4. 非事件窗口必须完全走 body；
5. 事件窗口中 body 成员应与 Raw 主体基本一致。

判读时同时使用两个参照：Raw body-tail 是正式基线，JSTD-Tail V1 是 H1
的直接父模型。只有优于直接父模型，才说明“连续事件假设”本身提供了新增
控制能力，而不是重复计入 JSTD V1 已经取得的收益。流水线会在
`h1_result_audit/` 中核验 oracle 标记、非事件零 Tail 路由、固定 Tail 配额，
并汇总相对两个参照的事件和普通质量指标。

判定：

- **H1 成功**：指定事件的 Tail 命中和 onset/duration/depth 明显改善，且 body 保持；进入 train-only 事件假设采样阶段。
- **H1 局部成功**：时间定位改善但幅度或持续时间仍差；只修改假设到修正的映射，不研究事件检索。
- **H1 失败**：给出 oracle 假设后仍无法定位；停止 JSTD Tail 路线，不再增加门控、检索或 CFG。

## 中断恢复

主流水线会完成训练、500成员生成、双基线评价和打包，且脚本内部会自动激活
`dm_env`。如果服务器在训练完成后中断，可用
`run_station24_jstd_event_hypothesis_h1_finalize.sh` 指向已有 pipeline 根目录。
它会复用 `model_best.pt`；完整生成结果直接复用，不完整的生成目录会保留并改用
带 `resumed` 后缀的新目录，不覆盖已有文件。打包只生成 `tar.gz`，不计算
SHA-256。
