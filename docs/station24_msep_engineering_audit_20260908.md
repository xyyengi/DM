# Station-24 MSEP 工程与 slow/fast 审计（2026-09-08）

## 结论先行

当前不能把 MSEP 的失败简单解释为“模型上限低”，也不能原样重跑。已定位到一个会直接破坏学习的实现缺陷：三个连续属性尺度头在初始化处被硬截断，梯度为零。与此同时，MSEP V1 将 H1 renderer 与新因果先验联合训练，renderer 训练时读取真实事件参数、正式生成时读取先验采样参数，导致先验失败与 renderer 漂移混在一起。

slow 与 fast **确实进入了计算图、产生非零修正并收到梯度**；目前没有发现二者数值抵消。可是尚无配对的最终采样消融，因此还不能说 slow 或 fast 分别改善了最终500成员。下一次付费实验必须先通过新版本预检，且先冻结 H1 renderer、只训练事件先验。

## 1. 已定位的实现缺陷

MSEP V1 对 duration、wind depth、solar depth 的尺度采用：

\[
\sigma=\operatorname{clamp}(\operatorname{softplus}(a),0.02,0.35).
\]

属性头零初始化时，\(\operatorname{softplus}(0)=0.693>0.35\)，输出被截为0.35；PyTorch 在该截断区对 \(a\) 的导数为0。直接检查得到尺度参数梯度为0，意味着训练不能从初始尺度0.35学回合理范围。duration 的0.35按168 h换算相当于约58.8 h标准差，与训练事件持续时间中位数7 h明显不匹配。

修复采用显式版本字段，旧配置缺失字段时仍走 `legacy_clamp`，确保历史检查点不会被悄悄重新解释。新路径 `transformed_normal_v2` 使用：

- duration：logit-normal，初始中位数7 h；
- signed wind/solar depth：tanh-normal；
- 尺度：平滑有界 sigmoid 映射，初始化和梯度均可检查；
- 采样：通过 sigmoid/tanh 自然落在物理范围内，不再先采高斯再硬裁剪。

## 2. MSEP V1 的训练混杂

V1 更新了整个 JSTD tail，包括已经由 H1 训练过的 slow/fast renderer 和新 segment prior。训练 renderer 时传入真实 `jstd_segment_hypotheses`，正式生成时却传入 segment prior 采样。这样会同时出现两个变化：

1. 因果先验是否学会 onset、duration、depth；
2. H1 renderer 是否在真实提示下继续漂移。

最终结果变差时无法区分是哪一项造成的。新代码加入 `train_jstd_segment_prior_only`：只让 `jstd_tail.segment_prior.*` 进入 optimizer，H1 renderer 的参数和 buffers 在微型优化前后必须逐项一致。先验验证目标也改为结构化的 `count + onset + mark`，不再由总扩散损失替它选择检查点。

## 3. slow/fast 是否发挥作用

在已下载的 MSEP 最佳 raw checkpoint（epoch 65）上，对4个验证事件窗口进行了标签辅助的离线 epsilon 前向/反向审计。该审计只判断分支是否参与，不把验证真实标签当作可部署条件。

| 扩散步 | slow RMS | fast RMS | slow/fast余弦 | 合成能量/分支能量和 | fast的12 h低频占比 | slow梯度 | fast梯度 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.0134 | 0.0294 | 0.003 | 1.002 | 20.9% | 0.0330 | 0.0388 |
| 100 | 0.0261 | 0.0457 | 0.061 | 1.053 | 21.4% | 0.0584 | 0.0686 |
| 300 | 0.0189 | 0.0151 | 0.074 | 1.072 | 26.8% | 0.1264 | 0.0166 |

由此可得：

- 两个分支都不是死分支；raw head 与 mask head 均收到非零梯度；
- slow/fast 余弦接近0且略为正，合成能量不低于分支能量和，没有发现抵消；
- fast 仍含21%～27%的12 h低频成分，说明移动平均的互补分解不是严格频谱正交；
- 冻结参数没有梯度；
- 最终采样的 `body / body+slow / body+fast / full` 配对消融尚未执行，因此“哪个分支改善了CRPS或事件命中”仍是 **NOT RUN**。

原始机器可读证据位于：

`outputs_shandong/station24/jstd_msep_20260907_222239/local_audit/branch_gradients.json`

## 4. 日志缺口与修复

V1 训练历史只记录总 `train_loss/val_loss`。内部已有的 epsilon、decomposition、mask、structure、outside-zero 以及 segment count/onset/mark 没有写入日志，导致“总损失下降”掩盖事件先验失败。

训练器现已为 JSTD 实验逐epoch记录以下分量：

- `diffusion_epsilon`；
- `diffusion_jstd_decomposition`；
- `diffusion_jstd_mask`；
- `diffusion_jstd_structure`；
- `diffusion_jstd_outside_zero`；
- `segment_count`、`segment_onset`、`segment_mark`、`segment_total`。

同时检查点和训练总结新增实际 optimizer 参数名、是否 prior-only、尺度参数化版本，避免把“结构上属于JSTD的参数”误写成“本次真正训练的参数”。

## 5. 尚未解决、不能伪装为已解决的问题

- 500成员最终采样中的 slow/fast 单独贡献没有证据；
- 两个事件槽独立采样，可能出现重叠；
- duration、wind depth、solar depth 当前为条件独立分布，未显式保留联合相关性；
- 固定10% tail配额不使用事件数头的 \(p_0\) 决定route；这是刻意分离“尾部预算”和“事件结构”，但 \(p_0\) 只能作为校准诊断，不能被表述为实际发生概率；
- 已建立独立本机CPU环境 `dm_preflight`；新路径单元测试、固定批次拟合、冻结状态、保存重载和3成员短生成均已通过。CUDA/AMP仍须在目标服务器验证。

## 6. 付费服务器启动门

当前状态：**不允许直接开展下一次正式训练**。

正式训练前必须依次满足：

1. 新配置显式使用 `transformed_normal_v2` 和 `train_jstd_segment_prior_only=true`；
2. CPU单元测试与固定小批量10步拟合通过；（已PASS：52项相关测试，损失2.9809降至2.0419）
3. 六个尺度通道梯度均非零，count/onset/attributes/encoder四组参数均实际更新；
4. 冻结 H1 renderer 的参数与buffers逐项不变；
5. 目标服务器CUDA+AMP执行同一测试、保存重载和3成员短生成；（NOT RUN）
6. 预检报告中必需项全部PASS，任何FAIL或必需NOT RUN都由脚本非零退出，停止正式训练。

即使这些工程门全部通过，也只证明“实验值得跑”，不保证科研指标一定改善。
