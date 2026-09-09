# Station-24 Joint Multiresidual Tail V1 实验说明

## 目标

本实验不是继续降低预测条件强度，也不是增加第三个专家。它保留已经验证较好的 Raw body 作为正式基线，只替换原来过于粗糙的 tail：让同一个联合 tail 同时学习风电与光伏残差中尚未被 Raw body 解释的慢尺度结构和快尺度变化。

主要检验两个问题：

1. 显式拆解发布预测后，tail 能否减少因预测错位造成的极端时刻滞后；
2. 慢、快残差分工后，能否同时改善持续深跌的 duration/depth 和 1/3/6 h ramp，而不只把区间撑宽。

## 结构与因果边界

- 主体：继承 Raw body-tail 的 Res-UNet、地理图、训练历史图、状态调制、FiLM、日历和 recent-error，参数与 buffers 冻结并保持 eval 模式。
- 预测拆解：已知的 168 h forecast 被拆成 8 h Haar 低频、严格互补高频，并额外提供 24 h 日尺度趋势。预测不是继续以一条原始曲线直接输入 tail。
- 残差拆解：tail 的修正被正交投影为 slow 与 fast；两者严格相加恢复总修正，内积应接近零，不能靠互相抵消隐藏错误。
- slow：学习持续偏差、深度、持续时间和慢恢复，使用 8 h 正交低频与 24 h 结构约束。
- fast：学习 1/3/6 h 急升急降、突然深跌与快速恢复。
- 风光联合：一个 tail 输出完整 `[24 stations, 168 h]` 修正；风/光局部通道与系统公共模式先融合再生成，不把风、光拆成两个互不相关模型。
- 生成条件仅含发布时可得的 forecast、calendar、lead、recent-error、state 与固定图。未来 actual/residual 和事件标签只参与训练目标或评价，不进入生成。
- 不依赖失效的二分类事件门控。正式生成固定分配 15% tail 成员，内部修正仍是连续的场站×时间输出，不是发布级硬 mask。
- 训练时使用 outside-identity 约束：事件之外回到 Raw body，无事件窗口整段保持 body；它只约束学习，不在生成时使用真实事件 mask。

## 与 Raw body-tail 的直接差别

Raw body-tail 的 tail 只知道“该成员走 tail”，并允许整段 168 h 被一个修正器改变，且主要面向风电。V1 的 tail 显式读取 forecast 的慢/快形态，并把残差修正限制为数学上互补的慢/快两部分，同时联合建模风、光和系统公共变化。

因此，这一版确实可能改善有预测形态线索但发生时刻偏移的事件；如果 forecast 和历史功率条件完全不含对应信息，它仍不能准确预言唯一时刻，只能在 500 成员中学习合理的时刻与形态分布。

## 训练与选择

- 初始化：Raw body-tail `model_state_dict`，不是 EMA。
- 优化参数：仅 `joint_multiresidual_tail.*`，约 2.06 万参数。
- Raw body 处于 eval，避免冻结参数但 Dropout 仍随机的移动目标缺陷。
- 事件连续标签按真实 onset、duration、depth 构造；加权采样只作用训练集。
- 普通 epsilon、slow/fast 分解和风光系统结构共同选择检查点。
- 最大 300 epoch，`validation_every=5`、`patience=40`；最大 epoch 不是强制训练到 300，最终记录最佳与停止 epoch。
- 正式生成使用同一验证集 23 个窗口、500 成员、seed 424242、raw checkpoint state。

## 评价

- 普通质量：wind/solar/renewable CRPS、90% coverage/width、Energy Score、空间相关 RMSE。
- 日尺度：第 1–7 天分别报告风、光、总新能源指标。
- 快尺度：风、光、总新能源的 1/3/6 h ramp CRPS 与覆盖。
- 慢尺度：12/24 h moving-mean CRPS 与覆盖。
- 连续事件：宽松/主要/严格三档 onset、区间覆盖、depth 命中，并报告 duration/depth/onset 误差；all/body/tail 分开。
- 极端与滞后：持续深跌代表图、风电事件 timing 诊断。
- 风光联合：同一成员的 wind-solar power/residual correlation RMSE。

## 证据门状态（本机，2026-09-09）

- G0 因果接口与训练/生成用途分离：PASS。
- G1 forecast 精确重构、slow/fast 正交、route=0 零修正：PASS。
- G1 Raw route=0 逐点等价：PASS，最大绝对误差 0。
- G2 新 head、上游和系统 loading 梯度/更新：PASS。
- G2 optimizer 仅包含新 tail，冻结参数和 buffers 不变：PASS。
- G3 固定小批量 4 步损失下降：PASS（0.56865 → 0.56611）。
- 保存—严格重载：PASS。
- 单元/回归测试：PASS，46 项。
- Bash 语法检查：NOT RUN（本机无 bash）；服务器由 `set -euo pipefail` 和正式预检继续检查。
- CUDA/AMP：NOT RUN，必须在目标服务器通过。
- 完整 500-step 采样：NOT RUN，必须在目标服务器通过。
- 科学效果：NOT RUN，必须由正式 500 成员结果判断。

当前本机证据不等于实验成功；服务器脚本会先执行 CUDA/AMP、完整采样、检查点重载和因果生成预检，任一失败都会在正式训练前退出。

## 正式判定

只有同时满足下列条件才保留该方案：

1. 相比 Raw body-tail，持续深跌的 onset/duration/depth 或 1/3/6 h ramp 至少一组得到实质改善；
2. 另一组不能出现明显系统性退化；
3. wind、solar、renewable 普通 CRPS 和覆盖不能靠无控制增宽换取命中；
4. 风光联合相关结构不能显著劣化。

否则结论写为结构学习失败或证据不足，不把“能运行”描述为突破。
