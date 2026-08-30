# 24场站两专家事件评分上限实验

## 1. 实验定位

本实验不是对 L1 调权重，而是直接构造一个成功后再向下消融的上限候选。初始化仍为当前最佳 Raw Body-tail 检查点，保持主体—尾部两个专家，不增加第三专家、动态图、Transformer、历史 Top-K 平均或动态预测中心。

L1 已证明：四成员按约 9.3% 风险概率独立路由时，约 67.5% 的训练成员组没有一个尾部成员；全 168 h Energy Score 又主要由普通小时贡献，最终虽然整体分数微升，但五个持续深跌事件的命中成员从 46 减为 29。因此本实验同时处理“尾部训练缺席”和“主体时间路径冻结”两个已定位瓶颈。

## 2. 训练成员与损失

每个被训练集事件回放选中的发布窗口固定生成六个最终 DDIM 成员：

\[
\{\hat x_b^{(1)},\hat x_b^{(2)}\}
\cup
\{\hat x_t^{(1)},\ldots,\hat x_t^{(4)}\}.
\]

前两个强制走主体路径，后四个强制走已有尾部路径。固定配额只用于降低稀有事件训练估计量的方差；正式生成仍由只使用发布时可得信息的因果风险门控进行 Bernoulli 路由。

在真实事件窗口以及向两侧扩展 3、6、12 h 的四个尺度上，分别计算尾部成员联合 Energy Score，再取平均：

\[
\mathcal L_{\mathrm{tail,local}}
=\frac{1}{4}\sum_{c\in\{0,3,6,12\}}
\left[
\frac{1}{K_t}\sum_k\|v_c(\hat x_t^{(k)})-v_c(y)\|
-\frac{1}{2K_t(K_t-1)}\sum_{k\ne l}
\|v_c(\hat x_t^{(k)})-v_c(\hat x_t^{(l)})\|
\right].
\]

其中窄窗口约束突变本体，宽窗口同时约束事件前状态和事件后恢复，避免只学到整周统一下移。

对 13 个风电场的容量加权聚合功率，使用 1/3/6 h 时间 Variogram：

\[
\mathcal L_{\mathrm{VS,time}}
=\frac{1}{3}\sum_{h\in\{1,3,6\}}
\left(
\mathbb E_k|\hat X_{t+h}^{(k)}-\hat X_t^{(k)}|
-|Y_{t+h}-Y_t|
\right)^2.
\]

它同时作用于主体组与尾部组，直接惩罚爬坡方向、幅度和时刻关系错误。

完整训练目标为：

\[
\mathcal L=
\mathcal L_{\epsilon,\mathrm{event-tail}}
+\mathcal L_{\mathrm{gate}}
+\lambda_A\mathcal L_{\epsilon,\mathrm{natural-body}}
+\lambda_E\left(
\mathcal L_{\mathrm{tail,local}}
+\lambda_T\mathcal L_{\mathrm{VS,time}}
\right).
\]

自然发布样本的主体 epsilon 项用于保住普通规律；事件真值、事件窗口和固定路由都只存在于训练损失，不会进入生成条件。

## 3. 参数更新边界

- 更新：已有尾部适配器、尾部门控、条件编码器、时间 Res-UNet 编码器/瓶颈/解码器及输出层；
- 冻结：地理图、历史功率图、空间传播模块、状态编码器、扩散噪声日程和所有静态场站信息；
- 尾部学习率：`1e-4`；
- 时间主体学习率：尾部的 `0.15`，即 `1.5e-5`；
- 正式输出：最佳检查点 Raw 参数，500成员、种子424242、验证集23个发布窗口。

这不是冻结主体的保守微调，也不是全模型无约束覆盖训练，而是针对已确认的时间条件路径进行大范围解冻，同时用自然样本锚定。

## 4. 自动审计与判定

流水线自动检查：

1. 源检查点必须是 Raw Body-tail；
2. 必须严格为2主体＋4尾部训练成员；
3. 尾部和时间主体都必须发生参数更新；
4. 空间图和状态编码器必须逐张量保持不变；
5. 正式生成必须为验证集500成员且不得使用测试集；
6. 自动输出总体、逐提前日、事件时刻归因和Top5持续深跌对比。

上限候选成功至少需要同时满足：

- 五个持续深跌事件命中总数高于 Raw 的46，且5/5保持99%覆盖；
- 聚合1/3/6 h事件命中有可见而非千分位改善；
- 聚合风电90%覆盖率不低于 Raw 的87.09%；
- 风/光CRPS、联合Energy和空间相关不出现明显退化；
- 代表窗口的错位突变包络得到实质改善。

若成功，再依次消融时间 Variogram、多尺度窗口、固定尾部配额和时间主体解冻；若失败，则证明现有功率条件下即使显式提供尾部训练质量和时间梯度，预测遗漏事件仍无法稳定恢复，需要把研究边界转向条件信息上限或定向风险生成。

## 5. 服务器入口

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
bash run_station24_event_score_upper_bound_pipeline.sh
```

若训练完成后服务器在生成或后处理阶段关闭：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
bash run_station24_event_score_upper_bound_resume.sh \
  outputs_shandong/station24/event_score_upper_bound_<时间戳>
```
