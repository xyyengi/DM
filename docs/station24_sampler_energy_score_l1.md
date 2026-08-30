# 24场站最终成员 Energy Score 尾部微调（L1）

## 实验目的

本实验只验证一个机制：让训练目标直接观察同一发布条件下的一组最终场景，能否在不破坏现有主体分布的前提下改善风电覆盖和成员多样性。

它不是重新训练第三个专家，也不加入动态图、Transformer、时间 Variogram 或新的状态指标。

## 初始化与冻结边界

- 初始化：当前 Raw Body-tail 最佳检查点的 `model_state_dict`；
- 冻结：主体 Res-UNet、固定双图、条件编码器、状态编码器和预测中心；
- 更新：现有 tail station/common adapters、tail common gate 和 causal risk gate；
- 旧的 event-x0 幅度、时间与同步辅助损失在 L1 中全部设为零。

审计脚本逐张量比较源检查点和候选检查点。任何非尾部状态发生变化都会使流水线失败。

## 训练目标

现有尾部 epsilon 与风险门控训练保持为锚点，唯一新增项为最终成员 Energy Score：

\[
\mathcal L_{L1}
=\mathcal L_{\epsilon,\mathrm{tail}}+\mathcal L_{\mathrm{gate}}
+\lambda_{ES}\widehat{ES}.
\]

Energy Score使用一个独立的自然顺序训练数据流，不使用事件重放采样。对同一个发布条件，从四个独立终端噪声生成四条最终 DDIM 场景：

\[
\widehat{ES}
=\frac{1}{K}\sum_{k=1}^{K}\lVert \hat{x}^{(k)}-y\rVert
-\frac{1}{2K(K-1)}\sum_{k\ne l}
\lVert \hat{x}^{(k)}-\hat{x}^{(l)}\rVert,
\qquad K=4.
\]

评分在13个风电场的逐站标幺功率上计算，并按有效维数进行 RMS 归一化。第一项约束成员接近真实轨迹，第二项防止成员坍缩。四个成员分别依据现有因果风险门控作独立的硬路由，前向过程与正式生成一致；反向采用 straight-through Binary Concrete 近似，让 Energy Score 能训练现有门控。未来真实值和事件标签都不参与路由或去噪条件。

为控制32 GB GPU显存，采用8步确定性 DDIM，并只对最后2个反向步骤保留梯度；前6步产生真实采样状态但截断计算图。训练中每4个epsilon批次从自然数据流取一个发布样本计算最终成员评分，并乘以4进行频率校正，使期望梯度仍对应配置中的 \(\lambda_{ES}\)。验证对全部23个发布窗口计算同口径评分。

## 与 L2 的边界

L1针对“成员集合有没有合理覆盖与离散度”，不宣称直接解决事件时刻错位。只有当 L1 保持主体质量并改善覆盖，但滞后仍存在时，才另开 L2 加入聚合风电1/3/6小时的时间 Variogram。L2不在本次代码和配置中。

## 正式协议

- 数据：train训练、val选模与比较，test保持封存；
- 正式生成：验证集23个发布窗口，每个窗口500成员；
- 生成随机种子：424242；
- 候选输出：使用候选最佳检查点的 Raw 参数，避免再次混入此前已经证实会压制尾部的旧EMA现象；
- 对比基准：同一套500成员 Raw Body-tail结果；
- 输出：常规指标、逐提前日指标、风电事件时刻诊断、预测失配归因和持续深跌Top-5审计。

## 服务器入口

```bash
bash run_station24_sampler_es_l1_pipeline.sh
```

如自动检索不到 Raw Body-tail 结果，可显式指定：

```bash
SOURCE_RAW_RESULT=/root/autodl-tmp/DM/outputs_shandong/station24/<raw-result-path> \
bash run_station24_sampler_es_l1_pipeline.sh
```

进度查看：

```bash
STATUS=$(ls -t logs/station24/station24_sampler_es_l1_*.status | head -n 1)
LOG="${STATUS%.status}.log"
cat "$STATUS"
tail -f "$LOG"
```

若训练已经完成但服务器在生成或比较阶段关闭，可从原目录继续，不会重新训练：

```bash
bash run_station24_sampler_es_l1_resume.sh \
  outputs_shandong/station24/sampler_es_l1_<时间戳>
```
