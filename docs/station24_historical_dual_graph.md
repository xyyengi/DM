# Station24 历史空间先验双图实验

本阶段只回答一个问题：在当前固定地理图已经存在的前提下，训练集历史实际功率关系和训练集标准化预测残差关系，哪一种能提供额外、可验证的空间信息。

## 1. 只读诊断结论

诊断严格只使用训练集。训练集 290 个发布样本中的重叠实际功率按目标时刻去重后，共得到 7,224 个唯一小时；验证集和测试集实际值均未参与构图。

- 历史实际功率相关：风—风平均 Pearson 相关系数为 0.483，光—光为 0.715。
- 条件标准化残差相关：风—风平均为 0.265，光—光为 0.544。
- 地理图与历史实际功率候选图的无向边 Jaccard 重合率约为 0.420。
- 地理图与标准化残差候选图的无向边 Jaccard 重合率约为 0.397。
- 历史实际功率图与标准化残差图的边重合率约为 0.744。
- 200 次分块 bootstrap 的正式审计与既有统计口径一致，参考复现关卡通过。

因此，两个历史候选都不是地理图的简单副本，满足进入独立训练比较的条件。低尾共同发生图、滞后边和动态图在本轮均不接入模型。

## 2. 两个实验严格分离

| 实验名称 | 第一张图 | 第二张图 | 配置 |
|---|---|---|---|
| `geo_history_actual_dual` | 固定地理图 | 训练集历史实际功率图 | `configs/station24_geo_history_actual_dual_168h.yaml` |
| `geo_history_residual_dual` | 固定地理图 | 训练集条件标准化残差图 | `configs/station24_geo_history_residual_dual_168h.yaml` |

两份配置除实验身份和第二张图来源外完全一致。训练、检查点、500 成员场景、评价结果和比较目录均使用完整变体名称，不使用字母编号。

## 3. 双图融合

每个既有图传播位置使用同一特征投影，只对两张固定图做凸组合：

\[
A^{mix}=\pi_{geo}A^{geo}+\pi_{hist}A^{hist},\qquad
(\pi_{geo},\pi_{hist})=\operatorname{softmax}(\alpha_{geo},\alpha_{hist}).
\]

初始化取 \((\alpha_{geo},\alpha_{hist})=(2,0)\)，即地理图权重约 0.881，历史图权重约 0.119。瓶颈图传播和 `encoder_0` 并行空间分支各增加两个 logit，总计仅增加 4 个标量参数；原 ResUNet、FiLM、State V1、条件残差尺度和损失函数不变。

如果第二张图也等于地理图，因为两权重之和恒为 1，传播结果与单地理图严格相同。这为后续按需补跑 `geo_geo_dual_control` 提供了干净的参数归因控制。

## 4. 服务器执行

提交并推送、服务器拉取本分支和代码后，只运行：

```bash
cd /root/autodl-tmp/DM
bash run_station24_historical_dual_graph_pipeline.sh
```

脚本将按以下顺序自动完成：训练集空间先验复算与审计 → `geo_history_actual_dual` 训练和 500 成员验证生成 → `geo_history_residual_dual` 训练和 500 成员验证生成 → 三组配对比较 → 打包。测试集保持封存。

查看日志：

```bash
LOG=$(ls -t logs/station24/station24_historical_dual_graph_*.log | head -n 1)
tail -f "$LOG"
```

查看状态：

```bash
STATUS=$(ls -t logs/station24/station24_historical_dual_graph_*.status | head -n 1)
cat "$STATUS"
```

停止整个后台流水线：

```bash
PID_FILE=$(ls -t logs/station24/station24_historical_dual_graph_*.pid | head -n 1)
kill -- -$(cat "$PID_FILE")
```

完成后查看下载路径：

```bash
RESULT=$(ls -t logs/station24/station24_historical_dual_graph_*.results.env | head -n 1)
cat "$RESULT"
```

## 5. 决策关卡

优先比较聚合风电覆盖、同步深跌命中、场站覆盖与 CRPS，再检查空间相关误差、Variogram/Energy Score 和光伏是否退化。若两个历史图都不能超过当前 500 成员地理图基准，或覆盖收益仅来自区间无差别变宽，则停止历史图路线，不继续叠加尾部图、滞后边或动态图。只有出现明确正向候选时，才补跑 `geo_geo_dual_control` 做参数归因。
