# 24场站风光联合扩散模型：空间消融实验说明

## 模型口径

历史V5-TF的准确名称是“基于FiLM条件调制的一维残差U-Net（Conditional ResUNet-1D）”。它同时具有：

- U-Net编码器、瓶颈、解码器和跨尺度跳连；
- 每一级内部的残差块；
- 扩散时间步与序列条件FiLM调制。

新的24场站模型保留上述结构，但将场站保留为独立空间维，不把24站简单压成普通特征通道。时间卷积参数在24站间共享。

## 三项严格消融

| 实验 | 配置 | 空间处理 | 参数量 |
|---|---|---|---:|
| 无空间基线 | `configs/station24_no_spatial_168h.yaml` | 无显式场站传播 | 735,905 |
| 固定图 | `configs/station24_fixed_graph_168h.yaml` | 瓶颈处一次固定地理图传播 | 735,906 |
| 类型门控固定图 | `configs/station24_type_gated_graph_168h.yaml` | 固定地理图加风—风、光—光、风—光三个门控 | 735,908 |

三份配置除实验说明、`spatial_mode`和门控初始化外完全一致。总流水线启动前会自动检查配置公平性。

共同训练设置：

- 训练集290个发布批次，验证集23个发布批次；
- 测试集锁定，不在流水线中读取；
- batch size 8；
- 梯度累积2次，有效batch size 16；
- AdamW，学习率 `1e-4`，weight decay `1e-4`；
- 最多300 epoch，每5 epoch固定噪声验证；
- early stopping patience 40 epoch；
- 梯度裁剪1.0，EMA 0.999；
- 500个扩散步；
- 每个验证发布批次生成80个联合场景；
- 训练种子2027，生成种子424242。

残差严格定义为：

```math
R = actual - forecast
```

按训练集逐站残差标准差做尺度平衡，不减均值。生成后执行逆尺度变换，并按下式还原：

```math
scenario = clip(forecast + generated_residual, 0, 1)
```

光伏场站随后使用各自经纬度、有效时间和太阳高度角执行夜间置零。该规则不读取实测功率。

## 服务器一键运行

```bash
cd /root/autodl-tmp/DM
bash run_station24_spatial_ablation_pipeline.sh
```

该命令立即返回提示符，三个实验会在独立后台进程组中严格串行执行。每个实验都会自动完成训练、验证集80成员生成和评价；最后自动生成三模型比较并打包。

## 查看进度

```bash
cd /root/autodl-tmp/DM
LOG=$(ls -t logs/station24/station24_three_experiments_*.log | head -n 1)
tail -f "$LOG"
```

退出实时查看但不停止实验：按 `Ctrl+C`。

查看状态：

```bash
cd /root/autodl-tmp/DM
STATUS=$(ls -t logs/station24/station24_three_experiments_*.status | head -n 1)
cat "$STATUS"
```

## 停止整个流水线

```bash
cd /root/autodl-tmp/DM
PID_FILE=$(ls -t logs/station24/station24_three_experiments_*.pid | head -n 1)
kill -- -"$(cat "$PID_FILE")"
```

这里终止的是整个后台进程组，包括当前训练或生成子进程，不会留下多个 `train_station24.py`。

## 完成后的结果

日志目录会生成一个 `.results.env`，其中记录：

- 三个训练目录；
- 三个验证集生成结果目录；
- 总比较目录；
- 最终压缩包路径。

查看：

```bash
cd /root/autodl-tmp/DM
RESULT=$(ls -t logs/station24/station24_three_experiments_*.results.env | head -n 1)
cat "$RESULT"
```

最终压缩包名称类似：

```text
outputs_shandong/station24/station24_three_experiments_YYYYMMDD_HHMMSS.tar.gz
```

比较结果包括逐站、分类型、分提前日、风/光/新能源总量指标，以及Energy Score、邻接Variogram Score、空间相关矩阵误差、时间自相关误差、物理越界率和典型验证窗口90%包络图。
