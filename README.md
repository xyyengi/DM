# 24场站风光联合条件扩散模型

本分支用于24个新能源场站（13个风电场、11个光伏场）的168小时联合场景生成。输入是一日发布的未来7天预测轨迹，模型生成与该预测条件对应的场站级联合功率场景。

## 当前实验

空间消融由三个结构逐级组成，但三个模型分别从头训练：

1. `no_spatial`：共享时间ResUNet，无显式场站传播；
2. `fixed_graph`：在基线上加入固定地理邻接图；
3. `type_gated_graph`：在固定图上区分风—风、光—光和风—光关系门控。

三者使用相同的数据划分、训练参数、随机种子、扩散过程和验证集生成设置。测试集保持封存。

## 目录结构

```text
DM_local/
├── configs/                         # 三项空间消融配置
├── docs/                            # 实验口径、服务器命令和交接说明
├── src/
│   ├── eval/physical_projection.py  # 太阳位置与物理投影公共函数
│   └── models/                      # 24场站条件ResUNet扩散模型
├── tests/                           # 模型、数据、生成和评价测试
├── tools/                           # Stage 0诊断与三模型结果比较
├── station_dataset.py               # 场站数据读取、残差尺度和日照掩码
├── station_evaluation.py            # 边际、联合、空间和时间指标
├── train_station24.py               # 单模型训练入口
├── generate_station24.py            # 场景生成、物理投影与评价入口
└── run_station24_spatial_ablation_pipeline.sh
                                      # 三实验后台串行流水线
```

本地数据位于 `diffusion_input_station/`，训练输出位于 `outputs_shandong/station24/`。两者均不进入Git。

## 服务器一键运行

```bash
cd /root/autodl-tmp/DM
bash run_station24_spatial_ablation_pipeline.sh
```

脚本会在后台依次完成：

```text
无空间模型：训练 -> 验证集场景生成 -> 评价
固定图模型：训练 -> 验证集场景生成 -> 评价
类型门控图模型：训练 -> 验证集场景生成 -> 评价
三模型比较 -> 可视化 -> 结果打包
```

详细命令、监控方式和结果目录见 `docs/station24_spatial_ablation.md`。

## 历史空间先验双图实验

当前分支的下一阶段使用两个完全独立的固定双图候选：`geo_history_actual_dual` 与 `geo_history_residual_dual`。两者保留同一地理图，仅第二张训练集历史图不同，并自动执行训练、500成员验证生成、比较和打包：

```bash
bash run_station24_historical_dual_graph_pipeline.sh
```

设计边界、诊断依据、监控和停止命令见 `docs/station24_historical_dual_graph.md`。

## 数据口径

- 每个发布批次包含未来168小时、24个场站的预测与实测；
- 残差定义为 `actual - forecast`；
- 残差尺度只在训练集逐站拟合；
- 生成结果先逆变换并叠加预测，再执行 `[0,1]` 物理投影；
- 光伏夜间置零由发布时间、经纬度和太阳高度角计算，不读取实测功率；
- 当前流水线只使用train/validation，禁止使用test调参。
