# 24场站尾部事件时刻定位实验

## 1. 目的

同一最佳检查点的 Raw 参数复核已经确认：主体—尾部分工能够保持普通概率质量，并在固定 Top5 持续深跌事件上将尾部命中从 EMA 的 4 条提高到 44 条、事件覆盖从 3/5 提高到 5/5。但 Raw 尾部成员在完整 168 h 上仍比主体普遍偏低，说明现有发布级标量门控只回答“是否进入尾部”，不能回答“尾部事件发生在哪个小时”。

本实验不重训主体、不重学尾部幅度，只增加成员级事件时刻定位：

\[
z_m\sim\operatorname{Bernoulli}(\pi(c)),\qquad
\tau_m\sim\operatorname{Categorical}(q_\psi(\tau\mid c)),
\]

\[
\hat\epsilon_{m,t}
=\hat\epsilon_{\mathrm{body},t}
+z_m K(t-\tau_m)\Delta\hat\epsilon_{\mathrm{tail},t}.
\]

其中同一成员的事件中心 \(\tau_m\) 在13个风电场间共享，站级幅值和形态继续由已经训练完成的 Raw 尾部适配器给出。

## 2. 参数隔离

初始化必须使用已完成主体—尾部模型最佳检查点中的 `model_state_dict`，不得使用压弱尾部的旧 EMA 参数。

| 模块 | 本实验状态 |
|---|---|
| 历史—空间主体 Res-UNet | 冻结 |
| 地理图与训练集历史功率图 | 冻结 |
| 条件编码器、状态编码器 | 冻结 |
| Raw 站级/公共尾部适配器 | 冻结 |
| Raw 发布级风险门控 | 冻结 |
| 新增逐小时时刻头 | 训练 |

逐小时时刻头复用现有六信号风险编码器在全局池化前的 `[B,C,168]` 表示，经一层时间卷积输出168个时刻 logit。初始输出为零，因此初始时刻分布为均匀分布。

## 3. 训练监督与数据边界

训练集实际功率仅用于构造独立持续深跌事件的6 h窗口软标签：

\[
\mathcal L_{\mathrm{loc}}
=-\sum_{t=1}^{168}y_t^{\mathrm{event}}
\log q_\psi(t\mid c).
\]

实际功率是训练标签，不是条件。验证与未来生成时，时刻分布只使用发布时已经可获得的当前预测、预测爬坡、可对齐修订、预测状态与最近已观测误差。测试集保持锁定。

生成时对每条尾部成员独立采样时刻，并使用半径9 h的余弦渐消窗口限制尾部修正；主体成员的修正严格为零。

## 4. 一次后台流水线

服务器已经存在主体—尾部源实验、Raw 500成员结果和历史空间500成员基准时，执行：

```bash
bash run_station24_tail_time_localization.sh
```

脚本自动完成：

1. 输入、分支、源Raw检查点和验证协议审计；
2. 仅训练时刻定位头；
3. 使用同一最佳检查点的 Raw 与预热 EMA 各生成500成员；
4. 保存发布级尾部概率、成员路由、168 h时刻分布和每个尾部成员的采样中心；
5. 完成历史基准、本次Raw基准、定位Raw和定位EMA的配对比较；
6. 完成参数冻结、Top5局部化、1/3/6 h时刻和持续深跌审计；
7. 打包最终结果。

若自动发现失败，可显式传入：

```bash
bash run_station24_tail_time_localization.sh \
  outputs_shandong/station24/body_tail_moe_20260824_191036 \
  outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242
```

若服务器中断，第三个参数传入日志中已经创建的 `PIPELINE_ROOT`。脚本会跳过已经完成的训练、生成和诊断；未完成的目录会先移动到 `incomplete_*` 留档，再从该步骤重新运行：

```bash
bash run_station24_tail_time_localization.sh \
  <source_body_tail_root> <previous_raw_result> <pipeline_root>
```

## 5. 判定重点

不能只看总覆盖率。必须同时检查：

- Top5持续深跌是否仍至少覆盖4/5，目标5/5；
- 事件窗口内/窗口外尾部修正比是否提高；
- 完整168 h尾部—主体平均偏移是否显著收缩；
- 风电聚合90%覆盖率是否保留大部分Raw收益；
- 区间宽度和投影前负值率是否低于Raw；
- 1/3/6 h事件时刻是否不再系统恶化；
- 主体参数、Raw尾部参数和发布级门控是否逐张量保持不变。

本实验只验证“尾部时刻定位”这一项因果假设，不同时加入动态图、Transformer、新状态指标或新的尾部幅度损失。
