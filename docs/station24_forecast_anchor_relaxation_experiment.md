# 24场站预测条件锚定松绑实验

## 1. 实验问题

历史空间图与条件残差尺度已经改善了总体覆盖，但部分风电突升、突降仍表现为“场景跟着发布预测错时或过度平滑”。前序预测失配加权仅小幅改善覆盖和时刻表现，同时扩大区间并损害 CRPS，因此本实验不继续增加损失权重，而是直接检验网络是否过度依赖未来 168 h 发布预测条件。

本实验只改预测条件的使用方式，保持以下内容不变：

- 24 场站、168 h、验证集 23 个发布样本；
- Res-UNet、固定地理图与训练集历史实际功率图；
- 残差扩散目标及风电条件残差尺度；
- 500 个反向扩散步、训练种子 2027、生成种子 424242；
- 每个发布样本生成 500 个成员；
- 物理投影、评价口径与历史空间基线一致；
- 测试集保持封存。

## 2. 方法

### 2.1 训练时预测条件丢弃

对每个训练样本采样

$$
m \sim \operatorname{Bernoulli}(1-p_{drop}), \qquad p_{drop}=0.10.
$$

当 $m=1$ 时，模型使用完整发布预测；当 $m=0$ 时，屏蔽发布预测曲线及由它计算的低/高出力和爬坡状态。日历、提前时距、场站身份、历史误差和空间图仍保留，因此该分支准确地说是“预测中性分支”，不是完全无条件模型。

同一个网络同时学习

$$
\epsilon_{cond}=\epsilon_\theta(x_t,t,c_{forecast},c_{known}),
$$

和

$$
\epsilon_{neutral}=\epsilon_\theta(x_t,t,0,c_{known}).
$$

条件丢弃不增加可训练参数，也不需要训练第二个模型。

### 2.2 生成时条件强度扫描

反向扩散每一步使用

$$
\hat\epsilon_\gamma
=\epsilon_{neutral}
+\gamma\left(\epsilon_{cond}-\epsilon_{neutral}\right),
\qquad \gamma\in\{1.0,0.75,0.5\}.
$$

- $\gamma=1.0$：完整预测条件，用于区分“训练时条件丢弃”本身的作用；
- $\gamma=0.75$：轻度松绑；
- $\gamma=0.5$：较强松绑。

三种强度共享一个检查点。弱条件生成需要在每个反向步分别计算完整预测分支和预测中性分支，因此 $\gamma<1$ 的生成计算量约为完整条件生成的两倍。

## 3. 重要边界

当前重构公式仍为

$$
\hat y^{(k)}=f+s\odot\hat z^{(k)},
$$

其中 $f$ 为发布预测，$s$ 为训练集拟合的残差尺度。本实验只放松 FiLM/状态编码中的预测条件，不移除最终加回的 $f$。因此：

- 若滞后显著改善，说明网络内部的预测条件锚定是主要原因之一；
- 若几乎不改善，说明更强的锚定来自残差目标和 $f+\hat e$ 重构，下一步才有依据测试直接功率目标或双目标头；
- 不能仅凭区间变宽宣称成功，必须同时检查时刻误差、深跌成员命中和概率评分。

## 4. 自动评价

流水线对三种条件强度分别完成：

1. 风、光及逐站 80%/90%/95%/99% 覆盖率与区间宽度；
2. CRPS、Energy Score、相关性与爬坡指标；
3. 风电事件时刻诊断（1/3/6 h、提前日、单站与聚合）；
4. 发布预测锚定归因；
5. 持续深跌与尾部成员命中可视化；
6. 与历史空间基线的成对对比及完整归档。

判断优先级为：先看真实事件附近是否出现及时的成员，再看 90% 覆盖率和上下越界，最后检查 CRPS、区间宽度和联合相关性是否付出过大代价。

## 5. 一键后台运行

```bash
cd /root/autodl-tmp/DM
git checkout experiment/24site-forecast-anchor-relaxation
git pull --ff-only origin experiment/24site-forecast-anchor-relaxation
bash run_station24_forecast_anchor_relaxation_pipeline.sh
```

脚本会自行查找最近一次合规的 500 成员历史空间基线。若自动查找失败，可显式指定：

```bash
REFERENCE_HISTORY_RESULT=/完整/validation_results/geo_history_actual_dual_val_n500_seed424242 \
bash run_station24_forecast_anchor_relaxation_pipeline.sh
```

启动后终端会返回日志和状态文件路径；退出 SSH 或关闭本地电脑不会中断后台进程。
