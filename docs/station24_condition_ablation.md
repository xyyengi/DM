# 24场站发布预测信息条件消融实验

## 实验目标

在当前 Fixed Graph 条件 ResUNet 扩散模型基础上，保持生成目标为：

```math
R = actual - current\_forecast
```

比较三种发布时可获得的信息条件：

| 变体 | 显式预测爬坡 | 当前版相对前一版修订量 | 上一发布日24h已实现误差 |
|---|---:|---:|---:|
| `revision_ramp` | 是，1/3/6h | 是，重叠144h | 否 |
| `history_ramp` | 是，1/3/6h | 否 | 是 |
| `revision_history_ramp` | 是，1/3/6h | 是 | 是 |

三类增量条件使用独立编码器和可学习门控，再与当前预测条件融合并用于各尺度 FiLM。它们不会通过固定公式直接修改预测值。

## 防止信息泄漏

- 当前发布日未来168h实际误差只作为扩散生成目标，不作为条件。
- 预测修订量只比较当前发布版本与前一日已经发布版本对同一有效时刻的预测。
- 两版7日预测仅有前144h重叠；最后24h修订量置零并通过 `revision_mask=0` 标记不可用。
- 近期误差只读取前一发布样本首24h已经实现的 `actual - forecast`。
- 若前一发布日缺失，则条件置零并通过可用性掩码标记。
- 流水线只使用 train/validation，test保持封存。

## 公平性设置

三项实验均使用：

- Fixed Graph 空间模块；
- 训练种子2027；
- 生成种子424242；
- 500步反向扩散；
- 每个验证发布批次80个24站联合场景；
- 相同训练超参数和早停规则；
- 80%/90%/95%概率区间；
- 相同物理投影。

## 服务器后台一键运行

```bash
cd /root/autodl-tmp/DM
bash run_station24_condition_ablation_pipeline.sh
```

命令会立即返回提示符。后台将严格依次执行：

```text
revision_ramp：训练 -> 80成员生成 -> 评价
history_ramp：训练 -> 80成员生成 -> 评价
revision_history_ramp：训练 -> 80成员生成 -> 评价
三模型比较 -> 可视化 -> tar.gz打包
```

关闭本地电脑或断开SSH不会终止后台任务。

## 查看进度

```bash
cd /root/autodl-tmp/DM
LOG=$(ls -t logs/station24/station24_condition_ablation_*.log | head -n 1)
tail -f "$LOG"
```

按 `Ctrl+C` 只退出日志查看，不会停止实验。

查看状态：

```bash
STATUS=$(ls -t logs/station24/station24_condition_ablation_*.status | head -n 1)
cat "$STATUS"
```

完成时应显示：

```text
state=completed
exit_code=0
```

## 停止整个流水线

```bash
cd /root/autodl-tmp/DM
PID_FILE=$(ls -t logs/station24/station24_condition_ablation_*.pid | head -n 1)
kill -- -"$(cat "$PID_FILE")"
```

该命令终止后台进程组，包括当前训练或生成子进程。

## 新增评价指标

除CRPS、80%/90%/95%覆盖率、区间宽度、Energy Score、Variogram Score、空间相关性和时间自相关外，本流水线还自动计算：

- 风电、光伏1h/3h/6h爬坡CRPS与覆盖率；
- 实际绝对爬坡幅值前10%的极端爬坡覆盖率；
- 风电、光伏每天实际最高出力时刻的概率区间覆盖率；
- 极端高峰超过场景上界的平均幅度；
- 条件门控训练后的实际数值；
- 预测修订和近期误差条件的可用样本审计。

## 完成结果

最终日志旁生成 `.results.env`，其中记录三个训练目录、三个生成评价目录、比较目录和压缩包路径。

压缩包形式：

```text
outputs_shandong/station24/station24_condition_ablation_YYYYMMDD_HHMMSS.tar.gz
```

下载该压缩包即可包含完整训练、场景、指标、CSV、可视化和模型比较结果。
