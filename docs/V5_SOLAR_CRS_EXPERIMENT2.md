# V5-TF实验2：光伏条件残差中心与尺度

## 目的

实验2检验：只改变光伏残差的标准化方式，能否改善白昼和高出力光伏的条件离散度，同时保持风电、负荷和V5-TF网络结构不变。

基线：

```text
V5-TF + 三通道全局残差标准化
```

实验模型：

```text
V5-TF + 风电/负荷全局残差标准化
      + 光伏forecast条件残差中心和尺度
```

不改变：

- V5-TF网络结构、参数量和FiLM条件注入；
- 168 h三通道联合状态；
- diffusion timestep、500步线性噪声日程和posterior反向方差；
- forecast/calendar条件；
- 数据划分、训练seed和生成seed；
- 物理投影；
- 风电与负荷标准化。

## Train-only条件统计

只从train的7,296个唯一小时拟合，不按7,129个重叠窗口重复加权。

对光伏forecast大于零的唯一小时：

1. 按forecast分成12个等频区间；
2. 每档约354–355个时刻，且强制不少于128个；
3. 每档计算内部残差 `forecast - actual` 的均值和标准差；
4. 以16个样本的等效权重轻度向全局光伏统计量收缩；
5. 在档位中位forecast之间做连续线性插值；
6. 夜间使用原全局光伏统计量；
7. 风电和负荷继续使用原全局均值、标准差。

训练变换：

\[
z_t=
\frac{r_t-\mu_r(f_t)}
{\sigma_r(f_t)}.
\]

生成逆变换：

\[
r_t^{(m)}
=
\mu_r(f_t)+
\sigma_r(f_t)z_t^{(m)}.
\]

\[
actual_t^{(m)}
=
forecast_t-r_t^{(m)}.
\]

所有拟合统计量写入训练目录：

```text
residual_standardization.json
config_used.yaml
```

validation和后续test只能读取该train统计量，不能重新拟合。

## 配置与流水线

配置：

```text
configs/v5_tf_solar_crs_stage2_168h.yaml
```

服务器后台连续流水线：

```text
run_v5_tf_solar_crs_80_pipeline.sh
```

流水线自动执行：

1. CUDA、分支、git状态和数据预检；
2. seed 2027训练；
3. 保存top-3 checkpoint；
4. 使用rank-1 checkpoint；
5. 在validation生成80成员；
6. 使用generation seed 424242；
7. 保存原始及物理投影后场景；
8. 写入validation元数据和完成状态。

test不会运行。

## 服务器启动

在服务器已pull到包含本实验的提交、工作区干净且已进入`dm_env`后：

```bash
cd /root/autodl-tmp/DM
bash run_v5_tf_solar_crs_80_pipeline.sh
```

命令会立即返回PID、日志和状态文件，训练与生成在后台连续执行。

监控最新日志：

```bash
tail -f "$(ls -t logs/v5_stage2/v5_tf_solar_crs_80_pipeline_*.log | head -n 1)"
```

查看最新状态：

```bash
cat "$(ls -t logs/v5_stage2/v5_tf_solar_crs_80_pipeline_*.status | head -n 1)"
```

完成后查看路径：

```bash
cat "$(ls -t logs/v5_stage2/v5_tf_solar_crs_80_pipeline_*.results.env | head -n 1)"
```

需要下载：

1. `RUN_DIR`：训练配置、条件统计、checkpoint选择和日志；
2. `RESULT_DIR`：80成员validation场景和完整指标；
3. 对应的`.results.env`、`.status`、`.log`和`.environment.txt`。

## 评价顺序

必须与原V5-TF seed 2027、rank-1、posterior、80成员结果配对比较：

1. 总体CRPS、MVES、Energy Score；
2. 80%/90%/95%覆盖率及宽度；
3. 风电、全部光伏、白昼光伏、高出力光伏、负荷；
4. 高出力光伏Rank Histogram；
5. 上下越界比例、条件bias和spread/skill；
6. 分预测时距覆盖率；
7. ACF、1 h/6 h爬坡、净负荷和跨变量相关；
8. 原始物理越界率与投影后指标；
9. 训练时间和80成员生成时间。

优先成功条件：

- 高出力光伏90%覆盖率明显高于原79.82%；
- Rank Histogram两端下降，但不能变成中间过度凸起；
- 宽度、WIS和CRPS不能像实验1一样明显恶化；
- 风电、负荷和联合时序指标不发生实质退化。

若只提高覆盖率却显著增加宽度或恶化概率评分，应判定为未通过。
