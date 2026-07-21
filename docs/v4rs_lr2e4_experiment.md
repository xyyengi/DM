# V4-RS低学习率验证实验

本实验是单变量对照：相对原V4-RS，仅将学习率从`5e-4`降低为`2e-4`。
模型结构、500个扩散步、posterior反向方差、分通道残差标准化、训练轮数、
patience、batch size、随机种子均保持不变。不使用guidance、事件采样或事件标签。

训练结束后只在validation上生成20个场景，用于模型选择；本阶段不评价test。

## 后台启动

```bash
conda activate dm_env
mkdir -p logs
LOG=logs/v4rs_lr2e4_$(date +%Y%m%d_%H%M%S).log
nohup env PYTHONUNBUFFERED=1 bash run_v4rs_lr2e4_val.sh > "$LOG" 2>&1 < /dev/null &
echo $! | tee "${LOG%.log}.pid"
echo "$LOG"
```

查看日志：

```bash
tail -f "$LOG"
```

退出日志查看不会中止训练：按`Ctrl+C`只会退出`tail -f`。

检查进程：

```bash
PID_FILE="${LOG%.log}.pid"
ps -fp "$(cat "$PID_FILE")"
```

## 需要带回本地的结果

- `outputs_shandong/<时间>_v4rs_lr2e4_no_guidance_168h/`
- `outputs_shandong/<时间>_v4rs_lr2e4_no_guidance_168h_val_posterior_n20_seed2026/`
- 对应的`logs/v4rs_lr2e4_*.log`

先比较validation loss、分通道CRPS、80/90/95%覆盖率、区间宽度，以及负荷按误差大小
分组后的覆盖率。只有验证集结果优于原V4-RS，才使用该checkpoint生成test的50成员结果。
