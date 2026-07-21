# V4-C负荷动态条件验证实验

该实验在原V4-RS基线上只增加两个由forecast计算的条件通道：

1. `load_ramp_1h`：归一化预测负荷的一阶差分，窗口第一个小时填0；
2. `net_load`：物理一致的预测净负荷，在负荷归一化尺度上计算。

输入由14通道变为16通道。模型仍生成风、光、荷三通道残差；学习率恢复为`5e-4`，保留
500扩散步、posterior反向方差和分通道残差标准化。不使用guidance、事件采样或事件标签。

两个新条件使用train的7296个唯一小时拟合均值和标准差。validation和test只复用train统计量。

## 后台运行

```bash
git fetch origin
git merge --ff-only origin/experiment/v4-posterior-residual-standardization
conda activate dm_env
mkdir -p logs

LOG=logs/v4c_load_features_$(date +%Y%m%d_%H%M%S).log
nohup env PYTHONUNBUFFERED=1 bash run_v4c_load_features_val.sh > "$LOG" 2>&1 < /dev/null &
echo $! | tee "${LOG%.log}.pid"
echo "$LOG"
tail -f "$LOG"
```

按`Ctrl+C`只退出日志查看，不会停止后台任务。

训练完成后脚本自动在validation上生成20成员结果，不评价test。需要下载：

- `outputs_shandong/<时间>_v4c_load_ramp1_netload_no_guidance_168h/`
- `outputs_shandong/<时间>_v4c_load_ramp1_netload_no_guidance_168h_val_posterior_n20_seed2026/`
- 对应的`logs/v4c_load_features_*.log`

模型筛选必须同时比较validation loss、CRPS、Energy Score、分通道覆盖率、物理区间宽度、
异常物理范围、按负荷预测误差等级的覆盖率和提前期覆盖率。不能只根据总体覆盖率判断改善。
