# V4-s 山东事件感知训练运行说明

V4-s只改变训练Sampler：V4残差目标、forecast+时间编码、网络、loss和guidance=0均保持不变。

## 默认正式运行

```bash
git pull
conda activate dm_env
bash launch_shandong_v4s.sh
```

启动脚本会打印PID、日志和PID文件。查看日志：

```bash
tail -f logs/shandong_v4s_*.log
```

检查进程：

```bash
ps -fp "$(cat "$(ls -t logs/shandong_v4s_*.pid | head -1)")"
```

停止进程：

```bash
kill "$(cat "$(ls -t logs/shandong_v4s_*.pid | head -1)")"
```

## 资源不足时

训练batch和生成batch可以独立设置：

```bash
BATCH=32 GEN_BATCH=2 bash launch_shandong_v4s.sh
```

训练batch变化会改变优化过程；优先只降低`GEN_BATCH`解决生成阶段显存不足。

## 场景数量

默认生成50个场景：

```bash
NSAMPLES=50 GEN_BATCH=4 bash launch_shandong_v4s.sh
```

快速诊断可使用20个：

```bash
NSAMPLES=20 GEN_BATCH=8 bash launch_shandong_v4s.sh
```

最终论文如需稳定估计95%区间，建议生成100个，同时降低生成batch：

```bash
NSAMPLES=100 GEN_BATCH=2 bash launch_shandong_v4s.sh
```

`NSAMPLES`只影响训练完成后的生成和评价，不改变训练本身。

## 输出检查

每个run的以下记录必须保留：

- `config_used.yaml`
- `logs/event_sampler_audit.json`
- `logs/event_sampling_epochs.jsonl`
- `checkpoints/model_best.pt`
- `actual_scenarios.npy`
- `actual_data.npy`
- `metrics.json`
- `samples/scenarios.npz`

`event_sampler_audit.json`记录train-only阈值、事件数量、提前期窗口数量和Sampler设置；`event_sampling_epochs.jsonl`记录每个epoch实际定向抽样比例、类型分布和单事件最大抽取次数。
