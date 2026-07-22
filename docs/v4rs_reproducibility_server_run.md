# V4-RS可重复性验证：双种子实验

本实验保留原V4-RS的14通道输入和全部月份编码，不加入负荷动态特征、guidance、事件采样或
事件标签。模型、残差目标、学习率、500扩散步和posterior反向方差均不改变。

训练协议修正：

- 显式设置Python、NumPy、PyTorch、CUDA和DataLoader随机种子；
- 每个epoch使用相同的validation扩散时刻和噪声（固定种子`314159`）；
- validation不消耗后续训练的随机状态；
- 保存固定噪声validation loss最好的Top-3 checkpoint及清单；
- 两个训练重复使用相同的validation生成种子`424242`和20个场景成员。

## 更新服务器

```bash
git fetch origin
git merge --ff-only origin/experiment/v4-posterior-residual-standardization
conda activate dm_env
mkdir -p logs
```

## 第一次训练：seed 2026

```bash
LOG=logs/v4rs_repro_seed2026_$(date +%Y%m%d_%H%M%S).log
nohup env PYTHONUNBUFFERED=1 SEED=2026 bash run_v4rs_repro_val.sh > "$LOG" 2>&1 < /dev/null &
echo $! | tee "${LOG%.log}.pid"
echo "$LOG"
tail -f "$LOG"
```

等待第一次训练和validation生成完全结束后，再启动第二次。不要同时占用同一块GPU。

## 第二次训练：seed 2027

```bash
LOG=logs/v4rs_repro_seed2027_$(date +%Y%m%d_%H%M%S).log
nohup env PYTHONUNBUFFERED=1 SEED=2027 bash run_v4rs_repro_val.sh > "$LOG" 2>&1 < /dev/null &
echo $! | tee "${LOG%.log}.pid"
echo "$LOG"
tail -f "$LOG"
```

按`Ctrl+C`只退出`tail -f`，不会终止后台任务。

## 需要下载

每个种子对应一个训练目录和一个validation结果目录：

- `outputs_shandong/<时间>_v4rs_repro_seed2026_no_guidance_168h/`
- `outputs_shandong/<时间>_v4rs_repro_seed2026_no_guidance_168h_val_posterior_n20_genseed424242/`
- `outputs_shandong/<时间>_v4rs_repro_seed2027_no_guidance_168h/`
- `outputs_shandong/<时间>_v4rs_repro_seed2027_no_guidance_168h_val_posterior_n20_genseed424242/`
- 两份`logs/v4rs_repro_seed*.log`

训练目录中的`checkpoints/top_checkpoints.json`给出Top-3的epoch、固定validation loss和路径。
本轮先自动生成rank 1（`model_best.pt`）的20成员validation结果；如果两个种子仍不稳定，再对
Top-2/Top-3进行场景级复核，不预先增加六套生成任务。
