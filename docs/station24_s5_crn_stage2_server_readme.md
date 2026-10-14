# Station-24 Original S5：单seed CRN第二阶段服务器运行说明

本文件只准备代码与命令。本地Codex未运行GPU门禁、正式生成或正式评价。

## 冻结协议

- validation完整23个issue；Test锁定。
- generation seed仅`271828`，不得自动扩展。
- 生成Raw 500条：前400条作为最终主体，后100条作为S5背景。
- 生成Full Independent Tail V2 100条，global member id为400–499，与Raw背景共享全路径DDPM噪声。
- 同一Raw500/Tail100成员池比较：Full V2全时域混合、Original S5 CRN identity、3次固定随机置换配对。
- permutation seeds固定为`104729 / 130363 / 155921`。
- duration seed固定为`424242`，Original S5最大修正能量窗及3 h半余弦边缘不变。
- 无训练、不改checkpoint、不访问Test、不开发S5-V3。
- CUDA/AMP门禁与正式生成是两条独立命令；门禁成功后不会自动生成。
- 正式生成保持历史FP32推理语义；AMP只用于额外CUDA兼容门禁。

所有路径与运行参数在`configs/station24_s5_crn_single_seed.yaml`中配置，不含Windows绝对路径。若服务器目录不同，只修改该配置副本，不修改脚本。

## 1. 拉取代码

```bash
cd /root/autodl-tmp/DM
git status --short
# 若这里有未提交修改，先停止并备份，不要强行checkout。
git fetch origin experiment/24site-s5-espc
git checkout experiment/24site-s5-espc
git pull --ff-only origin experiment/24site-s5-espc
git rev-parse --short HEAD
```

激活服务器既有环境：

```bash
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHON_BIN=/root/miniconda3/envs/dm_env/bin/python
export PYTHONPATH="$PWD"
CONFIG=configs/station24_s5_crn_single_seed.yaml
bash -n run_station24_s5_crn_gpu_gate.sh \
  run_station24_s5_crn_single_seed_generate.sh \
  run_station24_s5_crn_single_seed_finalize.sh
```

## 2. 环境、checkpoint、配置与磁盘检查

```bash
"$PYTHON_BIN" -c 'import torch,yaml,numpy,pandas; print("torch",torch.__version__,"cuda",torch.cuda.is_available(),"gpu",torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
"$PYTHON_BIN" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --dry-run
df -h /root/autodl-tmp
```

上述验证失败时不要运行后续命令。

## 3. 单独运行GPU门禁：1 issue × 4 Raw/Tail pairs

```bash
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHON_BIN=/root/miniconda3/envs/dm_env/bin/python
export PYTHONPATH="$PWD"
CONFIG=configs/station24_s5_crn_single_seed.yaml
GATE="outputs_shandong/station24/s5_crn_gpu_gate_$(date +%Y%m%d_%H%M%S)"
bash run_station24_s5_crn_gpu_gate.sh "$CONFIG" "$GATE"
echo "GATE=$GATE"
```

门禁结束后查看：

```bash
cat "$GATE/STATUS.env"
cat "$GATE/phase1_audit.json"
"$PYTHON_BIN" -m tools.validate_station24_s5_crn_stage2 \
  --config "$CONFIG" --gate-report "$GATE/phase1_audit.json"
```

必须同时满足：顶层`status=PASS`、`launch_eligible=true`、`device=cuda`、`amp=true`、`target_server_CUDA_AMP=PASS`。任何FAIL或NOT VERIFIED都不得启动正式生成。

## 4. 人工启动单seed正式生成

确认门禁报告后，手动执行另一条命令：

```bash
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHON_BIN=/root/miniconda3/envs/dm_env/bin/python
export PYTHONPATH="$PWD"
CONFIG=configs/station24_s5_crn_single_seed.yaml

# 如果换了终端，重新定位刚才通过的门禁目录：
GATE=$(ls -dt outputs_shandong/station24/s5_crn_gpu_gate_* | head -n 1)
ROOT="outputs_shandong/station24/s5_crn_single_seed_$(date +%Y%m%d_%H%M%S)"

bash run_station24_s5_crn_single_seed_generate.sh \
  "$CONFIG" "$ROOT" "$GATE/phase1_audit.json"

echo "ROOT=$ROOT"
cat "$ROOT/STATUS.env"
```

该命令只完成Raw500和Tail100生成及完整性检查，结束状态应为`state=generation_complete`、`phase=generation_complete_evaluation_not_run`。它不会自动启动评价。

查看生成进展：

```bash
STATUS=$(ls -t outputs_shandong/station24/s5_crn_single_seed_*/STATUS.env | head -n 1)
cat "$STATUS"
LOG=$(awk -F= '$1=="log"{print $2}' "$STATUS")
tail -f "$LOG"
```

## 5. 人工启动统一评价、归因、汇总与打包

确认Raw500和Tail100均完整后：

```bash
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHON_BIN=/root/miniconda3/envs/dm_env/bin/python
export PYTHONPATH="$PWD"
CONFIG=configs/station24_s5_crn_single_seed.yaml
ROOT=$(dirname "$(ls -t outputs_shandong/station24/s5_crn_single_seed_*/STATUS.env | head -n 1)")

cat "$ROOT/STATUS.env"
bash run_station24_s5_crn_single_seed_finalize.sh "$CONFIG" "$ROOT"
cat "$ROOT/STATUS.env"
```

该阶段会构造并统一评价：

1. Raw500；
2. 400 Raw＋100 Full V2全时域Tail；
3. Original S5全路径CRN identity；
4. Original S5固定置换1；
5. Original S5固定置换2；
6. Original S5固定置换3。

评价包含普通CRPS、coverage/width、Energy Score、Interval Score、Variogram Score、持续事件、时序、极端尾部、时空及风光相关结构、代表性图、成员级事件保留和配对差值方差。最后生成报告包与完整包。

主要结果：

```bash
cat "$ROOT/summary/CRN_SINGLE_SEED_REPORT.md"
ls -lh "${ROOT}"_reports_*.tar.gz "${ROOT}"_completed_*.tar.gz
```

## 6. 失败恢复

- 三个脚本均不覆盖已有正式结果。
- 不完整的生成/评价目录会改名为`*_incomplete_TIMESTAMP`后重建。
- 生成成功、评价失败时，只重新运行`run_station24_s5_crn_single_seed_finalize.sh`，不会重新生成。
- 门禁失败时，正式生成器会再次验证门禁JSON并拒绝运行。
- 不要删除`EXPERIMENT_MANIFEST.json`、`config_used.yaml`、`STATUS.env`或源生成数组。

## 7. 规模估计

- 模型生成：23 issues × (Raw500＋Tail100) = 13,800条168 h轨迹。
- DDPM工作量：约6.9 million member-step。
- 三次随机置换不重新生成模型成员，只对同一成员池重新配对和局部化。
- 相似GPU历史日志下，纯生成约50–70分钟；门禁约1–3分钟；五个候选的完整CPU评价、绘图和打包约30–90分钟，依机器而异。
- 建议至少预留12 GiB空闲磁盘。预计未压缩结果约4–6 GiB，报告包通常小于1 GiB，完整压缩包约2–5 GiB；以实际压缩率为准。

完成单seed报告后停止。不得自动追加第二个generation seed、训练、S5窗口搜索或新方法。
