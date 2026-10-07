# 91节点可靠通道数据：模型项目接入交接说明

## 应复制到实验项目的内容

将整个 `reliable_channel_training_v1` 目录复制到目标实验项目，例如：

```text
<目标项目根目录>/data/shandong91/reliable_channel_training_v1/
```

不要只复制 `windows168`。模型接入还需要：

- `windows168/`：训练、验证、测试的168小时样本；
- `static/`：91节点顺序、图结构、资源存在mask、训练通道mask和节点特征；
- `normalization_params.json`：反归一化到MW；
- `preprocessing_metadata.json`：数据语义和处理规则；
- `quality_report.json`：20项质量检查结果；
- `split_summary.json`：数据划分和shape；
- `output_manifest.json`：文件shape、dtype和SHA-256；
- `reports/`：缺失修复和mask审计；
- `hourly/`：调试、重建窗口和省级聚合检查；
- `15min/`：追溯与重新聚合使用，训练代码通常不直接读取。

如果只做模型运行且存储空间非常紧，可不复制 `15min/`；除此之外建议整目录复制。

## 复制后的推荐结构

```text
<目标项目根目录>/
  data/
    station24/                         # 原24场站数据，不动
    shandong91/
      reliable_channel_training_v1/   # 本次整目录
  configs/
    station24/                         # 原配置，不动
    shandong91/                        # 新增
  datasets/
    ...                                # 原24场站读取代码不动
    shandong91_reliable.py             # 新增91节点读取入口
  scripts/
    train_station24...                 # 原脚本不动
    train_shandong91...                # 新增独立脚本
```

## 可直接交给目标项目 Codex 的指令

以下内容从“任务”开始完整复制给目标项目中的 Codex：

---

### 任务

请在当前实验项目中接入“山东91节点风光荷可靠通道训练版”数据，为后续168h风光荷联合场景生成实验建立独立实验线。

数据目录：

```text
data/shandong91/reliable_channel_training_v1/
```

### 一、Git和隔离要求

1. 先检查当前Git状态、现有分支和未提交修改。
2. 在保留用户现有修改的前提下，新建并切换到独立分支：

```text
feat/shandong91-reliable-channel
```

3. 不允许修改、覆盖、移动或重命名任何24场站数据、配置、训练脚本、checkpoint和实验结果。
4. 91节点实验必须使用独立的数据集类、配置、运行脚本、日志目录、checkpoint目录和结果目录。
5. 如果当前工作区有无法安全隔离的未提交改动，先报告，不要reset、checkout丢弃或覆盖用户修改。

### 二、本轮工作范围

本轮只完成数据接入和模型兼容性改造，不正式训练模型，不更改扩散过程、噪声调度、采样算法、主损失原理或原有24场站实验逻辑。

允许因数据结构变化进行必要适配：

- 从24场站/扁平通道改为91节点、3资源通道；
- 加入物理图结构输入；
- 加入node_type_mask、channel_train_mask、valid_mask；
- 调整输入输出shape；
- 增加91节点专用配置与数据加载器；
- 增加masked loss和masked metrics；
- 增加基于normalization_params.json的MW反归一化。

不得把这些适配反向写入或破坏24场站默认路径。

### 三、必须先阅读的数据说明

接入前读取并核对：

```text
preprocessing_metadata.json
normalization_params.json
split_summary.json
quality_report.json
output_manifest.json
static/node_feature_columns.json
```

先验证`quality_report.json`中的20项检查全部通过，并按`output_manifest.json`检查训练需要的文件存在、shape正确。除非用户明确要求，不重新清洗数据，也不重新诊断异常倍率。

### 四、数据定义

动态张量维度统一为：

```text
[batch, 168, 91, 3]
```

资源通道固定为：

```text
0 = Wind
1 = Solar
2 = Load
```

窗口数据位置：

```text
windows168/train/
windows168/validation/
windows168/test/
```

每个划分读取：

```text
actual.npy
forecast.npy
residual.npy
actual_valid_mask.npy
forecast_valid_mask.npy
residual_valid_mask.npy
time_mark.npy
window_position.npy
window_starts.csv
```

其中：

- `actual/forecast/residual.npy`是normalized数据；
- 对应`*_mw.npy`是MW数据，仅用于检查、评价和结果导出；
- `residual = actual - forecast`；
- `forecast`是按目标timestamp拼接的滚动预测轨迹，不是真实一次性发布的7天预测；
- `window_position=0..167`只是窗口内位置，不是真实forecast lead time。

静态数据读取：

```text
static/node_order.csv
static/node_features.npy
static/node_type_mask.npy
static/channel_train_mask.npy
static/adjacency_binary_no_self_loop.npy
static/adjacency_binary_with_self_loop.npy
static/edge_index.npy
static/edge_features.npy
```

不要重排节点。数组位置0对应node_id=1，位置90对应node_id=91。

### 五、模型输入输出原则

联合场景生成的核心原理保持现有项目方案：以forecast为条件，学习或生成`residual = actual - forecast`，最后重构：

```text
generated_actual = forecast + generated_residual
```

优先保持模型输出为：

```text
[batch, 168, 91, 3]
```

如果现有模型只能处理扁平特征，可在91节点专用adapter内部转换：

```text
[B,168,91,3] <-> [B,168,273]
```

但必须在数据集边界保留91节点与3资源通道语义，不得永久丢失节点维度，也不得改变固定节点顺序。

图网络如非当前模型必需，可先把图结构完整加载并预留接口，不要为了本轮接入强行重构扩散模型主干。

### 六、Mask规则

训练监督有效位置必须为：

```python
effective_mask = (
    residual_valid_mask
    * channel_train_mask[None, None, :, :]
    * node_type_mask[None, None, :, :]
)
```

扩散噪声预测或残差预测损失必须逐元素计算，再执行：

```python
masked_loss = (element_loss * effective_mask).sum() / effective_mask.sum().clamp_min(1)
```

要求：

1. 不能先对全部元素求mean再乘mask。
2. 评价指标使用同一有效范围。
3. 数据中无效位置的存储值为中性值0，但不能被当成真实零功率。
4. 模型条件输入若包含无效位置，应同时提供对应mask，或明确将无效位置清零并把mask作为额外条件；不能让模型猜测0是无效值还是真实零值。
5. `node_type_mask=0`表示物理上不存在该资源，不能参与loss。
6. `channel_train_mask=0`表示该历史通道不提供可靠监督，但对应物理节点仍然存在。
7. 未来生成不能因为历史`valid_mask=0`永久删除节点。

### 七、E类关闭通道

以下通道必须保持`channel_train_mask=0`：

- Wind节点63；
- Solar节点40、54、56、58、60、67。

Solar节点81仍可训练，但其长缺口位置由`valid_mask=0`屏蔽。

不要自行补齐、复制邻节点、容量截断或改变这些mask。

### 八、归一化与反归一化

不要重新拟合scaler。直接使用数据中已经归一化的数组。

反归一化逐节点逐通道读取：

```text
normalization_params.json
```

统一关系：

```python
value_mw = value_normalized * scale
```

风光scale为节点容量；负荷scale为训练期actual有效观测的q99.5。Wind31和Solar48使用项目版暂定的机组明细汇总容量，且metadata中保留容量冲突信息。

禁止：

- 对normalized值clip到[0,1]；
- 给residual再次拟合独立归一化器；
- 使用validation/test重新拟合scale；
- actual与forecast使用不同scale。

### 九、建议新增而非修改的文件

请根据当前仓库实际结构命名，但优先新增：

```text
datasets/shandong91_reliable.py
configs/shandong91/<model_name>_168h.yaml
scripts/train_shandong91_<model_name>.py 或独立运行脚本
scripts/eval_shandong91_<model_name>.py
tests/test_shandong91_data_contract.py
```

91节点输出目录建议固定为：

```text
checkpoints/shandong91/<run_name>/
logs/shandong91/<run_name>/
results/shandong91/<run_name>/
```

不要复用24场站输出目录，避免覆盖checkpoint和结果。

### 十、本轮验收

实现后只运行轻量级检查，不启动正式训练：

1. 三个划分均可加载。
2. 样本shape分别为：
   - train `(298,168,91,3)`；
   - validation `(24,168,91,3)`；
   - test `(25,168,91,3)`。
3. 单个batch可通过模型forward或最小dry-run。
4. masked loss在手工构造的小样本上计算正确。
5. 改变无效位置的中性填充值不会改变masked loss。
6. E类关闭通道不参与loss。
7. 不存在资源的通道不参与loss。
8. normalized输出可以按节点正确恢复MW。
9. `actual = forecast + residual`在有效位置成立。
10. 原24场站的数据加载、配置导入和至少一个已有轻量测试仍通过。

完成后报告：

- 新分支名称；
- 新增和修改文件；
- 91节点数据流及张量shape；
- mask如何进入loss和metrics；
- 是否保持24场站路径完全独立；
- dry-run和兼容性测试结果；
- 尚未解决但会影响正式训练的问题。

不要自动开始正式训练，不要提交远程分支，不要修改24场站实验结果。

---

## 运行前人工确认

在目标项目中启动Codex前，确认：

1. 数据目录名没有改变；
2. `output_manifest.json`存在；
3. 目标项目是Git仓库；
4. 24场站当前分支和checkpoint已有备份或提交记录；
5. 目标机器至少预留约1 GB数据空间，以及训练产生checkpoint所需的额外空间。
