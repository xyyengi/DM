# 深度学习实验全流程检查规范

本规范由根目录 AGENTS.md 引用。目的：在服务器付费训练前暴露可检验的工程缺陷，并把工程通过与研究有效分开。不能保证实验成功，也不能以一份文档替代执行证据。

## 证据格式

每个实验保留独立配置、git提交、初始化来源、数据/图/阈值版本、软件硬件版本、随机种子和检查报告。每项状态只能是 PASS、FAIL、NOT RUN、NOT APPLICABLE，并附命令、数值或文件。FAIL 阻止正式启动；NOT RUN 的必需检查也阻止启动。修复后的实验使用独立名称，旧结果不覆盖。

## 本机检查环境

Windows工作区固定复用独立Conda环境 `dm_preflight`，不得把实验依赖装入 `base` 或 `paperread`：

```powershell
& 'C:\Users\mila2\miniconda3\Scripts\conda.exe' run -n dm_preflight python -c "import torch; print(torch.__version__)"
```

若环境被删除，重建命令为：

```powershell
& 'C:\Users\mila2\miniconda3\Scripts\conda.exe' create -y -n dm_preflight python=3.10 pip
& 'C:\Users\mila2\miniconda3\Scripts\conda.exe' run -n dm_preflight python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& 'C:\Users\mila2\miniconda3\Scripts\conda.exe' run -n dm_preflight python -m pip install numpy pandas scipy pyyaml matplotlib
```

本机相关回归测试至少运行：

```powershell
& 'C:\Users\mila2\miniconda3\Scripts\conda.exe' run -n dm_preflight python -m unittest tests.test_station24_jstd_targets tests.test_station24_jstd_tail tests.test_station24_jstd_h1 tests.test_station24_jstd_msep tests.test_station24_pipeline
```

涉及其他结构时，追加其对应测试模块。本机报告必须写明CPU-only；不能把它标成CUDA/AMP PASS。

## G0 数据与任务

- 核对形状轴顺序、单位、归一化、容量权重、时间间隔、缺失掩码和光伏夜间处理。
- 明确哪些量是生成条件、哪些是训练标签、哪些只用于评价。将未来 actual/residual 随机替换，固定种子后生成必须不变（oracle诊断例外，须明确标注）。
- 阈值、归一化、图、检索库只用 train 拟合。按真实时间检查窗口重叠；内部交叉验证不能把同一事件的重叠窗口分到训练和验证两侧。
- 统计独立事件数，不把24×168个相关位置称作独立样本。保留 onset、实际duration、方向、depth，不把观察尺度当事件类别。

## G1 参数与数学

- 列出新建、复用、冻结的参数与buffers，核对optimizer参数集合。
- 对每个变换检查范围、单位、初始化输出及导数。特别检查 clamp、detach、round、argmax、离散采样、硬mask和sigmoid饱和。
- 每个输出头分别测梯度是否有限、是否非零，并确认一次optimizer.step后参数实际改变。不能只检查“某个参数有grad”。
- 零初始化残差头允许首步上游零梯度，但必须验证头先更新后第二/第三步上游恢复梯度。记录例外，不能整体豁免。
- 训练概率分布与采样分布一致性：正值duration、窗口右边界截断、尺度范围、方向与风光属性相关性；裁剪高斯产生边界原子质量，不能称作未经限定的proper分布模型。

## G2 学习是否发生

- 使用固定小批量、有事件和无事件、风/光、正/负事件分别测试。
- 在小数据上短程拟合，逐项检查count/onset/duration/depth/sync，报告初始和结束的指标。无梯度、恒定输出、分支不更新、误差无法下降先查实现。
- 冻结参数及buffers在训练前后逐项比较。检查 train/eval、dropout、归一化层状态；相同条件、噪声和route下关闭tail应恢复body输出。
- 比较各loss的数值与梯度量级，确认加权和未掩盖关键任务。验证集没有重采样时不要直接沿用训练重采样的逆权重来声称校准。

## G3 slow/fast专项

- slow监督为12/24h低频结构，fast监督为1/3/6h变化；检查epsilon到x0的符号、SNR权重与物理尺度。
- 真checkpoint上记录slow/fast输出RMS、梯度范数、各自关闭后输出变化；同时报告噪声步和使用真实标签还是因果采样假设。
- 计算两分支相关性、低频泄漏、合成能量与分支能量比。移动平均不是正交投影，不能默认两个分支完全不重叠。
- 检查mask之前与之后的频率变化、事件边缘外溢、事件恢复后是否归零。
- 效果结论需固定checkpoint、初始噪声、反向噪声、route、事件假设，对比body / body+slow / body+fast / body+slow+fast。只测单步输出非零不代表最终500成员指标改善。
- 分别评价持续事件duration/depth/onset和短时ramp；同时检查普通CRPS、coverage、width、Energy及联合相关性。

## G4 服务器付费前

- CPU检查通过后，在目标CUDA/AMP环境执行同一梯度测试、数步训练、保存重载和生成冒烟测试。
- profiler记录数据等待、每步耗时、显存峰值和吞吐。先短探针估时，正式实验采用验证过的batch/chunk；未测不得承诺耗时。
- 预检必须退出码非零即停止；脚本激活环境，日志无缓冲，保存状态及恢复信息。不要预建会使评价脚本拒绝覆盖的输出目录。
- 不得为了进度跳过预检或因为单次GPU利用率低就改变科学配置。

## G5 训练和采样

- 每项关键loss与预测分布都写入历史日志。先验单独评价，不能只按总扩散loss选择“最佳先验”。
- 记录patience单位、最佳epoch和实际停止epoch；使用哪个raw/EMA状态需显式保存与验证。
- 固定tail配额会改变混合分布，冻结body不保证all-members质量不变；记录每窗口预算，而非只记录全局均值。
- chunk变化会改变随机数流和尾部配额舍入。配对因果比较须预先保存噪声与事件假设，不能把同seed自动当成相同成员。
- 元数据写明版本；训练目标名称不能误标为纯MSE。恢复脚本必须核对模型、成员数量、种子、结果文件完整性。

## G6 评价与结论

- 先检查N、K、站点顺序、实际值/forecast一致性、文件完整性和因果标志，再读指标。
- 事件按风/光、正/负、持续/急变分开，all/body/tail分开；宽松、主要、严格标准全部保留。
- 同时报告命中事件数、成员概率质量、持续时间和深度误差；“至少1条命中”不能代替概率校准。
- 真实事件区间的深度误差与发布预测残差事件命中不是同一指标，明确差别。重叠窗口不能当独立证据。
- 多随机种子或按独立时间块报告不确定度；少量事件不据AUC点估计宣称普遍可行/不可行。
- 先排除实现缺陷，再讨论过拟合、条件信息不足或结构局限。oracle成功只证明给定提示时有能力，不证明因果先验有效。

## 本轮已知事项（2026-09-08）

- MSEP V1 的 `softplus(raw_scale).clamp(max=0.35)` 在零初始化时位于截断区，duration/wind/solar 三个尺度头首步梯度均为零。这是实现缺陷，不是模型主动学出的不确定性。
- 代码已新增显式 `transformed_normal_v2`：duration 用 logit-normal，signed depth 用 tanh-normal，不再用裁剪高斯制造边界原子；缺失版本字段仍保留 `legacy_clamp`，旧检查点语义不变。正式新配置尚未锁定，状态不是“可以开跑”。
- 代码已新增 `train_jstd_segment_prior_only`，用于先冻结已完成的 H1 renderer、只训练因果事件先验，消除 MSEP V1 中“真实事件提示联合微调 renderer、生成时却使用采样提示”的混杂。该路径必须通过新预检。
- 真 MSEP checkpoint 的离线审计显示 slow/fast 均有非零输出和梯度，未发现相互抵消；fast 中仍有约 21%～27% 的 12 h 低频能量。该结论只证明分支机械参与，不证明最终场景获益。
- 仍需专项验证：body / body+slow / body+fast / full 的配对最终采样消融、多事件时间排序/重叠、duration与风光depth的联合依赖、CPU保存重载、服务器CUDA/AMP。
- 上述未完成事项不能当成已通过。本轮修复不等于授权立即开展新付费训练。
