# Diffusion-TS Joint V1：独立完整轨迹实验

## 本轮要回答什么

不再让冻结主体承担普通轨迹、尾部只承担补丁。让同一个可训练生成模型学习24场站、168小时完整实际功率，检验短时变化和持续结构能否同时改善。Raw body-tail 不改、不重新解释旧检查点，继续作为正式对照。

这是完整新骨干实验，不是将不兼容的 Raw 权重“解冻后装进 Transformer”。新模型从头训练全部603,996个参数。风光共同输入、共同去噪、共同生成；同一成员内的风光按容量相加，不独立抽取再拼接。

## 来源与明确的改动

- 官方仓库：https://github.com/Y-debug-sys/Diffusion-TS
- 固定来源提交：`566307e6cf2d8095e58de4c6e3a6ae965b69b5b5`。
- 复用官方 Transformer encoder/decoder、多项式趋势、Fourier季节成分及重构残差；保留上游许可证和源码来源说明于 `src/models/diffusion_ts_vendor/`。
- 参考官方 clean-series 重构、加权L1及复数Fourier损失，独立实现条件化扩散接口；采样明确使用DDIM eta=0、500步。
- 项目适配：发布forecast注入时间表示；发布前已观测24小时recent-error及可用性mask作为交叉注意力记忆；目标为归一化实际功率映射到[-1,1]。
- 不在最终结果上机械加回forecast，不输入未来actual/residual或未来事件标签。夜间光伏投影复用现有因果日照处理。
- 原始实现的趋势/周期/重构残差，不等于物理上唯一可辨识的slow/fast；不能宣称频段正交、不会抵消或自动学到急跌。需通过采样和事件评价验证。
- 不移植旧固定图和state/FiLM模块；全站通道混合和时间注意力提供新的联合路径。因此第一轮只能评价“整套替代模型”，不能把收益单独归因于分解。

## 锁定配置

配置：`configs/station24_diffusion_ts_joint_v1.yaml`；独立分支 `experiment/24site-diffusion-ts-joint-v1`。

维度64、4头、2层encoder/3层decoder、dropout0.1；batch8、累积2；AdamW学习率1e-4；上限220 epoch，每5 epoch验证、40 epoch无改善停止。选验证重构目标最优raw检查点，无EMA。该选择不是“事件最优检查点”的证明。

固定现有数据划分，23个验证窗口，每窗口500条联合场景，种子424242。生成chunk32；服务器预检必须先运行实际batch8和chunk32的短探针。不能依据本机CPU预测服务器耗时。

训练缺失值：逐点L1排除缺失标签；某场站168小时中含填补值时，该场站该窗口不进入Fourier损失，避免把填补轨迹当频谱真值。本数据39个训练场站序列被排除Fourier监督，不丢弃整个发布窗口。

## 本机证据与边界（2026-09-08）

环境：`C:\Users\mila2\miniconda3\envs\dm_preflight`，CPU-only。

|检查|状态|证据|
|---|---|---|
|既有回归＋新结构测试|PASS|57项，56通过、1项CUDA跳过，38.851秒|
|每个参数梯度及实际更新|PASS|无缺失梯度、无未更新参数|
|固定真实小批次短拟合|PASS|12步损失35.313→9.615；仅工程学习信号|
|未来target替换不影响生成|PASS|固定条件和初始噪声完全一致|
|forecast/history条件参与|PASS|替换条件可改变采样|
|风光跨类型条件依赖|PASS|双向梯度非零；不等于相关性已正确|
|保存重载|PASS|固定采样完全一致|
|分解参与|PASS|三项逐个移除均改变短步采样，记录RMS和相关性|
|两epoch→23窗口×3成员×4步生成|PASS|独立local-smoke输出，不是科研结果|
|连续事件评价接口|PASS|本机smoke样本跑通|
|联合/逐日评价接口|PASS|500成员Raw与自身对照跑通|
|极端绘图接口|PASS|3成员smoke与自身对照跑通；3对500被接口正确拒绝|
|Bash语法、diff空白检查|PASS|本机Git Bash|
|目标服务器CUDA/AMP及配置batch/chunk|NOT RUN|必须先通过，失败自动中止|
|正式500成员、科学收益、分解消融|NOT RUN|不得宣称胜过Raw|

最近预检：`outputs_shandong/station24/diffusion_ts_joint_v1_preflight_local_04/preflight.json`。
本机集成：`outputs_shandong/station24/diffusion_ts_joint_v1_integration_local/`。集成运行在增加服务器batch/chunk探针前完成，CPU训练生成逻辑未改变。

**当前不具备跳过服务器预检直接训练的资格。** 流水线先执行单测、CUDA预检、梯度与更新、保存重载、实际batch/chunk探针，再凭代码/配置/数据指纹匹配的报告允许训练。

## 正式运行与恢复

需先提交推送本分支，服务器拉到对应提交后才运行；本文件不代表已经推送。

```bash
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
python -m pip install -r requirements-diffusion-ts.txt
bash run_station24_diffusion_ts_joint_v1.sh run
```

脚本输出log/status路径。训练与生成、对比评价、打包串联，任何步骤异常立即停止。不会安装或更换服务器torch。若仅评价/打包中断且训练和生成已完整结束：

```bash
bash run_station24_diffusion_ts_joint_v1.sh resume outputs_shandong/station24/diffusion_ts_joint_v1_实际时间戳
```

当前恢复入口不续训中断的epoch，也不自动覆盖部分生成文件；遇到这两种情况保留现场并人工检查。不要盲目重启训练。打包为独立`.tar.gz`，不附加sha256文件。

## 如何判结果及后续消融

先报告风/光/总量普通CRPS、coverage、width、Energy、空间相关，以及同一成员的风光功率和残差相关；逐提前日1—7分别报告，不只看平均。

持续事件保留onset、连续duration、depth和原有三档标准；短时用1/3/6小时ramp。事件评价复用旧接口，输出中的body标签仅为兼容（全体成员），不能当成新模型有body/tail。当前持续深跌审计主要是风电；风光联合相关与光伏普通质量有评价，不宣称已完成光伏专门极端事件审计。

第一轮回答完整联合模型是否值得继续；若普通质量和事件结构均有希望，再训练：去Fourier损失、去趋势/季节分解、去recent-error记忆等独立消融。关闭已训练分量的配对采样仅诊断机械作用，不能替代重训消融。特别检查分量能量、相关性/抵消、1/3/6小时变化及12/24小时低频结构；不以分支非零冒充有效。

第一轮单seed和反复使用的val只能探索，后续需独立时间块/多seed检验；test暂不用于调参。增加500成员数量本身不会改善学到的分布。
