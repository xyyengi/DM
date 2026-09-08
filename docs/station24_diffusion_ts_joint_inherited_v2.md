# TS Joint Inherited V2 — 继承式完整轨迹实验

日期：2026-09-09。独立配置 `configs/station24_diffusion_ts_joint_inherited_v2.yaml`，模型版本 `diffusion_ts_joint_inherited_v2`。此前的条件较少V1代码和配置保留为后续消融，不覆盖Raw body-tail及历史结果。

## 2026-09-09 CUDA预检溢出修复

当前启动脚本改用独立配置`station24_diffusion_ts_joint_inherited_v2_bf16.yaml`及独立`...v2_bf16_时间戳`输出。原FP16配置保留。模型、条件、损失和采样设置不变；训练显式BF16 autocast，不使用FP16 GradScaler。缺失`amp_dtype`字段继续保持旧FP16行为。设备不支持BF16则停止，不静默回退。

服务器原错误发生于预检`nonfinite gradient: graph_gate`，尚未通过训练许可。首个坏参数不等于根因所在。本机同一真实批次、同一模型初始化和噪声的CPU数值对照：FP32 loss75.487、无坏梯度；FP16×65536 loss75.489、36个坏梯度，首个为graph_gate；FP16×1无坏梯度；BF16×1 loss75.724、无坏梯度。这支持“FP16梯度缩放溢出”的解释，但不替代目标CUDA复现。

参考PyTorch AMP文档：https://docs.pytorch.org/docs/stable/amp 。默认GradScaler初始scale为65536；FP16数值范围可能导致梯度溢出。此次不删有限性检查、不把NaN改成0、不跳过失败更新。

新增CPU BF16反传及实际更新测试，以及旧FP16/BF16精度策略测试；CUDA单测和预检使用与正式训练相同的BF16策略。CPU常规预检仍为FP32，报告明确CUDA NOT RUN；另有CPU BF16专项测试。新服务器日志在`precision`字段记录dtype和GradScaler开关。

修复后本机64项测试：62 PASS、2 CUDA NOT RUN（skip），56.868秒。新配置CPU预检PASS：`outputs_shandong/station24/diffusion_ts_joint_inherited_v2_bf16_preflight_local_01/preflight.json`；目标服务器BF16预检仍NOT RUN，未获免预检启动资格。

## 模型定义

**没有主体/尾部专家，没有二分类路由，也没有tail成员比例。** 所有成员由同一个全参数可训练模型生成24场站×168小时实际功率。趋势/周期/剩余重构是每条轨迹内部相加的成分，不是三个专家。

继承Raw的信息和先验，而非加载不兼容的ResUNet权重：

- 原forecast与发布前24h近期误差；
- 原8维calendar、2维lead；
- 原train-only阈值生成的四类forecast状态；
- 5维站点属性；
- 原冻结地理图及train历史功率图，检查来源manifest和SHA；
- 在TS每个encoder/decoder块前加入条件FiLM，非旧权重搬运。

场站级forecast、state、static先编码；双图归一化传播，训练融合权重；保留场站轴顺序后投影到时间特征。日历/提前时距加入条件，逐块FiLM调制，近期误差作为交叉注意力记忆。解码保留官方趋势、多频率Fourier成分及剩余重构路径。

继承资产来源固定为：
`outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/`。
服务器必须保留其中`graphs/`、`state_thresholds.json`及数据集，缺失直接报错，不伪造图。模型图、静态属性作为checkpoint buffers保存。当前重新建模/生成仍要求原继承资产可读并通过哈希验证。

## 训练目标：不只是更换骨干

基础沿用V1的扩散步加权clean功率L1 + Fourier损失；新增以下重构监督：

|项|定义|系数|
|---|---|---|
|ramp|重构与真实1/3/6h有符号差分的L1，三尺度平均|0.2|
|slow|重构与真实12/24h滑动均值的L1，两尺度平均|0.2|
|synchrony|同小时跨站预测误差乘积与真实误差乘积的L1，排除对角线|0.1|
|aggregate|按容量归一化后的风、光、合计轨迹重构L1|0.2|

上述系数是本轮锁定的实验超参数，不是论文给出的最优值。记录所有项及梯度量级，结果不好不能宣称方法有效。

同步项使用相对于当前forecast的残差：重构功率与forecast的差、真实功率与forecast的差。训练中使用真实答案合理；生成条件不允许输入这些未来差值。它显式约束同向/反向异常的瞬时二阶结构，**不是最终500成员分布的proper score，也不保证概率校准**。所有损失在有效标签上计算；跨时窗/跨站缺失位置不参加对应项。

没有原样移植Raw的6h事件重放、事件位置损失或tail gate。其事件训练目的改为作用于全部成员的多尺度结构与联合异常监督；这不能声称等价继承所有旧事件收益。没有强制将真实事件划为五类。12/24h窗口只用于监督，不进入生成条件；也不声称三个模型成分是正交slow/fast。

## 本机证据与启动资格

环境：隔离的`dm_preflight`，torch 2.14.0+cpu。共62个单测：60 PASS，2个CUDA测试NOT RUN（skip）。

- `diffusion_ts_joint_inherited_v2_preflight_local_02/preflight.json`：671,095参数，全部有梯度并更新；固定buffers不变；12步固定批次损失75.487→8.222。
- 每个新增结构项的x0梯度有限且非零；测试完美重构损失为零、缺失标签无直接监督、局部/持续偏差产生误差。
- forecast、history、calendar、lead、state及双图对输出有作用；未来actual/residual改变不会改变固定噪声生成。这里的条件移除只是机制探针，不是收益消融。
- 保存/重载一致；三成分逐个关闭影响短步采样；预检分量合成能量比约0.999，但不是训练后/全数据抵消结论。
- 两epoch本机短训练PASS，总损失55.072→32.181，全部损失逐项落日志。
- 23窗口×3成员×4步生成PASS；只做工程集成，不当正式效果。
- 真实短训checkpoint分量审计PASS，前三个val窗口、固定噪声8步；输出分量RMS、12/24h均值能量比例、相关性、合成能量比和逐项移除变化。
- 事件评价接口PASS；极端绘图以smoke与自身配对PASS；500成员Raw自对照的联合、逐日、多尺度评价PASS。
- 正式Raw对V2的500成员事件、ramp及联合评价：NOT RUN。
- 目标服务器CUDA/AMP、batch8与chunk32探针：NOT RUN，必须通过后才允许训练。
- Linux完整后台流程及打包：NOT RUN；本机Bash语法检查PASS。

本机结果目录：`outputs_shandong/station24/diffusion_ts_joint_inherited_v2_integration_local/`。
**当前不具备免预检训练资格。** 代码/配置/数据指纹与预检不符会阻断；服务器预检失败即停止，不靠放宽阈值通过。

## 正式流程

入口：`run_station24_diffusion_ts_joint_inherited_v2.sh`；同一独立TS分支`experiment/24site-diffusion-ts-joint-v1`内用V2配置/输出区分，V1不运行。

服务器拿到提交后：

```bash
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
python -m pip install -r requirements-diffusion-ts.txt
bash run_station24_diffusion_ts_joint_inherited_v2.sh run
```

顺序：单测→CUDA预检→训练→best checkpoint分量审计→500成员生成→风/光/合计及逐日、多尺度评价→连续事件三档评价→极端图→风电ramp时刻诊断→独立tar.gz。

正式上限220epoch、每5epoch验证、40epoch无改善停止；raw权重，不使用EMA。每窗口500条、500步DDIM eta0、chunk32；不是500个训练epoch。运行耗时以服务器探针为依据，不据本机估计承诺。

只完成训练/生成、后处理失败时可用：

```bash
bash run_station24_diffusion_ts_joint_inherited_v2.sh resume outputs_shandong/station24/diffusion_ts_joint_inherited_v2_实际时间戳
```

恢复入口不支持中断epoch续训，不覆盖部分生成；遇到这些情况保留现场检查。兼容旧事件评价的`tail_expert_route.npy`全零仅用于接口；其中body=all，不是真有主体专家，不解释其body/tail分组收益。

## 收口判据及消融

必须同时看短时ramp、持续事件onset/duration/depth、普通质量、风光联合相关与合计出力。低频监督下降不是持续事件已命中；至少一个成员命中不是概率校准。光伏/合计多尺度分布有输出，但独立光伏极端事件目录不是本轮已完成项。

先比较完整V2与Raw；再按结果做去双图、去state/FiLM、去结构损失、去TS分解的独立重训消融。保留较简V1作为整套少条件对照，不能把多处同时变化当单项消融。正式500成员配对分量关闭仍NOT RUN；本轮短步审计不替代它。单seed和val只作探索，test不参与调参。
