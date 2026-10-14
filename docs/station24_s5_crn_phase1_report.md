# Station-24 S5全路径CRN第一阶段实现与正确性报告

状态：**PHASE 1 COMPLETE / FORMAL CRN EXPERIMENT NOT RUN**  
日期：2026-10-14  
范围：24场站、168 h、Raw与Full Independent Joint Tail V2；validation issue 0；4对成员；正式500步linear ancestral DDPM。  

## 结论

现有正式ancestral DDPM采样器可以用小范围、可选接口实现全路径共同随机数控制，不需要改网络、checkpoint、噪声调度或反向公式。Raw与Tail现在可按`(base seed, issue, global member, reverse stage)`获得相同初始噪声和相同逐步反向噪声；噪声由固定100成员规范块即时再生，不保存501个完整噪声张量，也不依赖运行时batch/chunk切法。本地CPU上，1个validation issue、4对成员、500步真实checkpoint审计通过；Raw和Tail均输出`[1,4,24,168]`有限张量，参数及buffer哈希前后不变。目标服务器CUDA/AMP本轮未运行，记为`NOT VERIFIED`，因此进入第二阶段前仍应先做服务器CUDA小规模门禁。

## 1. 依据与现有随机数路径

`station24_s5_methodology_audit.md`已经指出：原S5的序号配对只是独立乘积耦合的有限样本实现，不能把`Tail_i - Raw_i`直接解释成同一路径的模型修正；相同generation seed或只共享initial noise均不足以建立DDPM全路径耦合。

代码审计结果如下：

1. 初始高斯噪声原位于`StationGaussianDiffusion.sample()`，历史路径调用`torch.randn([B*K,S,T])`。
2. 每一步噪声原位于`StationGaussianDiffusion.denoise_step()`，历史路径调用`torch.randn_like(noisy)`。
3. 500步正式设置下，t=499到t=1的随机噪声实际进入反向更新，共499次；t=0仍可生成噪声，但其乘数严格为0，不影响输出。
4. Raw与Full V2的采样状态均为`[B*K,24,168]`，均使用500步linear schedule，张量合同一致。
5. 当前Raw checkpoint另有member级body/tail Bernoulli路由；Full V2没有对应路由。它不是可在两模型间共享的同类潜变量，但在CRN模式下已改为由issue/member键确定，避免chunk改变路由。默认模式仍使用历史`torch.bernoulli`。
6. 这两个冻结checkpoint未启用event-memory、retrieval、JSTD hypothesis等其他随机分支。
7. 历史全局RNG会因issue batch、member chunk以及Raw特有路由的消费顺序而变化；不能满足CRN要求。
8. 两个正式checkpoint和config均以只读方式无损复用，均选择历史正式使用的`model_state_dict`。

## 2. 最小实现

### 2.1 随机键与chunk不变性

新增`keyed_ancestral_noise()`：

```text
key = SplitMix64(base_seed, issue_index, canonical_block_start, stage)
canonical_block_start = floor(global_member_id / 100) * 100
noise = torch.randn([100, station, time], generator=key)
return noise[global_member_id - canonical_block_start]
```

运行时可以把成员397–407一次生成，也可拆成397–398、399–404、405–407；每个global member得到的值逐元素相同。规范块只为随机数寻址，不改变成员概率，也不是S5窗口或模型超参数。

### 2.2 可选显式注入接口

- `sample(..., ancestral_initial_noise=..., ancestral_noise_provider=...)`：直接指定初始噪声，并按timestep即时提供反向噪声。
- `sample(..., ancestral_noise_seed=..., ancestral_issue_indices=..., ancestral_member_block_start=...)`：使用keyed即时生成。
- `denoise_step(..., ancestral_random_noise=...)`：只替代该步`randn_like`的来源；均值、posterior variance及DDPM公式不变。
- 所有新参数默认`None`。关闭接口时仍执行原`torch.randn`/`torch.randn_like`路径。

## 3. 修改文件与函数索引

| 文件 | 函数/位置 | 修改 |
|---|---|---|
| `src/models/station_conditioned_diffusion.py` | `keyed_ancestral_noise` | 稳定key、规范成员块、即时标准高斯 |
| 同上 | `StationGaussianDiffusion.denoise_step` | 可选单步噪声张量 |
| 同上 | `StationGaussianDiffusion.sample` | 可选初始噪声、逐步provider或keyed全路径 |
| 同上 | `Station24DiffusionModel.generate` | 透传CRN接口；Raw路由在CRN模式下keyed化 |
| `generate_station24.py` | generation CLI/循环/metadata | 可选`keyed_full_path`模式及global member offset；默认关闭 |
| `tests/test_station24_crn_sampling.py` | 8项单元测试 | replay、跨issue/member/stage、chunk/batch不变、默认路径、公式一致 |
| `tools/audit_station24_s5_crn_phase1.py` | 独立审计入口 | 1 issue × 4对成员的真实checkpoint只读审计 |

未修改Original S5实现、网络结构、训练代码语义、checkpoint或91节点工作区。此前草拟的正式三seed运行器已从本阶段变更中移除，避免误启动第二阶段。

## 4. A–H检查结果

| 检查 | 状态 | 证据 |
|---|---|---|
| A. Raw/Tail初始噪声逐元素一致 | PASS | stage=500纳入全路径哈希 |
| B. 所有实际随机反向步噪声一致 | PASS | t=499…1逐元素一致；t=0系数为0 |
| C. 不同member不同且可复现 | PASS | 4成员×501 stages；每stage四个成员哈希互异 |
| D. batch/chunk改变不影响key | PASS | 397:408一次生成与2+6+3拆分逐元素相等；issue batch拆分单测通过 |
| E. 旧公式与显式注入公式一致 | PASS | 正式Raw模型t=250，CPU `max_abs=0.0` |
| F. 关闭CRN后历史路径不变 | PASS | 默认参数均为None；历史global-RNG replay单测通过；generation smoke通过 |
| G. 输出shape、有限性、归一化 | PASS | Raw/Tail均`[1,4,24,168]`；全有限；归一化往返最大误差0 |
| H. 参数和buffer不变 | PASS | Raw与Tail完整`state_dict`哈希前后完全相同 |
| 目标服务器CUDA/AMP | NOT VERIFIED | 本机`dm_preflight`为CPU环境；不得用CPU结果替代CUDA门禁 |

全501阶段Raw/Tail噪声计划共同SHA-256：

`cabe5d86f8cf8ec6ef575cf7f769a078f62ac3135a98f25b6b7c24d2acd9a056`

4对global member为400、401、402、403；base seed为271828；validation `sample_index=0`。

完整机器可读结果：`outputs_shandong/station24/s5_crn_phase1_20261014_final/phase1_audit.json`。

## 5. checkpoint与状态保护

- Raw checkpoint SHA-256：`9a242d087dd7d31b48695db48c34e0354b0d2989e41aac1cf350c8ce76f98adb`
- Full V2 checkpoint SHA-256：`1a655f36bb21575e68b93542a0be5e41b851187db67a516349848b0ffc9bb799`
- Raw model state哈希前后：`5e61c87c...eac9e37b`，一致。
- Full V2 model state哈希前后：`2dc69a43...d79a1d5`，一致。
- 未构造optimizer，未执行backward/step，未写checkpoint，Test未访问。

第一次审计运行在最终单步公式检查处因审计脚本漏传Raw所需`recent_error`条件而终止；只补传正式条件张量后重跑通过。这是审计调用缺参，不是采样器或模型失败，也未触发训练或正式生成。

## 6. 修改范围与风险

修改集中在噪声来源注入与生成入口；反向均值、方差、schedule、condition encoder和网络前向均未改变。主要剩余风险是：

1. 本机没有验证CUDA随机数实现与CUDA/AMP执行；第二阶段前必须在服务器复跑同一4对门禁。
2. `torch.randn`在不同设备/不同PyTorch版本之间不承诺产生同一数值序列；本设计保证同一运行环境、同一key下可复现，不声称CPU与CUDA噪声字节相同。
3. CRN保证外生DDPM噪声一致，不保证Raw与Tail最终轨迹语义相似，也不保证差值方差必然降低；那是第二阶段才可检验的研究问题。
4. Raw专属body/tail路由只有Raw一侧存在。它已对issue/member可复现，但不能与Full V2共享“相同路由”；报告中不得把它描述成两个模型的共同潜变量。

## 7. 是否具备进入第二阶段的条件

**有条件具备。** 采样控制的CPU实现和真实checkpoint最小审计通过，且未发现需要重构采样器的问题。正式第二阶段的唯一前置条件是：将当前代码同步到目标服务器后，用CUDA/AMP重复同一1 issue × 4 pairs门禁并得到PASS；在此之前不得启动3-seed正式对照。

## 8. 第二阶段规模与成本估计

若第二阶段采用3个预注册generation seeds、23个validation issues、每seed生成Raw 500与Tail 100：

- 每seed：`23 × (500+100) = 13,800`条168 h场景轨迹；
- 三个seed：41,400条轨迹；
- 每seed DDPM member-step：`13,800 × 500 = 6.9 million`；
- 三个seed：20.7 million member-step；
- 若Raw按100成员分5块、Tail按100成员1块，则约69,000次batched denoiser step/seed，三seed约207,000次。

按该项目既有同类服务器日志粗估，纯生成约每seed50–70分钟，三seed约2.5–3.5小时；完整评价、I/O与打包再预留约0.5–1.5小时，总计约3–5小时。该数值仅为排期开机估计，首个seed完成后应以实测更新，不能当作性能承诺。

**第一阶段结束，尚未运行正式CRN实验，等待用户批准第二阶段。**
