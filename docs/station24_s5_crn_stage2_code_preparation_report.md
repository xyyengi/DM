# Station-24 S5 CRN 第二阶段代码准备报告

状态：**CODE PREPARATION COMPLETE；GPU FORMAL RUN NOT STARTED**

## 实现边界

- 仅针对24场站、168 h、Full Independent Joint Tail V2与Original S5。
- 没有训练、没有修改checkpoint、没有访问Test，也没有修改91节点工作区。
- 冻结为单个generation seed `271828`；正式生成量为Raw 500与Tail 100。
- 同一成员池比较CRN identity和3个固定随机置换配对；置换不触发重新生成。
- GPU门禁、正式生成、评价/汇总是三个独立的人工命令；门禁不会自动启动正式生成。

## 本地证据

| 检查 | 状态 | 说明 |
|---|---|---|
| Python语法编译 | PASS | 新增及修改的Python文件通过`py_compile` |
| CRN相关单元与回归测试 | PASS | 67项测试通过 |
| 配置冻结检查 | PASS | 400 Raw主体、100 Raw背景、100 Tail、单seed、3个置换seed均被强制校验 |
| CLI help / dry-run | PASS | 配置、路径与输出目录契约通过 |
| 本地CPU最小CRN复核 | PASS | 1 issue × 4 pairs；不代替服务器门禁 |
| Windows绝对路径检查 | PASS | 配置和运行脚本未硬编码本地Windows路径 |
| `git diff --check` | PASS | 无空白错误 |
| Bash语法检查 | NOT RUN locally | Windows环境无Bash；服务器GPU门禁首先执行`bash -n`，失败即停止 |
| CUDA/AMP门禁 | NOT RUN locally | 必须由服务器运行1 issue × 4 pairs门禁 |
| 正式Raw500/Tail100生成 | NOT RUN | 等待用户在门禁通过后手工启动 |
| 正式评价与报告 | NOT RUN | 等待生成完整后手工启动finalize |

## 关键保护

1. 正式生成器必须读取并验证GPU门禁JSON；要求`status=PASS`、`launch_eligible=true`、`device=cuda`、`amp=true`及`target_server_CUDA_AMP=PASS`。
2. CRN噪声以issue/member/timestep为键，并使用固定100成员规范块，避免batch/chunk改变随机流。
3. Raw背景global member id为400–499，Tail global member id同为400–499，从而共享初始噪声及所有随机反向步噪声。
4. 三个置换臂只改变Raw背景到同一Tail成员池的对应关系；成员边缘分布、duration分配和Original S5规则不变。
5. 正式生成保持历史FP32推理语义；AMP仅用于额外的目标服务器兼容门禁。
6. 任一阶段失败都会写状态并停止；finalize可幂等重跑，不重新生成模型成员。

服务器操作以`docs/station24_s5_crn_stage2_server_readme.md`为准。
