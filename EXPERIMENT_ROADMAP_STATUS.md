# Independent Joint Tail V2：机制归因与最小消融路线

更新时间：2026-09-21

## 范围冻结

本阶段不修改 Raw/body-tail 模型，不新增 onset-duration-depth 门控、检索、动态图或新 backbone；不启动新的训练。Raw、V2、V2+self-localization 作为封存参考。

## 固定参考

| 标记 | 参考 |
|---|---|
| A | `outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242` |
| B | `outputs_shandong/station24/independent_joint_tail_v2_event_balanced_20260910_200653/independent_tail_n100` 与 `mixture_body400_tail100_n500` |
| C | `outputs_shandong/station24/independent_joint_tail_v2_self_localized_20260911` |

## 当前低成本证据

| 项目 | 状态 | 结果位置/说明 |
|---|---|---|
| Raw normal/extreme gradient conflict | COMPLETE | `.../gradient_conflict`；无 optimizer step、无测试集 |
| Raw 与 mismatch/replay/V2 参数漂移 | COMPLETE | `.../parameter_drift`；V1 checkpoint 未在本地完整可读，标记 NOT RUN |
| tail 比例 5/10/15/20% | ORDINARY READY; EVENT EVAL RUNNING | `.../ratio_05`、`ratio_10`、`ratio_15`、`ratio_20`；完整事件/联合复评由 `full_low_cost_sensitivity` 后台执行；25/30% 不具备现有 V2 tail 成员池，NOT RUN |
| self-localization duration 0.75 | COMPLETE | `.../localize_d075` |
| self-localization duration 1.00 | COMPLETE | 使用既有 C 结果 |
| self-localization duration 1.25 | COMPLETE | `.../localize_d125` |
| duration 1.50 | EXISTING/VERIFY | 仅在已有 d150 输出确认后纳入 |

## 后续训练（尚未启动）

四角对照：Raw/no-aux、Sampling-only（事件采样 0.60，辅助损失关闭）、Auxiliary-only（自然采样，V2 辅助损失）、V2（采样+辅助）。另加一个冻结 Raw、只训练现有 tail/adapter 的轻量隔离对照。训练前必须通过服务器 CUDA/AMP、梯度非零、冻结参数不变、保存重载和因果检查。

建议最多先做 3 个新训练：Sampling-only、Auxiliary-only、轻量隔离；V2 已有结果作为第四角，不重复训练。每个配置锁定后再做 3 seeds。

### Sampling-only / Auxiliary-only 当前门禁

- 两个配置的本机 `dm_preflight` CPU 检查：PASS；梯度为非零，保存/重载与因果检查通过。
- CPU 检查不等价于服务器 CUDA/AMP 通过，因此目前仍不能把本机结果标为 launch eligible。
- Sampling-only 配置：`configs/station24_independent_joint_tail_v2_sampling_only_168h.yaml`。
- Auxiliary-only 配置：`configs/station24_independent_joint_tail_v2_auxiliary_only_168h.yaml`，其中 `independent_tail_event_sampling_fraction: null` 明确恢复自然采样；旧配置的默认 0.60 行为不变。
- 对应流水线：`run_station24_v2_sampling_only_pipeline.sh` 与 `run_station24_v2_auxiliary_only_pipeline.sh`。

## 解释边界

梯度余弦只能证明局部 batch 的方向关系，不能单独证明长期 catastrophic forgetting；参数漂移是描述性证据，不能作为因果结论。所有未执行项目必须保持 NOT RUN，不以通过形状/生成 smoke test 代替学习证据。
