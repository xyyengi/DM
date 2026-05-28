# 实验结果简要总结

## 1. 版本含义与条件引入方式

| 版本 | 实验名 | 扩散目标 x0 | 网络输入 | guidance | 最终 actual 构造 | 主要回答的问题 |
|---|---|---|---|---|---|---|
| V0 | `v0_uncond_ddpm_actual_168h` | actual | 仅 `x_t` | 不使用 | 模型直接输出 actual scenario | 没有 forecast 条件时，纯 DDPM 能达到什么 baseline？ |
| V1 | `v1_2023_guidance_actual_168h` | actual | 仅 `x_t` | 使用 forecast-error interval guidance | 模型直接输出 actual scenario | 不把 forecast 输入网络，只在反向采样阶段用 guidance 约束，是否能改善生成结果？ |
| V2 | `v2_csdi_cond_actual_given_forecast_168h` | actual | `[x_t, forecast]` | 不使用 | 模型直接输出 actual scenario | 把 forecast 作为 CSDI-like 条件输入网络，是否能提升精度和时序结构？ |
| Vmix | `v_mix_residual_forecast_concat_guidance` | residual | `[x_t, forecast, time_encoding]` | 使用 forecast-error interval guidance | `actual_scenario = forecast - generated_residual` | 同时使用 residual 建模、forecast 网络条件和 guidance，是否优于单独条件方式？ |

更具体地说，四个版本的区别不只是“有没有条件”，而是 forecast 进入模型的路径不同：

| 版本 | forecast 是否进入网络 | forecast 是否参与 guidance | 生成对象 | 说明 |
|---|---:|---:|---|---|
| V0 | 否 | 否 | actual | 完全无条件生成，用来衡量最低 baseline。 |
| V1 | 否 | 是 | actual | forecast 不作为神经网络输入，但在每一步反向扩散时通过区间约束修正采样方向。 |
| V2 | 是 | 否 | actual | forecast 被拼接到网络输入，模型在 denoising 网络内部学习 forecast 与 actual 的关系。 |
| Vmix | 是 | 是 | residual | forecast 同时作为网络输入和 guidance 参考，模型学习 forecast error/residual，再还原 actual。 |

当前残差定义：

```text
residual = forecast - actual
actual = forecast - residual
```

## 2. 主结果对比：n_samples = 20

| 版本 | 代表 run | mean_MAE | wind_MAE | pv_MAE | load_MAE | coverage | interval_width | acf_error | total_crps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | `20260525_001242_v0_uncond_ddpm_actual_168h` | 0.2148 | 0.3364 | 0.1891 | 0.1191 | 86.05 | 80.32 | 0.0890 | 0.1470 |
| V1 | `20260525_014620_v1_2023_guidance_actual_168h` | 0.1124 | 0.2255 | 0.0720 | 0.0397 | 63.14 | 19.70 | 0.1069 | 0.0882 |
| V2 | `20260525_033557_v2_csdi_cond_actual_given_forecast_168h` | 0.1230 | 0.2651 | 0.0643 | 0.0395 | **87.10** | 40.46 | **0.0457** | 0.0876 |
| Vmix | `20260525_050050_v_mix_residual_forecast_concat_guidance` | **0.0963** | **0.1995** | **0.0549** | **0.0347** | 68.39 | 19.29 | 0.0856 | **0.0743** |

## 3. 辅助对比：n_samples = 10

| 版本 | 代表 run | mean_MAE | coverage | interval_width | acf_error | total_crps |
|---|---|---:|---:|---:|---:|---:|
| V0 | `20260525_003053_v0_uncond_ddpm_actual_168h` | 0.1962 | 78.01 | 63.17 | 0.1714 | 0.1344 |
| V1 | `20260525_013839_v1_2023_guidance_actual_168h` | 0.1068 | 59.42 | 18.03 | 0.1067 | 0.0808 |
| V2 | `20260525_022900_v2_csdi_cond_actual_given_forecast_168h` | 0.1104 | 82.42 | 36.93 | 0.0762 | **0.0750** |
| Vmix | `20260525_033218_v_mix_residual_forecast_concat_guidance` | **0.1066** | 55.49 | **16.03** | 0.1222 | 0.0826 |

## 4. Vmix 与 V2 典型场景图

典型样本选择规则：选取 forecast MAE 最接近测试集中位数的测试样本。

- 样本编号：`493`
- median forecast MAE：`0.109628`
- selected forecast MAE：`0.109628`

图像文件：

![Vmix vs V2 typical case](comparison_figures/vmix_vs_v2_typical_case.png)

图中黑线是真实值，灰色虚线是 forecast，彩色实线是场景均值，浅色区域是 10%-90% 场景带。

## 5. 简要解读

Vmix 是当前综合表现最好的版本。它在 n_samples=20 下取得最低的 mean_MAE、最低的 total_crps，并且在 wind/PV/load 三个通道上都有较好的误差表现。

V2 的优势是 coverage 和 ACF。它的覆盖率最高，时序相关性误差最低，但代价是 interval_width 明显更宽，mean_MAE 也比 Vmix 高。

V1 明显优于 V0，是有效的 guidance-only baseline。n_samples=10 时 V1 和 Vmix 很接近，但 n_samples=20 时 Vmix 更稳。

V0 是合格的无条件 baseline，但准确率明显弱于条件版本。它的 coverage 较高主要来自很宽的区间，不代表生成质量更好。

Wind 是最难建模的变量。所有版本中 wind_MAE 都显著高于 PV/load，这也和 forecast quality 中 wind 相关性较低的现象一致。

## 6. 当前建议

下一阶段建议以 Vmix 作为主模型继续推进，同时保留 V2 作为 coverage/时序相关性较强的对照模型，V1 作为 guidance baseline，V0 作为无条件 baseline。

快速迭代可以继续使用 `n_samples=10`；最终对比建议至少使用 `n_samples=20`，如果时间允许，再对 Vmix 和 V2 补跑更大的 `n_samples` 做稳定性确认。
