#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多变量协同条件扩散模型 - 评估模块

核心场景指标：
1. Coverage Rate: 真实值落在预测区间内的比例
2. Scenario Width: 预测区间宽度（不确定性范围）
3. Energy Score: 评估概率分布整体质量
4. CRPS: CDF与真实值的距离
5. 多变量Energy Score: 风、光、负荷三者联合分布评估
6. ACF: 时间序列自相关结构保留（辅助验证）

移除的指标：
- 回归指标（R², RMSE, MAE）：场景生成不是确定性预测任务
- 分位数偏差：与Coverage/可靠性信息重叠
"""

import numpy as np


# ==================== 场景指标（论文公式） ====================


def compute_coverage_rate(samples, actual, quantile=1.0):
    """
    论文公式12: Coverage Rate
    
    CR_α = T_α / T × 100%
    
    其中T_α表示实际值落在α分位数区间内的次数
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值
        quantile: 分位数 (1.0=100%, 0.9=90%, 0.8=80%)
    Returns:
        coverage_rate: 覆盖率百分比
    """
    N, n_samples, L = samples.shape
    
    # 计算分位数区间边界
    alpha = quantile
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    
    # 计算每个时间点的上下界
    lower_bound = np.quantile(samples, lower_q, axis=1)  # (N, L)
    upper_bound = np.quantile(samples, upper_q, axis=1)  # (N, L)
    
    # 计算覆盖率
    covered = (actual >= lower_bound) & (actual <= upper_bound)
    T_alpha = np.sum(covered)
    T = N * L
    
    coverage_rate = T_alpha / T * 100
    
    return coverage_rate


def compute_scenario_width(samples, actual, quantile=1.0, global_range=None):
    """
    论文公式13: Scenario Width (PIAW)
    
    PIAW_α = Σ(P_up - P_down) / T
    
    【关键修复】使用全局极差归一化，避免 actual≈0 时 Width 爆炸：
    - 禁止使用 width / actual，因为残差趋于 0 时会导致数值无效
    - 改用测试集中该通道的全局极差 (Max - Min) 作为分母
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值
        quantile: 分位数
        global_range: 该通道的全局极差 (Max - Min)，用于归一化
    Returns:
        width_percent: 平均区间宽度百分比
    """
    N, n_samples, L = samples.shape
    
    # 计算分位数区间边界
    alpha = quantile
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    
    lower_bound = np.quantile(samples, lower_q, axis=1)  # (N, L)
    upper_bound = np.quantile(samples, upper_q, axis=1)  # (N, L)
    
    # 计算区间宽度
    width = upper_bound - lower_bound  # (N, L)
    
    # 【关键修复】使用全局极差归一化
    if global_range is not None and global_range > 0:
        # 使用测试集该通道的全局极差作为归一化基准
        width_normalized = width / global_range
    else:
        # 回退：如果没有提供全局极差，使用 actual 的全局极差
        actual_range = np.max(actual) - np.min(actual)
        if actual_range > 0:
            width_normalized = width / actual_range
        else:
            # 最后回退：使用样本自身的极差
            sample_range = np.max(width)
            width_normalized = width / max(sample_range, 1e-6)
    
    # 平均宽度百分比
    width_percent = np.mean(width_normalized) * 100
    
    return width_percent


def compute_energy_score(samples, actual, verbose=False):
    """
    标准经验 Energy Score（无偏 U-statistic 估计）：

    ES = E||X - Y|| - 0.5 E||X - X'||

    数值越小越好；除浮点舍入误差外应为非负数。实现与
    ``compute_multivariate_energy_score`` 使用相同定义。
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值
        verbose: 是否打印进度
    Returns:
        energy_score: 能量分数
    """
    N, n_samples, L = samples.shape
    
    if verbose:
        print(f"  计算Energy Score (n_samples={n_samples})...")
    
    # 第一项: 场景之间的距离
    term1 = 0
    pair_count = 0
    total_pairs = n_samples * (n_samples - 1) // 2
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            diff = samples[:, i, :] - samples[:, j, :]  # (N, L)
            dist = np.sqrt(np.sum(diff ** 2, axis=1))  # (N,)
            term1 += np.mean(dist)
            pair_count += 1
            if verbose and pair_count % 100 == 0:
                print(f"    场景对进度: {pair_count}/{total_pairs}")
    term1 = term1 / pair_count if pair_count > 0 else 0
    
    # 第二项: 场景与实际值的距离
    term2 = 0
    for i in range(n_samples):
        diff = samples[:, i, :] - actual  # (N, L)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))  # (N,)
        term2 += np.mean(dist)
    term2 = term2 / n_samples

    # Standard energy score: E||X-y|| - 0.5 E||X-X'||.
    # ``term1`` is the mean distance between distinct ensemble members and
    # ``term2`` is the mean distance between ensemble members and truth.
    energy_score = term2 - 0.5 * term1
    
    if verbose:
        print(f"  Energy Score 完成: {energy_score:.4f}")
    
    return energy_score


def compute_acf(samples, actual, max_lag=24, verbose=False):
    """
    论文公式15: Autocorrelation Coefficient (ACF)
    
    ρ_k = Σ(P_t - P̄)(P_{t+k} - P̄) / Σ(P_t - P̄)²
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值
        max_lag: 最大滞后阶数
        verbose: 是否打印进度
    Returns:
        acf_actual: 实际值的ACF
        acf_samples_mean: 生成场景的平均ACF
        acf_samples_std: 生成场景的ACF标准差
    """
    N, n_samples, L = samples.shape
    
    if verbose:
        print(f"  计算ACF (N={N}, n_samples={n_samples}, max_lag={max_lag})...")
    
    # 计算实际值的ACF
    acf_actual = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            acf_actual[lag] = 1.0
        else:
            # 对每个样本计算ACF然后平均
            acf_values = []
            for n in range(N):
                series = actual[n, :]
                if len(series) > lag:
                    mean = np.mean(series)
                    numerator = np.sum((series[:-lag] - mean) * (series[lag:] - mean))
                    denominator = np.sum((series - mean) ** 2)
                    if denominator > 0:
                        acf_values.append(numerator / denominator)
            acf_actual[lag] = np.mean(acf_values) if acf_values else 0
    
    if verbose:
        print(f"    实际值ACF完成")
    
    # 计算生成场景的ACF
    acf_samples = np.zeros((n_samples, max_lag))
    for s in range(n_samples):
        for lag in range(max_lag):
            if lag == 0:
                acf_samples[s, lag] = 1.0
            else:
                acf_values = []
                for n in range(N):
                    series = samples[n, s, :]
                    if len(series) > lag:
                        mean = np.mean(series)
                        numerator = np.sum((series[:-lag] - mean) * (series[lag:] - mean))
                        denominator = np.sum((series - mean) ** 2)
                        if denominator > 0:
                            acf_values.append(numerator / denominator)
                acf_samples[s, lag] = np.mean(acf_values) if acf_values else 0
        
        if verbose and (s + 1) % 10 == 0:
            print(f"    场景ACF进度: {s+1}/{n_samples}")
    
    acf_samples_mean = np.mean(acf_samples, axis=0)
    acf_samples_std = np.std(acf_samples, axis=0)
    
    if verbose:
        print(f"  ACF完成, MAE: {np.mean(np.abs(acf_actual - acf_samples_mean)):.4f}")
    
    return acf_actual, acf_samples_mean, acf_samples_std


def evaluate_all(samples, actual, quantiles=[1.0, 0.95, 0.9, 0.8]):
    """
    完整评估：计算所有指标
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值
        quantiles: 分位数列表
    Returns:
        metrics: 包含所有评估指标的字典
    """
    metrics = {}
    
    # Energy Score (不分位数)
    metrics['energy_score'] = compute_energy_score(samples, actual)
    
    # 各分位数下的Coverage Rate和Scenario Width
    for q in quantiles:
        q_name = f'{int(q*100)}%'
        metrics[f'coverage_{q_name}'] = compute_coverage_rate(samples, actual, q)
        metrics[f'width_{q_name}'] = compute_scenario_width(samples, actual, q)
    
    # ACF
    acf_actual, acf_mean, acf_std = compute_acf(samples, actual, max_lag=24)
    metrics['acf_actual'] = acf_actual
    metrics['acf_mean'] = acf_mean
    metrics['acf_std'] = acf_std
    metrics['acf_mae'] = np.mean(np.abs(acf_actual - acf_mean))  # ACF偏差
    
    return metrics


def evaluate_multichannel(samples, actual, channel_names=['wind', 'solar', 'load'], quantiles=[1.0, 0.95, 0.9, 0.8], verbose=True):
    """
    多通道完整评估：核心场景指标
    
    Args:
        samples: (N, n_samples, C, L) 生成的场景
        actual: (N, C, L) 实际值
        channel_names: 通道名称
        quantiles: 分位数列表
        verbose: 是否打印进度
    Returns:
        metrics: 包含所有评估指标的字典
    """
    N, n_samples, C, L = samples.shape
    metrics = {}
    
    if verbose:
        print(f"\n开始评估 (N={N}, n_samples={n_samples}, C={C})...")
    
    # 【关键修复】计算各通道的全局极差，用于 Width 归一化
    global_ranges = []
    for c in range(C):
        actual_c = actual[:, c, :]  # (N, L)
        global_range = np.max(actual_c) - np.min(actual_c)
        global_ranges.append(global_range)
        if verbose:
            print(f"  通道 {channel_names[c]} 全局极差: {global_range:.4f}")
    
    # 1. 多变量Energy Score（联合分布 - 最重要）
    if verbose:
        print("  【多变量联合分布】...")
    metrics['multivariate_es'] = compute_multivariate_energy_score(samples, actual, verbose=verbose)
    
    # 2. 各通道场景指标
    for c, name in enumerate(channel_names):
        if verbose:
            print(f"\n  【{name.capitalize()}】...")
        samples_c = samples[:, :, c, :]  # (N, n_samples, L)
        actual_c = actual[:, c, :]  # (N, L)
        
        # CRPS
        metrics[f'{name}_crps'] = compute_crps(samples_c, actual_c, verbose=verbose)
        
        # Energy Score (单通道)
        metrics[f'{name}_energy_score'] = compute_energy_score(samples_c, actual_c, verbose=verbose)
        
        # Coverage和Width - 使用全局极差归一化
        for q in quantiles:
            q_name = f'{int(q*100)}%'
            metrics[f'{name}_coverage_{q_name}'] = compute_coverage_rate(samples_c, actual_c, q)
            metrics[f'{name}_width_{q_name}'] = compute_scenario_width(samples_c, actual_c, q, global_range=global_ranges[c])
        
        # 可靠性检查（80%、90%、95%置信区间）
        reliability_dict = compute_reliability(samples_c, actual_c, confidence_levels=[0.80, 0.90, 0.95], verbose=verbose)
        for k, v in reliability_dict.items():
            metrics[f'{name}_{k}'] = v
        
        # ACF（辅助验证）
        acf_actual, acf_mean, acf_std = compute_acf(samples_c, actual_c, max_lag=24, verbose=False)
        metrics[f'{name}_acf_mae'] = np.mean(np.abs(acf_actual - acf_mean))
    
    # 3. 总体指标
    metrics['total_crps'] = np.mean([metrics[f'{name}_crps'] for name in channel_names])
    metrics['total_energy_score'] = np.mean([metrics[f'{name}_energy_score'] for name in channel_names])
    for q in quantiles:
        q_name = f'{int(q*100)}%'
        metrics[f'total_coverage_{q_name}'] = np.mean(
            [metrics[f'{name}_coverage_{q_name}'] for name in channel_names]
        )
        metrics[f'total_width_{q_name}'] = np.mean(
            [metrics[f'{name}_width_{q_name}'] for name in channel_names]
        )
    metrics['total_acf_mae'] = np.mean([metrics[f'{name}_acf_mae'] for name in channel_names])
    
    if verbose:
        print(f"\n评估完成!")
    
    return metrics


# ==================== 专业场景指标 ====================


def compute_crps(samples, actual, verbose=False):
    """
    CRPS (Continuous Ranked Probability Score)
    
    原理：评估生成的N个场景构成的CDF与真实单点观测值的距离
    
    公式：CRPS = ∫[F(x) - H(x-y)]² dx
    其中：
    - F(x): 生成场景的经验CDF
    - H(x-y): 真实值的阶跃函数（Heaviside函数）
    - y: 真实观测值
    
    简化计算：CRPS = E|X-y| - 0.5*E|X-X'|
    其中X, X'是两个独立生成的场景
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值
        verbose: 是否打印进度
    Returns:
        crps_mean: 平均CRPS值
    """
    N, n_samples, L = samples.shape
    
    if verbose:
        print(f"  计算CRPS (N={N}, n_samples={n_samples})...")
    
    crps_values = np.zeros(N * L)
    idx = 0
    
    for n in range(N):
        for l in range(L):
            # 获取该时间点的所有场景值
            x = samples[n, :, l]  # (n_samples,)
            y = actual[n, l]  # 真实值
            
            # 第一项：E|X-y|
            term1 = np.mean(np.abs(x - y))
            
            # 第二项：0.5*E|X-X'|
            # 计算所有场景对的平均距离
            term2 = 0
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    term2 += np.abs(x[i] - x[j])
            term2 = term2 / (n_samples * (n_samples - 1) / 2) if n_samples > 1 else 0
            term2 = 0.5 * term2
            
            crps_values[idx] = term1 - term2
            idx += 1
    
    crps_mean = np.mean(crps_values)
    
    if verbose:
        print(f"  CRPS完成: {crps_mean:.4f}")
    
    return crps_mean


def compute_multivariate_energy_score(samples, actual, verbose=False):
    """
    多变量Energy Score (评估风、光、负荷三者联合分布)
    
    原理：这是多变量场景生成的必测指标，用于检查CDM是否抓住了三者之间的相关性
    
    公式：ES = E||X-Y|| - 0.5*E||X-X'||  (欧几里得范数)
    
    对于多变量情况，||·||是跨所有通道的联合距离
    
    Args:
        samples: (N, n_samples, C, L) 生成的场景 (C=3通道)
        actual: (N, C, L) 实际值
        verbose: 是否打印进度
    Returns:
        es_value: 多变量Energy Score
    """
    N, n_samples, C, L = samples.shape
    
    if verbose:
        print(f"  计算多变量Energy Score (C={C}通道联合)...")
    
    # 将数据展平为 (N, n_samples, C*L) 以计算联合距离
    samples_flat = samples.reshape(N, n_samples, C * L)  # (N, n_samples, 504)
    actual_flat = actual.reshape(N, C * L)  # (N, 504)
    
    # 第一项：E||X-Y|| (场景与真实值的距离)
    term1 = 0
    for i in range(n_samples):
        diff = samples_flat[:, i, :] - actual_flat  # (N, C*L)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))  # (N,)
        term1 += np.mean(dist)
    term1 = term1 / n_samples
    
    # 第二项：0.5*E||X-X'|| (场景之间的距离)
    term2 = 0
    pair_count = 0
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            diff = samples_flat[:, i, :] - samples_flat[:, j, :]  # (N, C*L)
            dist = np.sqrt(np.sum(diff ** 2, axis=1))  # (N,)
            term2 += np.mean(dist)
            pair_count += 1
    term2 = term2 / pair_count if pair_count > 0 else 0
    term2 = 0.5 * term2
    
    es_value = term1 - term2
    
    if verbose:
        print(f"  多变量ES完成: {es_value:.4f}")
    
    return es_value


def compute_reliability(samples, actual, confidence_levels=[0.80, 0.90, 0.95], verbose=False):
    """
    可靠性/覆盖率检查
    
    原理：计算真实值落在生成的N个场景所形成的置信区间内的比例
    
    如果90%的区间只包含了70%的真实点，说明模型低估了波动性
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值
        confidence_levels: 置信水平列表
        verbose: 是否打印进度
    Returns:
        reliability_dict: 各置信水平的覆盖率字典
    """
    N, n_samples, L = samples.shape
    
    if verbose:
        print(f"  计算可靠性/覆盖率...")
    
    reliability_dict = {}
    
    for conf in confidence_levels:
        # 计算置信区间边界
        lower_q = (1 - conf) / 2
        upper_q = 1 - lower_q
        
        lower_bound = np.quantile(samples, lower_q, axis=1)  # (N, L)
        upper_bound = np.quantile(samples, upper_q, axis=1)  # (N, L)
        
        # 计算覆盖率
        covered = (actual >= lower_bound) & (actual <= upper_bound)
        coverage_rate = np.sum(covered) / (N * L) * 100
        
        # 理想覆盖率应该接近置信水平
        ideal_coverage = conf * 100
        coverage_deviation = coverage_rate - ideal_coverage
        
        reliability_dict[f'coverage_{int(conf*100)}%'] = coverage_rate
        reliability_dict[f'coverage_deviation_{int(conf*100)}%'] = coverage_deviation
        
        if verbose:
            status = "✓" if abs(coverage_deviation) < 5 else "⚠"
            print(f"    {int(conf*100)}%置信区间: 实际覆盖{coverage_rate:.1f}% (理想{ideal_coverage:.0f}%) {status}")
    
    return reliability_dict


def print_metrics(metrics, channel_names=['wind', 'solar', 'load']):
    """打印核心场景指标"""
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    
    # 1. 核心指标（总体）
    print("\n【核心指标 - 总体】")
    if 'multivariate_es' in metrics:
        print(f"  多变量Energy Score (联合分布): {metrics['multivariate_es']:.4f}")
    if 'total_crps' in metrics:
        print(f"  CRPS (平均):                   {metrics['total_crps']:.4f}")
    if 'total_energy_score' in metrics:
        print(f"  Energy Score (单通道平均):     {metrics['total_energy_score']:.4f}")
    if 'total_coverage_90%' in metrics:
        print(f"  Coverage (90% nominal):        {metrics['total_coverage_90%']:.2f}%")
    if 'total_width_90%' in metrics:
        print(f"  Width (90% nominal):           {metrics['total_width_90%']:.2f}%")
    if 'total_coverage_100%' in metrics:
        print(f"  Coverage (sample min-max):     {metrics['total_coverage_100%']:.2f}%")
    if 'total_width_100%' in metrics:
        print(f"  Width (sample min-max):        {metrics['total_width_100%']:.2f}%")
    if 'total_acf_mae' in metrics:
        print(f"  ACF MAE (辅助):                {metrics['total_acf_mae']:.4f}")
    
    # 2. 各通道详细指标
    for name in channel_names:
        if f'{name}_crps' in metrics:
            print(f"\n【{name.capitalize()}】")
            print(f"  CRPS:            {metrics[f'{name}_crps']:.4f}")
            print(f"  Energy Score:    {metrics[f'{name}_energy_score']:.4f}")
            print(f"  Coverage (100%): {metrics[f'{name}_coverage_100%']:.2f}%")
            print(f"  Coverage (90%):  {metrics[f'{name}_coverage_90%']:.2f}%")
            print(f"  Width (100%):    {metrics[f'{name}_width_100%']:.2f}%")
            print(f"  ACF MAE:         {metrics[f'{name}_acf_mae']:.4f}")
            # 可靠性
            if f'{name}_coverage_80%' in metrics:
                deviation = metrics.get(f'{name}_coverage_deviation_80%', 0)
                status = "✓" if abs(deviation) < 5 else "⚠"
                print(f"  可靠性 (80%区间): {metrics[f'{name}_coverage_80%']:.1f}% (理想80%) {status}")
            if f'{name}_coverage_90%' in metrics:
                deviation = metrics.get(f'{name}_coverage_deviation_90%', 0)
                status = "✓" if abs(deviation) < 5 else "⚠"
                print(f"  可靠性 (90%区间): {metrics[f'{name}_coverage_90%']:.1f}% (理想90%) {status}")
            if f'{name}_coverage_95%' in metrics:
                deviation = metrics.get(f'{name}_coverage_deviation_95%', 0)
                status = "✓" if abs(deviation) < 5 else "⚠"
                print(f"  可靠性 (95%区间): {metrics[f'{name}_coverage_95%']:.1f}% (理想95%) {status}")
    
    print("="*70)
