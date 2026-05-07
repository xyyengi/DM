#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多变量协同条件扩散模型 - 评估模块

包含两类评估指标：
1. 回归指标：R²、RMSE、MAE（评估场景均值/中位数的预测能力）
2. 场景指标：Coverage、Width、Energy Score、ACF（论文公式12-15）
"""

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ==================== 回归指标 ====================

def compute_regression_metrics(samples, actual, agg_method='median'):
    """
    计算回归指标（R²、RMSE、MAE）
    
    将多个场景聚合为单一预测值，然后计算与实际值的回归指标
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值
        agg_method: 聚合方法 ('median', 'mean', 'mean_of_middle')
    Returns:
        dict: 包含R²、RMSE、MAE的字典
    """
    N, n_samples, L = samples.shape
    
    # 聚合场景为单一预测
    if agg_method == 'median':
        predicted = np.median(samples, axis=1)  # (N, L)
    elif agg_method == 'mean':
        predicted = np.mean(samples, axis=1)  # (N, L)
    elif agg_method == 'mean_of_middle':
        # 取中间50%的场景均值
        sorted_samples = np.sort(samples, axis=1)
        mid_start = int(n_samples * 0.25)
        mid_end = int(n_samples * 0.75)
        predicted = np.mean(sorted_samples[:, mid_start:mid_end, :], axis=1)
    else:
        predicted = np.median(samples, axis=1)
    
    # 展平为一维
    actual_flat = actual.flatten()
    predicted_flat = predicted.flatten()
    
    # 计算指标
    r2 = r2_score(actual_flat, predicted_flat)
    rmse = np.sqrt(mean_squared_error(actual_flat, predicted_flat))
    mae = mean_absolute_error(actual_flat, predicted_flat)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'agg_method': agg_method
    }


def compute_regression_metrics_multichannel(samples, actual, channel_names=['wind', 'solar', 'load'], agg_method='median'):
    """
    多通道回归指标
    
    Args:
        samples: (N, n_samples, C, L) 生成的场景
        actual: (N, C, L) 实际值
        channel_names: 通道名称
        agg_method: 聚合方法
    Returns:
        dict: 包含所有通道回归指标的字典
    """
    N, n_samples, C, L = samples.shape
    metrics = {}
    
    for c, name in enumerate(channel_names):
        samples_c = samples[:, :, c, :]  # (N, n_samples, L)
        actual_c = actual[:, c, :]  # (N, L)
        
        reg_metrics = compute_regression_metrics(samples_c, actual_c, agg_method)
        metrics[f'{name}_r2'] = reg_metrics['r2']
        metrics[f'{name}_rmse'] = reg_metrics['rmse']
        metrics[f'{name}_mae'] = reg_metrics['mae']
    
    # 总体指标
    metrics['total_r2'] = np.mean([metrics[f'{name}_r2'] for name in channel_names])
    metrics['total_rmse'] = np.mean([metrics[f'{name}_rmse'] for name in channel_names])
    metrics['total_mae'] = np.mean([metrics[f'{name}_mae'] for name in channel_names])
    
    return metrics


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


def compute_scenario_width(samples, actual, quantile=1.0):
    """
    论文公式13: Scenario Width (PIAW)
    
    PIAW_α = Σ(P_up - P_down) / T
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值 (用于归一化)
        quantile: 分位数
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
    
    # 归一化（相对于实际值的百分比）
    max_val = np.maximum(np.abs(actual), 1e-6)
    width_normalized = width / max_val
    
    # 平均宽度百分比
    width_percent = np.mean(width_normalized) * 100
    
    return width_percent


def compute_energy_score(samples, actual):
    """
    论文公式14: Energy Score
    
    ES = (1/T) Σ Σ ||S_i - S_j|| - (2/T) Σ ||S_i - Y||
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值
    Returns:
        energy_score: 能量分数
    """
    N, n_samples, L = samples.shape
    T = N * L
    
    # 第一项: 场景之间的距离
    term1 = 0
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            diff = samples[:, i, :] - samples[:, j, :]  # (N, L)
            dist = np.sqrt(np.sum(diff ** 2, axis=1))  # (N,)
            term1 += np.mean(dist)
    term1 = term1 * 2 / (n_samples * (n_samples - 1)) if n_samples > 1 else 0
    
    # 第二项: 场景与实际值的距离
    term2 = 0
    for i in range(n_samples):
        diff = samples[:, i, :] - actual  # (N, L)
        dist = np.sqrt(np.sum(diff ** 2, axis=1))  # (N,)
        term2 += np.mean(dist)
    term2 = term2 * 2 / n_samples
    
    energy_score = term1 - term2
    
    return energy_score


def compute_acf(samples, actual, max_lag=24):
    """
    论文公式15: Autocorrelation Coefficient (ACF)
    
    ρ_k = Σ(P_t - P̄)(P_{t+k} - P̄) / Σ(P_t - P̄)²
    
    Args:
        samples: (N, n_samples, L) 生成的场景
        actual: (N, L) 实际值
        max_lag: 最大滞后阶数
    Returns:
        acf_actual: 实际值的ACF
        acf_samples_mean: 生成场景的平均ACF
        acf_samples_std: 生成场景的ACF标准差
    """
    N, n_samples, L = samples.shape
    
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
    
    acf_samples_mean = np.mean(acf_samples, axis=0)
    acf_samples_std = np.std(acf_samples, axis=0)
    
    return acf_actual, acf_samples_mean, acf_samples_std


def evaluate_all(samples, actual, quantiles=[1.0, 0.9, 0.8]):
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


def evaluate_multichannel(samples, actual, channel_names=['wind', 'solar', 'load'], quantiles=[1.0, 0.9, 0.8], agg_method='median'):
    """
    多通道完整评估：计算回归指标 + 场景指标
    
    Args:
        samples: (N, n_samples, C, L) 生成的场景
        actual: (N, C, L) 实际值
        channel_names: 通道名称
        quantiles: 分位数列表
        agg_method: 回归指标聚合方法
    Returns:
        metrics: 包含所有评估指标的字典
    """
    N, n_samples, C, L = samples.shape
    metrics = {}
    
    # 1. 回归指标（R²、RMSE、MAE）
    reg_metrics = compute_regression_metrics_multichannel(samples, actual, channel_names, agg_method)
    metrics.update(reg_metrics)
    
    # 2. 场景指标（论文公式）
    for c, name in enumerate(channel_names):
        samples_c = samples[:, :, c, :]  # (N, n_samples, L)
        actual_c = actual[:, c, :]  # (N, L)
        
        # Energy Score
        metrics[f'{name}_energy_score'] = compute_energy_score(samples_c, actual_c)
        
        # Coverage和Width
        for q in quantiles:
            q_name = f'{int(q*100)}%'
            metrics[f'{name}_coverage_{q_name}'] = compute_coverage_rate(samples_c, actual_c, q)
            metrics[f'{name}_width_{q_name}'] = compute_scenario_width(samples_c, actual_c, q)
        
        # ACF
        acf_actual, acf_mean, acf_std = compute_acf(samples_c, actual_c, max_lag=24)
        metrics[f'{name}_acf_actual'] = acf_actual
        metrics[f'{name}_acf_mean'] = acf_mean
        metrics[f'{name}_acf_std'] = acf_std
        metrics[f'{name}_acf_mae'] = np.mean(np.abs(acf_actual - acf_mean))
    
    # 总体场景指标
    metrics['total_energy_score'] = np.mean([metrics[f'{name}_energy_score'] for name in channel_names])
    metrics['total_coverage_100%'] = np.mean([metrics[f'{name}_coverage_100%'] for name in channel_names])
    metrics['total_width_100%'] = np.mean([metrics[f'{name}_width_100%'] for name in channel_names])
    metrics['total_acf_mae'] = np.mean([metrics[f'{name}_acf_mae'] for name in channel_names])
    
    return metrics


def print_metrics(metrics, channel_names=['wind', 'solar', 'load']):
    """打印评估指标（回归指标 + 场景指标）"""
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    
    # 1. 回归指标（总体）
    if 'total_r2' in metrics:
        print("\n【回归指标 - 总体】")
        print(f"  R²:   {metrics['total_r2']:.4f}")
        print(f"  RMSE: {metrics['total_rmse']:.4f}")
        print(f"  MAE:  {metrics['total_mae']:.4f}")
    
    # 2. 场景指标（总体）
    print("\n【场景指标 - 总体】")
    print(f"  Energy Score:  {metrics.get('total_energy_score', metrics.get('energy_score', 'N/A')):.4f}")
    print(f"  Coverage (100%): {metrics.get('total_coverage_100%', metrics.get('coverage_100%', 'N/A')):.2f}%")
    print(f"  Coverage (90%):  {metrics.get('total_coverage_90%', 'N/A'):.2f}%")
    print(f"  Width (100%):    {metrics.get('total_width_100%', metrics.get('width_100%', 'N/A')):.2f}%")
    print(f"  ACF MAE:         {metrics.get('total_acf_mae', metrics.get('acf_mae', 'N/A')):.4f}")
    
    # 3. 各通道详细指标
    for name in channel_names:
        if f'{name}_energy_score' in metrics:
            print(f"\n【{name.capitalize()}】")
            # 回归指标
            if f'{name}_r2' in metrics:
                print(f"  R²:   {metrics[f'{name}_r2']:.4f}")
                print(f"  RMSE: {metrics[f'{name}_rmse']:.4f}")
                print(f"  MAE:  {metrics[f'{name}_mae']:.4f}")
            # 场景指标
            print(f"  Energy Score:    {metrics[f'{name}_energy_score']:.4f}")
            print(f"  Coverage (100%): {metrics[f'{name}_coverage_100%']:.2f}%")
            print(f"  Coverage (90%):  {metrics[f'{name}_coverage_90%']:.2f}%")
            print(f"  Width (100%):    {metrics[f'{name}_width_100%']:.2f}%")
            print(f"  ACF MAE:         {metrics[f'{name}_acf_mae']:.4f}")
    
    print("="*70)
