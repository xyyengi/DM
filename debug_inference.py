#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理过程诊断脚本 - 定位 nan 来源 + 梯度监控
"""

import torch
import numpy as np
import yaml
import os

from dataset_multivariate import get_dataloader_multivariate
from diff_models_multivariate import MultiChannelCSDI


def debug_inference(exp_folder, data_path='./input_4.27/', debug_steps=[499, 400, 300, 200, 100, 50, 0],
                   guidance_scale=None, config_path=None):
    """诊断推理过程中的 nan 来源 + 梯度监控
    
    Args:
        exp_folder: 实验目录路径
        data_path: 数据路径
        debug_steps: 调试步骤列表
        guidance_scale: 覆盖配置中的 guidance_scale（可选）
        config_path: 指定配置文件路径（可选，默认使用 exp_folder/config_used.yaml）
    """
    
    # 加载配置
    if config_path is None:
        config_path = os.path.join(exp_folder, 'config_used.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖 guidance_scale（如果指定）
    if guidance_scale is not None:
        print(f"⚠ 覆盖 guidance_scale: {config['model']['guidance_scale']} -> {guidance_scale}")
        config['model']['guidance_scale'] = guidance_scale
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载模型
    model = MultiChannelCSDI(config['model'], device).to(device)
    ckpt_path = os.path.join(exp_folder, 'checkpoints', 'model_best.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"模型 epoch: {checkpoint.get('epoch')}")
    
    # 检查 diffusion 参数
    print(f"\n=== Diffusion 参数检查 ===")
    print(f"num_steps: {model.diffusion.num_steps}")
    print(f"beta 范围: [{model.diffusion.beta.min():.6f}, {model.diffusion.beta.max():.6f}]")
    print(f"alpha_hat[0]: {model.diffusion.alpha_hat[0]:.6f}")
    print(f"alpha_hat[-1]: {model.diffusion.alpha_hat[-1]:.6f}")
    print(f"guidance_scale: {model.diffusion.guidance_scale}")
    
    # 检查是否有 nan
    print(f"beta 有 nan: {torch.isnan(model.diffusion.beta).any()}")
    print(f"alpha 有 nan: {torch.isnan(model.diffusion.alpha).any()}")
    print(f"alpha_hat 有 nan: {torch.isnan(model.diffusion.alpha_hat).any()}")
    
    # 加载一个 batch
    test_loader, _, _ = get_dataloader_multivariate(
        data_path, 1, 'test', config['model']['n_intervals'])
    batch = next(iter(test_loader))
    
    print(f"\n=== 输入数据检查 ===")
    print(f"forecast_3ch: shape={batch['forecast_3ch'].shape}, nan={torch.isnan(batch['forecast_3ch']).any()}")
    print(f"residual_3ch: shape={batch['residual_3ch'].shape}, nan={torch.isnan(batch['residual_3ch']).any()}")
    print(f"cond_matrix: shape={batch['cond_matrix'].shape}, nan={torch.isnan(batch['cond_matrix']).any()}")
    print(f"time_encoding: shape={batch['time_encoding'].shape}, nan={torch.isnan(batch['time_encoding']).any()}")
    
    # 检查 cond_matrix 的值范围
    cond_matrix = batch['cond_matrix'].numpy()
    print(f"cond_matrix 范围: [{np.nanmin(cond_matrix):.4f}, {np.nanmax(cond_matrix):.4f}]")
    
    # 检查 residual_3ch 的值范围（与 cond_matrix 对比）
    residual_3ch = batch['residual_3ch'].numpy()
    print(f"residual_3ch 范围: [{np.nanmin(residual_3ch):.4f}, {np.nanmax(residual_3ch):.4f}]")
    
    # 【关键检查】坐标系一致性
    print(f"\n=== 坐标系一致性检查 ===")
    # 检查 cond_matrix 是否覆盖 residual_3ch 的范围
    c_down_min = np.min(cond_matrix[..., 0])
    c_up_max = np.max(cond_matrix[..., 1])
    res_min = np.min(residual_3ch)
    res_max = np.max(residual_3ch)
    
    print(f"cond_matrix 下界范围: [{np.min(cond_matrix[..., 0]):.4f}, {np.max(cond_matrix[..., 0]):.4f}]")
    print(f"cond_matrix 上界范围: [{np.min(cond_matrix[..., 1]):.4f}, {np.max(cond_matrix[..., 1]):.4f}]")
    print(f"residual_3ch 范围: [{res_min:.4f}, {res_max:.4f}]")
    
    # 计算覆盖率（residual 在 cond_matrix 区间内的比例）
    c_down = cond_matrix[..., 0]
    c_up = cond_matrix[..., 1]
    in_range = (residual_3ch >= c_down) & (residual_3ch <= c_up)
    coverage_ratio = np.mean(in_range)
    print(f"residual 在 cond_matrix 区间内的比例: {coverage_ratio:.2%}")
    
    if coverage_ratio < 0.5:
        print(f"⚠ 警告: residual 与 cond_matrix 坐标系可能不一致！")
    
    # 移动到设备
    forecast_3ch = batch['forecast_3ch'].to(device)
    time_encoding = batch['time_encoding'].to(device)
    cond_matrix = batch['cond_matrix'].to(device)
    timepoints = batch['timepoints'].to(device)
    
    cond_full = torch.cat([forecast_3ch, time_encoding], dim=1)
    time_feat = model.get_time_features(timepoints)
    
    print(f"\n=== 条件数据检查 ===")
    print(f"cond_full: nan={torch.isnan(cond_full).any()}, range=[{cond_full.min():.4f}, {cond_full.max():.4f}]")
    print(f"time_feat: nan={torch.isnan(time_feat).any()}")
    
    # 初始化噪声
    x_t = torch.randn(1, 3, 168, device=device)
    print(f"\n=== 初始噪声 ===")
    print(f"x_t: nan={torch.isnan(x_t).any()}, range=[{x_t.min():.4f}, {x_t.max():.4f}]")
    
    # 使用带调试的采样
    print(f"\n=== 逐步去噪检查（带梯度监控）===")
    
    with torch.no_grad():
        samples, debug_log = model.diffusion.sample(
            cond_full, cond_matrix, time_feat, n_samples=1, 
            debug=True, debug_steps=debug_steps
        )
    
    # 打印调试日志
    print(f"\n=== 梯度监控日志 ===")
    for info in debug_log:
        print(f"\nStep {info['step']}:")
        print(f"  alpha_t={info['alpha_t']:.6f}, alpha_hat_t={info['alpha_hat_t']:.6f}")
        print(f"  t_decay={info['t_decay']:.4f}, guidance_applied={info['guidance_applied']:.4f}")
        print(f"  x_t range: [{info['x_t_range'][0]:.4f}, {info['x_t_range'][1]:.4f}]")
        print(f"  c_down range: [{info['c_down_range'][0]:.4f}, {info['c_down_range'][1]:.4f}]")
        print(f"  c_up range: [{info['c_up_range'][0]:.4f}, {info['c_up_range'][1]:.4f}]")
        print(f"  gamma_ratio (超出区间比例): {info['gamma_ratio']:.2%}")
        print(f"  below_ratio (低于下界比例): {info['below_ratio']:.2%}")
        print(f"  above_ratio (高于上界比例): {info['above_ratio']:.2%}")
        print(f"  gradient range: [{info['gradient_range'][0]:.4f}, {info['gradient_range'][1]:.4f}]")
        print(f"  gradient mean: {info['gradient_mean']:.4f}")
        print(f"  x_prev range: [{info['x_prev_range'][0]:.4f}, {info['x_prev_range'][1]:.4f}]")
        print(f"  has_nan: {info['has_nan']}, has_inf: {info['has_inf']}")
    
    # 最终结果检查
    print(f"\n=== 最终结果 ===")
    final_sample = samples[0, 0].cpu().numpy()  # (3, 168)
    print(f"生成样本范围: [{final_sample.min():.4f}, {final_sample.max():.4f}]")
    print(f"生成样本有 nan: {np.isnan(final_sample).any()}")
    print(f"生成样本有 inf: {np.isinf(final_sample).any()}")
    
    # 计算最终覆盖率
    c_down_np = cond_matrix[0].cpu().numpy()  # (3, 168, 2)
    c_up_np = c_down_np[..., 1]
    c_down_np = c_down_np[..., 0]
    final_in_range = (final_sample >= c_down_np) & (final_sample <= c_up_np)
    final_coverage = np.mean(final_in_range)
    print(f"生成样本在 cond_matrix 区间内的比例: {final_coverage:.2%}")
    
    return debug_log


def check_kde_consistency(data_path='./input_4.27/'):
    """检查 KDE 条件矩阵与残差数据的一致性"""
    
    import pickle
    
    print(f"\n=== KDE 一致性检查 ===")
    
    # 加载 KDE 模型
    kde_path = os.path.join(data_path, 'kde_multivariate.pkl')
    if not os.path.exists(kde_path):
        print(f"KDE 模型不存在: {kde_path}")
        return
    
    with open(kde_path, 'rb') as f:
        kde_data = pickle.load(f)
    
    # 加载残差数据
    test_res = np.load(os.path.join(data_path, 'test_res.npy'))
    test_pred = np.load(os.path.join(data_path, 'test_pred.npy'))
    
    print(f"test_res shape: {test_res.shape}")
    print(f"test_pred shape: {test_pred.shape}")
    
    # 检查 error_stats
    for channel_name in ['wind', 'solar', 'load']:
        if channel_name in kde_data['error_stats']:
            means = kde_data['error_stats'][channel_name]['means']
            stds = kde_data['error_stats'][channel_name]['stds']
            print(f"\n{channel_name} error_stats:")
            print(f"  means: {means}")
            print(f"  stds: {stds}")
            
            # 计算条件区间宽度
            interval_widths = [2 * s for s in stds]  # k_h = 2.0 * std
            print(f"  interval widths (k_h=2.0*std): {interval_widths}")
    
    # 检查条件矩阵缓存
    for mode in ['train', 'val', 'test']:
        cache_path = os.path.join(data_path, f'cond_matrix_{mode}.npy')
        if os.path.exists(cache_path):
            cond_matrix = np.load(cache_path)
            print(f"\ncond_matrix_{mode} shape: {cond_matrix.shape}")
            print(f"  下界范围: [{cond_matrix[..., 0].min():.4f}, {cond_matrix[..., 0].max():.4f}]")
            print(f"  上界范围: [{cond_matrix[..., 1].min():.4f}, {cond_matrix[..., 1].max():.4f}]")
            
            # 计算区间宽度
            widths = cond_matrix[..., 1] - cond_matrix[..., 0]
            print(f"  区间宽度范围: [{widths.min():.4f}, {widths.max():.4f}]")
            print(f"  区间宽度均值: {widths.mean():.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='./save/run_beta04_lrsched_20260510_1656')
    parser.add_argument('--data_path', default='./input_4.27/')
    parser.add_argument('--check_kde', action='store_true', help='仅检查 KDE 一致性')
    parser.add_argument('--guidance_scale', type=float, default=None, help='覆盖配置中的 guidance_scale')
    parser.add_argument('--config', default=None, help='指定配置文件路径（默认使用 exp_folder/config_used.yaml）')
    args = parser.parse_args()
    
    if args.check_kde:
        check_kde_consistency(args.data_path)
    else:
        debug_inference(args.exp_folder, args.data_path, 
                       guidance_scale=args.guidance_scale, 
                       config_path=args.config)
