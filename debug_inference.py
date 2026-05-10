#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理过程诊断脚本 - 定位 nan 来源
"""

import torch
import numpy as np
import yaml
import os

from dataset_multivariate import get_dataloader_multivariate
from diff_models_multivariate import MultiChannelCSDI


def debug_inference(exp_folder, data_path='./input_4.27/'):
    """诊断推理过程中的 nan 来源"""
    
    # 加载配置
    config_path = os.path.join(exp_folder, 'config_used.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
    
    # 逐步去噪，检查每一步
    print(f"\n=== 逐步去噪检查 ===")
    nan_first_step = None
    
    with torch.no_grad():
        for t in range(model.diffusion.num_steps - 1, -1, -1):
            # 检查去噪前
            if torch.isnan(x_t).any():
                nan_first_step = t + 1
                print(f"⚠ nan 出现在 step {t+1} (去噪前)")
                break
            
            # 执行去噪
            x_prev = model.diffusion.denoise_step(x_t, t, cond_full, cond_matrix, time_feat)
            
            # 检查去噪后
            if torch.isnan(x_prev).any():
                nan_first_step = t
                print(f"⚠ nan 出现在 step {t} (去噪后)")
                
                # 详细检查
                alpha_t = model.diffusion.alpha[t]
                alpha_hat_t = model.diffusion.alpha_hat[t]
                print(f"  alpha_t={alpha_t:.6f}, alpha_hat_t={alpha_hat_t:.6f}")
                
                # 检查模型输出
                input_14ch = torch.cat([x_t, cond_full], dim=1)
                predicted_noise = model.unet(input_14ch, time_feat)
                print(f"  predicted_noise: nan={torch.isnan(predicted_noise).any()}")
                print(f"  predicted_noise range: [{predicted_noise.min():.4f}, {predicted_noise.max():.4f}]")
                
                # 检查 input_14ch
                print(f"  input_14ch: nan={torch.isnan(input_14ch).any()}, range=[{input_14ch.min():.4f}, {input_14ch.max():.4f}]")
                
                # 检查 time_feat
                print(f"  time_feat: nan={torch.isnan(time_feat).any()}")
                
                # 检查模型权重是否有 nan
                for name, param in model.unet.named_parameters():
                    if torch.isnan(param).any():
                        print(f"  ⚠ 模型权重 nan: {name}")
                
                # 检查去噪公式各部分
                print(f"\n  === 去噪公式分解 ===")
                print(f"  x_t range: [{x_t.min():.4f}, {x_t.max():.4f}]")
                print(f"  (1 - alpha_t) = {1 - alpha_t:.6f}")
                print(f"  (1 - alpha_hat_t).sqrt() = {(1 - alpha_hat_t).sqrt():.6f}")
                coef = (1 - alpha_t) / (1 - alpha_hat_t).sqrt()
                print(f"  噪声系数 = {coef:.6f}")
                
                # 手动计算 mean
                mean_part = x_t - coef * predicted_noise
                print(f"  x_t - coef*predicted_noise: nan={torch.isnan(mean_part).any()}")
                
                break
            
            x_t = x_prev
            
            # 每 100 步打印一次
            if t % 100 == 0:
                print(f"step {t}: x_t range=[{x_t.min():.4f}, {x_t.max():.4f}]")
    
    if nan_first_step is None:
        print(f"\n✓ 去噪完成，无 nan")
        print(f"最终 x_0: range=[{x_t.min():.4f}, {x_t.max():.4f}]")
    else:
        print(f"\n✗ nan 首次出现在 step {nan_first_step}")
    
    return nan_first_step


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='./save/run_beta04_lrsched_20260510_1656')
    parser.add_argument('--data_path', default='./input_4.27/')
    args = parser.parse_args()
    
    debug_inference(args.exp_folder, args.data_path)