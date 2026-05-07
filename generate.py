#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多变量协同条件扩散模型 - 预测脚本

使用方法:
    python predict.py --exp_name wind_scenario
    python predict.py --exp_name wind_scenario --ckpt_epoch 100 --n_samples 50
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml
from datetime import datetime

from dataset_multivariate import get_dataloader_multivariate
from diff_models_multivariate import MultiChannelCSDI


def find_experiment_folders(base_path, keyword=None):
    """查找实验文件夹"""
    if not os.path.exists(base_path):
        return []
    all_folders = [f for f in os.listdir(base_path) 
                   if f.startswith('run_') and os.path.isdir(os.path.join(base_path, f))]
    if keyword:
        return [f for f in all_folders if keyword in f]
    return all_folders


def get_checkpoint_path(exp_folder, ckpt_type='best'):
    """获取checkpoint路径"""
    ckpt_path = os.path.join(exp_folder, 'checkpoints')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"无checkpoint目录: {ckpt_path}")
    
    if ckpt_type == 'best':
        path = os.path.join(ckpt_path, 'model_best.pt')
        if os.path.exists(path):
            return path
    
    # 选择最新的checkpoint
    ckpts = [f for f in os.listdir(ckpt_path) if f.startswith('model_epoch_') and f.endswith('.pt')]
    if not ckpts:
        raise FileNotFoundError(f"无可用checkpoint")
    ckpts.sort(key=lambda x: int(x.replace('model_epoch_', '').replace('.pt', '')))
    return os.path.join(ckpt_path, ckpts[-1])


def generate_scenarios(model, test_loader, device, n_samples=10):
    """生成场景"""
    model.eval()
    all_samples, all_forecast, all_residual = [], [], []
    
    print(f"生成场景 (n_samples={n_samples})...")
    
    with torch.no_grad():
        for batch in test_loader:
            samples = model.generate(batch, n_samples=n_samples)
            all_samples.append(samples.cpu().numpy())
            all_forecast.append(batch['forecast_3ch'].numpy())
            all_residual.append(batch['residual_3ch'].numpy())
    
    return np.concatenate(all_samples), np.concatenate(all_forecast), np.concatenate(all_residual)


def evaluate_and_save(samples, forecast, residual, max_values, save_path):
    """评估并保存结果"""
    max_values_3ch = max_values[:3]
    samples_denorm = samples * max_values_3ch.reshape(1, 1, 3, 1)
    residual_denorm = residual * max_values_3ch.reshape(1, 3, 1)
    
    metrics = {}
    N, n_samples, C, L = samples.shape
    
    # Energy Score
    distances = np.sqrt(np.sum((samples_denorm - residual_denorm.reshape(N, 1, 3, L)) ** 2, axis=(2, 3)))
    metrics['energy_score'] = np.mean(distances)
    
    # Coverage
    for c, name in enumerate(['wind', 'solar', 'load']):
        sc = samples_denorm[:, :, c, :]
        ac = residual_denorm[:, c, :]
        up, down = np.max(sc, axis=1), np.min(sc, axis=1)
        metrics[f'{name}_coverage_100'] = np.mean((ac >= down) & (ac <= up))
    
    print(f"\n评估结果:")
    print(f"  Energy Score: {metrics['energy_score']:.4f}")
    for name in ['wind', 'solar', 'load']:
        print(f"  {name.capitalize()} Coverage: {metrics[f'{name}_coverage_100']:.4f}")
    
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'generated_samples.npy'), samples)
    np.save(os.path.join(save_path, 'forecast_data.npy'), forecast)
    np.save(os.path.join(save_path, 'residual_data.npy'), residual)
    
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"样本数: {n_samples}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    
    print(f"\n结果保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='多变量协同条件扩散模型 - 预测')
    parser.add_argument('--config', default='config/wind_scenario.yaml')
    parser.add_argument('--data_path', default='./input_4.27/')
    parser.add_argument('--save_path', default='./save/')
    parser.add_argument('--exp_name', default='wind_scenario', help='实验名称或文件夹名')
    parser.add_argument('--ckpt_epoch', type=int, default=None, help='指定epoch，默认使用最佳模型')
    parser.add_argument('--n_samples', type=int, default=10, help='生成样本数')
    args = parser.parse_args()
    
    # 查找实验文件夹
    if args.exp_name.startswith('run_'):
        exp_folder = os.path.join(args.save_path, args.exp_name)
    else:
        folders = find_experiment_folders(args.save_path, args.exp_name)
        if not folders:
            print(f"未找到实验: {args.exp_name}")
            sys.exit(1)
        exp_folder = os.path.join(args.save_path, folders[0])
    
    print(f"实验文件夹: {exp_folder}")
    
    # 加载checkpoint
    if args.ckpt_epoch:
        ckpt_path = os.path.join(exp_folder, 'checkpoints', f'model_epoch_{args.ckpt_epoch}.pt')
    else:
        ckpt_path = get_checkpoint_path(exp_folder, 'best')
    
    print(f"加载模型: {ckpt_path}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载配置
    config_path = os.path.join(exp_folder, 'config_used.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # 数据加载
    test_loader, _, max_values = get_dataloader_multivariate(
        args.data_path, config['train']['batch_size'], 'test', config['model']['n_intervals'])
    
    # 模型
    model = MultiChannelCSDI(config['model'], device).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # 生成
    samples, forecast, residual = generate_scenarios(model, test_loader, device, args.n_samples)
    
    # 保存
    result_folder = os.path.join(exp_folder, 'results', f'predict_{datetime.now().strftime("%Y%m%d_%H%M")}')
    evaluate_and_save(samples, forecast, residual, max_values, result_folder)
    
    print(f"\n完成!")


if __name__ == '__main__':
    main()