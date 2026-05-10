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
from evaluation import evaluate_multichannel, print_metrics


def find_experiment_folders(base_path, keyword=None):
    """查找实验文件夹"""
    if not os.path.exists(base_path):
        return []
    all_folders = [f for f in os.listdir(base_path) 
                   if f.startswith('run_') and os.path.isdir(os.path.join(base_path, f))]
    if keyword:
        return [f for f in all_folders if keyword in f]
    return all_folders


def list_experiments(base_path):
    """列出所有实验及其 checkpoint 概况（紧凑格式）"""
    folders = find_experiment_folders(base_path)
    if not folders:
        print("无实验记录")
        return
    
    print("\n" + "="*110)
    print("可用实验列表")
    print("="*110)
    print(f"{'实验文件夹':<48} | {'最新epoch':<14} | {'best (epoch, loss)':<35}")
    print("-" * 110)
    
    for folder in sorted(folders, reverse=True):
        exp_path = os.path.join(base_path, folder)
        ckpt_dir = os.path.join(exp_path, 'checkpoints')
        best_path = os.path.join(ckpt_dir, 'model_best.pt')
        epoch_ckpts = []

        latest_info = "--"
        if os.path.exists(ckpt_dir):
            for name in os.listdir(ckpt_dir):
                if name.startswith('model_epoch_') and name.endswith('.pt'):
                    try:
                        epoch = int(name.replace('model_epoch_', '').replace('.pt', ''))
                        epoch_ckpts.append((epoch, os.path.join(ckpt_dir, name)))
                    except ValueError:
                        continue

        if epoch_ckpts:
            epoch_ckpts.sort(key=lambda item: item[0])
            latest_epoch, latest_path = epoch_ckpts[-1]
            latest_info = f"epoch {latest_epoch}"

        best_info = "--"
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location='cpu')
            epoch = ckpt.get('epoch', 'unknown')
            val_loss = ckpt.get('val_loss', 'unknown')
            if isinstance(val_loss, float):
                best_info = f"epoch {epoch}, loss {val_loss:.4f}"
            else:
                best_info = f"epoch {epoch}"

        print(f"{folder:<48} | {latest_info:<14} | {best_info:<35}")
    
    print("="*110)


def get_checkpoint_path(exp_folder, ckpt_type='best'):
    """获取checkpoint路径。

    优先级（默认使用最佳模型）：
    1. 默认优先返回 model_best.pt（最佳验证损失模型）
    2. 如果指定 ckpt_type='latest'，选择最新的 model_epoch_*.pt
    3. 如果没有 model_best.pt，回退到最新的 epoch checkpoint
    4. 最后选取 checkpoints 目录中最新修改的 .pt 文件
    """
    ckpt_path = os.path.join(exp_folder, 'checkpoints')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"无checkpoint目录: {ckpt_path}")

    # 优先查找 model_best.pt
    best_path = os.path.join(ckpt_path, 'model_best.pt')
    if os.path.exists(best_path) and ckpt_type == 'best':
        return best_path

    # 查找所有 epoch checkpoint
    epoch_ckpts = []
    for name in os.listdir(ckpt_path):
        if name.startswith('model_epoch_') and name.endswith('.pt'):
            try:
                epoch = int(name.replace('model_epoch_', '').replace('.pt', ''))
                epoch_ckpts.append((epoch, os.path.join(ckpt_path, name)))
            except ValueError:
                continue

    if epoch_ckpts:
        epoch_ckpts.sort(key=lambda item: item[0])
        latest_epoch_path = epoch_ckpts[-1][1]
        # 如果请求 latest，或者没有 best，使用最新的 epoch
        if ckpt_type == 'latest' or not os.path.exists(best_path):
            return latest_epoch_path
    
    # 回退到 model_best.pt（即使 ckpt_type 不是 'best'）
    if os.path.exists(best_path):
        return best_path
    
    # 最后尝试任意 .pt 文件
    any_pt = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.endswith('.pt')]
    if any_pt:
        any_pt.sort(key=lambda p: os.path.getmtime(p))
        return any_pt[-1]

    raise FileNotFoundError(f"无可用checkpoint: {ckpt_path}")


def generate_scenarios(model, test_loader, device, n_samples=10):
    """生成场景"""
    model.eval()
    all_samples, all_forecast, all_residual = [], [], []
    
    total_batches = len(test_loader)
    print(f"生成场景 (n_samples={n_samples}, 总批次: {total_batches})...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            samples = model.generate(batch, n_samples=n_samples)
            all_samples.append(samples.cpu().numpy())
            all_forecast.append(batch['forecast_3ch'].numpy())
            all_residual.append(batch['residual_3ch'].numpy())
            
            # 进度提示
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                print(f"  已完成 {batch_idx + 1}/{total_batches} 批次")
    
    print(f"生成完成!")
    return np.concatenate(all_samples), np.concatenate(all_forecast), np.concatenate(all_residual)


def evaluate_and_save(samples, forecast, residual, max_values, save_path):
    """评估并保存结果（使用论文公式12-15）"""
    N, n_samples, C, L = samples.shape
    
    # 使用evaluation模块计算完整指标
    metrics = evaluate_multichannel(samples, residual)
    print_metrics(metrics)
    
    # 保存结果
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'generated_samples.npy'), samples)
    np.save(os.path.join(save_path, 'forecast_data.npy'), forecast)
    np.save(os.path.join(save_path, 'residual_data.npy'), residual)
    
    # 保存ACF数据
    for c, name in enumerate(['wind', 'solar', 'load']):
        if f'{name}_acf_actual' in metrics:
            np.save(os.path.join(save_path, f'{name}_acf.npy'), {
                'actual': metrics[f'{name}_acf_actual'],
                'mean': metrics[f'{name}_acf_mean'],
                'std': metrics[f'{name}_acf_std']
            })
    
    # 保存评估指标
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"样本数: {n_samples}\n\n")
        f.write("="*60 + "\n")
        f.write("论文公式评估指标\n")
        f.write("="*60 + "\n\n")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
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
    parser.add_argument('--list', action='store_true', help='列出所有可用实验')
    args = parser.parse_args()
    
    # 列出所有实验
    if args.list:
        list_experiments(args.save_path)
        return
    
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
        if not os.path.exists(ckpt_path):
            print(f"警告: 未找到指定 epoch checkpoint，改为自动选择最新文件: {ckpt_path}")
            ckpt_path = get_checkpoint_path(exp_folder, 'latest')
    else:
        ckpt_path = get_checkpoint_path(exp_folder, 'latest')
    
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
    
    # 使用strict=False，因为beta/alpha/alpha_hat是buffer，已在模型初始化时创建
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        # 过滤掉buffer相关的missing keys（这些是正常的）
        buffer_keys = ['diffusion.beta', 'diffusion.alpha', 'diffusion.alpha_hat']
        real_missing = [k for k in missing_keys if k not in buffer_keys]
        if real_missing:
            print(f"警告: 缺失关键参数 {real_missing}")
    
    print(f"模型epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # 生成
    samples, forecast, residual = generate_scenarios(model, test_loader, device, args.n_samples)
    
    # 保存
    result_folder = os.path.join(exp_folder, 'results', f'predict_{datetime.now().strftime("%Y%m%d_%H%M")}')
    evaluate_and_save(samples, forecast, residual, max_values, result_folder)
    
    print(f"\n完成!")


if __name__ == '__main__':
    main()