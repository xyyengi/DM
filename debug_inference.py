#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试推理脚本：检查数据归一化、模型权重加载与一次生成的数值统计

用法示例：
    python debug_inference.py --data_path ./input_4.27/ --exp_folder ./save/run_test_run_20260506_1534/ --n_samples 1 --device cpu
"""
import argparse
import os
import sys
import numpy as np

import torch

from dataset_multivariate import MultiChannelWindScenarioDataset, get_dataloader_multivariate
from diff_models_multivariate import MultiChannelCSDI


def print_stats(name, arr):
    try:
        a = np.array(arr)
        print(f"{name}: shape={a.shape}, min={a.min():.6f}, max={a.max():.6f}, mean={a.mean():.6f}, std={a.std():.6f}")
    except Exception as e:
        print(f"{name}: 无法计算统计: {e}")


def resolve_checkpoint_path(exp_folder, ckpt=None):
    """优先使用显式指定的 checkpoint；否则自动选择最新的 model_epoch_*.pt。"""
    if ckpt:
        if os.path.isabs(ckpt):
            return ckpt
        return os.path.join(exp_folder, ckpt)

    ckpt_dir = os.path.join(exp_folder, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f'Checkpoint directory not found: {ckpt_dir}')

    epoch_ckpts = []
    for name in os.listdir(ckpt_dir):
        if name.startswith('model_epoch_') and name.endswith('.pt'):
            try:
                epoch = int(name.replace('model_epoch_', '').replace('.pt', ''))
                epoch_ckpts.append((epoch, os.path.join(ckpt_dir, name)))
            except ValueError:
                continue

    if epoch_ckpts:
        epoch_ckpts.sort(key=lambda item: item[0])
        return epoch_ckpts[-1][1]

    best_path = os.path.join(ckpt_dir, 'model_best.pt')
    if os.path.exists(best_path):
        return best_path

    any_pt = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if any_pt:
        any_pt.sort(key=lambda p: os.path.getmtime(p))
        return any_pt[-1]

    raise FileNotFoundError(f'No checkpoint files found in: {ckpt_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./input_4.27/')
    parser.add_argument('--exp_folder', default='./save/run_test_run_20260506_1534/')
    parser.add_argument('--ckpt', default=None, help='可选：指定checkpoint路径；不指定时自动选最新的 model_epoch_*.pt')
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--device', default='cpu', help='cpu 或 cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    print('Using device:', device)

    # 1) 数据集检查
    print('\n== 数据集检查 ==')
    ds = MultiChannelWindScenarioDataset(data_path=args.data_path, mode='test')
    print('dataset.max_values[:3]=', ds.max_values[:3])
    sample = ds[0]
    print_stats('sample residual_3ch (numpy)', sample['residual_3ch'].numpy())
    print_stats('sample forecast_3ch (numpy)', sample['forecast_3ch'].numpy())
    print_stats('sample cond_matrix (numpy)', sample['cond_matrix'].numpy())

    # 2) 加载模型与checkpoint
    print('\n== 模型检查 ==')
    ckpt_path = resolve_checkpoint_path(args.exp_folder, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    print('Loading checkpoint:', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_state = ckpt.get('model_state_dict', ckpt)
    ckpt_config = ckpt.get('config', None)
    if ckpt_config is None:
        print('注意：Checkpoint 未包含 config，脚本将使用默认配置构建模型（可能与训练时不一致）')
        ckpt_config = {
            'in_channels': 14,
            'out_channels': 3,
            'base_channels': 128,
            'num_layers': 4,
            'd_time': 64,
            'num_steps': 50,
            'beta_start': 0.0001,
            'beta_end': 0.5,
            'schedule': 'quad',
            'guidance_scale': 1.0
        }

    model = MultiChannelCSDI(ckpt_config, device=str(device)).to(device)

    print('Loading state_dict with strict=False ...')
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    print('missing keys count:', len(missing))
    if missing:
        print('  some missing keys (first 10):', missing[:10])
    print('unexpected keys count:', len(unexpected))
    if unexpected:
        print('  some unexpected keys (first 10):', unexpected[:10])

    # 打印前若干参数统计
    print('\n参数统计（前20个参数）:')
    for i, (n, p) in enumerate(model.named_parameters()):
        if i >= 20:
            break
        data = p.data.cpu().numpy()
        print(f'  {n}: shape={data.shape}, mean={data.mean():.6e}, std={data.std():.6e}, norm={np.linalg.norm(data):.6e}')

    # 3) 数据加载器与一次生成
    print('\n== 一次生成测试 ==')
    # 使用 batch_size=1 快速测试
    test_loader, _, max_values = get_dataloader_multivariate(args.data_path, batch_size=1, mode='test')

    batch = next(iter(test_loader))
    # batch 中为 torch.Tensor
    b_forecast = batch['forecast_3ch']
    b_res = batch['residual_3ch']
    b_cond = batch['cond_matrix']

    print('batch forecast stats (tensor):', 'min', float(b_forecast.min()), 'max', float(b_forecast.max()), 'abs_max', float(b_forecast.abs().max()))
    print('batch residual stats (tensor):', 'min', float(b_res.min()), 'max', float(b_res.max()), 'abs_max', float(b_res.abs().max()))
    print('batch cond_matrix stats (tensor):', 'min', float(b_cond.min()), 'max', float(b_cond.max()))
    print('max_values (from dataloader):', max_values[:3])

    model.eval()
    with torch.no_grad():
        samples = model.generate(batch, n_samples=args.n_samples)

    # samples: (B, n_samples, 3, 168) - torch.Tensor
    print('\n生成样本统计:')
    print('  shape:', samples.shape)
    print('  min:', float(samples.min()), ' max:', float(samples.max()), ' mean:', float(samples.mean()), ' std:', float(samples.std()))
    print('  any NaN:', bool(torch.isnan(samples).any()))

    # 简单比较：生成样本与真实 residual 范围
    sample_res = b_res.numpy()
    print('真实 residual stats (numpy): min', sample_res.min(), 'max', sample_res.max(), 'mean', sample_res.mean(), 'std', sample_res.std())

    # 若需要，可以保存少量样本到文件以供分析
    out_dir = os.path.join(args.exp_folder, 'debug_out')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'batch_forecast.npy'), b_forecast.numpy())
    np.save(os.path.join(out_dir, 'batch_residual.npy'), b_res.numpy())
    np.save(os.path.join(out_dir, 'generated_samples.npy'), samples.cpu().numpy())
    print('已保存 debug 输出到:', out_dir)

    print('\nDone')


if __name__ == '__main__':
    main()
