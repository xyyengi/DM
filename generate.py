#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多变量协同条件扩散模型 - 预测脚本

使用方法:
    python predict.py --exp_name wind_scenario
    python predict.py --exp_name wind_scenario --ckpt_epoch 100 --n_samples 50
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import yaml
from datetime import datetime

from dataset_multivariate import get_dataloader_multivariate
from diff_models_multivariate import MultiChannelCSDI
from evaluation import evaluate_multichannel, print_metrics
from src.eval.experiment_logger import (
    append_summary,
    build_summary_row,
    save_metrics_json,
    timestamp_now,
)
from src.training.residual_standardization import inverse_standardize_residual


def apply_experiment_switches(config):
    """Copy top-level target/condition switches into model config for the current code path."""
    model_cfg = config.setdefault('model', {})
    target_cfg = config.get('target', {})
    condition_cfg = config.get('condition', {})
    guidance_cfg = config.get('guidance', {})
    sampling_cfg = config.get('sampling', {})

    if 'type' in target_cfg:
        model_cfg['target_type'] = target_cfg['type']
    model_cfg['residual_standardization_enabled'] = bool(
        target_cfg.get('residual_standardization', {}).get('enabled', False)
    )
    if 'mode' in condition_cfg:
        model_cfg['condition_mode'] = condition_cfg['mode']
    if 'use_forecast' in condition_cfg:
        model_cfg['use_forecast'] = condition_cfg['use_forecast']
    if 'use_network_condition' in condition_cfg:
        model_cfg['use_network_condition'] = condition_cfg['use_network_condition']
    if 'use_guidance' in condition_cfg:
        model_cfg['use_guidance'] = condition_cfg['use_guidance']
    if 'cond_mask' in condition_cfg:
        model_cfg['cond_mask'] = condition_cfg['cond_mask']
    if 'forecast_features' in condition_cfg:
        model_cfg['forecast_features'] = condition_cfg['forecast_features']
    if 'enable' in guidance_cfg:
        model_cfg['use_guidance'] = guidance_cfg['enable']
    if {'wind_scale', 'pv_scale', 'load_scale'} <= set(guidance_cfg):
        model_cfg['guidance_scales'] = [
            guidance_cfg['wind_scale'],
            guidance_cfg['pv_scale'],
            guidance_cfg['load_scale'],
        ]
        model_cfg['guidance_scale'] = max(model_cfg['guidance_scales'])
    if 'input_channels' in model_cfg:
        model_cfg['in_channels'] = model_cfg['input_channels']
    if 'reverse_variance_type' in sampling_cfg:
        model_cfg['reverse_variance_type'] = sampling_cfg['reverse_variance_type']
    return config


def find_experiment_folders(base_path, keyword=None):
    """查找实验文件夹"""
    if not os.path.exists(base_path):
        return []
    all_folders = [
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
        and os.path.exists(os.path.join(base_path, f, 'checkpoints'))
    ]
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


def generate_scenarios(model, data_loader, device, n_samples=10, max_batches=None):
    """生成场景"""
    model.eval()
    all_samples, all_forecast, all_residual, all_actual = [], [], [], []
    
    total_batches = len(data_loader)
    print(f"生成场景 (n_samples={n_samples}, 总批次: {total_batches})...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            total_msg = max_batches if max_batches is not None else total_batches
            print(f"  generating batch {batch_idx + 1}/{total_msg} ...", flush=True)
            samples = model.generate(batch, n_samples=n_samples)
            all_samples.append(samples.cpu().numpy())
            all_forecast.append(batch['forecast_3ch'].numpy())
            all_residual.append(batch['residual_3ch'].numpy())
            all_actual.append(batch['actual_3ch'].numpy())
            
            # 进度提示
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                total_msg = max_batches if max_batches is not None else total_batches
                print(f"  已完成 {batch_idx + 1}/{total_msg} 批次")
    
    print(f"生成完成!")
    return (
        np.concatenate(all_samples),
        np.concatenate(all_forecast),
        np.concatenate(all_residual),
        np.concatenate(all_actual),
    )


def model_output_to_residual(samples, target_type, residual_standardizer=None):
    """Convert model-space output to normalized-power residual coordinates."""
    if target_type != 'residual':
        return samples
    if residual_standardizer is None:
        return samples
    return inverse_standardize_residual(
        samples,
        residual_standardizer,
        channel_axis=2,
    ).astype(samples.dtype, copy=False)


def model_output_to_actual(
    samples,
    forecast,
    target_type,
    residual_standardizer=None,
):
    """Convert model output to normalized-power actual scenarios."""
    if target_type == 'residual':
        residual_samples = model_output_to_residual(
            samples, target_type, residual_standardizer
        )
        # residual = forecast - actual, therefore actual = forecast - residual.
        return forecast[:, None, :, :] - residual_samples
    return samples


def load_denormalization_scales(data_path, fallback_max_values):
    """Return [wind, solar, load] multiplicative scales and their provenance."""
    params_path = os.path.join(data_path, 'normalization_params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        scales = np.asarray([
            params['wind_total_capacity'],
            params['solar_total_capacity'],
            params['load_denominator'],
        ], dtype=np.float64)
        source = params_path
    else:
        scales = np.asarray(fallback_max_values[:3], dtype=np.float64)
        source = 'dataset.max_values (legacy fallback)'

    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError(f'Invalid denormalization scales from {source}: {scales}')
    return scales, source


def denormalize_channels(values, scales):
    """Denormalize arrays whose channel axis is second-to-last: [..., C, L]."""
    shape = [1] * values.ndim
    shape[-2] = 3
    return (values * scales.reshape(shape)).astype(values.dtype, copy=False)


def save_scenarios_npz(actual_samples, forecast, save_path):
    """Save UC-facing scenario file with flattened equal-probability scenarios."""
    N, n_samples, C, L = actual_samples.shape
    flat = actual_samples.reshape(N * n_samples, C, L)
    prob = np.ones(N * n_samples, dtype=np.float64) / float(N * n_samples)
    samples_dir = os.path.join(save_path, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    scenario_path = os.path.join(samples_dir, 'scenarios.npz')
    np.savez(
        scenario_path,
        wind=flat[:, 0, :],
        pv=flat[:, 1, :],
        load=flat[:, 2, :],
        prob=prob,
        forecast_wind=forecast[0, 0, :],
        forecast_pv=forecast[0, 1, :],
        forecast_load=forecast[0, 2, :],
    )
    return scenario_path


def add_basic_mae(metrics, actual_samples, actual):
    """Add scenario-mean MAE fields expected by experiment_summary.csv."""
    scenario_mean = np.mean(actual_samples, axis=1)
    abs_err = np.abs(scenario_mean - actual)
    channel_mae = np.mean(abs_err, axis=(0, 2))
    metrics['wind_MAE'] = float(channel_mae[0])
    metrics['pv_MAE'] = float(channel_mae[1])
    metrics['load_MAE'] = float(channel_mae[2])
    metrics['mean_MAE'] = float(np.mean(channel_mae))
    return metrics


def resolve_result_folder(exp_folder, output_dir, data_split, n_samples, seed):
    """Choose a safe result folder; validation never overwrites a training run."""
    if output_dir:
        return output_dir
    if data_split == 'val':
        return f"{exp_folder}_val_n{n_samples}_seed{seed}"
    return exp_folder


def evaluate_and_save(
    samples,
    forecast,
    residual,
    actual,
    max_values,
    data_path,
    save_path,
    config,
    config_path=None,
    checkpoint_path=None,
    target_type='residual',
    residual_standardizer=None,
    data_split='test',
):
    """评估并保存结果（使用论文公式12-15）"""
    N, n_samples, C, L = samples.shape
    residual_samples_norm = model_output_to_residual(
        samples, target_type, residual_standardizer
    )
    actual_samples_norm = model_output_to_actual(
        samples, forecast, target_type, residual_standardizer
    )
    scales, scale_source = load_denormalization_scales(data_path, max_values)

    samples_physical = denormalize_channels(residual_samples_norm, scales)
    actual_samples = denormalize_channels(actual_samples_norm, scales)
    forecast_physical = denormalize_channels(forecast, scales)
    residual_physical = denormalize_channels(residual, scales)
    target = denormalize_channels(actual, scales)

    print(
        'Denormalization: '
        f'wind={scales[0]:.6f}, solar={scales[1]:.6f}, '
        f'load={scales[2]:.6f}; source={scale_source}'
    )
    
    # 使用evaluation模块计算完整指标
    metrics = evaluate_multichannel(actual_samples, target)
    metrics = add_basic_mae(metrics, actual_samples, target)
    print_metrics(metrics)
    
    # 保存结果
    os.makedirs(save_path, exist_ok=True)
    # Canonical result files are in physical power units (MW).
    np.save(os.path.join(save_path, 'generated_samples.npy'), samples_physical)
    np.save(os.path.join(save_path, 'actual_scenarios.npy'), actual_samples)
    np.save(os.path.join(save_path, 'forecast_data.npy'), forecast_physical)
    np.save(os.path.join(save_path, 'residual_data.npy'), residual_physical)
    np.save(os.path.join(save_path, 'actual_data.npy'), target)

    # Keep normalized-space arrays for reproducibility and diagnostics.
    np.save(os.path.join(save_path, 'generated_samples_normalized.npy'), residual_samples_norm)
    if residual_standardizer is not None:
        np.save(os.path.join(save_path, 'generated_samples_standardized.npy'), samples)
    np.save(os.path.join(save_path, 'actual_scenarios_normalized.npy'), actual_samples_norm)
    np.save(os.path.join(save_path, 'forecast_data_normalized.npy'), forecast)
    np.save(os.path.join(save_path, 'residual_data_normalized.npy'), residual)
    np.save(os.path.join(save_path, 'actual_data_normalized.npy'), actual)

    with open(os.path.join(save_path, 'denormalization_used.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'channel_order': ['wind', 'solar', 'load'],
            'scales': scales.tolist(),
            'source': scale_source,
            'output_unit': 'MW',
            'normalized_copies_suffix': '_normalized.npy',
            'residual_standardization': residual_standardizer,
            'generated_samples_normalized_semantics': 'forecast_minus_actual residual in normalized power coordinates',
            'data_split': data_split,
        }, f, ensure_ascii=False, indent=2)

    scenario_path = save_scenarios_npz(actual_samples, forecast_physical, save_path)
    
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

    run_id = os.path.basename(os.path.abspath(save_path))
    timestamp = config.get('experiment', {}).get('timestamp', timestamp_now())
    figure_dir = os.path.join(save_path, 'figures')
    metrics.update({
        'run_id': run_id,
        'timestamp': timestamp,
        'checkpoint_path': checkpoint_path or 'NA',
        'scenario_path': scenario_path,
        'figure_dir': figure_dir,
        'scenario_shape': str([int(N * n_samples), int(C), int(L)]),
        'reverse_variance_type': config.get('model', {}).get('reverse_variance_type', 'beta'),
        'residual_standardization_enabled': residual_standardizer is not None,
        'data_split': data_split,
    })
    metrics_path = save_metrics_json(metrics, save_path)
    summary_row = build_summary_row(
        config=config,
        run_id=run_id,
        timestamp=timestamp,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        scenario_path=scenario_path,
        figure_dir=figure_dir,
        metrics=metrics,
        notes=f'evaluated split={data_split}',
    )
    summary_path = append_summary(os.path.dirname(os.path.abspath(save_path)), summary_row)
    
    print(f"\n结果保存至: {save_path}")
    print(f"metrics.json: {metrics_path}")
    print(f"实验汇总已追加: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='多变量协同条件扩散模型 - 预测')
    parser.add_argument('--config', default='config/wind_scenario.yaml')
    parser.add_argument('--data_path', default='./input_4.27/')
    parser.add_argument('--save_path', default='./outputs/')
    parser.add_argument('--exp_name', default='wind_scenario', help='实验名称或文件夹名')
    parser.add_argument('--ckpt_epoch', type=int, default=None, help='指定epoch，默认使用最佳模型')
    parser.add_argument('--n_samples', type=int, default=10, help='生成样本数')
    parser.add_argument('--max_batches', type=int, default=None, help='最多生成多少个测试批次，用于CPU smoke test')
    parser.add_argument('--list', action='store_true', help='列出所有可用实验')
    parser.add_argument('--guidance_scale', type=float, default=None, help='覆盖配置中的guidance_scale')
    parser.add_argument('--batch_size', type=int, default=None, help='generate/evaluate batch size; default=min(train batch, 8)')
    parser.add_argument('--reverse_variance_type', choices=['beta', 'posterior'], default=None,
                        help='Override reverse-process variance without retraining')
    parser.add_argument('--output_dir', default=None,
                        help='Write generated arrays separately instead of overwriting the checkpoint run')
    parser.add_argument('--seed', type=int, default=2026, help='Generation random seed')
    parser.add_argument('--split', choices=['val', 'test'], default='test',
                        help='Data split used for generation/evaluation; tune on val and reserve test for final evaluation')
    args = parser.parse_args()
    print(f"Evaluation data split: {args.split}")
    
    # 列出所有实验
    if args.list:
        list_experiments(args.save_path)
        return
    
    # 查找实验文件夹
    if os.path.isdir(os.path.join(args.save_path, args.exp_name)):
        exp_folder = os.path.join(args.save_path, args.exp_name)
    else:
        folders = find_experiment_folders(args.save_path, args.exp_name)
        if not folders:
            print(f"未找到实验: {args.exp_name}")
            sys.exit(1)
        exp_folder = os.path.join(args.save_path, folders[0])
    
    print(f"实验文件夹: {exp_folder}")
    
    # 加载checkpoint（默认使用最佳模型 model_best.pt）
    if args.ckpt_epoch:
        ckpt_path = os.path.join(exp_folder, 'checkpoints', f'model_epoch_{args.ckpt_epoch}.pt')
        if not os.path.exists(ckpt_path):
            print(f"警告: 未找到指定 epoch checkpoint: {ckpt_path}")
            print(f"改为使用最佳模型...")
            ckpt_path = get_checkpoint_path(exp_folder, 'best')
    else:
        # 默认使用最佳模型（model_best.pt）
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
    config = apply_experiment_switches(config)
    if args.reverse_variance_type is not None:
        config.setdefault('sampling', {})['reverse_variance_type'] = args.reverse_variance_type
        config.setdefault('model', {})['reverse_variance_type'] = args.reverse_variance_type

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 数据加载
    build_kde = bool(config['model'].get('use_guidance', False))
    generate_batch_size = args.batch_size or min(int(config['train']['batch_size']), 16)
    print(f"生成batch size: {generate_batch_size}")
    data_loader, _, max_values = get_dataloader_multivariate(
        args.data_path, generate_batch_size, args.split, config['model']['n_intervals'],
        build_kde=build_kde,
        residual_standardization=config.get('target', {}).get(
            'residual_standardization', {'enabled': False}
        ))
    
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
    print(f"Reverse variance type: {model.diffusion.reverse_variance_type}")
    
    # 覆盖guidance_scale（如果命令行指定）
    if args.guidance_scale is not None:
        original_gs = model.diffusion.guidance_scale
        model.diffusion.guidance_scale = args.guidance_scale
        model.diffusion.guidance_scales.fill_(args.guidance_scale)
        print(f"覆盖guidance_scale: {original_gs} -> {args.guidance_scale}")
    
    # 生成
    samples, forecast, residual, actual = generate_scenarios(
        model,
        data_loader,
        device,
        args.n_samples,
        max_batches=args.max_batches,
    )

    result_folder = resolve_result_folder(
        exp_folder, args.output_dir, args.split, args.n_samples, args.seed
    )
    os.makedirs(result_folder, exist_ok=True)
    config.setdefault('evaluation', {})['evaluated_split'] = args.split
    config['evaluation']['n_samples'] = args.n_samples
    generation_config_path = os.path.join(result_folder, 'generation_config_used.yaml')
    with open(generation_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    
    # 保存
    evaluate_and_save(
        samples,
        forecast,
        residual,
        actual,
        max_values,
        args.data_path,
        result_folder,
        config=config,
        config_path=generation_config_path,
        checkpoint_path=ckpt_path,
        target_type=config['model'].get('target_type', 'residual'),
        residual_standardizer=data_loader.dataset.residual_standardizer,
        data_split=args.split,
    )
    
    print(f"\n完成!")


if __name__ == '__main__':
    main()
