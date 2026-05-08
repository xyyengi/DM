#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本 - 排查场景生成流程问题

检查项目：
1. 数据加载是否正确
2. 模型生成是否正常
3. 数据尺度是否一致
4. 评估流程是否正确
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 导入项目模块
from dataset_multivariate import MultiChannelWindScenarioDataset, get_dataloader_multivariate
from diff_models_multivariate import ResUNet, GaussianDiffusionMultivariate, MultiChannelCSDI


def check_data_loading(data_path='./input_4.27/'):
    """检查数据加载"""
    print("\n" + "="*60)
    print("【1. 数据加载检查】")
    print("="*60)
    
    # 检查文件是否存在
    required_files = ['train_pred.npy', 'train_res.npy', 'test_pred.npy', 'test_res.npy']
    for f in required_files:
        path = os.path.join(data_path, f)
        if os.path.exists(path):
            data = np.load(path)
            print(f"✓ {f}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")
        else:
            print(f"✗ {f}: 文件不存在")
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = MultiChannelWindScenarioDataset(data_path=data_path, mode='test')
    
    # 检查单个样本
    sample = dataset[0]
    print(f"\n样本结构:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}]")
    
    # 检查max_values
    print(f"\nmax_values (前3个通道): {dataset.max_values[:3]}")
    
    return dataset


def check_model_generation(exp_folder, dataset, device='cuda'):
    """检查模型生成"""
    print("\n" + "="*60)
    print("【2. 模型生成检查】")
    print("="*60)
    
    # 查找checkpoint
    ckpt_path = os.path.join(exp_folder, 'checkpoints')
    if not os.path.exists(ckpt_path):
        print(f"✗ checkpoint目录不存在: {ckpt_path}")
        return None
    
    # 找最佳模型
    best_path = os.path.join(ckpt_path, 'model_best.pt')
    if os.path.exists(best_path):
        model_path = best_path
        print(f"✓ 找到最佳模型: model_best.pt")
    else:
        # 找最新的epoch模型
        ckpts = [f for f in os.listdir(ckpt_path) if f.startswith('model_epoch_')]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.replace('model_epoch_', '').replace('.pt', '')))
            model_path = os.path.join(ckpt_path, ckpts[-1])
            print(f"⚠ 未找到model_best.pt，使用: {ckpts[-1]}")
        else:
            print(f"✗ 无可用checkpoint")
            return None
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    print(f"\nCheckpoint内容: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        print(f"  epoch: {epoch}, loss: {loss}")
    else:
        state_dict = checkpoint
    
    # 从checkpoint的config创建完整模型
    ckpt_config = checkpoint.get('config', {})
    
    # 创建完整的MultiChannelCSDI模型
    model = MultiChannelCSDI(
        config=ckpt_config if ckpt_config else {
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
        },
        device=device
    ).to(device)
    
    # 使用strict=False加载，因为checkpoint可能缺少buffer
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"✓ MultiChannelCSDI模型加载成功")
    if missing_keys:
        print(f"  缺失的keys (buffer): {missing_keys[:3]}...")
        # 手动初始化缺失的buffer（beta, alpha, alpha_hat）
        # 这些是扩散过程的核心参数！
        print(f"  正在重新初始化扩散参数...")
    if unexpected_keys:
        print(f"  多余的keys: {unexpected_keys[:3]}...")
    model.eval()
    
    # 检查diffusion的buffer值
    print(f"\n扩散参数检查:")
    print(f"  beta范围: [{model.diffusion.beta.min():.6f}, {model.diffusion.beta.max():.6f}]")
    print(f"  alpha范围: [{model.diffusion.alpha.min():.6f}, {model.diffusion.alpha.max():.6f}]")
    print(f"  alpha_hat范围: [{model.diffusion.alpha_hat.min():.6f}, {model.diffusion.alpha_hat.max():.6f}]")
    print(f"  alpha_hat[-1] (最终值): {model.diffusion.alpha_hat[-1]:.6f}")
    
    # 检查beta_end是否过大
    if model.diffusion.beta.max() > 0.1:
        print(f"  ⚠ 警告：beta_end={model.diffusion.beta.max():.4f} 过大！标准值应<0.1")
        print(f"  这会导致扩散过程不稳定，生成结果范围异常")
    
    # 测试生成
    print(f"\n测试生成...")
    sample = dataset[0]
    
    # 准备输入
    input_14ch = sample['input_14ch'].unsqueeze(0).to(device)  # (1, 14, 168)
    forecast_3ch = sample['forecast_3ch'].unsqueeze(0).to(device)  # (1, 3, 168)
    cond_matrix = sample['cond_matrix'].unsqueeze(0).to(device)  # (1, 3, 168, 2)
    time_encoding = sample['time_encoding'].unsqueeze(0).to(device)  # (1, 8, 168)
    
    print(f"输入形状:")
    print(f"  input_14ch: {input_14ch.shape}")
    print(f"  forecast_3ch: {forecast_3ch.shape}")
    print(f"  cond_matrix: {cond_matrix.shape}")
    
    # 生成 - 使用MultiChannelCSDI的generate方法
    with torch.no_grad():
        # 准备batch
        batch = {
            'forecast_3ch': forecast_3ch,
            'time_encoding': time_encoding,
            'cond_matrix': cond_matrix,
            'timepoints': torch.arange(168, device=device).unsqueeze(0).float()
        }
        
        # 使用model.generate生成
        samples = model.generate(batch, n_samples=1)
        generated = samples[0, 0].cpu().numpy()  # (3, 168)
    
    print(f"\n生成结果:")
    print(f"  shape: {generated.shape}")
    print(f"  range: [{generated.min():.4f}, {generated.max():.4f}]")
    print(f"  mean: {generated.mean():.4f}, std: {generated.std():.4f}")
    
    # 对比真实残差
    residual_3ch = sample['residual_3ch'].numpy()  # (3, 168)
    print(f"\n真实残差:")
    print(f"  shape: {residual_3ch.shape}")
    print(f"  range: [{residual_3ch.min():.4f}, {residual_3ch.max():.4f}]")
    print(f"  mean: {residual_3ch.mean():.4f}, std: {residual_3ch.std():.4f}")
    
    return generated, residual_3ch


def check_evaluation_scale(data_path='./input_4.27/'):
    """检查评估时的数据尺度"""
    print("\n" + "="*60)
    print("【3. 数据尺度检查】")
    print("="*60)
    
    # 加载原始数据
    test_pred = np.load(os.path.join(data_path, 'test_pred.npy'))
    test_res = np.load(os.path.join(data_path, 'test_res.npy'))
    
    print(f"原始数据 (未归一化):")
    print(f"  test_pred: shape={test_pred.shape}, range=[{test_pred.min():.4f}, {test_pred.max():.4f}]")
    print(f"  test_res: shape={test_res.shape}, range=[{test_res.min():.4f}, {test_res.max():.4f}]")
    
    # 检查各通道
    channel_names = ['wind', 'solar', 'load']
    for c, name in enumerate(channel_names):
        pred_c = test_pred[:, :, c]
        res_c = test_res[:, :, c]
        print(f"\n  【{name}】通道:")
        print(f"    pred: range=[{pred_c.min():.4f}, {pred_c.max():.4f}], mean={pred_c.mean():.4f}")
        print(f"    res:  range=[{res_c.min():.4f}, {res_c.max():.4f}], mean={res_c.mean():.4f}")
    
    # 模拟归一化
    max_values = np.max(np.abs(test_pred), axis=(0, 1))
    max_values = np.maximum(max_values, 1e-6)
    
    pred_norm = test_pred / max_values
    res_norm = test_res / max_values
    
    print(f"\n归一化后:")
    print(f"  pred_norm: range=[{pred_norm.min():.4f}, {pred_norm.max():.4f}]")
    print(f"  res_norm: range=[{res_norm.min():.4f}, {res_norm.max():.4f}]")
    print(f"  max_values: {max_values[:3]}")
    
    # 检查反归一化
    res_denorm = res_norm * max_values
    print(f"\n反归一化验证:")
    print(f"  差异: {np.abs(res_denorm - test_res).max():.10f} (应接近0)")
    
    return max_values


def visualize_generation(generated, residual, save_path='diagnosis_plot.png'):
    """可视化生成结果"""
    print("\n" + "="*60)
    print("【4. 可视化检查】")
    print("="*60)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    channel_names = ['Wind', 'Solar', 'Load']
    
    for c in range(3):
        # 生成 vs 真实
        axes[c, 0].plot(generated[c], label='Generated', alpha=0.7)
        axes[c, 0].plot(residual[c], label='Real', alpha=0.7)
        axes[c, 0].set_title(f'{channel_names[c]} - Generated vs Real')
        axes[c, 0].legend()
        axes[c, 0].set_xlabel('Time')
        axes[c, 0].set_ylabel('Residual (normalized)')
        
        # 误差分布
        error = generated[c] - residual[c]
        axes[c, 1].hist(error, bins=30, alpha=0.7, edgecolor='black')
        axes[c, 1].axvline(x=0, color='red', linestyle='--')
        axes[c, 1].set_title(f'{channel_names[c]} - Error Distribution')
        axes[c, 1].set_xlabel('Error')
        axes[c, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ 可视化保存至: {save_path}")
    plt.close()


def main():
    print("="*60)
    print("场景生成流程诊断")
    print("="*60)
    
    # 1. 检查数据加载
    dataset = check_data_loading()
    
    # 2. 检查数据尺度
    max_values = check_evaluation_scale()
    
    # 3. 检查模型生成
    # 请修改为你的实验文件夹路径
    exp_folder = './save/run_test_run_20260506_1534/'
    if os.path.exists(exp_folder):
        result = check_model_generation(exp_folder, dataset)
        if result:
            generated, residual = result
            visualize_generation(generated, residual)
    else:
        print(f"\n⚠ 实验文件夹不存在: {exp_folder}")
        print("请使用 --exp_folder 参数指定正确的路径")
    
    print("\n" + "="*60)
    print("诊断完成")
    print("="*60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', type=str, default='./save/run_test_run_20260506_1534/', help='实验文件夹路径')
    parser.add_argument('--data_path', type=str, default='./input_4.27/', help='数据路径')
    args = parser.parse_args()
    
    # 更新路径
    if args.exp_folder:
        import diagnose
        # 重新运行检查
        dataset = check_data_loading(args.data_path)
        max_values = check_evaluation_scale(args.data_path)
        if os.path.exists(args.exp_folder):
            result = check_model_generation(args.exp_folder, dataset)
            if result:
                generated, residual = result
                visualize_generation(generated, residual)
    
    main()