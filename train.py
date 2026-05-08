#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多变量协同条件扩散模型 - 训练脚本

使用方法:
    python train.py --exp_name my_experiment
    python train.py --exp_name test --epochs 100 --patience 5 --use_lr_scheduler
"""

import argparse
import os
import torch
import yaml
from datetime import datetime

from dataset_multivariate import get_dataloader_multivariate
from diff_models_multivariate import MultiChannelCSDI
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class EarlyStopping:
    """早停机制类"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Args:
            patience: 验证损失不改善的最大轮数
            min_delta: 认为是改善的最小损失减少量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.stop_training = False
        
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """检查是否应该早停"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
            return self.stop_training
    
    def get_patience_info(self) -> str:
        """获取patience进度信息"""
        return f"patience: {self.wait}/{self.patience}"


def create_experiment_folder(base_path, exp_name):
    """创建带时间戳的实验文件夹"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    folder_name = f'run_{exp_name}_{timestamp}'
    exp_folder = os.path.join(base_path, folder_name)
    
    if os.path.exists(exp_folder):
        import random
        suffix = random.randint(100, 999)
        folder_name = f'run_{exp_name}_{timestamp}_{suffix}'
        exp_folder = os.path.join(base_path, folder_name)
    
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'logs'), exist_ok=True)
    
    print(f"实验文件夹: {exp_folder}")
    return exp_folder


def create_lr_scheduler(optimizer, config):
    """
    创建学习率调度器 (warmup + step decay)
    
    Args:
        optimizer: 优化器
        config: 配置字典
    Returns:
        学习率调度器
    """
    lr_config = config.get('lr_scheduler', {})
    
    if not lr_config.get('enabled', False):
        return None
    
    initial_lr = lr_config.get('initial_lr', 1e-4)
    target_lr = config['train']['lr']
    warmup_epochs = lr_config.get('warmup_epochs', 10)
    decay_epochs = lr_config.get('decay_epochs', 50)
    decay_factor = lr_config.get('decay_factor', 0.5)
    min_lr = lr_config.get('min_lr', 1e-6)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup阶段: 线性增长
            warmup_factor = (epoch + 1) / warmup_epochs
            return target_lr / initial_lr * warmup_factor
        else:
            # 衰减阶段
            decay_times = (epoch - warmup_epochs) // decay_epochs
            decay_factor_total = decay_factor ** decay_times
            final_lr = target_lr / initial_lr * decay_factor_total
            min_lr_ratio = min_lr / initial_lr
            return max(final_lr, min_lr_ratio)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"学习率调度策略:")
    print(f"  - 初始学习率: {initial_lr}")
    print(f"  - 目标学习率: {target_lr}")
    print(f"  - Warmup轮数: {warmup_epochs}")
    print(f"  - 衰减间隔: {decay_epochs}轮")
    print(f"  - 衰减因子: {decay_factor}")
    print(f"  - 最小学习率: {min_lr}")
    
    return scheduler


def train(model, train_loader, val_loader, config, device, exp_folder, save_every=50, patience=5, use_lr_scheduler=False):
    """训练模型（带早停机制和学习率调度）"""
    epochs = config['train']['epochs']
    lr = config['train']['lr']
    
    # 学习率配置
    if use_lr_scheduler and config.get('lr_scheduler', {}).get('enabled', False):
        initial_lr = config['lr_scheduler'].get('initial_lr', 1e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(optimizer, config)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
        print(f"使用固定学习率: {lr}")
    
    log_file = os.path.join(exp_folder, 'logs', 'train_log.txt')
    
    # 早停机制
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)
    
    print(f"开始训练: epochs={epochs}, patience={patience}")
    
    # 记录 loss 曲线
    train_losses = []
    val_losses = []

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"训练开始: {datetime.now()}\nEpochs: {epochs}\nPatience: {patience}\n")
        f.write(f"LR: {lr}\nLR Scheduler: {use_lr_scheduler}\n\n")

    # 计算并打印 alpha_hat[-1]（用于检查噪声强度）
    mcfg = config.get('model', {})
    T_check = mcfg.get('num_steps', 50)
    beta_start = mcfg.get('beta_start', 0.0001)
    beta_end = mcfg.get('beta_end', 0.02)
    schedule = mcfg.get('schedule', 'quad')
    if schedule == 'quad':
        beta_vec = np.linspace(beta_start**0.5, beta_end**0.5, T_check) ** 2
    else:
        beta_vec = np.linspace(beta_start, beta_end, T_check)
    alpha = 1.0 - beta_vec
    alpha_hat = np.cumprod(alpha)
    alpha_hat_last = float(alpha_hat[-1])
    print(f"alpha_hat[-1] (T={T_check}, schedule={schedule}): {alpha_hat_last:.6e}")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"alpha_hat_last: {alpha_hat_last}\n")
    if alpha_hat_last > 0.001:
        warn_msg = ("WARNING: alpha_hat[-1] > 0.001 — the final signal retention is large; "
                    "noise may be too small. Consider increasing beta_end.")
        print(warn_msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(warn_msg + "\n")
    
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = model(batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # ========== 日志记录 ==========
        current_lr = optimizer.param_groups[0]['lr']
        patience_info = early_stopping.get_patience_info()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, {patience_info}")
            if scheduler:
                print(f"  学习率: {current_lr:.6f}")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, LR={current_lr:.6f}\n")
        
        # ========== 早停检查 ==========
        if not early_stopping(avg_val_loss, epoch + 1):
            # 保存最佳模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, os.path.join(exp_folder, 'checkpoints', 'model_best.pt'))
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  → 最佳模型已保存 (Val Loss: {early_stopping.best_loss:.4f})")
        else:
            print(f"\n早停触发! 最佳epoch: {early_stopping.best_epoch}, 最佳Val Loss: {early_stopping.best_loss:.4f}")
            break
        
        # ========== 学习率调度 ==========
        if scheduler is not None:
            scheduler.step()
        
        # ========== 定期保存 ==========
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, os.path.join(exp_folder, 'checkpoints', f'model_epoch_{epoch+1}.pt'))
    
    # ========== 训练结束：绘制 Loss 曲线并检查是否过拟合 ==========
    loss_plot_path = os.path.join(exp_folder, 'logs', 'loss_curve.png')
    try:
        plot_loss_curve(train_losses, val_losses, loss_plot_path)
        # 检查验证集loss是否有“不降反升”趋势
        min_val = min(val_losses) if val_losses else float('inf')
        final_val = val_losses[-1] if val_losses else None
        overfit = False
        if final_val is not None and final_val > min_val + 1e-4:
            overfit = True
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Loss plot saved: {loss_plot_path}\n")
                f.write(f"Val min: {min_val:.6f}, Val final: {final_val:.6f}, Overfit: {overfit}\n")
        print(f"Loss plot saved to {loss_plot_path}. Overfitting detected: {overfit}")
    except Exception as e:
        print('Failed to plot loss curve:', e)

    return early_stopping.best_epoch


def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_sampling_progress(model, batch, device, exp_folder, record_steps=None, n_samples=1):
    """
    记录并绘制一次采样过程中指定时间步的 x_t 波形。
    record_steps: list of integer timesteps to record (e.g., [499,249,99,49,0])
    """
    if record_steps is None:
        record_steps = [model.diffusion.num_steps - 1, model.diffusion.num_steps // 2,
                        max(0, model.diffusion.num_steps // 5), max(0, model.diffusion.num_steps // 10), 0]

    model.eval()
    # 准备条件
    forecast_3ch = batch['forecast_3ch'].to(device)
    time_encoding = batch['time_encoding'].to(device)
    cond_full = torch.cat([forecast_3ch, time_encoding], dim=1)
    cond_matrix = batch['cond_matrix'].to(device)
    timepoints = batch['timepoints'].to(device)
    time_feat = model.get_time_features(timepoints)

    B = cond_full.shape[0]
    rec = {t: [] for t in record_steps}

    with torch.no_grad():
        for s in range(n_samples):
            x_t = torch.randn(B, 3, cond_full.shape[-1], device=device)
            for t in range(model.diffusion.num_steps - 1, -1, -1):
                if t in record_steps:
                    rec[t].append(x_t.detach().cpu().numpy())
                x_t = model.diffusion.denoise_step(x_t, t, cond_full, cond_matrix, time_feat)
            # final
            if 0 in record_steps:
                rec[0].append(x_t.detach().cpu().numpy())

    # 选择第一个样本的记录绘图
    steps_sorted = sorted(record_steps, reverse=True)
    sample0 = {t: np.squeeze(np.array(rec[t])[0], axis=0) for t in steps_sorted}

    # 绘制：3行 x len(steps)列
    n_cols = len(steps_sorted)
    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9))
    for i, t in enumerate(steps_sorted):
        data = sample0[t]  # shape (3, L)
        L = data.shape[1]
        for ch in range(3):
            ax = axes[ch, i] if n_cols > 1 else axes[ch]
            ax.plot(np.arange(L), data[ch], lw=1)
            if ch == 0:
                ax.set_title(f't={t}')
            if i == 0:
                ax.set_ylabel(f'Channel {ch}')
            ax.grid(True)
    plt.tight_layout()
    out_path = os.path.join(exp_folder, 'logs', 'sampling_progress.png')
    fig.savefig(out_path)
    plt.close(fig)

    # 统计最终生成样本的物理边界
    final = sample0[0]  # (3, L)
    stats = {
        'min': float(final.min()),
        'max': float(final.max()),
        'mean': float(final.mean())
    }
    with open(os.path.join(exp_folder, 'logs', 'sampling_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(str(stats) + '\n')

    print('Sampling stats:', stats)
    # 检查是否超出物理范围 [-1, 1]
    if stats['min'] < -1.0 or stats['max'] > 1.0:
        msg = 'WARNING: generated residuals exceed physical range [-1,1]'
        print(msg)
        with open(os.path.join(exp_folder, 'logs', 'sampling_stats.txt'), 'a', encoding='utf-8') as f:
            f.write(msg + '\n')


def main():
    parser = argparse.ArgumentParser(description='多变量协同条件扩散模型 - 训练')
    parser.add_argument('--config', default='config/wind_scenario.yaml')
    parser.add_argument('--data_path', default='./input_4.27/')
    parser.add_argument('--save_path', default='./save/')
    parser.add_argument('--exp_name', default='wind_scenario')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--use_lr_scheduler', action='store_true', help='启用学习率调度器')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖
    if args.epochs: config['train']['epochs'] = args.epochs
    if args.lr: config['train']['lr'] = args.lr
    if args.batch_size: config['train']['batch_size'] = args.batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 数据加载
    train_loader, _, _ = get_dataloader_multivariate(
        args.data_path, config['train']['batch_size'], 'train', config['model']['n_intervals'])
    val_loader, _, _ = get_dataloader_multivariate(
        args.data_path, config['train']['batch_size'], 'val', config['model']['n_intervals'])
    
    # 模型
    model = MultiChannelCSDI(config['model'], device).to(device)
    
    # 训练
    exp_folder = create_experiment_folder(args.save_path, args.exp_name)
    with open(os.path.join(exp_folder, 'config_used.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    best_epoch = train(model, train_loader, val_loader, config, device, exp_folder, 
                       args.save_every, args.patience, args.use_lr_scheduler)
    
    print(f"\n训练完成! 最佳epoch: {best_epoch}")
    print(f"模型保存: {exp_folder}/checkpoints/model_best.pt")
    print(f"\n提示: 使用 generate.py 在测试集上生成场景并计算评估指标")

    # 采样过程深度可视化（记录若干中间时间步波形）
    try:
        val_iter = iter(val_loader)
        sample_batch = next(val_iter)
        record_steps = [model.diffusion.num_steps - 1, model.diffusion.num_steps // 2,
                        max(0, model.diffusion.num_steps // 5), max(0, model.diffusion.num_steps // 10), 0]
        visualize_sampling_progress(model, sample_batch, device, exp_folder, record_steps=record_steps, n_samples=1)
        print(f"采样进程图已保存到: {os.path.join(exp_folder, 'logs', 'sampling_progress.png')}")
    except Exception as e:
        print('采样可视化失败:', e)


if __name__ == '__main__':
    main()