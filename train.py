#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多变量协同条件扩散模型 - 训练脚本

使用方法:
    python train.py --exp_name my_experiment
    python train.py --exp_name test --epochs 100 --patience 5
"""

import argparse
import os
import torch
import yaml
from datetime import datetime

from dataset_multivariate import get_dataloader_multivariate
from diff_models_multivariate import MultiChannelCSDI


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


def train(model, train_loader, val_loader, config, device, exp_folder, save_every=50, patience=5):
    """训练模型（带早停机制）"""
    epochs = config['train']['epochs']
    lr = config['train']['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log_file = os.path.join(exp_folder, 'logs', 'train_log.txt')
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print(f"开始训练: epochs={epochs}, lr={lr}, patience={patience}")
    
    with open(log_file, 'w') as f:
        f.write(f"训练开始: {datetime.now()}\nEpochs: {epochs}\nLR: {lr}\nPatience: {patience}\n\n")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = model(batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        # 日志
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}\n")
        
        # 早停判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, os.path.join(exp_folder, 'checkpoints', 'model_best.pt'))
            print(f"  → 最佳模型已保存 (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发! 最佳epoch: {best_epoch}, 最佳Val Loss: {best_val_loss:.4f}")
                break
        
        # 定期保存
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, os.path.join(exp_folder, 'checkpoints', f'model_epoch_{epoch+1}.pt'))
    
    return best_epoch


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
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
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
    with open(os.path.join(exp_folder, 'config_used.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    best_epoch = train(model, train_loader, val_loader, config, device, exp_folder, args.save_every, args.patience)
    
    print(f"\n训练完成! 最佳epoch: {best_epoch}")
    print(f"模型保存: {exp_folder}/checkpoints/model_best.pt")


if __name__ == '__main__':
    main()