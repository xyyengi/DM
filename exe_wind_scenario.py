# ============================================================================
# 多变量协同条件扩散模型执行脚本 - 实验留痕版本
# ============================================================================

import argparse
import os
import sys
import numpy as np
import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader

from dataset_multivariate import get_dataloader_multivariate
from diff_models_multivariate import MultiChannelCSDI


def create_experiment_folder(base_path, exp_name):
    """创建带时间戳的独立实验文件夹: run_[exp_name]_[YYYYMMDD_HHMM]/"""
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
    os.makedirs(os.path.join(exp_folder, 'results'), exist_ok=True)
    os.makedirs(os.path.join(exp_folder, 'logs'), exist_ok=True)
    
    print(f"实验文件夹已创建: {exp_folder}")
    return exp_folder


def find_experiment_folders(base_path, keyword=None):
    """查找所有实验文件夹"""
    if not os.path.exists(base_path):
        return []
    all_folders = [f for f in os.listdir(base_path) 
                   if f.startswith('run_') and os.path.isdir(os.path.join(base_path, f))]
    if keyword:
        return [f for f in all_folders if keyword in f]
    return all_folders


def list_checkpoints(exp_folder):
    """列出可用的checkpoint"""
    ckpt_path = os.path.join(exp_folder, 'checkpoints')
    if not os.path.exists(ckpt_path):
        return []
    ckpts = [f for f in os.listdir(ckpt_path) if f.startswith('model_epoch_') and f.endswith('.pt')]
    ckpts.sort(key=lambda x: int(x.replace('model_epoch_', '').replace('.pt', '')))
    return ckpts


def get_checkpoint_path(exp_folder, ckpt_epoch):
    """获取特定epoch的checkpoint路径"""
    ckpt_path = os.path.join(exp_folder, 'checkpoints', f'model_epoch_{ckpt_epoch}.pt')
    if not os.path.exists(ckpt_path):
        ckpts = list_checkpoints(exp_folder)
        if ckpts:
            latest = ckpts[-1]
            latest_epoch = int(latest.replace('model_epoch_', '').replace('.pt', ''))
            ckpt_path = os.path.join(exp_folder, 'checkpoints', latest)
            print(f"自动选择最新checkpoint: epoch {latest_epoch}")
        else:
            raise FileNotFoundError(f"无可用checkpoint")
    return ckpt_path


def train(model, train_loader, config, device, exp_folder, save_every=10):
    """训练模型"""
    epochs = config['train']['epochs']
    lr = config['train']['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log_file = os.path.join(exp_folder, 'logs', 'train_log.txt')
    
    print(f"开始训练，总Epochs: {epochs}")
    
    with open(log_file, 'w') as f:
        f.write(f"训练开始: {datetime.now()}\nEpochs: {epochs}\nLR: {lr}\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}: {avg_loss:.4f}\n")
        
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'config': config
            }, os.path.join(exp_folder, 'checkpoints', f'model_epoch_{epoch+1}.pt'))
    
    return model, epochs


def generate_scenarios(model, test_loader, device, n_samples=10):
    """生成场景"""
    model.eval()
    all_samples, all_forecast, all_residual = [], [], []
    
    print(f"生成场景 (n_samples={n_samples})")
    
    with torch.no_grad():
        for batch in test_loader:
            samples = model.generate(batch, n_samples=n_samples)
            all_samples.append(samples.cpu().numpy())
            all_forecast.append(batch['forecast_3ch'].numpy())
            all_residual.append(batch['residual_3ch'].numpy())
    
    return np.concatenate(all_samples), np.concatenate(all_forecast), np.concatenate(all_residual)


def evaluate_and_save(samples, forecast, residual, max_values, save_path):
    """评估并保存结果"""
    samples_denorm = samples * max_values.reshape(1, 1, 3, 1)
    residual_denorm = residual * max_values.reshape(1, 3, 1)
    
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
    
    print(f"Energy Score: {metrics['energy_score']:.4f}")
    
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'generated_samples.npy'), samples)
    np.save(os.path.join(save_path, 'forecast_data.npy'), forecast)
    np.save(os.path.join(save_path, 'metrics.txt'), str(metrics))
    print(f"结果保存至: {save_path}")


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    train_loader, kde, max_values = get_dataloader_multivariate(
        args.data_path, config['train']['batch_size'], 'train', config['model']['n_intervals'])
    test_loader, _, _ = get_dataloader_multivariate(
        args.data_path, config['train']['batch_size'], 'test', config['model']['n_intervals'])
    
    model = MultiChannelCSDI(config['model'], device).to(device)
    
    if args.mode == 'train':
        exp_folder = create_experiment_folder(args.save_path, args.exp_name)
        with open(os.path.join(exp_folder, 'config_used.yaml'), 'w') as f:
            yaml.dump(config, f)
        
        model, final_epoch = train(model, train_loader, config, device, exp_folder, args.save_every)
        samples, forecast, residual = generate_scenarios(model, test_loader, device, args.n_samples)
        evaluate_and_save(samples, forecast, residual, max_values, os.path.join(exp_folder, 'results'))
        
        print(f"完成! 实验文件夹: {exp_folder}")
    
    elif args.mode == 'predict':
        if args.exp_name.startswith('run_'):
            exp_folder = os.path.join(args.save_path, args.exp_name)
        else:
            folders = find_experiment_folders(args.save_path, args.exp_name)
            if not folders:
                print(f"未找到实验: {args.exp_name}")
                sys.exit(1)
            exp_folder = os.path.join(args.save_path, folders[0])
        
        ckpt_path = get_checkpoint_path(exp_folder, args.ckpt_epoch)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载: {ckpt_path}")
        
        samples, forecast, residual = generate_scenarios(model, test_loader, device, args.n_samples)
        predict_folder = os.path.join(exp_folder, 'results', f'predict_{datetime.now().strftime("%Y%m%d_%H%M")}')
        evaluate_and_save(samples, forecast, residual, max_values, predict_folder)
        
        print(f"完成! 结果: {predict_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多变量协同条件扩散模型')
    parser.add_argument('--config', default='config/wind_scenario.yaml')
    parser.add_argument('--data_path', default='./input_4.27/')
    parser.add_argument('--save_path', default='./save/')
    parser.add_argument('--exp_name', default='wind_scenario', help='实验名称')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--ckpt_epoch', type=int, default=200, help='加载的epoch')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=50, help='checkpoint保存间隔')
    
    main(parser.parse_args())