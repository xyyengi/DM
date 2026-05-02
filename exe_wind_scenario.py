# ============================================================================
# 多变量协同条件扩散模型执行脚本
# 
# 论文"2023-Conditional_Diffusion_Model.pdf"复现
# 
# 功能：
# 1. 训练多通道CSDI模型
# 2. 生成风、光、负荷协同场景
# 3. 评估生成场景质量
# ============================================================================

import argparse
import os
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset_multivariate import MultiChannelWindScenarioDataset, get_dataloader_multivariate
from diff_models_multivariate import MultiChannelCSDI


def get_next_version(save_path):
    """
    获取下一个模型版本编号
    
    Args:
        save_path: 模型保存路径
    Returns:
        version: 版本编号 (从1开始)
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 查找现有模型文件
    existing_models = [f for f in os.listdir(save_path) if f.startswith('model_v') and f.endswith('.pth')]
    
    if not existing_models:
        return 1
    
    # 提取版本号
    versions = []
    for f in existing_models:
        try:
            v = int(f.replace('model_v', '').replace('.pth', ''))
            versions.append(v)
        except:
            pass
    
    return max(versions) + 1 if versions else 1


def get_latest_model_path(save_path):
    """
    获取最新模型的路径
    
    Args:
        save_path: 模型保存路径
    Returns:
        model_path: 最新模型路径，如果没有则返回None
    """
    if not os.path.exists(save_path):
        return None
    
    existing_models = [f for f in os.listdir(save_path) if f.startswith('model_v') and f.endswith('.pth')]
    
    if not existing_models:
        return None
    
    # 找到最大版本号
    max_version = 0
    for f in existing_models:
        try:
            v = int(f.replace('model_v', '').replace('.pth', ''))
            if v > max_version:
                max_version = v
        except:
            pass
    
    if max_version > 0:
        return os.path.join(save_path, f'model_v{max_version}.pth')
    return None


def train(model, train_loader, config, device, save_path):
    """
    训练多通道CSDI模型
    
    Args:
        model: MultiChannelCSDI模型
        train_loader: 训练数据加载器
        config: 配置字典
        device: 设备
        save_path: 模型保存路径
    """
    epochs = config['train']['epochs']
    lr = config['train']['lr']
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 获取版本编号
    version = get_next_version(save_path)
    
    print("=" * 60)
    print(f"开始训练多变量协同条件扩散模型 (版本 v{version})")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            loss = model(batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 保存模型（带版本编号）
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f'model_v{version}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")
    
    # 同时保存一个latest链接
    latest_path = os.path.join(save_path, 'model_latest.pth')
    torch.save(model.state_dict(), latest_path)
    
    return model, version


def generate_scenarios(model, test_loader, config, device, n_samples=10):
    """
    生成风、光、负荷协同场景
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        config: 配置字典
        device: 设备
        n_samples: 每个条件生成的场景数量
    Returns:
        generated_samples: 生成的残差场景
        forecast_3ch: 预测值（风、光、负荷）
        actual_residual: 实际残差
    """
    model.eval()
    
    all_samples = []
    all_forecast_3ch = []
    all_residual = []
    
    print("=" * 60)
    print(f"开始生成场景 (每个条件生成 {n_samples} 个场景)")
    print("=" * 60)
    
    with torch.no_grad():
        for batch in test_loader:
            # 生成场景
            samples = model.generate(batch, n_samples=n_samples)
            
            all_samples.append(samples.cpu().numpy())
            all_forecast_3ch.append(batch['forecast_3ch'].numpy())  # 使用forecast_3ch (3, 168)
            all_residual.append(batch['residual_3ch'].numpy())  # 使用residual_3ch (3, 168)
    
    # 合并所有批次
    generated_samples = np.concatenate(all_samples, axis=0)  # (N_test, n_samples, 3, 168)
    forecast_3ch = np.concatenate(all_forecast_3ch, axis=0)  # (N_test, 3, 168)
    actual_residual = np.concatenate(all_residual, axis=0)  # (N_test, 3, 168)
    
    print(f"生成完成: {generated_samples.shape}")
    
    return generated_samples, forecast_3ch, actual_residual


def evaluate_scenarios(generated_samples, actual_residual, max_values):
    """
    评估生成场景质量
    
    论文Section IV.B评估指标：
    1. Energy Score (公式14)
    2. Scenario Width (PIAW)
    3. Coverage Rate
    
    Args:
        generated_samples: (N, n_samples, 3, 168) 生成的残差场景
        actual_residual: (N, 3, 168) 实际残差
        max_values: (3,) 归一化最大值
    Returns:
        metrics: 评估指标字典
    """
    N, n_samples, C, L = generated_samples.shape
    
    # 反归一化
    generated_samples_denorm = generated_samples * max_values.reshape(1, 1, 3, 1)
    actual_residual_denorm = actual_residual * max_values.reshape(1, 3, 1)
    
    metrics = {}
    
    # 1. Energy Score (论文公式14)
    # ES = (1/S) Σ ||S_i - Y|| - (1/2S²) ΣΣ ||S_i - S_j||
    # 简化计算：只计算第一项
    
    # 计算每个场景与实际值的距离
    distances = np.zeros((N, n_samples))
    for s in range(n_samples):
        diff = generated_samples_denorm[:, s] - actual_residual_denorm
        distances[:, s] = np.sqrt(np.sum(diff ** 2, axis=(1, 2)))  # Euclidean norm
    
    energy_score = np.mean(distances)
    metrics['energy_score'] = energy_score
    
    # 2. Scenario Width (PIAW) - 论文公式13
    # 计算生成场景的区间宽度
    for c in range(C):
        channel_name = ['wind', 'solar', 'load'][c]
        
        # 计算每个时间点的场景区间
        samples_c = generated_samples_denorm[:, :, c, :]  # (N, n_samples, L)
        
        # 100% quantile interval
        c_up_100 = np.max(samples_c, axis=1)  # (N, L)
        c_down_100 = np.min(samples_c, axis=1)
        width_100 = np.mean(c_up_100 - c_down_100)
        
        # 80% quantile interval
        c_up_80 = np.percentile(samples_c, 90, axis=1)
        c_down_80 = np.percentile(samples_c, 10, axis=1)
        width_80 = np.mean(c_up_80 - c_down_80)
        
        metrics[f'{channel_name}_width_100'] = width_100
        metrics[f'{channel_name}_width_80'] = width_80
    
    # 3. Coverage Rate - 实际值落在区间内的比例
    for c in range(C):
        channel_name = ['wind', 'solar', 'load'][c]
        
        samples_c = generated_samples_denorm[:, :, c, :]
        actual_c = actual_residual_denorm[:, c, :]
        
        # 100% interval coverage
        c_up_100 = np.max(samples_c, axis=1)
        c_down_100 = np.min(samples_c, axis=1)
        covered_100 = (actual_c >= c_down_100) & (actual_c <= c_up_100)
        coverage_100 = np.mean(covered_100)
        
        # 80% interval coverage
        c_up_80 = np.percentile(samples_c, 90, axis=1)
        c_down_80 = np.percentile(samples_c, 10, axis=1)
        covered_80 = (actual_c >= c_down_80) & (actual_c <= c_up_80)
        coverage_80 = np.mean(covered_80)
        
        metrics[f'{channel_name}_coverage_100'] = coverage_100
        metrics[f'{channel_name}_coverage_80'] = coverage_80
    
    # 打印评估结果
    print("=" * 60)
    print("场景评估结果")
    print("=" * 60)
    print(f"Energy Score: {metrics['energy_score']:.4f}")
    print()
    print("Scenario Width (PIAW):")
    for c in ['wind', 'solar', 'load']:
        print(f"  {c} - 100%: {metrics[f'{c}_width_100']:.2f}, 80%: {metrics[f'{c}_width_80']:.2f}")
    print()
    print("Coverage Rate:")
    for c in ['wind', 'solar', 'load']:
        print(f"  {c} - 100%: {metrics[f'{c}_coverage_100']*100:.2f}%, 80%: {metrics[f'{c}_coverage_80']*100:.2f}%")
    
    return metrics


def save_results(generated_samples, forecast_data, actual_residual, metrics, save_path):
    """
    保存生成结果和评估指标
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 保存生成场景
    np.save(os.path.join(save_path, 'generated_samples.npy'), generated_samples)
    np.save(os.path.join(save_path, 'forecast_data.npy'), forecast_data)
    np.save(os.path.join(save_path, 'actual_residual.npy'), actual_residual)
    
    # 保存评估指标
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"结果已保存至: {save_path}")


def main(args):
    """
    主函数
    """
    # 加载配置
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    data_path = args.data_path
    save_path = args.save_path
    
    # 加载数据
    print("加载数据...")
    train_loader, kde, max_values = get_dataloader_multivariate(
        data_path=data_path,
        batch_size=config['train']['batch_size'],
        mode='train',
        n_intervals=config['model']['n_intervals']
    )
    
    test_loader, _, _ = get_dataloader_multivariate(
        data_path=data_path,
        batch_size=config['train']['batch_size'],
        mode='test',
        n_intervals=config['model']['n_intervals']
    )
    
    # 创建模型
    model_config = config['model']
    model = MultiChannelCSDI(model_config, device).to(device)
    
    # 训练或加载预训练模型
    if args.mode == 'train':
        model, version = train(model, train_loader, config, device, save_path)
    else:
        # 加载预训练模型（优先加载最新版本）
        model_path = get_latest_model_path(save_path)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"已加载预训练模型: {model_path}")
        else:
            print("未找到预训练模型，开始训练...")
            model, version = train(model, train_loader, config, device, save_path)
    
    # 生成场景
    n_samples = args.n_samples
    generated_samples, forecast_data, actual_residual = generate_scenarios(
        model, test_loader, config, device, n_samples
    )
    
    # 评估场景
    metrics = evaluate_scenarios(generated_samples, actual_residual, max_values)
    
    # 保存结果
    save_results(generated_samples, forecast_data, actual_residual, metrics, save_path)
    
    print("=" * 60)
    print("任务完成!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多变量协同条件扩散模型')
    
    parser.add_argument('--config', type=str, default='config/wind_scenario.yaml',
                        help='配置文件路径')
    parser.add_argument('--data_path', type=str, default='./input_4.27/',
                        help='数据路径')
    parser.add_argument('--save_path', type=str, default='./save/wind_scenario/',
                        help='模型保存路径')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='运行模式')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='每个条件生成的场景数量')
    
    args = parser.parse_args()
    
    main(args)