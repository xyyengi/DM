# ============================================================================
# 多变量协同条件扩散模型数据集 - 论文"2023-Conditional_Diffusion_Model.pdf"复现
# 
# 核心改进：将单变量条件扩展为风、光、负荷三通道协同条件
# 
# 数据结构：
# - pred.npy: (N, 168, 3) - 训练集预测值
# - test_pred.npy: (N, 168, 3) - 测试集预测值  
# - test_res.npy: (N, 168, 3) - 测试集残差 (预测 - 真实)
# - 通道映射: 0=风电, 1=光伏, 2=负荷
# 
# 论文公式对应：
# - 公式7: 区间划分与条件概率 P(e|f)
# - 公式8: 核密度估计 K_h(x)
# - 公式9: 条件区间构造 c = [c_down, c_up]
# ============================================================================

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy import stats
import pickle
import os


class MultiChannelKDE:
    """
    论文公式7-8: 多通道核密度估计
    
    对风、光、负荷三个通道分别进行KDE拟合，
    并考虑光伏夜间、负荷周末的特殊性。
    """
    
    def __init__(self, n_intervals=10):
        self.n_intervals = n_intervals
        self.kde_models = {}  # 存储每个通道、每个区间的KDE模型
        self.interval_bounds = {}  # 存储区间边界
        self.error_stats = {}  # 存储误差统计量
        
    def fit(self, forecast_data, residual_data):
        """
        拟合多通道KDE模型
        
        Args:
            forecast_data: (N, L, 3) 预测值
            residual_data: (N, L, 3) 残差值 (预测 - 真实)
        """
        n_channels = forecast_data.shape[2]
        
        for c in range(n_channels):
            channel_name = ['wind', 'solar', 'load'][c]
            
            # 获取该通道的数据
            f_flat = forecast_data[:, :, c].flatten()
            e_flat = residual_data[:, :, c].flatten()
            
            # 论文公式7: 按预测值划分区间
            percentiles = np.linspace(0, 100, self.n_intervals + 1)
            bounds = np.percentile(f_flat, percentiles)
            self.interval_bounds[channel_name] = bounds
            
            # 存储每个区间的KDE模型
            self.kde_models[channel_name] = []
            self.error_stats[channel_name] = {'means': [], 'stds': []}
            
            for i in range(self.n_intervals):
                mask = (f_flat >= bounds[i]) & (f_flat < bounds[i+1])
                interval_errors = e_flat[mask]
                
                if len(interval_errors) > 10:
                    # 论文公式8: 核密度估计
                    kde = stats.gaussian_kde(interval_errors)
                    self.kde_models[channel_name].append(kde)
                    self.error_stats[channel_name]['means'].append(np.mean(interval_errors))
                    self.error_stats[channel_name]['stds'].append(np.std(interval_errors))
                else:
                    self.kde_models[channel_name].append(None)
                    self.error_stats[channel_name]['means'].append(0)
                    self.error_stats[channel_name]['stds'].append(1)
    
    def get_interval_index(self, forecast_value, channel_idx):
        """获取预测值所属的区间索引"""
        channel_name = ['wind', 'solar', 'load'][channel_idx]
        bounds = self.interval_bounds[channel_name]
        
        for i in range(len(bounds) - 1):
            if bounds[i] <= forecast_value < bounds[i+1]:
                return i
        return len(bounds) - 2
    
    def get_error_expectation(self, forecast_value, channel_idx):
        """
        论文公式7: 计算条件误差期望
        φ_i = Σ e * P(e|f∈D_i) / P(f∈D_i)
        """
        interval_idx = self.get_interval_index(forecast_value, channel_idx)
        channel_name = ['wind', 'solar', 'load'][channel_idx]
        return self.error_stats[channel_name]['means'][interval_idx]
    
    def get_conditional_interval(self, forecast_value, channel_idx):
        """
        论文公式9: 构建条件区间 c = [c_down, c_up]
        c_up = min(1, f + K_h(f))
        c_down = max(0, f - K_h(f))
        """
        error_exp = self.get_error_expectation(forecast_value, channel_idx)
        k_h = abs(error_exp)
        
        c_up = min(1.0, forecast_value + k_h)
        c_down = max(0.0, forecast_value - k_h)
        return c_down, c_up
    
    def save(self, path):
        """保存KDE模型"""
        with open(path, 'wb') as f:
            pickle.dump({
                'kde_models': self.kde_models,
                'interval_bounds': self.interval_bounds,
                'error_stats': self.error_stats,
                'n_intervals': self.n_intervals
            }, f)
    
    def load(self, path):
        """加载KDE模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kde_models = data['kde_models']
            self.interval_bounds = data['interval_bounds']
            self.error_stats = data['error_stats']
            self.n_intervals = data['n_intervals']


class MultiChannelWindScenarioDataset(Dataset):
    """
    多变量协同条件扩散模型数据集
    张量结构: (Batch, Channels, Length=168)
    
    数据文件结构 (input_4.27):
    - train_pred.npy: (18917, 168, 11) 训练集预测值
    - train_res.npy: (18917, 168, 11) 训练集残差
    - val_pred.npy: (2608, 168, 11) 验证集预测值
    - val_res.npy: (2608, 168, 11) 验证集残差
    - test_pred.npy: (5381, 168, 11) 测试集预测值
    - test_res.npy: (5381, 168, 11) 测试集残差
    """
    
    def __init__(self, data_path='./input_4.27/', 
                 mode='train', seq_length=168, n_intervals=10, n_channels=11):
        self.data_path = data_path
        self.mode = mode
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.n_intervals = n_intervals
        
        self._load_data()
        self._normalize_data()
        
        self.kde = MultiChannelKDE(n_intervals=n_intervals)
        if mode == 'train':
            self.kde.fit(self.forecast_norm, self.residual_norm)
            self.kde.save(os.path.join(data_path, 'kde_multivariate.pkl'))
        else:
            kde_path = os.path.join(data_path, 'kde_multivariate.pkl')
            if os.path.exists(kde_path):
                self.kde.load(kde_path)
            else:
                self.kde.fit(self.forecast_norm, self.residual_norm)
    
    def _load_data(self):
        """
        加载预测值和残差数据
        
        数据文件结构 (input_4.27):
        - train_pred.npy: (18917, 168, 11) 训练集预测值
        - train_res.npy: (18917, 168, 11) 训练集残差
        - val_pred.npy: (2608, 168, 11) 验证集预测值
        - val_res.npy: (2608, 168, 11) 验证集残差
        - test_pred.npy: (5381, 168, 11) 测试集预测值
        - test_res.npy: (5381, 168, 11) 测试集残差
        
        11维特征定义：
        - Channel [0:3]: 风、光、负荷残差 (Residuals) - 生成核心主体
        - Channel [3:11]: 8维时间周期编码 (Sin/Cos) - 环境背景条件
        """
        # 加载所有数据文件
        train_pred_path = os.path.join(self.data_path, 'train_pred.npy')
        train_res_path = os.path.join(self.data_path, 'train_res.npy')
        val_pred_path = os.path.join(self.data_path, 'val_pred.npy')
        val_res_path = os.path.join(self.data_path, 'val_res.npy')
        test_pred_path = os.path.join(self.data_path, 'test_pred.npy')
        test_res_path = os.path.join(self.data_path, 'test_res.npy')
        
        # 检查并加载文件
        self.train_pred = np.load(train_pred_path) if os.path.exists(train_pred_path) else None
        self.train_res = np.load(train_res_path) if os.path.exists(train_res_path) else None
        self.val_pred = np.load(val_pred_path) if os.path.exists(val_pred_path) else None
        self.val_res = np.load(val_res_path) if os.path.exists(val_res_path) else None
        self.test_pred = np.load(test_pred_path) if os.path.exists(test_pred_path) else None
        self.test_res = np.load(test_res_path) if os.path.exists(test_res_path) else None
        
        # 根据模式选择数据
        if self.mode == 'train':
            self.forecast_data = self.train_pred if self.train_pred is not None else self.test_pred
            self.residual_data = self.train_res if self.train_res is not None else self.test_res
        elif self.mode == 'val':
            self.forecast_data = self.val_pred if self.val_pred is not None else self.test_pred
            self.residual_data = self.val_res if self.val_res is not None else self.test_res
        else:  # test mode
            self.forecast_data = self.test_pred
            self.residual_data = self.test_res
        
        # 确保数据维度正确
        if self.forecast_data is None:
            raise FileNotFoundError(f"未找到数据文件，请检查 {self.data_path} 目录")
        
        self.num_samples = self.forecast_data.shape[0]
        self.n_channels = self.forecast_data.shape[2]  # 应为11
        
    def _normalize_data(self):
        """归一化数据到[0,1]范围"""
        self.max_values = np.max(np.abs(self.forecast_data), axis=(0, 1))
        self.max_values = np.maximum(self.max_values, 1e-6)
        
        self.forecast_norm = self.forecast_data / self.max_values
        self.residual_norm = self.residual_data / self.max_values
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        """
        返回单个样本
        
        11维特征定义：
        - Channel [0:3]: 风、光、负荷残差 (Residuals) - 生成核心主体
        - Channel [3:11]: 8维时间周期编码 (Sin/Cos) - 环境背景条件
        
        Returns:
            forecast: (11, 168) 归一化预测值（完整11维）
            residual: (11, 168) 归一化残差（完整11维）
            residual_3ch: (3, 168) 仅风、光、负荷残差（用于扩散模型输出）
            cond_matrix: (3, 168, 2) 条件矩阵 [c_down, c_up]（仅对前3维构建）
            timepoints: (168,) 时间点索引
        """
        # 获取完整11维数据，转置为 (11, 168)
        forecast = self.forecast_norm[index].transpose(1, 0)  # (168, 11) -> (11, 168)
        residual = self.residual_norm[index].transpose(1, 0)  # (168, 11) -> (11, 168)
        
        # 提取前3维（风、光、负荷残差）用于扩散模型
        residual_3ch = residual[:3, :]  # (3, 168)
        forecast_3ch = forecast[:3, :]  # (3, 168)
        
        # 论文公式9: 仅对前3维（风、光、负荷）构建条件矩阵
        # KDE只针对残差通道构建条件区间
        cond_down = np.zeros((3, self.seq_length))
        cond_up = np.zeros((3, self.seq_length))
        
        for c in range(3):  # 仅前3个通道
            for t in range(self.seq_length):
                f_val = forecast_3ch[c, t]
                c_down, c_up = self.kde.get_conditional_interval(f_val, c)
                cond_down[c, t] = c_down
                cond_up[c, t] = c_up
        
        # 条件矩阵: (3, 168, 2) - 仅风、光、负荷
        cond_matrix = np.stack([cond_down, cond_up], axis=-1)
        
        return {
            'forecast': torch.FloatTensor(forecast),        # (11, 168) 完整输入
            'residual': torch.FloatTensor(residual),        # (11, 168) 完整残差
            'residual_3ch': torch.FloatTensor(residual_3ch), # (3, 168) 扩散目标
            'cond_matrix': torch.FloatTensor(cond_matrix),   # (3, 168, 2) 条件
            'timepoints': torch.FloatTensor(np.arange(self.seq_length)),
        }


def get_dataloader_multivariate(data_path='./wind_solar_load_168_FEDformer/',
                                batch_size=16, mode='train', n_intervals=10):
    """获取多通道数据加载器"""
    dataset = MultiChannelWindScenarioDataset(
        data_path=data_path, mode=mode, n_intervals=n_intervals
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'))
    return loader, dataset.kde, dataset.max_values