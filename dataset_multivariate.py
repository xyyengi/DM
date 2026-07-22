# ============================================================================
# 多变量协同条件扩散模型数据集 - 论文"2023-Conditional_Diffusion_Model.pdf"复现
# 
# 核心改进：将单变量条件扩展为风、光、负荷三通道协同条件
# 
# 数据结构：
# - pred.npy: (N, 168, 3) - 训练集预测值
# - test_pred.npy: (N, 168, 3) - 测试集预测值  
# - test_res.npy: (N, 168, 3) - 测试集残差 (forecast - actual)
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
import json
import random

from src.training.residual_standardization import (
    fit_residual_standardizer,
    standardize_residual,
    validate_standardizer,
)


def _seed_dataloader_worker(worker_id):
    """Seed NumPy/Python from the deterministic PyTorch worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MultiChannelKDE:
    """
    论文公式7-8: 多通道核密度估计
    
    对风、光、负荷三个通道分别进行KDE拟合，
    并考虑光伏夜间、负荷周末的特殊性。
    """
    
    CACHE_VERSION = 2

    def __init__(self, n_intervals=10):
        self.n_intervals = n_intervals
        self.cache_version = self.CACHE_VERSION
        self.kde_models = {}  # 存储每个通道、每个区间的KDE模型
        self.interval_bounds = {}  # 存储区间边界
        self.error_stats = {}  # 存储误差统计量
        self.precomputed_quantiles = {}  # 预计算的分位数 {channel: {interval: (c_down, c_up)}
        self.interval_counts = {}  # 存储每个区间的样本数

    def _get_neighbor_fallback_stats(self, channel_name, interval_idx, interval_counts, interval_means, interval_stds, dense_threshold=5):
        """Use nearby dense bins to estimate fallback mean/std for sparse intervals."""
        counts = interval_counts
        means = interval_means
        stds = interval_stds

        candidates = []
        max_radius = max(interval_idx, len(counts) - 1 - interval_idx)
        for radius in range(1, max_radius + 1):
            left = interval_idx - radius
            right = interval_idx + radius
            if 0 <= left < len(counts) and counts[left] > dense_threshold:
                candidates.append((left, radius))
            if 0 <= right < len(counts) and counts[right] > dense_threshold:
                candidates.append((right, radius))
            if candidates:
                break

        if candidates:
            weights = []
            candidate_means = []
            candidate_vars = []
            for neighbor_idx, radius in candidates:
                count = max(counts[neighbor_idx], 1)
                weight = count / float(radius)
                weights.append(weight)
                candidate_means.append(means[neighbor_idx])
                candidate_vars.append(stds[neighbor_idx] ** 2)

            weights = np.asarray(weights, dtype=np.float64)
            candidate_means = np.asarray(candidate_means, dtype=np.float64)
            candidate_vars = np.asarray(candidate_vars, dtype=np.float64)
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                fallback_mean = float(np.sum(weights * candidate_means) / weight_sum)
                second_moment = float(np.sum(weights * (candidate_vars + candidate_means ** 2)) / weight_sum)
                fallback_var = max(second_moment - fallback_mean ** 2, 1e-6)
                return fallback_mean, float(np.sqrt(fallback_var))

        # 如果周围没有足够密集的箱，则退回到通道全局统计
        finite_means = np.asarray([m for m in means if np.isfinite(m)], dtype=np.float64)
        finite_stds = np.asarray([s for s in stds if np.isfinite(s)], dtype=np.float64)
        global_mean = float(np.mean(finite_means)) if len(finite_means) > 0 else 0.0
        global_std = float(np.sqrt(np.mean(finite_stds ** 2))) if len(finite_stds) > 0 else 1.0
        return global_mean, max(global_std, 1e-6)

    def _build_dynamic_grid(self, center_mean, center_std, interval_errors=None, channel_mean=None, channel_std=None, n_points=1000):
        """Build a KDE integration grid that adapts to the local error scale."""
        local_std = max(float(center_std), 1e-6)
        span = max(4.0 * local_std, 1e-3)

        if channel_std is not None:
            span = max(span, 2.0 * float(channel_std))

        residual_min = center_mean - span
        residual_max = center_mean + span

        if interval_errors is not None and len(interval_errors) > 0:
            sample_min = float(np.min(interval_errors))
            sample_max = float(np.max(interval_errors))
            residual_min = min(residual_min, sample_min)
            residual_max = max(residual_max, sample_max)

        if channel_mean is not None and channel_std is not None:
            channel_span = max(4.0 * float(channel_std), 1e-3)
            residual_min = min(residual_min, float(channel_mean) - channel_span)
            residual_max = max(residual_max, float(channel_mean) + channel_span)

        if not np.isfinite(residual_min) or not np.isfinite(residual_max) or residual_max <= residual_min:
            residual_min = center_mean - 1.0
            residual_max = center_mean + 1.0

        return np.linspace(residual_min, residual_max, n_points)
        
    def fit(self, forecast_data, residual_data):
        """
        拟合多通道KDE模型
        
        仅对前3个通道（风、光、负荷）进行KDE拟合
        
        Args:
            forecast_data: (N, L, 11) 预测值
            residual_data: (N, L, 11) 残差值 (forecast - actual)
        """
        # 仅对前3个通道（风、光、负荷）进行KDE拟合
        n_channels_kde = 3  # 风、光、负荷
        rng = np.random.default_rng(42)
        max_kde_samples = 50000  # 每个区间最多使用的样本数，避免 gaussian_kde 在超大数据上过慢/过耗内存
        
        for c in range(n_channels_kde):
            channel_name = ['wind', 'solar', 'load'][c]
            print(f"    [KDE] fitting channel {c} ({channel_name})...")
            
            # 获取该通道的数据
            f_flat = forecast_data[:, :, c].flatten()
            e_flat = residual_data[:, :, c].flatten()
            print(f"    [KDE] data size: {len(f_flat)} points")
            
            # 论文公式7: 按预测值划分区间
            percentiles = np.linspace(0, 100, self.n_intervals + 1)
            bounds = np.percentile(f_flat, percentiles)
            self.interval_bounds[channel_name] = bounds
            
            interval_errors_list = []
            interval_counts = []
            interval_means = []
            interval_stds = []

            for i in range(self.n_intervals):
                mask = (f_flat >= bounds[i]) & (f_flat < bounds[i+1])
                interval_errors = e_flat[mask]
                count = len(interval_errors)
                interval_errors_list.append(interval_errors)
                interval_counts.append(count)
                interval_means.append(float(np.mean(interval_errors)) if count > 0 else np.nan)
                interval_stds.append(float(np.std(interval_errors)) if count > 0 else np.nan)

            self.interval_counts[channel_name] = interval_counts

            # 存储每个区间的KDE模型
            self.kde_models[channel_name] = []
            self.error_stats[channel_name] = {'means': [], 'stds': []}

            for i in range(self.n_intervals):
                interval_errors = interval_errors_list[i]
                count = interval_counts[i]

                # 如果样本数过少，使用邻近密集箱的统计量作为后备
                if count > 5:
                    # 论文公式8: 核密度估计
                    if count > max_kde_samples:
                        sample_idx = rng.choice(count, size=max_kde_samples, replace=False)
                        kde_errors = interval_errors[sample_idx]
                        print(f"    [KDE] interval {i}: {count} points, subsample {max_kde_samples}, fitting...")
                    else:
                        kde_errors = interval_errors
                        print(f"    [KDE] interval {i}: {count} points, fitting...")
                    kde = stats.gaussian_kde(kde_errors)
                    self.kde_models[channel_name].append(kde)
                    interval_mean = interval_means[i]
                    interval_std = max(interval_stds[i], 1e-6)
                    self.error_stats[channel_name]['means'].append(interval_mean)
                    self.error_stats[channel_name]['stds'].append(interval_std)
                    print(f"    [KDE] interval {i}: done")
                else:
                    self.kde_models[channel_name].append(None)
                    fallback_mean, fallback_std = self._get_neighbor_fallback_stats(
                        channel_name,
                        i,
                        interval_counts,
                        interval_means,
                        interval_stds,
                    )
                    self.error_stats[channel_name]['means'].append(fallback_mean)
                    self.error_stats[channel_name]['stds'].append(fallback_std)
            
            # 预计算该通道所有区间的分位数（避免每次调用都计算CDF）
            print(f"    [KDE] precomputing quantiles for {channel_name}...")
            self.precomputed_quantiles[channel_name] = {}
            for i in range(self.n_intervals):
                kde = self.kde_models[channel_name][i]
                if kde is not None:
                    # 预计算第10和第90分位数，网格范围随该区间误差分布自适应
                    error_mean = self.error_stats[channel_name]['means'][i]
                    error_std = self.error_stats[channel_name]['stds'][i]
                    x_grid = self._build_dynamic_grid(
                        error_mean,
                        error_std,
                        interval_errors=interval_errors_list[i],
                        channel_mean=float(np.mean(e_flat)) if len(e_flat) > 0 else None,
                        channel_std=float(np.std(e_flat)) if len(e_flat) > 0 else None,
                        n_points=1000,
                    )
                    pdf_values = kde(x_grid)
                    cdf_values = np.cumsum(pdf_values) * (x_grid[1] - x_grid[0])
                    if cdf_values[-1] <= 0 or not np.isfinite(cdf_values[-1]):
                        self.precomputed_quantiles[channel_name][i] = (
                            error_mean - 1.28 * error_std,
                            error_mean + 1.28 * error_std
                        )
                        continue
                    cdf_values = cdf_values / cdf_values[-1]
                    
                    c_down_idx = max(0, min(np.searchsorted(cdf_values, 0.10), len(x_grid) - 1))
                    c_up_idx = max(0, min(np.searchsorted(cdf_values, 0.90), len(x_grid) - 1))
                    
                    self.precomputed_quantiles[channel_name][i] = (
                        x_grid[c_down_idx], x_grid[c_up_idx]
                    )
                else:
                    # 使用统计量作为后备
                    error_mean = self.error_stats[channel_name]['means'][i]
                    error_std = self.error_stats[channel_name]['stds'][i]
                    self.precomputed_quantiles[channel_name][i] = (
                        error_mean - 1.28 * error_std,
                        error_mean + 1.28 * error_std
                    )
            print(f"    [KDE] channel {c} ({channel_name}) done")
    
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
    
    def get_conditional_interval(self, forecast_value, channel_idx, lower_percentile=10, upper_percentile=90):
        """
        论文公式9: 构建条件区间 c = [c_down, c_up]
        
        【优化版本】直接使用预计算的分位数，避免每次调用都计算CDF
        
        Args:
            forecast_value: 归一化的预测功率值（范围 [0, 1]）
            channel_idx: 通道索引（0=风电, 1=光伏, 2=负荷）
            lower_percentile: 下分位数（默认10，对应80%置信区间下界）
            upper_percentile: 上分位数（默认90，对应80%置信区间上界）
        
        Returns:
            c_down: 残差下界（残差空间，可以是负数）
            c_up: 残差上界（残差空间）
        """
        interval_idx = self.get_interval_index(forecast_value, channel_idx)
        channel_name = ['wind', 'solar', 'load'][channel_idx]
        
        # 【优化】直接使用预计算的分位数
        if channel_name in self.precomputed_quantiles and interval_idx in self.precomputed_quantiles[channel_name]:
            return self.precomputed_quantiles[channel_name][interval_idx]
        
        # 后备方案：如果预计算不存在，使用统计量
        error_mean = self.error_stats[channel_name]['means'][interval_idx]
        error_std = self.error_stats[channel_name]['stds'][interval_idx]
        z_score = 1.28 if (upper_percentile - lower_percentile) == 80 else 1.64
        return error_mean - z_score * error_std, error_mean + z_score * error_std
    
    def save(self, path):
        """保存KDE模型（包括预计算的分位数）"""
        with open(path, 'wb') as f:
            pickle.dump({
                'cache_version': self.cache_version,
                'kde_models': self.kde_models,
                'interval_bounds': self.interval_bounds,
                'error_stats': self.error_stats,
                'interval_counts': self.interval_counts,
                'n_intervals': self.n_intervals,
                'precomputed_quantiles': self.precomputed_quantiles
            }, f)
    
    def load(self, path):
        """加载KDE模型（包括预计算的分位数）"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            cache_version = data.get('cache_version', 1)
            if cache_version != self.CACHE_VERSION:
                raise ValueError(f"KDE cache version mismatch: {cache_version} != {self.CACHE_VERSION}")
            self.kde_models = data['kde_models']
            self.interval_bounds = data['interval_bounds']
            self.error_stats = data['error_stats']
            self.interval_counts = data.get('interval_counts', {})
            self.n_intervals = data['n_intervals']
            # 兼容旧版本：如果没有预计算分位数，标记为空
            self.precomputed_quantiles = data.get('precomputed_quantiles', {})


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
                 mode='train', seq_length=168, n_intervals=10, n_channels=11,
                 build_kde=True, residual_standardization=None):
        self.data_path = data_path
        self.mode = mode
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.n_intervals = n_intervals
        self.build_kde = build_kde
        self.residual_standardization_config = residual_standardization or {'enabled': False}
        self.residual_standardizer = None
        self._load_export_metadata()
        
        print(f"  [Dataset] mode={mode}, loading data...")
        self._load_data()
        print(f"  [Dataset] normalizing data...")
        self._normalize_data()
        self._prepare_residual_target()
        
        self.kde = None
        if self.build_kde:
            print(f"  [Dataset] initializing KDE...")
            self.kde = MultiChannelKDE(n_intervals=n_intervals)
            kde_path = os.path.join(data_path, 'kde_multivariate.pkl')
            
            # 优先使用缓存，避免重复拟合（train模式也先检查缓存）
            if os.path.exists(kde_path):
                print(f"  [Dataset] loading KDE from cache: {kde_path}")
                try:
                    self.kde.load(kde_path)
                    if self.kde.n_intervals != n_intervals:
                        raise ValueError(f"KDE n_intervals mismatch: {self.kde.n_intervals} != {n_intervals}")
                    print(f"  [Dataset] KDE loaded from cache.")
                except (ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
                    print(f"  [Dataset] KDE cache invalid, refitting: {exc}")
                    self.kde = MultiChannelKDE(n_intervals=n_intervals)
                    self.kde.fit(self.forecast_norm, self.residual_norm)
                    self.kde.save(kde_path)
                    print(f"  [Dataset] KDE rebuilt and saved to {kde_path}")
            else:
                print(f"  [Dataset] fitting KDE (no cache, this may take a moment)...")
                self.kde.fit(self.forecast_norm, self.residual_norm)
                self.kde.save(kde_path)
                print(f"  [Dataset] KDE saved to {kde_path}")

            # 预计算条件矩阵（在KDE创建之后）
            print(f"  [Dataset] precomputing condition matrix...")
            self._precompute_cond_matrix()
        else:
            print(f"  [Dataset] KDE/guidance disabled; using zero condition matrix.")
            self.cond_matrix_all = np.zeros((self.num_samples, 3, self.seq_length, 2), dtype=np.float32)
        print(f"  [Dataset] initialization complete.")

    def _load_export_metadata(self):
        """Load optional export semantics while keeping legacy datasets compatible."""
        metadata_path = os.path.join(self.data_path, 'export_metadata.json')
        self.pre_normalized = False
        self.residual_definition = 'forecast_minus_actual'

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.pre_normalized = bool(metadata.get('pre_normalized', False))
            self.residual_definition = metadata.get(
                'residual_definition', 'forecast_minus_actual'
            )

        supported = {'forecast_minus_actual', 'actual_minus_forecast'}
        if self.residual_definition not in supported:
            raise ValueError(
                f"Unsupported residual_definition={self.residual_definition!r}; "
                f"expected one of {sorted(supported)}"
            )
        print(
            "  [Dataset] export semantics: "
            f"pre_normalized={self.pre_normalized}, "
            f"residual_definition={self.residual_definition}"
        )
    
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

        # Internally the model consistently uses residual = forecast - actual.
        # New exports may use the opposite sign; only the three physical channels
        # are residuals (channels 3:11 are compatibility time features).
        if self.residual_data is not None and self.residual_definition == 'actual_minus_forecast':
            self.residual_data = self.residual_data.copy()
            self.residual_data[:, :, :3] *= -1.0
        
        # 确保数据维度正确
        if self.forecast_data is None:
            raise FileNotFoundError(f"未找到数据文件，请检查 {self.data_path} 目录")
        
        self.num_samples = self.forecast_data.shape[0]
        self.n_channels = self.forecast_data.shape[2]  # 应为11
        
    def _normalize_data(self):
        """
        归一化数据到[0,1]范围。

        数据定义：
        residual = forecast - actual
        actual = forecast - residual
        """
        if self.pre_normalized:
            self.max_values = np.ones(self.forecast_data.shape[2], dtype=np.float32)
        else:
            self.max_values = np.max(np.abs(self.forecast_data), axis=(0, 1))
            self.max_values = np.maximum(self.max_values, 1e-6)

        reconstructed_actual = self.forecast_data[:, :, :3] - self.residual_data[:, :, :3]
        assert np.isfinite(reconstructed_actual).all(), "reconstructed actual contains NaN/Inf"
        print(
            f"  [Dataset] reconstructed actual ({self.mode}) "
            f"shape={reconstructed_actual.shape}, "
            f"min={reconstructed_actual.min():.6f}, "
            f"max={reconstructed_actual.max():.6f}, "
            f"mean={reconstructed_actual.mean():.6f}"
        )
        
        self.forecast_norm = self.forecast_data / self.max_values
        self.residual_norm = self.residual_data / self.max_values

    def _prepare_residual_target(self):
        """Standardize only the diffusion target using train unique-hour stats."""
        settings = dict(self.residual_standardization_config or {})
        if not bool(settings.get('enabled', False)):
            self.residual_target_norm = self.residual_norm[:, :, :3]
            return

        fitted = settings.get('fitted_stats')
        if fitted is None:
            fitted = fit_residual_standardizer(
                self.train_res,
                residual_definition=self.residual_definition,
                normalization_divisors=self.max_values[:3],
                epsilon=float(settings.get('epsilon', 1e-6)),
            )
        self.residual_standardizer = validate_standardizer(fitted)
        self.residual_target_norm = standardize_residual(
            self.residual_norm[:, :, :3],
            self.residual_standardizer,
            channel_axis=2,
        ).astype(np.float32, copy=False)
        print(
            "  [Dataset] standardized residual target from train unique hours: "
            f"mean={self.residual_standardizer['mean']}, "
            f"std={self.residual_standardizer['std']}"
        )
    
    def _precompute_cond_matrix(self):
        """
        预计算所有样本的条件矩阵
        将504次KDE查询从__getitem__移到初始化阶段，大幅加速训练
        
        【修复】条件区间从残差区间改为功率值区间（符合论文公式9）
        论文公式9: c = [c_down, c_up] = [f - K_h(f), f + K_h(f)]
        即功率值区间 = 预测值 ± 残差范围
        
        支持缓存：如果已有缓存文件则直接加载，避免重复计算
        """
        # 检查缓存文件
        cache_path = os.path.join(self.data_path, f'cond_matrix_{self.mode}_res_forecast_minus_actual.npy')
        
        if os.path.exists(cache_path):
            print(f"加载缓存的条件矩阵: {cache_path}")
            self.cond_matrix_all = np.load(cache_path)
            print(f"条件矩阵加载完成，形状: {self.cond_matrix_all.shape}")
            return
        
        print(f"预计算条件矩阵... (样本数: {self.num_samples})")
        print(f"  【修复】条件区间改为功率值区间（预测值±残差范围）")
        self.cond_matrix_all = np.zeros((self.num_samples, 3, self.seq_length, 2), dtype=np.float32)
        
        for idx in range(self.num_samples):
            forecast_3ch = self.forecast_norm[idx, :, :3]  # (168, 3)
            
            for c in range(3):
                for t in range(self.seq_length):
                    f_val = forecast_3ch[t, c]
                    # residual = forecast - actual, so actual = forecast - residual.
                    residual_down, residual_up = self.kde.get_conditional_interval(f_val, c)
                    c_down = f_val - residual_up
                    c_up = f_val - residual_down
                    self.cond_matrix_all[idx, c, t, 0] = c_down
                    self.cond_matrix_all[idx, c, t, 1] = c_up
            
            if (idx + 1) % 5000 == 0:
                print(f"  已处理 {idx + 1}/{self.num_samples} 样本")
        
        print(f"条件矩阵预计算完成，形状: {self.cond_matrix_all.shape}")
        
        # 保存缓存
        np.save(cache_path, self.cond_matrix_all)
        print(f"条件矩阵已缓存至: {cache_path}")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        """
        返回单个样本
        
        14通道输入结构：
        - Channel [0:3]: Target Residuals (正在去噪的风、光、负荷残差 x_t)
        - Channel [3:6]: Base Prediction (来自FEDformer的风、光、负荷预测趋势)
        - Channel [6:14]: Time Encoding (8维时间周期特征)
        
        Returns:
            input_14ch: (14, 168) 14通道完整输入
            actual_3ch: (3, 168) 风、光、负荷真实值（默认扩散目标）
            residual_3ch: (3, 168) 仅风、光、负荷残差（可选扩散目标）
            forecast_3ch: (3, 168) 预测趋势（用于条件）
            time_encoding/calendar_8ch: (8, 168) 真实日历sin/cos编码
            cond_matrix: (3, 168, 2) 条件矩阵 [c_down, c_up]（仅对前3维构建）
            timepoints: (168,) V4兼容位置索引
            relative_positions: (168,) V5窗口内相对位置，不代表真实日历
        """
        # 获取完整11维数据，转置为 (11, 168)
        forecast = self.forecast_norm[index].transpose(1, 0)  # (168, 11) -> (11, 168)
        residual = self.residual_norm[index].transpose(1, 0)  # (168, 11) -> (11, 168)
        residual_target_3ch = self.residual_target_norm[index].transpose(1, 0)
        
        # 提取各部分特征
        residual_3ch = residual[:3, :]  # (3, 168) Target Residuals
        forecast_3ch = forecast[:3, :]  # (3, 168) Base Prediction
        actual_3ch = forecast_3ch - residual_3ch  # (3, 168) actual = forecast - residual
        time_encoding = forecast[3:11, :]  # (8, 168) Time Encoding

        assert actual_3ch.ndim == 2 and actual_3ch.shape == (3, self.seq_length)
        assert residual_3ch.ndim == 2 and residual_3ch.shape == (3, self.seq_length)
        assert forecast_3ch.ndim == 2 and forecast_3ch.shape == (3, self.seq_length)
        
        # 构建14通道输入: [Target Residuals, Base Prediction, Time Encoding]
        input_14ch = np.concatenate([residual_target_3ch, forecast_3ch, time_encoding], axis=0)  # (14, 168)
        
        # 使用预计算的条件矩阵（已移除504次KDE查询）
        cond_matrix = self.cond_matrix_all[index]  # (3, 168, 2)
        
        return {
            'input_14ch': torch.FloatTensor(input_14ch),     # (14, 168) 14通道完整输入
            'actual_3ch': torch.FloatTensor(actual_3ch),     # (3, 168) 真实值
            'residual_target_3ch': torch.FloatTensor(residual_target_3ch), # model-space target
            'residual_3ch': torch.FloatTensor(residual_3ch), # normalized-power residual for reconstruction/evaluation
            'forecast_3ch': torch.FloatTensor(forecast_3ch), # (3, 168) 预测趋势
            'time_encoding': torch.FloatTensor(time_encoding), # (8, 168) 时间编码
            'calendar_8ch': torch.FloatTensor(time_encoding), # (8, 168) 真实calendar sin/cos
            'cond_matrix': torch.FloatTensor(cond_matrix),   # (3, 168, 2) 条件
            'timepoints': torch.FloatTensor(np.arange(self.seq_length)),
            'relative_positions': torch.FloatTensor(np.arange(self.seq_length)),
        }


def get_dataloader_multivariate(data_path='./wind_solar_load_168_FEDformer/',
                                batch_size=16, mode='train', n_intervals=10,
                                num_workers=None, pin_memory=None,
                                build_kde=True, residual_standardization=None,
                                seed=None):
    """
    获取多通道数据加载器
    
    Args:
        num_workers: 多进程数据加载（None=自动检测：Windows=0, Linux=4）
        pin_memory: 锁页内存，加速GPU数据传输（None=自动检测：GPU=True）
    """
    import platform
    import torch
    
    # 自动检测最佳配置
    if num_workers is None:
        num_workers = 0 if platform.system() == 'Windows' else 4
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    dataset = MultiChannelWindScenarioDataset(
        data_path=data_path, mode=mode, n_intervals=n_intervals, build_kde=build_kde,
        residual_standardization=residual_standardization,
    )
    generator = None
    if seed is not None:
        split_offset = {'train': 0, 'val': 10_000, 'test': 20_000}.get(mode, 30_000)
        generator = torch.Generator()
        generator.manual_seed(int(seed) + split_offset)

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(mode=='train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=_seed_dataloader_worker if seed is not None else None,
    )
    return loader, dataset.kde, dataset.max_values
