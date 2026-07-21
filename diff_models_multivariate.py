# ============================================================================
# 多变量协同条件扩散模型 - 论文"2023-Conditional_Diffusion_Model.pdf"复现
# 
# 核心架构改进：
# 1. Res-UNet + 空洞卷积（感受野覆盖168点）
# 2. 时间特征注入（小时、周几、月份）
# 3. 多通道条件引导（公式10的Frobenius范数梯度）
# 
# 论文公式对应：
# - 公式10: 反向去噪中的条件梯度引导
#   ∇_{x_t} ||γ·x_t - c||²_F (Frobenius范数)
# ============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_FORECAST_FEATURES = ('load_ramp_1h', 'net_load')


def build_forecast_dynamic_features(forecast_3ch, feature_config):
    """Build leakage-free forecast features in standardized train coordinates."""
    if not bool(feature_config.get('enabled', False)):
        return None

    names = tuple(feature_config.get('names', ()))
    unsupported = sorted(set(names) - set(SUPPORTED_FORECAST_FEATURES))
    if unsupported:
        raise ValueError(f"Unsupported forecast features: {unsupported}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate forecast feature names: {names}")
    if not names:
        raise ValueError("forecast_features.enabled=true requires at least one feature name")

    assert forecast_3ch.ndim == 3 and forecast_3ch.shape[1] == 3, (
        f"forecast_3ch must be [B, 3, L], got {tuple(forecast_3ch.shape)}"
    )
    wind = forecast_3ch[:, 0, :]
    solar = forecast_3ch[:, 1, :]
    load = forecast_3ch[:, 2, :]
    raw = []
    for name in names:
        if name == 'load_ramp_1h':
            feature = torch.zeros_like(load)
            feature[:, 1:] = load[:, 1:] - load[:, :-1]
        elif name == 'net_load':
            scale = feature_config.get('net_load_scale', {})
            wind_to_load = float(scale['wind_to_load'])
            solar_to_load = float(scale['solar_to_load'])
            feature = load - wind_to_load * wind - solar_to_load * solar
        raw.append(feature)

    features = torch.stack(raw, dim=1)
    normalization = feature_config.get('normalization', {})
    means = normalization.get('mean')
    stds = normalization.get('std')
    if means is None or stds is None or len(means) != len(names) or len(stds) != len(names):
        raise ValueError("forecast feature normalization mean/std must match feature names")
    mean = torch.as_tensor(means, dtype=features.dtype, device=features.device).view(1, -1, 1)
    std = torch.as_tensor(stds, dtype=features.dtype, device=features.device).view(1, -1, 1)
    if torch.any(std <= 0):
        raise ValueError("forecast feature normalization std must be positive")
    return (features - mean) / std


# ============================================================================
# 时间特征注入模块
# ============================================================================

class TimeFeatureEmbedding(nn.Module):
    """
    时间特征注入：小时、周几、月份三个尺度的Embedding
    
    解决误差的异方差性问题：
    - 小时：捕获日内周期性（光伏白天高、负荷峰谷）
    - 周几：捕获周周期性（负荷周末低）
    - 月份：捕获季节性（风电冬春高、光伏夏季高）
    """
    
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        # 计算每个嵌入维度，确保拼接后等于d_model
        # 使用 d_model // 3 + 1 来弥补余数
        embed_dim = d_model // 3
        remainder = d_model - embed_dim * 3
        
        # 小时嵌入 (0-23) - 分配余数给第一个
        self.hour_embed = nn.Embedding(24, embed_dim + remainder)
        
        # 周几嵌入 (0-6, 0=周一)
        self.weekday_embed = nn.Embedding(7, embed_dim)
        
        # 月份嵌入 (0-11)
        self.month_embed = nn.Embedding(12, embed_dim)
        
        # 合并后的投影层（输入维度已经是d_model）
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, hour, weekday, month):
        """
        Args:
            hour: (B, L) 小时索引 0-23
            weekday: (B, L) 周几索引 0-6
            month: (B, L) 月份索引 0-11
        Returns:
            time_feat: (B, L, d_model)
        """
        h_emb = self.hour_embed(hour)  # (B, L, d_model/3)
        w_emb = self.weekday_embed(weekday)
        m_emb = self.month_embed(month)
        
        # 拼接三个时间特征
        time_feat = torch.cat([h_emb, w_emb, m_emb], dim=-1)  # (B, L, d_model)
        time_feat = self.proj(time_feat)
        
        return time_feat


class SinusoidalPositionEmbedding(nn.Module):
    """
    标准的Sinusoidal位置编码，用于序列位置
    """
    
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, positions):
        """
        Args:
            positions: (B, L) 或 (L,) 位置索引
        Returns:
            pos_emb: (B, L, d_model) 或 (L, d_model)
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
            
        B, L = positions.shape
        device = positions.device
        
        pos_emb = torch.zeros(B, L, self.d_model, device=device)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * 
                             (-math.log(10000.0) / self.d_model))
        
        pos_emb[:, :, 0::2] = torch.sin(positions.unsqueeze(-1) * div_term)
        pos_emb[:, :, 1::2] = torch.cos(positions.unsqueeze(-1) * div_term)
        
        return pos_emb


# ============================================================================
# 空洞卷积模块（感受野增强）
# ============================================================================

class DilatedConvBlock(nn.Module):
    """
    空洞卷积序列：空洞率 [1, 2, 4, 8]
    
    确保感受野覆盖168个点，捕获周循环特征：
    - dilation=1: 局部特征
    - dilation=2: 2小时间隔
    - dilation=4: 4小时间隔
    - dilation=8: 8小时间隔（覆盖约64小时）
    
    累计感受野: 1 + 2 + 4 + 8 = 15 * kernel_size
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 4, 8]):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList()
        for d in dilations:
            # 空洞卷积 + 批归一化
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                           padding=(kernel_size-1)*d//2, dilation=d)
            self.conv_blocks.append(nn.Sequential(
                conv,
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ))
        
        # 融合层：将多个空洞卷积结果合并
        self.fusion = nn.Conv1d(out_channels * len(dilations), out_channels, 1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            out: (B, C, L)
        """
        outputs = []
        for conv_block in self.conv_blocks:
            outputs.append(conv_block(x))
        
        # 拼接所有空洞卷积结果
        concat = torch.cat(outputs, dim=1)  # (B, C*len(dilations), L)
        out = self.fusion(concat)  # (B, C, L)
        
        return out


# ============================================================================
# Residual Block
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual Block with time embedding injection
    """
    
    def __init__(self, in_channels, out_channels, d_time=64, kernel_size=3, dropout=0.1):
        super().__init__()
        
        # 主卷积路径
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # 时间特征注入
        self.time_proj = nn.Linear(d_time, out_channels)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Dropout正则化
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_feat):
        """
        Args:
            x: (B, C_in, L)
            time_feat: (B, L, d_time)
        Returns:
            out: (B, C_out, L)
        """
        # 时间特征注入到卷积中
        time_emb = self.time_proj(time_feat)  # (B, L, C_out)
        time_emb = time_emb.permute(0, 2, 1)  # (B, C_out, L)
        
        # 主路径
        h = self.conv1(x)
        h = self.bn1(h)
        h = h + time_emb  # 注入时间特征
        h = F.relu(h)
        h = self.dropout(h)  # Dropout正则化
        
        h = self.conv2(h)
        h = self.bn2(h)
        h = h + time_emb
        h = F.relu(h)
        h = self.dropout(h)  # Dropout正则化
        
        # 残差连接
        res = self.residual(x)
        
        return h + res


# ============================================================================
# Res-UNet 架构
# ============================================================================

class ResUNet(nn.Module):
    """
    Res-UNet架构用于多通道扩散模型
    
    特点：
    1. Encoder-Decoder结构
    2. Bottleneck使用空洞卷积
    3. 时间特征注入到每个ResBlock
    4. 支持14维特征解耦输入
    
    输入维度定义 (14维张量)：
    - Channel [0:3]: Target Residuals (正在去噪的风、光、负荷残差 x_t)
    - Channel [3:6]: Base Prediction (来自FEDformer的风、光、负荷预测趋势)
    - Channel [6:14]: Time Encoding (8维时间周期特征)
    
    输出维度定义：
    - Channel [0:3]: 风、光、负荷残差噪声预测（只预测前3维）
    """
    
    def __init__(self, in_channels=14, out_channels=3, d_time=64, 
                 base_channels=64, num_layers=3):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_time = d_time
        self.num_layers = num_layers
        
        # Encoder - 简化为3层
        # Layer 0: 14 → 64
        # Layer 1: 64 → 128 (下采样)
        # Layer 2: 128 → 256 (下采样)
        
        self.enc_conv0 = ResidualBlock(in_channels, base_channels, d_time)
        self.enc_down1 = nn.Conv1d(base_channels, base_channels, 4, stride=2, padding=1)
        self.enc_conv1 = ResidualBlock(base_channels, base_channels * 2, d_time)
        self.enc_down2 = nn.Conv1d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        self.enc_conv2 = ResidualBlock(base_channels * 2, base_channels * 4, d_time)
        
        # Bottleneck with Dilated Convolution
        self.bottleneck = DilatedConvBlock(base_channels * 4, base_channels * 4, dilations=[1, 2, 4, 8])
        
        # Decoder
        # Layer 2: 256 → 128 (上采样) + skip from enc_conv1 (128)
        # Layer 1: 128 → 64 (上采样) + skip from enc_conv0 (64)
        
        self.dec_up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec_conv2 = ResidualBlock(base_channels * 4, base_channels * 2, d_time)  # 256+128=384 → 128
        
        self.dec_up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec_conv1 = ResidualBlock(base_channels * 2, base_channels, d_time)  # 128+64=192 → 64
        
        # Output layer
        self.output_conv = nn.Conv1d(base_channels, out_channels, 1)
        
    def forward(self, x, time_feat):
        """
        Args:
            x: (B, 14, 168) 多通道输入
            time_feat: (B, 168, d_time) 时间特征
        Returns:
            out: (B, 3, 168) 多通道输出
        """
        # Encoder Layer 0: 14 → 64, L=168
        enc0 = self.enc_conv0(x, time_feat)  # (B, 64, 168)
        
        # Encoder Layer 1: 64 → 128, L=84
        x1 = self.enc_down1(enc0)  # (B, 64, 84)
        time_feat1 = F.interpolate(
            time_feat.permute(0, 2, 1),
            scale_factor=0.5,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # (B, 84, d_time)
        enc1 = self.enc_conv1(x1, time_feat1)  # (B, 128, 84)
        
        # Encoder Layer 2: 128 → 256, L=42
        x2 = self.enc_down2(enc1)  # (B, 128, 42)
        time_feat2 = F.interpolate(
            time_feat1.permute(0, 2, 1),
            scale_factor=0.5,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # (B, 42, d_time)
        enc2 = self.enc_conv2(x2, time_feat2)  # (B, 256, 42)
        
        # Bottleneck: 256 → 256, L=42
        x = self.bottleneck(enc2)  # (B, 256, 42)
        
        # Decoder Layer 2: 256 → 128, L=84
        x = self.dec_up2(x)  # (B, 128, 84)
        time_feat_up2 = F.interpolate(
            time_feat2.permute(0, 2, 1),
            scale_factor=2,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # (B, 84, d_time)
        x = torch.cat([x, enc1], dim=1)  # (B, 256, 84) - 128+128=256
        x = self.dec_conv2(x, time_feat_up2)  # (B, 128, 84)
        
        # Decoder Layer 1: 128 → 64, L=168
        x = self.dec_up1(x)  # (B, 64, 168)
        time_feat_up1 = F.interpolate(
            time_feat_up2.permute(0, 2, 1),
            scale_factor=2,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # (B, 168, d_time)
        x = torch.cat([x, enc0], dim=1)  # (B, 128, 168) - 64+64=128
        x = self.dec_conv1(x, time_feat_up1)  # (B, 64, 168)
        
        # Output: 64 → 3
        out = self.output_conv(x)  # (B, 3, 168)
        
        return out


# ============================================================================
# 多通道扩散模型核心类
# ============================================================================

class GaussianDiffusionMultivariate(nn.Module):
    """
    多变量协同条件扩散模型
    
    论文公式10实现：反向去噪中的条件梯度引导
    
    核心公式：
    x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ(x_t, t, c)) + σ_t·z
    
    条件梯度修正：
    ∇_{x_t} ||γ·x_t - c||²_F
    
    其中：
    - x_t: 当前生成值 (B, 3, 168)
    - c: 条件矩阵 (B, 3, 168, 2) 包含[c_down, c_up]
    - γ: 引导强度系数
    - ||·||_F: Frobenius范数
    """
    
    def __init__(self, model, num_steps=500, beta_start=0.0001, beta_end=0.04,
                 schedule='quad', guidance_scale=1.0, guidance_scales=None,
                 target_type='residual', reverse_variance_type='beta'):
        super().__init__()
        
        self.model = model  # ResUNet
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.target_type = target_type
        if reverse_variance_type not in {'beta', 'posterior'}:
            raise ValueError(
                f"reverse_variance_type must be 'beta' or 'posterior', got {reverse_variance_type!r}"
            )
        self.reverse_variance_type = reverse_variance_type
        self._shape_debug_printed = False
        if guidance_scales is None:
            guidance_scales = [guidance_scale, guidance_scale, guidance_scale]
        self.register_buffer('guidance_scales', torch.as_tensor(guidance_scales, dtype=torch.float32).view(1, 3, 1))
        
        # Beta schedule - 使用register_buffer确保张量随模型移动到GPU
        if schedule == 'quad':
            beta = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps) ** 2
        elif schedule == 'linear':
            beta = torch.linspace(beta_start, beta_end, num_steps)
        
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        
        # 注册为buffer，这样会自动随模型移动到正确设备
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_hat', alpha_hat)

    def reverse_variance(self, t):
        """Return fixed-large beta variance or the DDPM posterior variance."""
        if self.reverse_variance_type == 'beta':
            return self.beta[t]
        if t <= 0:
            return torch.zeros_like(self.beta[0])
        alpha_hat_prev = self.alpha_hat[t - 1]
        posterior = self.beta[t] * (1.0 - alpha_hat_prev) / (1.0 - self.alpha_hat[t])
        return posterior.clamp(min=0.0)
        
    def add_noise(self, x0, t):
        """
        前向扩散：添加噪声
        
        x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε
        
        Args:
            x0: (B, 3, 168) 原始数据
            t: (B,) 时间步索引
        Returns:
            x_t: 加噪后的数据
            noise: 添加的噪声
        """
        assert x0.ndim == 3, f"x0 must be [B, 3, 168], got {tuple(x0.shape)}"
        assert x0.shape[1] == 3, f"x0 channel dim must be 3, got {x0.shape[1]}"
        assert x0.shape[2] == 168, f"x0 length dim must be 168, got {x0.shape[2]}"
        assert t.ndim == 1 and t.shape[0] == x0.shape[0], f"t must be [B], got {tuple(t.shape)}"

        noise = torch.randn_like(x0)
        
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1)
        x_t = alpha_hat_t.sqrt() * x0 + (1 - alpha_hat_t).sqrt() * noise
        
        return x_t, noise
    
    def compute_conditional_gradient(self, x_t, cond_matrix, forecast=None, debug=False):
        """
        论文公式10: 计算条件梯度
        
        条件区间是 actual 功率值区间。
        target_type == actual 时，x_t 表示 actual。
        target_type == residual 时，x_t 表示 residual，且 residual = forecast - actual。
        
        γ 是 0-1 二值系数：
        - 当 power_t 在条件区间 [c_down, c_up] 内时，γ=0（不修正）
        - 当 power_t 超出条件区间时，γ=1（修正）
        
        梯度方向：引导 power_t 向条件区间边界移动
        
        Args:
            x_t: (B, 3, 168) 当前生成值（残差）
            cond_matrix: (B, 3, 168, 2) 条件矩阵 [c_down, c_up]（功率值区间）
            forecast: (B, 3, 168) 预测值（用于计算功率值 = forecast + x_t）
            debug: 是否打印调试信息
        Returns:
            gradient: (B, 3, 168) 条件梯度（作用于残差空间）
            gamma_mask: (B, 3, 168) 二值系数掩码
            debug_info: dict 调试信息（仅当 debug=True）
        """
        c_down = cond_matrix[..., 0]  # (B, 3, 168) 功率值下界
        c_up = cond_matrix[..., 1]    # (B, 3, 168) 功率值上界
        
        if self.target_type == 'residual':
            if forecast is None:
                raise ValueError("forecast is required when target_type='residual'")
            power_t = forecast - x_t  # residual = forecast - actual
            residual_target = True
        else:
            power_t = x_t
            residual_target = False
        
        # 计算二值系数 γ
        # 当 power_t < c_down 或 power_t > c_up 时，γ=1
        # 当 power_t 在 [c_down, c_up] 内时，γ=0
        gamma_mask = ((power_t < c_down) | (power_t > c_up)).float()
        
        # 计算更新方向。actual target 时直接推 actual；residual target 时方向取反，
        # 因为 actual = forecast - residual。
        gradient = torch.zeros_like(x_t)
        
        below_mask = (power_t < c_down).float()
        if residual_target:
            gradient = gradient + below_mask * (power_t - c_down)
        else:
            gradient = gradient + below_mask * (c_down - power_t)
        
        above_mask = (power_t > c_up).float()
        if residual_target:
            gradient = gradient + above_mask * (power_t - c_up)
        else:
            gradient = gradient + above_mask * (c_up - power_t)
        
        # 调试信息
        debug_info = None
        if debug:
            debug_info = {
                'x_t_range': (x_t.min().item(), x_t.max().item()),
                'power_t_range': (power_t.min().item(), power_t.max().item()),
                'c_down_range': (c_down.min().item(), c_down.max().item()),
                'c_up_range': (c_up.min().item(), c_up.max().item()),
                'gamma_ratio': gamma_mask.mean().item(),  # 超出区间的比例
                'below_ratio': below_mask.mean().item(),  # 低于下界的比例
                'above_ratio': above_mask.mean().item(),  # 高于上界的比例
                'gradient_range': (gradient.min().item(), gradient.max().item()),
                'gradient_mean': gradient.mean().item(),
            }
        
        return gradient, gamma_mask, debug_info
    
    def denoise_step(self, x_t, t, model_input, time_feat, cond_matrix=None, forecast=None, debug=False):
        """
        论文公式10: 反向去噪一步
        
        x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ) + σ_t·z
        
        条件梯度修正：
        mean = mean - guidance_scale · ∇_{x_t} ||γ·x_t - c||²_F
        
        Args:
            x_t: (B, 3, 168) 当前噪声数据 (Target Residuals)
            t: 时间步索引
            model_input: (B, C_in, 168) 模型输入
            cond_matrix: (B, 3, 168, 2) KDE条件矩阵，仅 guidance 使用
            forecast: (B, 3, 168) 预测值，仅 residual guidance 使用
            time_feat: (B, 168, d_time) 时间特征
            debug: 是否返回调试信息
        Returns:
            x_{t-1}: 去噪一步后的数据
            debug_info: dict 调试信息（仅当 debug=True）
        """
        assert x_t.ndim == 3, f"x_t must be [B, 3, 168], got {tuple(x_t.shape)}"
        assert x_t.shape[1] == 3, f"x_t channel dim must be 3, got {x_t.shape[1]}"
        assert x_t.shape[2] == 168, f"x_t length dim must be 168, got {x_t.shape[2]}"
        assert model_input.ndim == 3 and model_input.shape[0] == x_t.shape[0] and model_input.shape[2] == 168, (
            f"model_input must be [B, C_in, 168], got {tuple(model_input.shape)}"
        )
        if self.guidance_scale > 0:
            assert cond_matrix is not None, "cond_matrix is required when guidance is enabled"
            assert cond_matrix.ndim == 4 and cond_matrix.shape == (x_t.shape[0], 3, 168, 2), (
                f"cond_matrix must be [B, 3, 168, 2], got {tuple(cond_matrix.shape)}"
            )

        B = x_t.shape[0]
        device = x_t.device
        debug_info = None

        # 预测噪声（模型输出3维）
        predicted_noise = self.model(model_input, time_feat)
        assert predicted_noise.shape == x_t.shape, (
            f"epsilon_theta shape must match x_t shape, got {tuple(predicted_noise.shape)} vs {tuple(x_t.shape)}"
        )

        if not self._shape_debug_printed:
            print(
                "[Shape] "
                f"xt={tuple(x_t.shape)}, model_input={tuple(model_input.shape)}, "
                f"epsilon_theta={tuple(predicted_noise.shape)}"
            )
            self._shape_debug_printed = True
        
        # 计算去噪均值
        alpha_t = self.alpha[t]
        alpha_hat_t = self.alpha_hat[t]
        
        # 基础去噪公式 - 添加数值稳定性保护
        coef = (1 - alpha_t) / (1 - alpha_hat_t).sqrt()
        mean = (1 / alpha_t.sqrt()) * (x_t - coef * predicted_noise)
        
        # 论文公式10: 条件梯度修正
        # γ 是 0-1 二值系数：超出条件区间时才修正
        if self.guidance_scale > 0:
            # 【修复】时间步衰减系数: t 越小（越接近最后去噪），约束越强
            # 原来: t_decay = (t + 1) / num_steps，导致 t=0 时只有 0.002
            # 改成: t_decay = 1 - t / num_steps，t=0 时为 1.0，t=499 时为 0.002
            t_decay = 1.0 - t / self.num_steps if self.num_steps > 0 else 1.0
            
            # 计算条件梯度（返回梯度方向和二值掩码）
            cond_gradient, gamma_mask, grad_debug = self.compute_conditional_gradient(x_t, cond_matrix, forecast=forecast, debug=debug)
            
            # 【修复】放宽梯度裁剪范围
            cond_gradient_clamped = torch.clamp(cond_gradient, min=-1.0, max=1.0)
            
            # 应用梯度修正：只在超出区间的地方修正
            # 【关键修复】gradient 已经是(目标 - 当前)的向量，代表更新的正确方向，应当是加上而不是减去！
            scale = self.guidance_scales.to(x_t.device)
            mean = mean + t_decay * scale * cond_gradient_clamped * gamma_mask
            
            if debug:
                debug_info = {
                    'step': t,
                    'alpha_t': alpha_t.item(),
                    'alpha_hat_t': alpha_hat_t.item(),
                    't_decay': t_decay,
                    'predicted_noise_range': (predicted_noise.min().item(), predicted_noise.max().item()),
                    'mean_before_guidance_range': (mean.min().item(), mean.max().item()),
                    'guidance_applied': (scale * t_decay).detach().cpu().tolist(),
                    **grad_debug
                }
        
        # 添加噪声（除了最后一步）
        if t > 0:
            sigma = self.reverse_variance(t).sqrt()
            noise = torch.randn_like(x_t)
            # 移除错误限制: noise = torch.clamp(noise, min=-1.0, max=1.0)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean
        
        if debug and debug_info:
            debug_info['reverse_variance_type'] = self.reverse_variance_type
            debug_info['reverse_sigma'] = float(sigma.item()) if t > 0 else 0.0
            debug_info['x_prev_range'] = (x_prev.min().item(), x_prev.max().item())
            debug_info['has_nan'] = torch.isnan(x_prev).any().item()
            debug_info['has_inf'] = torch.isinf(x_prev).any().item()
        
        return x_prev, debug_info
    
    def sample(self, build_model_input_fn, batch_size, device, time_feat, cond_matrix=None,
               forecast=None, n_samples=1, debug=False, debug_steps=None):
        """
        完整采样过程：从噪声生成场景
        
        Args:
            build_model_input_fn: callable that maps x_t -> model input
            batch_size: batch size B
            time_feat: (B, 168, d_time) 时间特征
            n_samples: 生成样本数量
            debug: 是否启用调试模式
            debug_steps: 调试信息收集的步骤列表（如 [499, 400, 300, 200, 100, 50, 0]）
        Returns:
            samples: (B, n_samples, 3, 168) 生成的场景
            debug_log: list 调试日志（仅当 debug=True）
        """
        B = batch_size
        
        # 初始化纯噪声
        samples = torch.zeros(B, n_samples, 3, 168, device=device)
        debug_log = []
        
        for s in range(n_samples):
            x_t = torch.randn(B, 3, 168, device=device)
            
            # 逐步去噪
            for t in range(self.num_steps - 1, -1, -1):
                # 判断是否需要收集调试信息
                need_debug = debug and (debug_steps is None or t in debug_steps)
                model_input = build_model_input_fn(x_t)
                x_t, step_debug = self.denoise_step(
                    x_t, t, model_input, time_feat,
                    cond_matrix=cond_matrix, forecast=forecast, debug=need_debug
                )
                
                if need_debug and step_debug:
                    debug_log.append(step_debug)
            
            samples[:, s] = x_t
        
        if debug:
            return samples, debug_log
        return samples
    
    def forward(self, x0, model_input_fn, time_feat, t=None):
        """
        训练时的前向传播：计算损失
        
        Args:
            x0: (B, 3, 168) 原始残差数据 (Target Residuals)
            model_input_fn: callable that maps x_t -> model input
            time_feat: (B, 168, d_time) 时间特征
            cond_matrix: (B, 3, 168, 2) KDE条件矩阵 (可选)
            t: 时间步（可选，随机采样）
        Returns:
            loss: 训练损失
        """
        assert x0.ndim == 3, f"x0 must be [B, 3, 168], got {tuple(x0.shape)}"
        assert x0.shape[1] == 3, f"x0 channel dim must be 3, got {x0.shape[1]}"
        assert x0.shape[2] == 168, f"x0 length dim must be 168, got {x0.shape[2]}"
        B = x0.shape[0]
        device = x0.device
        
        # 随机采样时间步
        if t is None:
            t = torch.randint(0, self.num_steps, (B,), device=device)
        
        # 添加噪声（只对前3维Target Residuals添加噪声）
        x_t, noise = self.add_noise(x0, t)
        
        model_input = model_input_fn(x_t)
        
        # 预测噪声（模型输出3维）
        predicted_noise = self.model(model_input, time_feat)
        assert predicted_noise.shape == noise.shape, (
            f"epsilon_theta shape must match noise shape, got {tuple(predicted_noise.shape)} vs {tuple(noise.shape)}"
        )
        if not self._shape_debug_printed:
            print(
                "[Shape] "
                f"x0={tuple(x0.shape)}, xt={tuple(x_t.shape)}, noise={tuple(noise.shape)}, "
                f"model_input={tuple(model_input.shape)}, epsilon_theta={tuple(predicted_noise.shape)}"
            )
            self._shape_debug_printed = True
        
        # 【修改】干干净净，只算噪声的基准 MSE 损失
        # 移除训练时的条件梯度纠缠，把约束释放到纯推理阶段
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss


# ============================================================================
# 完整的多通道CSDI模型
# ============================================================================

class MultiChannelCSDI(nn.Module):
    """
    多变量协同条件扩散模型完整类
    
    整合：
    1. 时间特征注入（小时、周几、月份）
    2. Res-UNet + 空洞卷积
    3. 多通道条件引导（公式10）
    
    输入维度定义 (14维张量)：
    - Channel [0:3]: Target Residuals (正在去噪的风、光、负荷残差 x_t)
    - Channel [3:6]: Base Prediction (来自FEDformer的风、光、负荷预测趋势)
    - Channel [6:14]: Time Encoding (8维时间周期特征)
    
    输出维度定义：
    - Channel [0:3]: 风、光、负荷残差噪声预测（只预测前3维）
    
    KDE拟合：仅对Channel [0:3]（物理残差）进行概率密度建模和梯度引导
    """
    
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.target_type = config.get('target_type', 'residual')
        self.condition_mode = config.get('condition_mode', 'mix')
        self.use_forecast = config.get('use_forecast', True)
        self.use_network_condition = config.get('use_network_condition', True)
        self.use_guidance = config.get('use_guidance', True)
        self.cond_mask = config.get('cond_mask', [1, 1, 1])
        self.forecast_feature_config = config.get('forecast_features', {'enabled': False})
        self.forecast_feature_names = tuple(
            self.forecast_feature_config.get('names', ())
            if self.forecast_feature_config.get('enabled', False) else ()
        )
        
        # 输入输出通道定义。V0/V1: x_t only; V2: [x_t, forecast]; Vmix: [x_t, forecast, time_encoding].
        self.in_channels = config.get('in_channels', self._infer_input_channels())
        self.out_channels = config.get('out_channels', 3)  # 3维输出（风、光、负荷残差）
        
        # 时间特征嵌入维度
        self.d_time = config.get('d_time', 64)
        
        # 时间特征注入模块
        self.time_feature_embed = TimeFeatureEmbedding(self.d_time)
        self.pos_embed = SinusoidalPositionEmbedding(self.d_time)
        
        # Res-UNet模型 - 14维输入，3维输出
        self.unet = ResUNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            d_time=self.d_time,
            base_channels=config.get('base_channels', 128),
            num_layers=config.get('num_layers', 4)
        )
        
        # 扩散过程
        self.diffusion = GaussianDiffusionMultivariate(
            model=self.unet,
            num_steps=config.get('num_steps', 50),
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.02),
            schedule=config.get('schedule', 'quad'),
            guidance_scale=config.get('guidance_scale', 1.0),
            guidance_scales=config.get('guidance_scales', None),
            target_type=self.target_type,
            reverse_variance_type=config.get('reverse_variance_type', 'beta'),
        )

        if not self.use_guidance or self.condition_mode == 'none':
            self.diffusion.guidance_scale = 0.0
            self.diffusion.guidance_scales.zero_()
        
        # 保存配置用于一致性检查
        self.diffusion_config = {
            'num_steps': config.get('num_steps', 50),
            'beta_start': config.get('beta_start', 0.0001),
            'beta_end': config.get('beta_end', 0.02),
            'schedule': config.get('schedule', 'quad'),
            'guidance_scale': config.get('guidance_scale', 1.0),
            'reverse_variance_type': config.get('reverse_variance_type', 'beta'),
        }

    def _infer_input_channels(self):
        n_forecast_features = len(self.forecast_feature_names)
        if self.use_network_condition and self.condition_mode == 'mix':
            return 14 + n_forecast_features
        if self.use_network_condition:
            return 6 + n_forecast_features
        return 3

    def _select_target(self, batch):
        """Return diffusion target x0 with shape [B, 3, 168]."""
        if self.target_type == 'actual':
            x0 = batch['actual_3ch'].to(self.device)
        elif self.target_type == 'residual':
            x0 = batch.get('residual_target_3ch', batch['residual_3ch']).to(self.device)
        else:
            raise ValueError(f"Unsupported target_type: {self.target_type}")

        assert x0.ndim == 3, f"x0 must be [B, 3, 168], got {tuple(x0.shape)}"
        assert x0.shape[1] == 3, f"x0 channel dim must be 3, got {x0.shape[1]}"
        assert x0.shape[2] == 168, f"x0 length dim must be 168, got {x0.shape[2]}"
        return x0

    def _masked_forecast(self, forecast_3ch):
        assert forecast_3ch.ndim == 3 and forecast_3ch.shape[1:] == (3, 168), (
            f"forecast_3ch must be [B, 3, 168], got {tuple(forecast_3ch.shape)}"
        )
        mask = torch.as_tensor(self.cond_mask, dtype=forecast_3ch.dtype, device=forecast_3ch.device).view(1, 3, 1)
        return forecast_3ch * mask

    def build_model_input(self, x_t, forecast_3ch=None, time_encoding=None):
        """Build model input according to version config."""
        assert x_t.ndim == 3 and x_t.shape[1:] == (3, 168), (
            f"x_t must be [B, 3, 168], got {tuple(x_t.shape)}"
        )
        if not self.use_network_condition:
            model_input = x_t
        else:
            assert forecast_3ch is not None, "forecast_3ch is required when use_network_condition=True"
            forecast_3ch = self._masked_forecast(forecast_3ch)
            dynamic_features = build_forecast_dynamic_features(
                forecast_3ch, self.forecast_feature_config
            )
            if self.condition_mode == 'mix':
                assert time_encoding is not None, "time_encoding is required for condition_mode='mix'"
                assert time_encoding.ndim == 3 and time_encoding.shape[1:] == (8, 168), (
                    f"time_encoding must be [B, 8, 168], got {tuple(time_encoding.shape)}"
                )
                parts = [x_t, forecast_3ch]
                if dynamic_features is not None:
                    parts.append(dynamic_features)
                parts.append(time_encoding)
                model_input = torch.cat(parts, dim=1)
            else:
                parts = [x_t, forecast_3ch]
                if dynamic_features is not None:
                    parts.append(dynamic_features)
                model_input = torch.cat(parts, dim=1)

        assert model_input.shape[1] == self.in_channels, (
            f"model input channels {model_input.shape[1]} != configured in_channels {self.in_channels}"
        )
        return model_input

    def _build_condition(self, forecast_3ch, time_encoding):
        """Compatibility helper for old visualization code."""
        assert forecast_3ch.ndim == 3 and forecast_3ch.shape[1:] == (3, 168), (
            f"forecast_3ch must be [B, 3, 168], got {tuple(forecast_3ch.shape)}"
        )
        assert time_encoding.ndim == 3 and time_encoding.shape[1:] == (8, 168), (
            f"time_encoding must be [B, 8, 168], got {tuple(time_encoding.shape)}"
        )
        if self.condition_mode == 'none' or not self.use_forecast:
            forecast_3ch = torch.zeros_like(forecast_3ch)
            time_encoding = torch.zeros_like(time_encoding)
        else:
            forecast_3ch = self._masked_forecast(forecast_3ch)
        return torch.cat([forecast_3ch, time_encoding], dim=1)
        
    def get_time_features(self, timepoints):
        """
        从时间点提取小时、周几、月份特征
        
        Args:
            timepoints: (B, L) 时间点索引（假设是小时索引）
        Returns:
            time_feat: (B, L, d_time)
        """
        B, L = timepoints.shape
        device = timepoints.device
        
        # 假设timepoints是相对于某个起始点的偏移
        # 需要根据实际数据调整
        
        # 小时: timepoints % 24
        hour = (timepoints % 24).long()
        
        # 周几: (timepoints // 24) % 7
        weekday = (torch.div(timepoints, 24, rounding_mode='floor') % 7).long()
        
        # 月份: 假设数据跨度一年，简化处理
        month = (torch.div(timepoints, 24 * 30, rounding_mode='floor') % 12).long()
        
        # 时间特征嵌入
        time_feat = self.time_feature_embed(hour, weekday, month)
        
        # 加上位置编码
        pos_feat = self.pos_embed(torch.arange(L, device=device).unsqueeze(0).expand(B, -1))
        time_feat = time_feat + pos_feat
        
        return time_feat
    
    def forward(self, batch):
        """
        训练时的前向传播
        
        14通道输入结构：
        - Channel [0:3]: Target Residuals (正在去噪的风、光、负荷残差 x_t)
        - Channel [3:6]: Base Prediction (来自FEDformer的风、光、负荷预测趋势)
        - Channel [6:14]: Time Encoding (8维时间周期特征)
        
        Args:
            batch: 包含 'residual_3ch', 'forecast_3ch', 'time_encoding', 'cond_matrix', 'timepoints'
        Returns:
            loss: 训练损失
        """
        # x0: actual by default for V0, residual only for explicit residual experiments.
        x0 = self._select_target(batch)  # (B, 3, 168)
        
        forecast_3ch = batch['forecast_3ch'].to(self.device)  # (B, 3, 168)
        time_encoding = batch['time_encoding'].to(self.device)  # (B, 8, 168)
        if self.use_forecast:
            assert forecast_3ch.shape == x0.shape, f"forecast shape {forecast_3ch.shape} must match x0 {x0.shape}"
        
        residual = batch['residual_3ch'].to(self.device)
        if self.target_type == 'residual':
            reconstructed_actual = forecast_3ch - residual
            assert reconstructed_actual.shape == x0.shape
        
        # 时间点索引
        timepoints = batch['timepoints'].to(self.device)  # (B, 168)
        
        # 获取时间特征
        time_feat = self.get_time_features(timepoints)
        
        def model_input_fn(x_t):
            return self.build_model_input(
                x_t,
                forecast_3ch=forecast_3ch if self.use_forecast else None,
                time_encoding=time_encoding,
            )

        loss = self.diffusion(x0, model_input_fn, time_feat)
        
        return loss
    
    def generate(self, batch, n_samples=10):
        """
        生成场景
        
        Args:
            batch: 包含 'forecast_3ch', 'time_encoding', 'cond_matrix', 'timepoints'
            n_samples: 生成样本数量
        Returns:
            samples: (B, n_samples, 3, 168) 生成的残差场景
        """
        forecast_3ch = batch['forecast_3ch'].to(self.device)  # (B, 3, 168)
        time_encoding = batch['time_encoding'].to(self.device)  # (B, 8, 168)
        
        cond_matrix = batch['cond_matrix'].to(self.device) if self.use_guidance else None
        timepoints = batch['timepoints'].to(self.device)

        B = forecast_3ch.shape[0]

        # Vectorize scenario samples by folding the sample dimension into batch.
        # This avoids n_samples separate reverse-diffusion loops per test batch.
        if n_samples > 1:
            forecast_for_model = forecast_3ch.repeat_interleave(n_samples, dim=0)
            time_encoding_for_model = time_encoding.repeat_interleave(n_samples, dim=0)
            timepoints_for_model = timepoints.repeat_interleave(n_samples, dim=0)
            cond_matrix_for_model = (
                cond_matrix.repeat_interleave(n_samples, dim=0)
                if cond_matrix is not None else None
            )
            effective_batch = B * n_samples
        else:
            forecast_for_model = forecast_3ch
            time_encoding_for_model = time_encoding
            timepoints_for_model = timepoints
            cond_matrix_for_model = cond_matrix
            effective_batch = B

        time_feat = self.get_time_features(timepoints_for_model)

        def model_input_fn(x_t):
            return self.build_model_input(
                x_t,
                forecast_3ch=forecast_for_model if self.use_forecast else None,
                time_encoding=time_encoding_for_model,
            )
        
        with torch.no_grad():
            samples = self.diffusion.sample(
                model_input_fn,
                batch_size=effective_batch,
                device=self.device,
                time_feat=time_feat,
                cond_matrix=cond_matrix_for_model,
                forecast=forecast_for_model if self.use_forecast else None,
                n_samples=1,
            )

        if n_samples > 1:
            samples = samples[:, 0].reshape(B, n_samples, 3, 168)

        return samples
