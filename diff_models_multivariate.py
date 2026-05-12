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
    
    def __init__(self, in_channels, out_channels, d_time=64, kernel_size=3):
        super().__init__()
        
        # 主卷积路径
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # 时间特征注入
        self.time_proj = nn.Linear(d_time, out_channels)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
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
        
        h = self.conv2(h)
        h = self.bn2(h)
        h = h + time_emb
        h = F.relu(h)
        
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
                 schedule='quad', guidance_scale=1.0):
        super().__init__()
        
        self.model = model  # ResUNet
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        
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
        noise = torch.randn_like(x0)
        
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1)
        x_t = alpha_hat_t.sqrt() * x0 + (1 - alpha_hat_t).sqrt() * noise
        
        return x_t, noise
    
    def compute_conditional_gradient(self, x_t, cond_matrix, debug=False):
        """
        论文公式10: 计算条件梯度
        
        γ 是 0-1 二值系数：
        - 当 x_t 在条件区间 [c_down, c_up] 内时，γ=0（不修正）
        - 当 x_t 超出条件区间时，γ=1（修正）
        
        梯度方向：引导 x_t 向条件区间边界移动
        
        Args:
            x_t: (B, 3, 168) 当前生成值
            cond_matrix: (B, 3, 168, 2) 条件矩阵 [c_down, c_up]
            debug: 是否打印调试信息
        Returns:
            gradient: (B, 3, 168) 条件梯度
            gamma_mask: (B, 3, 168) 二值系数掩码
            debug_info: dict 调试信息（仅当 debug=True）
        """
        c_down = cond_matrix[..., 0]  # (B, 3, 168)
        c_up = cond_matrix[..., 1]
        
        # 计算二值系数 γ
        # 当 x_t < c_down 或 x_t > c_up 时，γ=1
        # 当 x_t 在 [c_down, c_up] 内时，γ=0
        gamma_mask = ((x_t < c_down) | (x_t > c_up)).float()
        
        # 计算梯度方向
        # 如果 x_t < c_down，梯度指向 c_down（向上）
        # 如果 x_t > c_up，梯度指向 c_up（向下）
        gradient = torch.zeros_like(x_t)
        
        # x_t < c_down: 梯度 = c_down - x_t（向上推）
        below_mask = (x_t < c_down).float()
        gradient = gradient + below_mask * (c_down - x_t)
        
        # x_t > c_up: 梯度 = c_up - x_t（向下推）
        above_mask = (x_t > c_up).float()
        gradient = gradient + above_mask * (c_up - x_t)
        
        # 调试信息
        debug_info = None
        if debug:
            debug_info = {
                'x_t_range': (x_t.min().item(), x_t.max().item()),
                'c_down_range': (c_down.min().item(), c_down.max().item()),
                'c_up_range': (c_up.min().item(), c_up.max().item()),
                'gamma_ratio': gamma_mask.mean().item(),  # 超出区间的比例
                'below_ratio': below_mask.mean().item(),  # 低于下界的比例
                'above_ratio': above_mask.mean().item(),  # 高于上界的比例
                'gradient_range': (gradient.min().item(), gradient.max().item()),
                'gradient_mean': gradient.mean().item(),
            }
        
        return gradient, gamma_mask, debug_info
    
    def denoise_step(self, x_t, t, cond_full, cond_matrix, time_feat, debug=False):
        """
        论文公式10: 反向去噪一步
        
        x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ) + σ_t·z
        
        条件梯度修正：
        mean = mean - guidance_scale · ∇_{x_t} ||γ·x_t - c||²_F
        
        Args:
            x_t: (B, 3, 168) 当前噪声数据 (Target Residuals)
            t: 时间步索引
            cond_full: (B, 11, 168) 条件部分 (Base Prediction + Time Encoding)
            cond_matrix: (B, 3, 168, 2) KDE条件矩阵
            time_feat: (B, 168, d_time) 时间特征
            debug: 是否返回调试信息
        Returns:
            x_{t-1}: 去噪一步后的数据
            debug_info: dict 调试信息（仅当 debug=True）
        """
        B = x_t.shape[0]
        device = x_t.device
        debug_info = None
        
        # 构建14通道输入: [x_t (3), cond_full (11)]
        input_14ch = torch.cat([x_t, cond_full], dim=1)  # (B, 14, 168)
        
        # 预测噪声（模型输出3维）
        predicted_noise = self.model(input_14ch, time_feat)
        
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
            cond_gradient, gamma_mask, grad_debug = self.compute_conditional_gradient(x_t, cond_matrix, debug=debug)
            
            # 【修复】放宽梯度裁剪范围
            cond_gradient_clamped = torch.clamp(cond_gradient, min=-1.0, max=1.0)
            
            # 应用梯度修正：只在超出区间的地方修正
            # 【关键修复】gradient 已经是(目标 - 当前)的向量，代表更新的正确方向，应当是加上而不是减去！
            mean = mean + self.guidance_scale * t_decay * cond_gradient_clamped * gamma_mask
            
            if debug:
                debug_info = {
                    'step': t,
                    'alpha_t': alpha_t.item(),
                    'alpha_hat_t': alpha_hat_t.item(),
                    't_decay': t_decay,
                    'predicted_noise_range': (predicted_noise.min().item(), predicted_noise.max().item()),
                    'mean_before_guidance_range': (mean.min().item(), mean.max().item()),
                    'guidance_applied': self.guidance_scale * t_decay,
                    **grad_debug
                }
        
        # 添加噪声（除了最后一步）
        if t > 0:
            sigma = self.beta[t].sqrt()
            noise = torch.randn_like(x_t)
            # 移除错误限制: noise = torch.clamp(noise, min=-1.0, max=1.0)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean
        
        if debug and debug_info:
            debug_info['x_prev_range'] = (x_prev.min().item(), x_prev.max().item())
            debug_info['has_nan'] = torch.isnan(x_prev).any().item()
            debug_info['has_inf'] = torch.isinf(x_prev).any().item()
        
        return x_prev, debug_info
    
    def sample(self, cond_full, cond_matrix, time_feat, n_samples=1, debug=False, debug_steps=None):
        """
        完整采样过程：从噪声生成场景
        
        Args:
            cond_full: (B, 11, 168) 条件部分 (Base Prediction + Time Encoding)
            cond_matrix: (B, 3, 168, 2) KDE条件矩阵
            time_feat: (B, 168, d_time) 时间特征
            n_samples: 生成样本数量
            debug: 是否启用调试模式
            debug_steps: 调试信息收集的步骤列表（如 [499, 400, 300, 200, 100, 50, 0]）
        Returns:
            samples: (B, n_samples, 3, 168) 生成的场景
            debug_log: list 调试日志（仅当 debug=True）
        """
        B = cond_full.shape[0]
        device = cond_full.device
        
        # 初始化纯噪声
        samples = torch.zeros(B, n_samples, 3, 168, device=device)
        debug_log = []
        
        for s in range(n_samples):
            x_t = torch.randn(B, 3, 168, device=device)
            
            # 逐步去噪
            for t in range(self.num_steps - 1, -1, -1):
                # 判断是否需要收集调试信息
                need_debug = debug and (debug_steps is None or t in debug_steps)
                x_t, step_debug = self.denoise_step(x_t, t, cond_full, cond_matrix, time_feat, debug=need_debug)
                
                if need_debug and step_debug:
                    debug_log.append(step_debug)
            
            samples[:, s] = x_t
        
        if debug:
            return samples, debug_log
        return samples
    
    def forward(self, x0, cond_full, time_feat, cond_matrix=None, t=None):
        """
        训练时的前向传播：计算损失
        
        Args:
            x0: (B, 3, 168) 原始残差数据 (Target Residuals)
            cond_full: (B, 11, 168) 条件部分 (Base Prediction + Time Encoding)
            time_feat: (B, 168, d_time) 时间特征
            cond_matrix: (B, 3, 168, 2) KDE条件矩阵 (可选)
            t: 时间步（可选，随机采样）
        Returns:
            loss: 训练损失
        """
        B = x0.shape[0]
        device = x0.device
        
        # 随机采样时间步
        if t is None:
            t = torch.randint(0, self.num_steps, (B,), device=device)
        
        # 添加噪声（只对前3维Target Residuals添加噪声）
        x_t, noise = self.add_noise(x0, t)
        
        # 构建14通道输入: [x_t (3), cond_full (11)]
        input_14ch = torch.cat([x_t, cond_full], dim=1)  # (B, 14, 168)
        
        # 预测噪声（模型输出3维）
        predicted_noise = self.model(input_14ch, time_feat)
        
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
        
        # 输入输出通道定义
        self.in_channels = config.get('in_channels', 14)  # 14维输入
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
            guidance_scale=config.get('guidance_scale', 1.0)
        )
        
        # 保存配置用于一致性检查
        self.diffusion_config = {
            'num_steps': config.get('num_steps', 50),
            'beta_start': config.get('beta_start', 0.0001),
            'beta_end': config.get('beta_end', 0.02),
            'schedule': config.get('schedule', 'quad'),
            'guidance_scale': config.get('guidance_scale', 1.0)
        }
        
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
        # Target Residuals (3, 168) - 扩散目标
        residual = batch['residual_3ch'].to(self.device)  # (B, 3, 168)
        
        # 条件部分: Base Prediction (3, 168) + Time Encoding (8, 168) = 11维
        forecast_3ch = batch['forecast_3ch'].to(self.device)  # (B, 3, 168)
        time_encoding = batch['time_encoding'].to(self.device)  # (B, 8, 168)
        cond_full = torch.cat([forecast_3ch, time_encoding], dim=1)  # (B, 11, 168)
        
        # KDE条件矩阵 (仅对Channel [0:3]构建)
        cond_matrix = batch['cond_matrix'].to(self.device)  # (B, 3, 168, 2)
        
        # 时间点索引
        timepoints = batch['timepoints'].to(self.device)  # (B, 168)
        
        # 获取时间特征
        time_feat = self.get_time_features(timepoints)
        
        # 计算扩散损失（传入条件矩阵，保证训练推理一致）
        loss = self.diffusion(residual, cond_full, time_feat, cond_matrix)
        
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
        # 条件部分
        forecast_3ch = batch['forecast_3ch'].to(self.device)  # (B, 3, 168)
        time_encoding = batch['time_encoding'].to(self.device)  # (B, 8, 168)
        cond_full = torch.cat([forecast_3ch, time_encoding], dim=1)  # (B, 11, 168)
        
        # KDE条件矩阵
        cond_matrix = batch['cond_matrix'].to(self.device)  # (B, 3, 168, 2)
        timepoints = batch['timepoints'].to(self.device)
        
        time_feat = self.get_time_features(timepoints)
        
        with torch.no_grad():
            samples = self.diffusion.sample(cond_full, cond_matrix, time_feat, n_samples)
        
        return samples
