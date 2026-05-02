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
        
        # 小时嵌入 (0-23)
        self.hour_embed = nn.Embedding(24, d_model // 3)
        
        # 周几嵌入 (0-6, 0=周一)
        self.weekday_embed = nn.Embedding(7, d_model // 3)
        
        # 月份嵌入 (0-11)
        self.month_embed = nn.Embedding(12, d_model // 3)
        
        # 合并后的投影层
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
    4. 支持11维特征解耦输入
    
    输入维度定义 (11维张量)：
    - Channel [0:3]: 风、光、负荷残差 (Residuals) - 生成核心主体
    - Channel [3:11]: 8维时间周期编码 (Sin/Cos) - 环境背景条件
    
    输出维度定义：
    - Channel [0:3]: 风、光、负荷残差生成结果
    """
    
    def __init__(self, in_channels=11, out_channels=3, d_time=64, 
                 base_channels=128, num_layers=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_time = d_time
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        
        channels = base_channels
        for i in range(num_layers):
            self.encoder_blocks.append(
                ResidualBlock(in_channels if i == 0 else channels, 
                             channels * 2 if i < num_layers - 1 else channels,
                             d_time)
            )
            if i < num_layers - 1:
                self.encoder_downsample.append(nn.Conv1d(channels * 2, channels * 2, 4, stride=2, padding=1))
                channels *= 2
        
        # Bottleneck with Dilated Convolution
        self.bottleneck = DilatedConvBlock(channels, channels, dilations=[1, 2, 4, 8])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.decoder_upsample.append(nn.ConvTranspose1d(channels, channels // 2, 4, stride=2, padding=1))
            channels //= 2
            self.decoder_blocks.append(
                ResidualBlock(channels * 2, channels, d_time)  # *2 for skip connection
            )
        
        # Output layer
        self.output_conv = nn.Conv1d(channels, out_channels, 1)
        
    def forward(self, x, time_feat):
        """
        Args:
            x: (B, 3, 168) 多通道输入
            time_feat: (B, 168, d_time) 时间特征
        Returns:
            out: (B, 3, 168) 多通道输出
        """
        # Encoder
        encoder_outputs = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, time_feat)
            encoder_outputs.append(x)
            if i < len(self.encoder_downsample):
                x = self.encoder_downsample[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, block in enumerate(self.decoder_blocks):
            x = self.decoder_upsample[i](x)
            # Skip connection
            skip = encoder_outputs[-(i+2)]
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_feat)
        
        # Output
        out = self.output_conv(x)
        
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
    
    def __init__(self, model, num_steps=50, beta_start=0.0001, beta_end=0.5,
                 schedule='quad', guidance_scale=1.0):
        super().__init__()
        
        self.model = model  # ResUNet
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        
        # Beta schedule
        if schedule == 'quad':
            self.beta = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps) ** 2
        elif schedule == 'linear':
            self.beta = torch.linspace(beta_start, beta_end, num_steps)
        
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
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
    
    def compute_conditional_gradient(self, x_t, cond_matrix, gamma):
        """
        论文公式10: 计算条件梯度
        
        ∇_{x_t} ||γ·x_t - c||²_F
        
        其中条件c取区间中点: c_mid = (c_down + c_up) / 2
        
        Args:
            x_t: (B, 3, 168) 当前生成值
            cond_matrix: (B, 3, 168, 2) 条件矩阵 [c_down, c_up]
            gamma: 引导强度
        Returns:
            gradient: (B, 3, 168) 条件梯度
        """
        # 计算条件中点
        c_down = cond_matrix[..., 0]  # (B, 3, 168)
        c_up = cond_matrix[..., 1]
        c_mid = (c_down + c_up) / 2
        
        # Frobenius范数梯度
        # ||γ·x_t - c||²_F = Σ(γ·x_t - c)²
        # ∇ = 2γ(γ·x_t - c)
        gradient = 2 * gamma * (gamma * x_t - c_mid)
        
        return gradient
    
    def denoise_step(self, x_t, t, cond_matrix, time_feat):
        """
        论文公式10: 反向去噪一步
        
        x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t)·ε_θ) + σ_t·z
        
        条件梯度修正：
        mean = mean - guidance_scale · ∇_{x_t} ||γ·x_t - c||²_F
        
        Args:
            x_t: (B, 3, 168) 当前噪声数据
            t: 时间步索引
            cond_matrix: (B, 3, 168, 2) 条件矩阵
            time_feat: (B, 168, d_time) 时间特征
        Returns:
            x_{t-1}: 去噪一步后的数据
        """
        B = x_t.shape[0]
        device = x_t.device
        
        # 预测噪声
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        predicted_noise = self.model(x_t, time_feat)
        
        # 计算去噪均值
        alpha_t = self.alpha[t]
        alpha_hat_t = self.alpha_hat[t]
        
        # 基础去噪公式
        mean = (1 / alpha_t.sqrt()) * (x_t - (1 - alpha_t) / (1 - alpha_hat_t).sqrt() * predicted_noise)
        
        # 论文公式10: 条件梯度修正
        if self.guidance_scale > 0:
            cond_gradient = self.compute_conditional_gradient(x_t, cond_matrix, gamma=1.0)
            mean = mean - self.guidance_scale * cond_gradient
        
        # 添加噪声（除了最后一步）
        if t > 0:
            sigma = self.beta[t].sqrt()
            noise = torch.randn_like(x_t)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean
        
        return x_prev
    
    def sample(self, cond_matrix, time_feat, n_samples=1):
        """
        完整采样过程：从噪声生成场景
        
        Args:
            cond_matrix: (B, 3, 168, 2) 条件矩阵
            time_feat: (B, 168, d_time) 时间特征
            n_samples: 生成样本数量
        Returns:
            samples: (B, n_samples, 3, 168) 生成的场景
        """
        B = cond_matrix.shape[0]
        device = cond_matrix.device
        
        # 初始化纯噪声
        samples = torch.zeros(B, n_samples, 3, 168, device=device)
        
        for s in range(n_samples):
            x_t = torch.randn(B, 3, 168, device=device)
            
            # 逐步去噪
            for t in range(self.num_steps - 1, -1, -1):
                x_t = self.denoise_step(x_t, t, cond_matrix, time_feat)
            
            samples[:, s] = x_t
        
        return samples
    
    def forward(self, x0, cond_matrix, time_feat, t=None):
        """
        训练时的前向传播：计算损失
        
        Args:
            x0: (B, 3, 168) 原始残差数据
            cond_matrix: (B, 3, 168, 2) 条件矩阵
            time_feat: (B, 168, d_time) 时间特征
            t: 时间步（可选，随机采样）
        Returns:
            loss: 训练损失
        """
        B = x0.shape[0]
        device = x0.device
        
        # 随机采样时间步
        if t is None:
            t = torch.randint(0, self.num_steps, (B,), device=device)
        
        # 添加噪声
        x_t, noise = self.add_noise(x0, t)
        
        # 预测噪声
        predicted_noise = self.model(x_t, time_feat)
        
        # 计算损失（MSE）
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
    
    输入维度定义 (11维张量)：
    - Channel [0:3]: 风、光、负荷残差 (Residuals) - 生成核心主体
    - Channel [3:11]: 8维时间周期编码 (Sin/Cos) - 环境背景条件
    
    输出维度定义：
    - Channel [0:3]: 风、光、负荷残差生成结果
    """
    
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        
        # 输入输出通道定义
        self.in_channels = config.get('in_channels', 11)  # 11维输入
        self.out_channels = config.get('out_channels', 3)  # 3维输出（风、光、负荷残差）
        
        # 时间特征嵌入维度
        self.d_time = config.get('d_time', 64)
        
        # 时间特征注入模块
        self.time_feature_embed = TimeFeatureEmbedding(self.d_time)
        self.pos_embed = SinusoidalPositionEmbedding(self.d_time)
        
        # Res-UNet模型 - 11维输入，3维输出
        self.unet = ResUNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            d_time=self.d_time,
            base_channels=config.get('base_channels', 128),  # 确保容量足够
            num_layers=config.get('num_layers', 4)
        )
        
        # 扩散过程
        self.diffusion = GaussianDiffusionMultivariate(
            model=self.unet,
            num_steps=config.get('num_steps', 50),
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.5),
            schedule=config.get('schedule', 'quad'),
            guidance_scale=config.get('guidance_scale', 1.0)
        )
        
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
        weekday = ((timepoints // 24) % 7).long()
        
        # 月份: 假设数据跨度一年，简化处理
        month = ((timepoints // (24 * 30)) % 12).long()
        
        # 时间特征嵌入
        time_feat = self.time_feature_embed(hour, weekday, month)
        
        # 加上位置编码
        pos_feat = self.pos_embed(torch.arange(L, device=device).unsqueeze(0).expand(B, -1))
        time_feat = time_feat + pos_feat
        
        return time_feat
    
    def forward(self, batch):
        """
        训练时的前向传播
        
        Args:
            batch: 包含 'residual', 'cond_matrix', 'timepoints'
        Returns:
            loss: 训练损失
        """
        residual = batch['residual'].to(self.device)  # (B, 3, 168)
        cond_matrix = batch['cond_matrix'].to(self.device)  # (B, 3, 168, 2)
        timepoints = batch['timepoints'].to(self.device)  # (B, 168)
        
        # 获取时间特征
        time_feat = self.get_time_features(timepoints)
        
        # 计算扩散损失
        loss = self.diffusion(residual, cond_matrix, time_feat)
        
        return loss
    
    def generate(self, batch, n_samples=10):
        """
        生成场景
        
        Args:
            batch: 包含 'cond_matrix', 'timepoints'
            n_samples: 生成样本数量
        Returns:
            samples: (B, n_samples, 3, 168) 生成的残差场景
        """
        cond_matrix = batch['cond_matrix'].to(self.device)
        timepoints = batch['timepoints'].to(self.device)
        
        time_feat = self.get_time_features(timepoints)
        
        with torch.no_grad():
            samples = self.diffusion.sample(cond_matrix, time_feat, n_samples)
        
        return samples