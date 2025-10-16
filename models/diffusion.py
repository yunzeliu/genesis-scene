import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        horizon: int = 1,
        n_action_dims: int = 3,
        n_diffusion_steps: int = 100,
        obs_encoder_group_norm: bool = True,
        width: int = 32,
        ):
        super().__init__()
        
        # 配置参数
        self.horizon = horizon
        self.n_action_dims = n_action_dims
        self.n_diffusion_steps = n_diffusion_steps
        
        # 图像编码器
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width) if obs_encoder_group_norm else nn.Identity(),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width) if obs_encoder_group_norm else nn.Identity(),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width) if obs_encoder_group_norm else nn.Identity(),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width) if obs_encoder_group_norm else nn.Identity(),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(width * 20 * 20, 512),
            nn.SiLU()
        )
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(width),
            nn.Linear(width, width * 4),
            nn.SiLU(),
            nn.Linear(width * 4, width * 4)
        )
        
        # 动作解码器
        self.action_decoder = nn.Sequential(
            nn.Linear(512 + width * 4 + n_action_dims, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, n_action_dims)
        )
        
        # 设置beta schedule
        self.beta_schedule = cosine_beta_schedule(n_diffusion_steps)
        self.alphas = 1. - self.beta_schedule
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
    def forward(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        Args:
            obs: 图像观测 (B, C, H, W)
            action: 动作 (B, A)
            t: 时间步 (B,)
        Returns:
            预测的噪声
        """
        # 编码观测
        obs_feat = self.obs_encoder(obs)
        
        # 时间嵌入
        time_emb = self.time_mlp(t)
        
        # 合并特征
        x = torch.cat([obs_feat, time_emb, action], dim=1)
        
        # 解码预测噪声
        pred_noise = self.action_decoder(x)
        return pred_noise
    
    def get_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算损失函数
        Args:
            obs: 图像观测 (B, C, H, W)
            action: 动作 (B, A)
        Returns:
            loss
        """
        batch_size = obs.shape[0]
        
        # 采样时间步并确保是长整型
        t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=obs.device, dtype=torch.long)
        
        # 采样噪声
        noise = torch.randn_like(action)
        
        # 对动作添加噪声
        noisy_action = self.q_sample(action, t, noise)
        
        # 预测噪声
        pred_noise = self(obs, noisy_action, t)
        
        # 计算损失
        loss = nn.functional.mse_loss(pred_noise, noise)
        return loss
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        从q(x_t|x_0)采样
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, obs: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        从p(x_{t-1}|x_t)采样
        """
        # 确保t是长整型
        t = t.long()
        
        betas_t = extract(self.beta_schedule, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        
        # 方差
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self(obs, x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.beta_schedule, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        """
        采样动作
        Args:
            obs: 图像观测 (B, C, H, W)
        Returns:
            采样的动作
        """
        device = obs.device
        batch_size = obs.shape[0]
        
        # 从标准正态分布开始
        x = torch.randn((batch_size, self.n_action_dims), device=device)
        
        # 逐步去噪
        for t in reversed(range(self.n_diffusion_steps)):
            t_batch = torch.ones(batch_size, device=device, dtype=torch.long) * t
            x = self.p_sample(obs, x, t_batch)
            
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def extract(a, t, x_shape):
    """
    从a中提取适当的索引以匹配x_shape
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine调度
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
