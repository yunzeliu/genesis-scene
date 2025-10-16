import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
from models.diffusion import DiffusionPolicy
from dataset import RoboticDataset
import wandb

def train(
    data_dir: str = "processed_dataset",
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    save_dir: str = "checkpoints",
    log_interval: int = 10,
    save_interval: int = 10000,
    use_wandb: bool = False
):
    """
    训练Diffusion Policy
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        save_dir: 模型保存目录
        log_interval: 日志记录间隔
        save_interval: 模型保存间隔
        use_wandb: 是否使用wandb记录
    """
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化wandb
    if use_wandb:
        wandb.init(
            project="diffusion_policy",
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs
            }
        )
    
    # 创建数据加载器
    dataset = RoboticDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = DiffusionPolicy(
        n_action_dims=3,  # 3D位置差
        n_diffusion_steps=100,
        width=32
    ).to(device)
    
    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 训练循环
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # 将数据移到设备上
            image = batch['image'].to(device)
            action = batch['action'].to(device)
            
            # 计算损失
            loss = model.get_loss(image, action)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录日志
            if global_step % log_interval == 0:
                progress_bar.set_postfix({'loss': loss.item()})
                if use_wandb:
                    wandb.log({
                        'loss': loss.item(),
                        'epoch': epoch,
                        'global_step': global_step
                    })
            
            # 保存模型
            if global_step % save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, save_dir / f'model_step_{global_step}.pt')
            
            global_step += 1
        
        # 每个epoch结束后保存一次
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, save_dir / f'model_epoch_{epoch}.pt')
    
    if use_wandb:
        wandb.finish()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="processed_dataset")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--save_dir', type=str, default="checkpoints")
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--use_wandb', action='store_true')
    
    args = parser.parse_args()
    train(**vars(args))

if __name__ == "__main__":
    main()
