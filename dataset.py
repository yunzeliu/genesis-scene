import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class RoboticDataset(Dataset):
    def __init__(self, data_dir):
        """
        机器人数据集
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.pairs = list(self.data_dir.glob("pairs/traj_*_step_*"))
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair_dir = self.pairs[idx]
        
        # 加载图像和动作
        image = np.load(pair_dir / "image.npy")
        action = np.load(pair_dir / "action.npy")
        
        # 转换为torch张量
        image = torch.from_numpy(image).float()
        action = torch.from_numpy(action).float()
        
        return {
            'image': image,
            'action': action
        }
        
    def get_trajectory(self, traj_id):
        """
        获取完整的轨迹数据
        
        Args:
            traj_id: 轨迹ID
            
        Returns:
            trajectory_data: 包含完整轨迹的字典
        """
        traj_pairs = sorted([p for p in self.pairs if f"traj_{traj_id}_" in p.name])
        
        images = []
        actions = []
        
        for pair_dir in traj_pairs:
            image = np.load(pair_dir / "image.npy")
            action = np.load(pair_dir / "action.npy")
            
            images.append(torch.from_numpy(image).float())
            actions.append(torch.from_numpy(action).float())
            
        return {
            'images': torch.stack(images),
            'actions': torch.stack(actions)
        }
