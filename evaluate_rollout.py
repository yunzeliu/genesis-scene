import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from models.diffusion import DiffusionPolicy
from dataset import RoboticDataset

class RolloutEvaluator:
    def __init__(self, model_path, device="cuda"):
        """初始化评估器"""
        self.device = device
        
        # 加载模型
        self.model = DiffusionPolicy(
            n_action_dims=3,
            n_diffusion_steps=100,
            width=32
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def generate_random_start_position(self):
        """生成随机初始位置"""
        # 参考process_dataset.py中的范围
        pos = np.array([0.55, -0.05, 0.125])
        poffset = np.random.uniform(-0.1, 0.1, size=(2,))
        pos = pos + np.array([poffset[0] * 0.5, poffset[1], 0.0])
        return pos
    
    def rollout_trajectory(self, start_pos, num_steps=2000):
        """从给定起始位置生成轨迹
        
        Args:
            start_pos: 起始位置
            num_steps: 推演步数
            
        Returns:
            positions: 位置序列
            actions: 动作序列
        """
        positions = [start_pos]
        actions = []
        current_pos = start_pos
        
        # 创建初始观测图像
        image = self.create_observation(current_pos)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        
        for _ in tqdm(range(num_steps), desc="Generating trajectory"):
            with torch.no_grad():
                # 使用模型预测下一个动作
                action = self.model.sample(image_tensor)
                action = action.cpu().numpy()[0]  # (3,)
                
                # 更新位置
                current_pos = current_pos + action
                
                # 记录
                positions.append(current_pos.copy())
                actions.append(action)
                
                # 更新观测图像
                image = self.create_observation(current_pos)
                image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        
        return np.array(positions), np.array(actions)
    
    def create_observation(self, position):
        """创建观测图像（简化版本，实际使用时需要匹配训练数据的图像生成方式）
        
        Args:
            position: 当前位置 (3,)
            
        Returns:
            observation: 图像数组 (C, H, W)
        """
        # 创建一个简单的可视化图像
        H, W = 320, 320
        image = np.zeros((H, W, 3), dtype=np.float32)
        
        # 将3D位置映射到2D图像坐标
        x, y, z = position
        px = int(np.clip((x + 1) * W / 2, 0, W-1))  # 将x从[-1,1]映射到[0,W]并限制范围
        py = int(np.clip((y + 1) * H / 2, 0, H-1))  # 将y从[-1,1]映射到[0,H]并限制范围
        
        # 绘制当前位置
        cv2.circle(image, (int(px), int(py)), 5, (1.0, 0.0, 0.0), -1)
        
        # 转换为(C,H,W)格式
        image = image.transpose(2, 0, 1)
        return image
    
    def save_trajectory_video(self, positions, output_path, fps=30):
        """保存轨迹视频
        
        Args:
            positions: 位置序列 (N, 3)
            output_path: 输出视频路径
            fps: 视频帧率
        """
        H, W = 320, 320
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
        
        for pos in tqdm(positions, desc="Saving video"):
            # 创建帧
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            
            # 将3D位置映射到2D图像坐标
            x, y, z = pos
            px = int(np.clip((x + 1) * W / 2, 0, W-1))
            py = int(np.clip((y + 1) * H / 2, 0, H-1))
            
            # 绘制轨迹
            cv2.circle(frame, (int(px), int(py)), 5, (0, 0, 255), -1)
            
            # 写入帧
            out.write(frame)
        
        out.release()
    
    def plot_trajectory_3d(self, positions, output_path):
        """绘制3D轨迹图
        
        Args:
            positions: 位置序列 (N, 3)
            output_path: 输出图像路径
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-')
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', marker='o', label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', marker='o', label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        plt.savefig(output_path)
        plt.close()

def evaluate_rollouts(
    model_path: str,
    output_dir: str = "rollout_results",
    num_rollouts: int = 10,
    num_steps: int = 2000,
    device: str = "cuda"
):
    """评估多个轨迹
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
        num_rollouts: 轨迹数量
        num_steps: 每个轨迹的步数
        device: 运行设备
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建评估器
    evaluator = RolloutEvaluator(model_path, device)
    
    # 生成多个轨迹
    for i in range(num_rollouts):
        print(f"\nGenerating rollout {i+1}/{num_rollouts}")
        
        # 创建轨迹目录
        rollout_dir = output_dir / f"rollout_{i}"
        rollout_dir.mkdir(exist_ok=True)
        
        # 生成随机起始位置
        start_pos = evaluator.generate_random_start_position()
        
        # 生成轨迹
        positions, actions = evaluator.rollout_trajectory(start_pos, num_steps)
        
        # 保存轨迹数据
        np.save(rollout_dir / "positions.npy", positions)
        np.save(rollout_dir / "actions.npy", actions)
        
        # 保存视频
        evaluator.save_trajectory_video(positions, rollout_dir / "trajectory.mp4")
        
        # 绘制3D轨迹图
        evaluator.plot_trajectory_3d(positions, rollout_dir / "trajectory_3d.png")
        
        # 保存起始位置
        np.save(rollout_dir / "start_position.npy", start_pos)
        
        print(f"Saved results to {rollout_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default="rollout_results", help='Output directory')
    parser.add_argument('--num_rollouts', type=int, default=10, help='Number of rollouts to generate')
    parser.add_argument('--num_steps', type=int, default=2000, help='Number of steps per rollout')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run evaluation on')
    
    args = parser.parse_args()
    evaluate_rollouts(**vars(args))

if __name__ == "__main__":
    main()
