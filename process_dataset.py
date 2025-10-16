import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

class DatasetProcessor:
    def __init__(self, experiment_dir, output_dir="processed_dataset"):
        """
        Args:
            experiment_dir: 实验数据目录路径
            output_dir: 处理后数据的保存目录
        """
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        
        # 创建必要的子目录
        self.pairs_dir = self.output_dir / "pairs"
        self.pairs_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_image(self, frame):
        """预处理图像用于训练
        
        Args:
            frame: BGR格式的图像
            
        Returns:
            processed: 预处理后的图像数组 (C, H, W)
        """
        # 转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为float32并归一化到[0,1]
        frame = frame.astype(np.float32) / 255.0
        
        # 调整通道顺序为(C,H,W)以适应PyTorch
        frame = frame.transpose(2, 0, 1)
        
        return frame
    
    def compute_action(self, curr_target, next_target):
        """计算相邻时刻的动作（位置差）
        
        Args:
            curr_target: 当前时刻的目标位置
            next_target: 下一时刻的目标位置
            
        Returns:
            action: 位置差向量
        """
        curr_target = np.array(curr_target)
        next_target = np.array(next_target)
        
        # 验证数据
        if curr_target.shape != (3,) or next_target.shape != (3,):
            raise ValueError(f"Invalid target shape: curr_target {curr_target.shape}, next_target {next_target.shape}")
            
        # 计算位置差
        action = next_target - curr_target
        
        # 验证动作范围
        action_norm = np.linalg.norm(action)
        if action_norm > 1.0:  # 假设动作不应该太大
            print(f"Warning: Large action detected: {action_norm:.3f}")
            
        return action
    
    def extract_frames(self, video_path):
        """从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            frames: 预处理后的视频帧列表
        """
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"Video info for {video_path.name}:")
        print(f"  - Total frames: {total_frames}")
        print(f"  - FPS: {fps}")
        print(f"  - Duration: {duration:.2f}s")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 检查帧的有效性
            if frame is None or frame.size == 0:
                print(f"Warning: Invalid frame {frame_count} in {video_path.name}")
                continue
                
            try:
                # 预处理图像
                frame = self.preprocess_image(frame)
                frames.append(frame)
                frame_count += 1
            except Exception as e:
                print(f"Error processing frame {frame_count} in {video_path.name}: {str(e)}")
                continue
            
        cap.release()
        
        if frame_count != total_frames:
            print(f"Warning: Expected {total_frames} frames but got {frame_count} in {video_path.name}")
            
        return frames
    
    def process_trajectory(self, trajectory_id):
        """处理单个轨迹的数据
        
        Args:
            trajectory_id: 轨迹ID
        """
        # 读取轨迹数据
        trajectory_path = self.experiment_dir / "trajectories" / f"trajectory_{trajectory_id}.json"
        with open(trajectory_path, 'r') as f:
            trajectory_data = json.load(f)
        
        # 读取场景视频数据
        scene_video_path = self.experiment_dir / "videos" / f"scene_view_{trajectory_id}.mp4"
        
        # 提取视频帧
        scene_frames = self.extract_frames(scene_video_path)
        
        # 获取end_targ数据
        end_targs = trajectory_data['end_targ']
        
        # 打印调试信息
        print(f"Trajectory {trajectory_id}:")
        print(f"  - Frame count: {len(scene_frames)}")
        print(f"  - Trajectory data points: {len(end_targs)}")
        
        # 取两者中较小的长度，确保我们有完整的帧-动作对
        seq_length = min(len(scene_frames), len(end_targs)) - 1  # -1 因为我们需要下一时刻的目标
        
        # 处理每一帧
        processed_pairs = 0
        for i in range(seq_length):  # 使用对齐后的长度
            # 计算动作（位置差）
            action = self.compute_action(end_targs[i], end_targs[i+1])
            
            # 创建pair目录
            pair_dir = self.pairs_dir / f"traj_{trajectory_id}_step_{i}"
            pair_dir.mkdir(exist_ok=True)
            
            # 保存图像数据（.npy格式）
            image_path = pair_dir / "image.npy"
            np.save(str(image_path), scene_frames[i])
            
            # 保存动作数据（.npy格式）
            action_path = pair_dir / "action.npy"
            np.save(str(action_path), action)
            
            # 保存元数据
            metadata = {
                'trajectory_id': trajectory_id,
                'step': i,
                'current_target': end_targs[i],
                'next_target': end_targs[i+1],
                'image_shape': scene_frames[i].shape,
                'action_shape': action.shape
            }
            
            with open(pair_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            processed_pairs += 1
            
        return processed_pairs
    
    def process_all_trajectories(self):
        """处理所有轨迹数据"""
        # 获取所有轨迹文件
        trajectory_files = list(self.experiment_dir.glob("trajectories/trajectory_*.json"))
        total_pairs = 0
        
        # 进度条显示处理进度
        pbar = tqdm(trajectory_files, desc="Processing trajectories")
        for traj_file in pbar:
            trajectory_id = int(traj_file.stem.split('_')[1])
            pairs = self.process_trajectory(trajectory_id)
            total_pairs += pairs
            pbar.set_postfix({'pairs': total_pairs})
            
        # 保存数据集信息
        dataset_info = {
            'num_trajectories': len(trajectory_files),
            'num_pairs': total_pairs,
            'data_format': {
                'image': 'RGB normalized float32 array (C,H,W)',
                'action': '3D position difference vector',
            }
        }
        
        with open(self.output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        return dataset_info

def main():
    # 设置实验目录
    experiment_dir = "/mnt/nas/yunze/reseach/genesis/genesis-scene/datasets/experiment_20251016_141822"
    output_dir = "processed_dataset"
    
    # 创建处理器并处理数据
    processor = DatasetProcessor(experiment_dir, output_dir)
    dataset_info = processor.process_all_trajectories()
    
    print("\nDataset processing complete!")
    print(f"Total trajectories: {dataset_info['num_trajectories']}")
    print(f"Total pairs: {dataset_info['num_pairs']}")
    print("\nData format:")
    print("- Image: RGB normalized float32 array (C,H,W)")
    print("- Action: 3D position difference vector")

if __name__ == "__main__":
    main()
