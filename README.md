# Genesis Push-T: Robot Manipulation Trajectory Learning

##TODO:实现一个评估脚本，能够评估训练出的模型的性能。目前4.中的evaluate_rollout.py还不可用，需要额外effort或者重写评估框架。

本项目实现了基于dp的机器人操作轨迹学习系统。包含轨迹生成、数据处理、模型训练和评估等完整流程。

## 项目结构

```
.
├── models/
│   └── diffusion.py      # 扩散策略模型定义
├── dataset.py            # 数据集类定义
├── trajectory_generator.py # 轨迹生成脚本
├── process_dataset.py    # 数据处理脚本
├── train.py             # 训练脚本
└── evaluate_rollout.py  # 评估脚本
```

## 1. 生成轨迹数据

使用`trajectory_generator.py`生成原始轨迹数据：

```bash
python trajectory_generator.py
```

生成的数据将保存在`data/rigid/`目录下，包含：
- `episode{seed}.json`: 轨迹数据（关节角度、速度等）
- `hand_record{seed}.mp4`: 手部视角视频
- `scene_record{seed}.mp4`: 场景视角视频

配置参数：
- 轨迹数量：修改`main()`中的`range(100)`
- 随机种子范围：可以设置不同的seed值
- 机器人参数：可在`TaskEnv`类中调整

## 2. 处理数据集

使用`process_dataset.py`将原始轨迹数据处理为训练格式：

```bash
python process_dataset.py
```

处理后的数据将保存在`processed_dataset/`目录下：
```
processed_dataset/
├── pairs/
│   ├── traj_0_step_0/
│   │   ├── image.npy    # 场景图像 (C,H,W)格式
│   │   ├── action.npy   # 动作数据（位置差）
│   │   └── metadata.json # 相关元数据
│   ├── traj_0_step_1/
│   └── ...
```

数据格式：
- 图像：RGB格式，归一化到[0,1]，(C,H,W)排列
- 动作：3D位置差向量
- 元数据：包含轨迹ID、步骤等信息

## 3. 训练模型

使用`train.py`训练扩散策略模型：

```bash
python train.py \
    --data_dir processed_dataset \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --device cuda \
    --save_dir checkpoints \
    --use_wandb  # 可选，启用wandb日志
```

主要参数：
- `batch_size`: 批次大小
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `save_dir`: 模型保存目录
- `log_interval`: 日志记录间隔
- `save_interval`: 模型保存间隔

训练过程将保存：
- 模型检查点：`checkpoints/model_epoch_{epoch}.pt`
- 训练日志：wandb（如果启用）或本地日志

## 4. 评估模型

使用`evaluate_rollout.py`评估训练好的模型：

```bash
python evaluate_rollout.py \
    --model_path checkpoints/model_epoch_99.pt \
    --output_dir rollout_results \
    --num_rollouts 10 \
    --num_steps 2000
```

评估结果将保存在`rollout_results/`目录下：
```
rollout_results/
├── rollout_0/
│   ├── positions.npy      # 位置序列
│   ├── actions.npy       # 动作序列
│   ├── start_position.npy # 起始位置
│   ├── trajectory.mp4    # 轨迹视频
│   └── trajectory_3d.png # 3D轨迹可视化
├── rollout_1/
└── ...
```

主要参数：
- `model_path`: 要评估的模型检查点路径
- `num_rollouts`: 要生成的轨迹数量
- `num_steps`: 每个轨迹的步数
- `output_dir`: 结果保存目录

## 环境要求

```
torch
numpy
opencv-python
wandb (可选，用于实验跟踪)
matplotlib
tqdm
```
