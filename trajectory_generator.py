import os
import json
import logging
import numpy as np
import genesis as gs
from datetime import datetime
from config import Config
from main import TaskEnv

class TrajectoryGenerator:
    def __init__(self, config):
        self.config = config
        self.exp_dir = config.create_experiment_dir()
        self.setup_logging()
        
        # Initialize Genesis in headless mode
        gs.init(backend=gs.gpu)
        
        # Create environment
        self.env = TaskEnv(
            debug_mode=config.debug_mode,
            material=config.material
        )
        
        self.logger.info(f"Initialized TrajectoryGenerator with experiment dir: {self.exp_dir}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.exp_dir, "logs", "generation.log")
        
        # Create logger
        self.logger = logging.getLogger("TrajectoryGenerator")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def save_metadata(self):
        """Save experiment metadata"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "num_trajectories": self.config.num_trajectories,
            "material": self.config.material,
            "robot_config": self.config.robot_config,
            "camera_config": self.config.camera_config
        }
        
        metadata_file = os.path.join(self.exp_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
        
        self.logger.info(f"Saved metadata to {metadata_file}")
    
    def generate_trajectories(self):
        """Generate multiple trajectories"""
        self.logger.info(f"Starting trajectory generation for {self.config.num_trajectories} trajectories")
        
        # Save experiment metadata
        self.save_metadata()
        
        successful_trajectories = 0
        failed_trajectories = 0
        
        for seed in range(self.config.num_trajectories):
            try:
                self.logger.info(f"Generating trajectory {seed+1}/{self.config.num_trajectories}")
                
                # Generate trajectory
                self.env.run(seed=seed)
                
                # Move generated files to experiment directory
                self._organize_trajectory_files(seed)
                
                successful_trajectories += 1
                self.logger.info(f"Successfully generated trajectory {seed}")
                
            except Exception as e:
                failed_trajectories += 1
                self.logger.error(f"Failed to generate trajectory {seed}: {str(e)}")
                continue
        
        # Log final statistics
        self.logger.info(f"""
        Trajectory Generation Complete:
        - Successful trajectories: {successful_trajectories}
        - Failed trajectories: {failed_trajectories}
        - Success rate: {(successful_trajectories/self.config.num_trajectories)*100:.2f}%
        """)
    
    def _organize_trajectory_files(self, seed):
        """Organize generated files into the experiment directory"""
        # Source file paths
        source_dir = os.path.join("data", self.config.material)
        trajectory_file = os.path.join(source_dir, f"episode{seed}.json")
        hand_video = os.path.join(source_dir, f"hand_record{seed}.mp4")
        scene_video = os.path.join(source_dir, f"scene_record{seed}.mp4")
        
        # Destination paths
        dest_traj_dir = os.path.join(self.exp_dir, "trajectories")
        dest_video_dir = os.path.join(self.exp_dir, "videos")
        
        # Move files
        if os.path.exists(trajectory_file):
            os.rename(trajectory_file, 
                     os.path.join(dest_traj_dir, f"trajectory_{seed}.json"))
        
        if os.path.exists(hand_video):
            os.rename(hand_video, 
                     os.path.join(dest_video_dir, f"hand_view_{seed}.mp4"))
        
        if os.path.exists(scene_video):
            os.rename(scene_video, 
                     os.path.join(dest_video_dir, f"scene_view_{seed}.mp4"))

def main():
    # Initialize configuration
    config = Config()
    
    # Create and run trajectory generator
    generator = TrajectoryGenerator(config)
    generator.generate_trajectories()

if __name__ == "__main__":
    main()
