import os
from datetime import datetime

class Config:
    def __init__(self):
        # Base directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "datasets")
        
        # Experiment settings
        self.num_trajectories = 100
        self.material = "rigid"  # Options: "rigid", "mpm", "fem", "pbd"
        self.debug_mode = False
        
        # Robot settings
        self.robot_config = {
            "kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
            "kv": [450, 450, 350, 350, 200, 200, 200, 10, 10],
            "force_range": {
                "min": [-87, -87, -87, -87, -12, -12, -12, -100, -100],
                "max": [87, 87, 87, 87, 12, 12, 12, 100, 100]
            }
        }
        
        # Camera settings
        self.camera_config = {
            "hand_cam": {
                "fov": 70,
                "res": (320, 320)
            },
            "scene_cam": {
                "fov": 40,
                "res": (320, 320),
                "pos": (2, 0, 1.5),
                "lookat": (0.0, 0.0, 0.0)
            }
        }
    
    def create_experiment_dir(self):
        """Create a new experiment directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"experiment_{timestamp}"
        exp_dir = os.path.join(self.data_dir, exp_name)
        
        # Create directory structure
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "trajectories"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
        
        return exp_dir
