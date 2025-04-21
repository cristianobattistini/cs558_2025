import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from a1_rl_env import A1GymEnv
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

log_dir = "./ppo_logs"
os.makedirs(log_dir, exist_ok=True)

## SET TRAINING PARAMETERS, AUTO MODEL NAME SAVER
use_obstacles = False
total_timesteps = 480000
model_name = f"ppo_model_{int(total_timesteps/1000)}k_no_obstacles" if not use_obstacles else f"ppo_model_{int(total_timesteps/1000)}k_with_obstacles"


# Wrap environment with Monitor and DummyVecEnv
env = DummyVecEnv([
    lambda: Monitor(A1GymEnv(use_obstacles=use_obstacles, robot='roomba', sim_step=1./240., cell_size=1.2), filename=os.path.join(log_dir, "monitor.csv"))
])

# Initialize and train the PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    device="cuda",
    n_steps=24000,       # Total timesteps collected before each training phase 3600
    batch_size=2400,     # Batch size for training (use a multiple of 64 ideally)
    n_epochs=10         # Number of passes over the rollout data
)
"""
If not specified, two layer of 64 neurons are created, with increased lidar dimension a higher number of neurons could be usefull 
    model = PPO(
    "MlpPolicy", 
    env,
    policy_kwargs=dict(net_arch=[128, 128]),  # 2 hidden layers, 128 neurons each
    verbose=1
)
"""
model.learn(total_timesteps=total_timesteps)
model.save(model_name)
env.close()
