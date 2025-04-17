import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from a1_rl_env import A1GymEnv
import matplotlib.pyplot as plt
from a1_rl_env import A1GymEnv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

log_dir = "./ppo_logs"
os.makedirs(log_dir, exist_ok=True)

# Wrap environment with Monitor and DummyVecEnv
env = DummyVecEnv([
    lambda: Monitor(A1GymEnv(robot='roomba', sim_step=0.01, cell_size=1.2), filename=os.path.join(log_dir, "monitor.csv"))
])

# Initialize and train the PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
"""
If not specified, two layer of 64 neurons are created, with increased lidar dimension a higher number of neurons could be usefull 
    model = PPO(
    "MlpPolicy", 
    env,
    policy_kwargs=dict(net_arch=[128, 128]),  # 2 hidden layers, 128 neurons each
    verbose=1
)
"""
model.learn(total_timesteps=10000)
model.save("ppo_model_10k")

env.close()
