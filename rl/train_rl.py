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
model.learn(total_timesteps=500000)
model.save("ppo_model_500k_1604250900")


# ===== Evaluating the Trained Agent =====
model = PPO.load("ppo_model_500k_1604250900")
obs = env.reset()
done = False
x_vals = []
y_vals = []

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    x_vals.append(obs[0][0])  # DummyVecEnv adds batch dimension
    y_vals.append(obs[0][1])

plt.figure()
plt.plot(x_vals, y_vals, marker='o', label='RL Agent Path')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Trajectory of RL Agent')
plt.legend()
plt.grid(True)
plt.show()

env.close()
