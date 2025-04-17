# evaluate_rl.py
import sys
import os
import time
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from a1_rl_env import A1GymEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1) DummyVecEnv
eval_env = DummyVecEnv([lambda: A1GymEnv(robot='roomba', sim_step=0.01, cell_size=1.2, gui=True)])

model = PPO.load("ppo_model_10k")

obs = eval_env.reset()  # shape (1, obs_dim)
done = [False]
x_vals, y_vals = [], []
total_reward = 0

while not done[0]:
    action, _ = model.predict(obs, deterministic=True)  # shape (1, act_dim)
    obs, rewards, done, trunc, info = eval_env.step(action)   # obs shape (1, obs_dim)
    total_reward += rewards[0]
    x_vals.append(obs[0][0])
    y_vals.append(obs[0][1])

    time.sleep(0.01)

print(f"Total episode reward: {total_reward:.2f}")

plt.figure()
plt.plot(x_vals, y_vals, marker='o', label='RL Agent Path')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('RL Agent Trajectory')
plt.legend()
plt.grid(True)
plt.show()

eval_env.close()
