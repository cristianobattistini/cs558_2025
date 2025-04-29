import sys
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from a1_rl_env import A1GymEnv

# Allow OpenMP duplicates
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==== Discretize actions (same as in training) ====
wheel_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
actions = list(itertools.product(wheel_levels, wheel_levels))  # 25 discrete actions

# ==== Environment setup ====
base_env = A1GymEnv(use_obstacles=False, robot='roomba', sim_step=1./240., cell_size=1.2, gui=True)

class DiscreteToContinuousActionEnvWrapper:
    def __init__(self, env, actions):
        self.env = env
        self.actions = actions
    
    def reset(self):
        obs, info = self.env.reset()
        return obs

    def step(self, action_idx):
        cont_action = np.array(self.actions[action_idx], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(cont_action)
        done = terminated or truncated
        return obs, reward, done, info

# Wrap the environment for discrete control
eval_env = DummyVecEnv([lambda: base_env])
env_wrapper = DiscreteToContinuousActionEnvWrapper(base_env, actions)

# ==== Load model ====
model = DQN.load("dqn_model_3000k_no_obstacles")

# ==== Evaluation loop ====
obs = eval_env.reset()  # shape (1, obs_dim)
done = False
x_vals, y_vals = [], []
total_reward = 0

while not done:
    action_idx, _ = model.predict(obs, deterministic=False)  # Predict discrete action index
    obs, reward, done, _ = env_wrapper.step(action_idx[0])  # Use wrapper to map index to wheel velocities
    obs = np.expand_dims(obs, axis=0)  # DummyVecEnv expects shape (1, obs_dim)
    
    total_reward += reward
    x_vals.append(obs[0][0])  # X position
    y_vals.append(obs[0][1])  # Y position

print(f"Total episode reward: {total_reward:.2f}")

# ==== Evaluate goal reachability ====
goal_pos = base_env.goal  # Already part of env
final_pos = np.array([obs[0][0], obs[0][1]])
distance = np.linalg.norm(final_pos - goal_pos)

print(f"\nTotal episode reward: {total_reward:.2f}")
if distance < base_env.goal_threshold:
    print(f"✅ Goal reached! Final distance: {distance:.2f} m")
else:
    print(f"❌ Goal not reached. Final distance: {distance:.2f} m")

# ==== Plot trajectory ====
plt.figure()
plt.plot(x_vals, y_vals, marker='o', label='RL Agent Path')
plt.plot(goal_pos[0], goal_pos[1], 'rx', markersize=10, label='Goal')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('RL Agent Trajectory')
plt.legend()
plt.grid(True)
plt.show()
