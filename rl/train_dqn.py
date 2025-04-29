import sys
import os
import numpy as np
import itertools
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from a1_rl_env import A1GymEnv

# Allow OpenMP duplicates
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==== Discretize continuous actions ====
wheel_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
actions = list(itertools.product(wheel_levels, wheel_levels))  # 25 discrete actions

class DiscreteToContinuousActionEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.actions = actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        cont_action = np.array(self.actions[action], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(cont_action)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

# ==== Parameters ====
use_obstacles = False
total_timesteps = 3_000_000
model_name = f"dqn_model_{int(total_timesteps/1000)}k_no_obstacles"
log_dir = "./dqn_logs"
os.makedirs(log_dir, exist_ok=True)

# ==== Create env and wrap ====
base_env = A1GymEnv(use_obstacles=use_obstacles,
                    robot='roomba',
                    sim_step=1./240.,
                    cell_size=1.2,
                    gui=False)

env = DiscreteToContinuousActionEnv(base_env)
env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))

# ==== DQN model setup ====
# model = DQN(
#     policy="MlpPolicy",
#     env=env,
#     learning_starts=5_000,
#     buffer_size=100_000,
#     learning_rate=3e-4,
#     batch_size=128,
#     gamma=0.99,
#     target_update_interval=500,
#     train_freq=4,
#     gradient_steps=4,
#     verbose=1,
#     tensorboard_log=log_dir,
#     policy_kwargs=dict(net_arch=[128, 128, 64]),
#     exploration_initial_eps=1.0,
#     exploration_final_eps=0.05,
#     exploration_fraction=0.2,
#     device="cuda"
# )

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,             # lower LR for stability
    buffer_size=200_000,            # bigger buffer helps large envs
    learning_starts=10_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,               # slow update → more stable
    target_update_interval=150,     # update target network more often
    exploration_fraction=0.3,       # explore for longer
    exploration_final_eps=0.05,
    exploration_initial_eps=1.0,
    policy_kwargs=dict(
        net_arch=[256, 256]     # deeper network for harder task
    ),
    verbose=1,
    device="cuda"
)


# ==== Train ====
model.learn(total_timesteps=total_timesteps)

# ==== Save ====
model.save(model_name)
env.close()
