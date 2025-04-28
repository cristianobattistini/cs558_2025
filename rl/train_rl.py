import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.pretrain import pretrain
from a1_rl_env import A1GymEnv
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

log_dir = "./ppo_logs"
os.makedirs(log_dir, exist_ok=True)

## SET TRAINING PARAMETERS, AUTO MODEL NAME SAVER
use_obstacles = False
total_timesteps = 500000
model_name = f"ppo_model_{int(total_timesteps/1000)}k_no_obstacles" if not use_obstacles else f"ppo_model_{int(total_timesteps/1000)}k_with_obstacles"

load_existing_model = False  # Set to True to load an existing model
pretrained_model_path = "ppo_model_1000k_no_obstacles"  # Path to the pretrained model

# Wrap environment with Monitor and DummyVecEnv
env = DummyVecEnv([
    lambda: Monitor(A1GymEnv(use_obstacles=use_obstacles, robot='roomba', sim_step=1./240., cell_size=1.2), filename=os.path.join(log_dir, "monitor.csv"))
])
# Load or create a new model
if load_existing_model:
    if not os.path.exists(pretrained_model_path + ".zip"):
        raise FileNotFoundError(f"Pretrained model {pretrained_model_path} not found")
    print("\nLoading pretrained model ", pretrained_model_path, "\n")
    model = PPO.load(pretrained_model_path, env=env, device="cuda")
    model.set_env(env)  # Ensure the environment is properly set
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True) 
else:
    print("\nCreating a new model\n")
    # Initialize and train the PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        #policy_kwargs=dict(net_arch=[128, 128])  # 2 hidden layers, 128 neurons each
        n_steps=6000,  # Number of steps to collect before updating the model
        n_epochs=10,  # Number of epochs to train on each batch

    )

    ## PRETRAININ USING BC
    expert_data_path = "expert_data.npy"  # Path to saved expert data
    # Load expert data
    expert_data = np.load("expert_data.npy", allow_pickle=True)
    observations = np.array([d[0] for d in expert_data])
    actions = np.array([d[1] for d in expert_data])

    # Pretrain the policy
    pretrain(
        model,
        observations,
        actions,
        n_epochs=100,  # Adjust based on performance
        batch_size=64,
        learning_rate=1e-4,
        verbose=1
    )

    model.learn(total_timesteps=total_timesteps)


model.save(model_name)
env.close()
