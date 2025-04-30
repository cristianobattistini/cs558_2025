import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer
from a1_rl_env import A1GymEnv
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

def pretrain_bc(
    model,
    observations: np.ndarray,
    actions:      np.ndarray,
    n_epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-4,
):
    """
    Simple Behavior Cloning on a SB3 policy:
     - observations: [N x obs_dim]
     - actions:      [N x act_dim]
    """
    device = model.device
    # 1) build a PyTorch dataloader
    obs_t = torch.tensor(observations, dtype=torch.float32, device=device)
    act_t = torch.tensor(actions,      dtype=torch.float32, device=device)
    ds    = TensorDataset(obs_t, act_t)
    loader= DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 2) grab the same optimizer PPO uses (it will update both actor & critic)
    optimizer = model.policy.optimizer
    loss_fn   = torch.nn.MSELoss()   # for Box actions; for Discrete you could use CrossEntropyLoss

    # 3) supervised loop
    for epoch in range(1, n_epochs+1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            dist = model.policy.get_distribution(xb)  # your policy’s action distribution
                        # continuous: match the mean; discrete: you could do dist.log_prob(yb).neg().mean()
            # for a DiagGaussianDistribution, get the torch.distributions.Normal under the hood:
            pred = dist.distribution.mean
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[BC] epoch {epoch:3d}/{n_epochs}   loss = {total_loss/len(loader):.4f}")



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

log_dir = "./ppo_logs"
os.makedirs(log_dir, exist_ok=True)

## SET TRAINING PARAMETERS, AUTO MODEL NAME SAVER
use_obstacles = True
total_timesteps = 5000000
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
        device="cuda",
        n_steps=4096,              # Smaller, more frequent rollouts
        batch_size=1024,           # A quarter of n_steps
        n_epochs=10,               # Keep 10 for now
        learning_rate=1e-3,        # More aggressive updates
        gamma=0.98,                # Slightly less long-term oriented
        gae_lambda=0.95,           # Keep for now
        clip_range=0.2,            # Standard
        ent_coef=0.01,             # Encourage exploration
        policy_kwargs=dict(net_arch=[128, 128])  # Slightly wider network
        #policy_kwargs=dict(net_arch=[128, 128])  # 2 hidden layers, 128 neurons each
    )

    # ## PRETRAININ USING BC
    # expert_data_path = "expert_data.npy"  # Path to saved expert data
    # # Load expert data
    # expert_data = np.load("expert_data.npy", allow_pickle=True)
    # observations = np.array([d[0] for d in expert_data])
    # actions = np.array([d[1] for d in expert_data])

    # # Pretrain the policy
    # pretrain(
    #     model,
    #     observations,
    #     actions,
    #     n_epochs=100,  # Adjust based on performance
    #     batch_size=64,
    #     learning_rate=1e-4,
    #     verbose=1
    # )


    # model.learn(total_timesteps=total_timesteps)


    # … right before your call to pretrain(), replace everything from expert_data_path onward with:
    import glob
    # ——— load all expert .npz files ———
    expert_dir = "expert_data"   # or "duqi_logs/expert_data" if that's your folder
    pattern    = os.path.join(expert_dir, "expert_dataset_*eps.npz")
    files      = sorted(glob.glob(pattern))

    obs_list = []
    act_list = []
    for fpath in files:
        with np.load(fpath) as data:
            # inspect your keys once to confirm their names:
            # print(fpath, "keys =", data.files)

            # most demos save arrays named e.g. 'observations' and 'actions'
            # change below to match your key names if different
            obs = data["observations"]  
            acts = data["actions"]

            obs_list.append(obs)
            act_list.append(acts)

    # concatenate into one big array
    observations = np.concatenate(obs_list, axis=0)
    actions      = np.concatenate(act_list, axis=0)

    print(f"Loaded {len(files)} expert files → {observations.shape[0]} state-action pairs")

    # ——— Behavioral Cloning pre-training ———
    print("Pretraining policy via Behavioral Cloning…")
# ——— Behavioral Cloning pre-training ———
    print("Pretraining policy via Behavioral Cloning…")
    pretrain_bc(
        model,
        observations,
        actions,
        n_epochs=10,
        batch_size=64,
        lr=1e-4,
    )




    # then continue with RL fine-tuning
    model.learn(total_timesteps=total_timesteps)



model.save(model_name)
env.close()
