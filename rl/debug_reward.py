import sys
import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from a1_rl_env import A1GymEnv
import matplotlib.pyplot as plt

# Setup paths and environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create environment (no obstacles for now, Roomba robot, GUI enabled)
env = DummyVecEnv([lambda: A1GymEnv(use_obstacles=False, robot='roomba', sim_step=1./240., cell_size=1.2, gui=True)])

obs = env.reset()
done = [False]
total_reward = 0
step_count = 0

# For logging reward components
reward_log = {
    'progress': [],
    'speed': [],
    'collision_penalty': [],
    'proximity_penalty': [],
    'goal_bonus': [],
    'total_reward': [],
    'distance_to_goal': [],
    'min_lidar': []
}

# ------------------------
# Debug loop: fixed action (go straight)
# ------------------------
while not done[0]:
    action = np.array([[1.0, 1.0]])  # Forward motion (batch dimension)

    # Step the environment
    obs, reward, done, info = env.step(action)

    # Extract useful debug info
    distance_to_goal = obs[0][3]  # Assuming 4th element is goal x position (adjust if needed)
    min_lidar = np.min(obs[0][5:])  # Assuming LiDAR starts at index 5 (adjust if needed)

    # ---------------------
    # Compute reward components (same as env logic)
    pos_x, pos_y = obs[0][0], obs[0][1]
    vel_x, vel_y = obs[0][2], obs[0][3]
    speed = np.sqrt(vel_x**2 + vel_y**2)
    distance = np.sqrt((obs[0][0] - obs[0][3])**2 + (obs[0][1] - obs[0][4])**2)  # dist to goal

    # Components from environment (manual calculation)
    delta_distance = 0.0 if step_count == 0 else reward_log['distance_to_goal'][-1] - distance
    progress_reward = 30 * delta_distance
    speed_reward = 0.009 * speed
    collision_penalty = 0.0  # No collision in this test (if obstacles enabled, you'd compute)
    proximity_penalty = 0.0 if min_lidar > 0.2 else (-(0.02 - min_lidar/10))
    goal_bonus = 3000.0 if distance < env.envs[0].goal_threshold else 0.0

    total_step_reward = progress_reward + speed_reward + collision_penalty + proximity_penalty + goal_bonus

    # ---------------------
    # Logging
    reward_log['progress'].append(progress_reward)
    reward_log['speed'].append(speed_reward)
    reward_log['collision_penalty'].append(collision_penalty)
    reward_log['proximity_penalty'].append(proximity_penalty)
    reward_log['goal_bonus'].append(goal_bonus)
    reward_log['total_reward'].append(total_step_reward)
    reward_log['distance_to_goal'].append(distance)
    reward_log['min_lidar'].append(min_lidar)

    # Terminal print
    print(f"Step {step_count:04d} | Action: {action} | Total Reward: {total_step_reward:.4f} | Distance: {distance:.2f} | Min LIDAR: {min_lidar:.2f}")

    total_reward += reward[0]
    step_count += 1

# ------------------------
# End of episode summary
print(f"\nTotal episode reward: {total_reward:.2f}")
env.close()

# ------------------------
# Plot reward components
plt.figure(figsize=(12, 6))
plt.plot(reward_log['progress'], label='Progress Reward')
plt.plot(reward_log['speed'], label='Speed Reward')
plt.plot(reward_log['proximity_penalty'], label='Proximity Penalty')
plt.plot(reward_log['goal_bonus'], label='Goal Bonus')
plt.plot(reward_log['total_reward'], label='Total Reward')
plt.xlabel('Steps')
plt.ylabel('Reward Value')
plt.title('Reward Breakdown Over Time (Fixed Action)')
plt.legend()
plt.grid(True)
plt.show()
