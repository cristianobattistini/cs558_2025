# What goes in each folder?
## assets/

URDF Files: Put a1.urdf here (the robot model). If you have a pre-made environment URDF (e.g., a more complex world with ramps or random obstacles), store it here too (like environment1.urdf).

Any other meshes, textures, or resource files needed at runtime.

## classical_planning/

path_planner.py: Implement your RRT/PRM or any classical path-planning algorithm here.

controller.py: Implement your PID (or other) controller logic that steers the robot along the planned path.

run_classical.py: A script that ties the planner and controller together. It should:

Load the PyBullet simulation (plane + environment URDF + A1).

Run the planner to get a path.

Use the controller to follow that path.

Print or log results.

## rl/

a1_rl_env.py: A custom Gym environment (if you go that route) or a script that sets up the RL environment in PyBullet for the A1 robot (observation space, action space, reward function, etc.).

train_rl.py: Script for training the RL agent (e.g., PPO or SAC via Stable Baselines3).

evaluate_rl.py: Code for evaluating the trained RL model on test scenarios.
(For your immediate focus on classical planning, you can leave RL scripts mostly empty and return to them later.)

## utils/ (Optional but recommended)

common.py: Helper functions used by both classical planning and RL (e.g., functions for getting the robot’s base position, converting quaternions to Euler angles, collision checks, etc.).

visualization.py: Functions to draw debug lines in PyBullet, display plots, or handle logging data about performance metrics.

## tests/ (Optional)

test_all.py: Quick test script to ensure your modules and assets are in the correct paths, environment loads, etc.

## requirements.txt

List of Python dependencies (e.g., pybullet, stable-baselines3, ompl-py, numpy, etc.).

## README.md

A short description of your project, instructions on how to run each module (classical vs. RL), environment details, etc.