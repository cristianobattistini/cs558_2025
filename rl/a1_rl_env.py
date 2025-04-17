import gymnasium as gym
from gymnasium import spaces

import numpy as np

class A1GymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, robot='a1', sim_step=0.01, cell_size=1.2, gui=False):
        super(A1GymEnv, self).__init__()

        # Import simulation only when the environment is initialized (avoids cyclic imports)
        from utils.environment import A1Simulation

        # Create simulation instance
        self.sim = A1Simulation(gui=gui, time_step=sim_step, robot=robot)
        self.cell_size = cell_size

        # Define action space: 2 continuous values (left and right wheel velocities)
        self.max_wheel_vel = 30.0
        self.action_space = spaces.Box(low=-self.max_wheel_vel, high=self.max_wheel_vel, shape=(2,), dtype=np.float32)

        # Define observation space:
        self.lidar_dim = 18
        # [x, y, θ, ẋ, ẏ, goal_x, goal_y, LIDAR_measurement*lidar_dim]
        # The first seven values are unbounded, while the LIDAR measurements are bounded by lidar_range but are normalized for better training
        obs_low = np.array([-np.inf] * 7 + [0] * self.lidar_dim)  
        obs_high = np.array([np.inf] * 7 + [1] * self.lidar_dim)  
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Threshold for goal-reaching
        self.goal_threshold = 0.1
        self.current_step = 0



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the simulation and robot position
        self.sim.reset_robot()
        start_pos, goal_pos, obstacles = self.sim.create_maze()
        self.obstacle_ids = obstacles

        # Teleport robot to starting position
        self.sim.teleport_robot([start_pos[0], start_pos[1], 0.3])

        # Get current state and construct the first observation
        state = self.sim.get_base_state()
        lidar_scan = self.sim.get_lidar_scan(num_rays=self.lidar_dim) 
        x, y, theta = state[0], state[1], state[2]
        observation_without_lidar = np.array([x, y, theta, 0.0, 0.0, goal_pos[0], goal_pos[1]], dtype=np.float32)
        observation = np.concatenate([observation_without_lidar, lidar_scan])

        # Store goal and reset counters
        self.goal = np.array([goal_pos[0], goal_pos[1]], dtype=np.float32)
        self.last_position = np.array([x, y])
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.last_position - self.goal)

        return observation, {}



    def step(self, action):
        # 1. Scale the action from [-1, 1] to [-max_wheel_vel, max_wheel_vel]
        scaled_action = action * self.max_wheel_vel
        self.sim.set_wheel_velocities(scaled_action[0], scaled_action[1])
        self.sim.step_simulation(steps=10)

        # 2. Get current base state
        state = self.sim.get_base_state()
        lidar_scan = self.sim.get_lidar_scan(num_rays=self.lidar_dim) 
        x, y, theta = state[0], state[1], state[2]

        # 3. Compute estimated linear velocity (x_dot, y_dot)
        pos_now = np.array([x, y])
        vel = (pos_now - self.last_position) / (self.sim.time_step * 10)
        x_dot, y_dot = vel[0], vel[1]
        self.last_position = pos_now

        # 4. Construct the observation
        observation_without_lidar = np.array([x, y, theta, x_dot, y_dot, self.goal[0], self.goal[1]], dtype=np.float32)
        observation = np.concatenate([observation_without_lidar, lidar_scan])
        

        # 5. Compute reward using shaping
        distance = np.linalg.norm(pos_now - self.goal)
        delta_distance = self.prev_distance - distance
        speed = np.sqrt(x_dot**2 + y_dot**2)

        # Check for collision
        collision = self.sim.check_collision_roomba(self.obstacle_ids)

        reward = 0.0
        reward += -distance                   # Penalize distance to goal
        reward += 0.2 * delta_distance        # Encourage getting closer
        reward += 0.05 * speed                # Encourage speed
        reward -= 0.01                        # Time penalty
        if collision:
            reward -= 10                     # Collision penalty
        if distance < self.goal_threshold:
            reward += 100                   # Goal reward

        self.prev_distance = distance

        # 6. Termination conditions
        terminated = distance < self.goal_threshold
        truncated = self.current_step >= 300  # Max episode length. Since step is 0.01 and each
                                              # trainig step is 10 sim steps, this is 30 seconds.

        self.current_step += 1
        info = {}

        return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.disconnect()
        return
