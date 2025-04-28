import gymnasium as gym
from gymnasium import spaces

import numpy as np

class A1GymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, use_obstacles, current_steps=0, robot='a1', sim_step=1./240., cell_size=1.2, gui=False):
        super(A1GymEnv, self).__init__()

        # Import simulation only when the environment is initialized (avoids cyclic imports)
        from utils.environment import A1Simulation

        # Create simulation instance
        self.sim = A1Simulation(gui=gui, time_step=sim_step, robot=robot)
        self.cell_size = cell_size
        # Define action space: 2 continuous values (left and right wheel velocities)
        self.max_wheel_vel = 45.0
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Define observation space:
        self.lidar_dim = 18
        # [x, y, θ, goal_x, goal_y, LIDAR_measurement*lidar_dim]
        # The first seven values are unbounded, while the LIDAR measurements are bounded by lidar_range but are normalized for better training
        obs_low = np.array([-np.inf] * 5 + [0] * self.lidar_dim)  
        obs_high = np.array([np.inf] * 5 + [1] * self.lidar_dim)  
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Threshold for goal-reaching
        self.goal_threshold = 1
        self.current_step = current_steps

        # Initialize variables for tracking the robot's state
        self.last_position = np.array([0.0, 0.0])
        self.prev_distance = np.inf
        self.goal = np.array([0.0, 0.0], dtype=np.float32)
        self.obstacle_ids = []
        self.use_obstacles = use_obstacles



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the simulation and robot position
        start_pos, goal_pos, obstacles = self.sim.create_maze(self.use_obstacles)
        self.obstacle_ids = obstacles
        self.sim.reset_robot([start_pos[0], start_pos[1], 0.4])

        # Teleport robot to starting position
        self.sim.teleport_robot([start_pos[0], start_pos[1], 0.3])

        # Get current state and construct the first observation
        state = self.sim.get_base_state()
        lidar_scan = self.sim.get_lidar_scan(num_rays=self.lidar_dim) 
        x, y, theta = state[0], state[1], state[2]
        observation_without_lidar = np.array([x, y, theta, goal_pos[0], goal_pos[1]], dtype=np.float32)
        observation = np.concatenate([observation_without_lidar, lidar_scan])

        # Store goal and reset counters
        self.goal = np.array([goal_pos[0], goal_pos[1]], dtype=np.float32)
        self.last_position = np.array([x, y])
        self.current_step = 0
        self.prev_distance = np.sqrt((self.goal[0]-x)**2 + (self.goal[1]-y)**2) 

        return observation, {}



    def step(self, action):
        # 1. Scale the action from [-1, 1] to [-max_wheel_vel, max_wheel_vel]
        scaled_action = action * self.max_wheel_vel
        self.sim.set_wheel_velocities(scaled_action[0], scaled_action[1])
        self.sim.step_simulation()

        # 2. Get current base state
        state = self.sim.get_base_state()
        lidar_scan = self.sim.get_lidar_scan(num_rays=self.lidar_dim) 
        x, y, theta = state[0], state[1], state[2]

        # 3. Compute estimated linear velocity (x_dot, y_dot)
        pos_now = np.array([x, y])
        vel = (pos_now - self.last_position) / self.sim.time_step
        x_dot, y_dot = vel[0], vel[1]
        self.last_position = pos_now

        # 4. Construct the observation
        observation_without_lidar = np.array([x, y, theta, self.goal[0], self.goal[1]], dtype=np.float32)
        observation = np.concatenate([observation_without_lidar, lidar_scan])
        

        # 5. Compute reward using shaping
        distance_x = self.goal[0] - pos_now[0]
        distance_y = self.goal[1] - pos_now[1]
        distance = np.sqrt(distance_x**2 + distance_y**2)
        delta_distance = self.prev_distance - distance
        speed = np.sqrt(x_dot**2 + y_dot**2)

        
        # Compute reward
        reward = 0.0
        # print("Delta distance: ", 30*delta_distance)
        # print("Speed: ", 0.01*speed)
        # print("Distance: ", distance)
        # print("Time penalty: ", -0.01)
        # print("\n")
        reward += 30 * delta_distance   # Incentive for progress towards goal
        reward += 0.06 * speed          # Reward high speeds
        #reward -= 0.01 * distance       # Penalty to discourage wandering
        #reward -= 0.01                  # Time penalty to encourage faster episodes
        #if distance < 3:
            #reward += 5 * (1 - (distance / 3) ** 2)  # Peaks at 10 when distance=0
                
            #print("Distance reward: ", 10 * (1 - (distance / 3) ** 2))

        # Check for collision
        if self.use_obstacles:
            collision = self.sim.check_collision_roomba(self.obstacle_ids)
            min_lidar_distance = np.min(lidar_scan)
            if min_lidar_distance < 0.2:  # Too close to obstacle
                reward -= (0.02 - min_lidar_distance/10)
            if collision:
                reward -= 10                     # Collision penalty
        if distance < self.goal_threshold:
            reward += 3000                   # Goal reward

        self.prev_distance = distance

        # 6. Termination conditions
        terminated = distance < self.goal_threshold
        truncated = self.current_step >= 6000  # Max episode length. Since step is 1/240 and each
                                              # trainig step is 1 sim steps, this is 50 seconds.

        self.current_step += 1
        info = {}

        return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.disconnect()
        return
