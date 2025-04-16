import gymnasium as gym
from gymnasium import spaces

import numpy as np

class A1GymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, robot='a1', sim_step=0.01, cell_size=1.2, gui=False):
        super(A1GymEnv, self).__init__()
        from utils.environment import A1Simulation  # import locale per evitare cicli
        self.sim = A1Simulation(gui=gui, time_step=sim_step, robot=robot)
        self.cell_size = cell_size

        max_wheel_vel = 10.0
        self.action_space = spaces.Box(low=-max_wheel_vel, high=max_wheel_vel, shape=(2,), dtype=np.float32)

        obs_low = np.array([-np.inf, -np.inf, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf])
        obs_high = np.array([np.inf, np.inf, np.pi, np.inf, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.goal_threshold = 0.1
        self.current_step = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim.reset_robot()
        start_pos, goal_pos, _ = self.sim.create_maze()
        self.sim.teleport_robot([start_pos[0], start_pos[1], 0.3])

        state = self.sim.get_base_state()
        x, y, theta = state[0], state[1], state[2]
        observation = np.array([x, y, theta, 0.0, 0.0, goal_pos[0], goal_pos[1]], dtype=np.float32)
        self.goal = np.array([goal_pos[0], goal_pos[1]], dtype=np.float32)
        self.last_position = np.array([x, y])
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.last_position - self.goal)

        return observation, {}


    def step(self, action):
        # 1) Scaled action to avoid excessively small outputs
        scaled_action = action * 3.0
        self.sim.set_wheel_velocities(scaled_action[0], scaled_action[1])
        self.sim.step_simulation(steps=10)

        # 2) Current state from the simulator
        state = self.sim.get_base_state()
        x, y, theta = state[0], state[1], state[2]

        # 3) Compute linear velocity in the plane (x_dot, y_dot)
        pos_now = np.array([x, y])
        vel = (pos_now - self.last_position) / (self.sim.time_step * 10)
        x_dot, y_dot = vel[0], vel[1]
        self.last_position = pos_now

        # 4) Build observation to return
        observation = np.array([x, y, theta, x_dot, y_dot, self.goal[0], self.goal[1]], dtype=np.float32)

        # 5) Compute reward with shaping
        distance = np.linalg.norm(pos_now - self.goal)
        delta_distance = self.prev_distance - distance
        speed = np.sqrt(x_dot**2 + y_dot**2)

        # Example of collision method (to be implemented):
        collision = self.check_collision() if hasattr(self, 'check_collision') else False

        reward = 0.0
        reward = -distance                     # (a) Distance penalty  
        reward += 0.2 * delta_distance         # (b) Bonus for getting closer  
        reward += 0.05 * speed                 # (c) Bonus for movement  
        reward -= 0.01                         # (d) Time penalty per step  
        reward -= 10 if collision else 0       # (e) Collision penalty  
        reward += 100 if distance < self.goal_threshold else 0  # (f) Goal reached bonus


        # Save the distance for the next iteration
        self.prev_distance = distance

        # 6) Check termination
        terminated = distance < self.goal_threshold
        truncated = self.current_step >= 1000
        if terminated:
            # If needed, you could reset something here
            pass

        self.current_step += 1
        info = {}

        return observation, reward, terminated, truncated, info



    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.disconnect()
