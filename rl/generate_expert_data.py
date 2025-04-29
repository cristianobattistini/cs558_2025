import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from tqdm import tqdm
from utils.environment import *
from classical_planning.controller import *
from classical_planning.rtt_star_path_planner import *
from classical_planning.maze_generator import *

class ExpertDataGenerator:
    def __init__(self):
        self.cell_size = 1.2
        self.rows = 10
        self.cols = 10
        self.robot_dims = (0.5, 0.4, 0.3)
        self.max_wheel_vel = 45.0
        self.lidar_dim = 18  # Must match RL environment
        
        # Create output directory
        self.output_dir = "expert_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize empty dataset
        self.dataset = {
            'observations': [],
            'actions': [],
            'episode_starts': [],
            'timeouts': []
        }
        self.episode_counter = 0

    def run_episode(self, episode_idx):
        """Run one complete navigation episode"""
        # Initialize simulation (no GUI for speed)
        sim = A1Simulation(gui=False, time_step=1./240., robot='roomba')
        
        # Generate random maze (different each episode)
        maze = MazeGenerator(rows=self.rows, cols=self.cols, cell_size=self.cell_size)
        start_pos, goal_pos, obstacles = maze.create_simplified_maze(use_obstacles=True)
        
        # Plan path
        path = run_rrt_star(
            start_conf=(start_pos[0], start_pos[1], 0),
            goal_conf=(goal_pos[0], goal_pos[1], 0),
            obstacles=obstacles,
            maze_bounds=(0, self.cols*self.cell_size, -self.rows*self.cell_size, 0),
            robot_dims=self.robot_dims,
            max_iter=1000,
            goal_threshold=0.6
        )
        
        if not path:
            sim.disconnect()
            return False  # Skip failed planning episodes
            
        path = [p[:2] for p in path]  # Remove angles for Roomba
        
        # Initialize controller
        controller = Controller(1./240., robot='roomba')
        
        # Reset robot
        sim.reset_robot([start_pos[0], start_pos[1], 0.3])
        
        # Episode data collection
        episode_data = {
            'obs': [],
            'acts': [],
            'timeouts': []
        }
        
        # Execute path
        for waypoint in path:
            desired_state = np.array([waypoint[0], waypoint[1]])
            close_to_goal = False
            
            while not close_to_goal:
                # Get state (matches RL observation space)
                state = sim.get_base_state()
                lidar_scan = sim.get_lidar_scan(num_rays=self.lidar_dim)
                x, y, theta = state[0], state[1], state[2]
                
                # Construct observation
                obs = np.concatenate([
                    [x, y, theta, goal_pos[0], goal_pos[1]],
                    lidar_scan
                ])
                
                # Get expert action
                current_state = np.array([x, y, theta, 0, 0])
                left_vel, right_vel = controller.compute_control(current_state, desired_state)
                action = np.array([
                    left_vel / self.max_wheel_vel,
                    right_vel / self.max_wheel_vel
                ])
                
                # Store transition
                episode_data['obs'].append(obs)
                episode_data['acts'].append(action)
                
                # Apply control
                sim.set_wheel_velocities(left_vel, right_vel)
                sim.step_simulation()
                
                # Check termination
                dist_to_goal = np.linalg.norm([x - goal_pos[0], y - goal_pos[1]])
                if dist_to_goal < 1.0:  # Success
                    episode_data['timeouts'].append(False)
                    close_to_goal = True
                    break
                elif len(episode_data['obs']) > 10000:  # Timeout
                    episode_data['timeouts'].append(True)
                    close_to_goal = True
                    break
                
                # Check waypoint reached
                if (x - waypoint[0])**2 + (y - waypoint[1])**2 < 0.1:
                    close_to_goal = True
        
        # Process episode data
        if len(episode_data['obs']) > 0:
            self._add_episode_to_dataset(episode_data)
            self.episode_counter += 1
            
        sim.disconnect()
        return True

    def _add_episode_to_dataset(self, episode_data):
        """Add episode data to the main dataset"""
        n_steps = len(episode_data['obs'])
        
        self.dataset['observations'].extend(episode_data['obs'])
        self.dataset['actions'].extend(episode_data['acts'])
        self.dataset['episode_starts'].extend([True] + [False]*(n_steps-1))
        self.dataset['timeouts'].extend(episode_data['timeouts'])
        
        # Save incremental backups
        if self.episode_counter % 100 == 0:
            self.save_dataset()

    def save_dataset(self):
        """Save dataset to numpy format"""
        np.savez_compressed(
            os.path.join(self.output_dir, f'expert_dataset_{self.episode_counter}eps.npz'),
            observations=np.array(self.dataset['observations'], dtype=np.float32),
            actions=np.array(self.dataset['actions'], dtype=np.float32),
            episode_starts=np.array(self.dataset['episode_starts'], dtype=bool),
            timeouts=np.array(self.dataset['timeouts'], dtype=bool)
        )
        print(f"\nSaved dataset with {self.episode_counter} episodes")

if __name__ == "__main__":
    generator = ExpertDataGenerator()
    
    # Run 1000 episodes with progress bar
    for _ in tqdm(range(1000), desc="Generating expert data"):
        success = generator.run_episode(_)
        if not success:
            tqdm.write("Episode skipped, failed during planning")
    
    # Final save
    generator.save_dataset()
    print("Expert data generation complete!")