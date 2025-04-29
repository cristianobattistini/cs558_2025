import pybullet as p
import random
from numpy import sin, cos, pi
class MazeGenerator:
    def __init__(self, rows=10, cols=10, cell_size=0.4, seed=None):
        """
        Initialize the maze generator.

        Args:
            rows (int): Number of rows in the maze grid.
            cols (int): Number of columns in the maze grid.
            cell_size (float): Physical size of each cell in PyBullet units.
            seed (int, optional): Seed for reproducible maze generation.
        """
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.wall_height = 0.3
        self.obstacle_ids = [] 

        if seed is not None:
            random.seed(seed)


    def create_simplified_maze(self, use_obstacles, random_goal=True):
        """
        Create a denser maze by placing smaller block obstacles in multiple rows along the y-direction.
        Obstacles are scattered along the x-axis and multiple y rows, making a denser corridor-like passage.
        The goal can either be placed randomly around a radius or directly in front of the robot.
        """
        self.obstacle_ids = []
        
        if use_obstacles:
            num_y_layers = 10
            y_offsets = [self.cell_size * (i - num_y_layers // 2) for i in range(num_y_layers)]
            possible_x = [self.cell_size * i for i in range(2, 8)]

            for y in y_offsets:
                num_obs = random.randint(2, 6)
                x_positions = random.sample(possible_x, num_obs)
                for x in x_positions:
                    pos = [x, y, self.wall_height / 2]
                    size = [self.cell_size / 4.5, self.cell_size / 4.5, self.wall_height / 2]
                    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
                    vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.7, 0.7, 0.7, 1])
                    body_id = p.createMultiBody(baseMass=0,
                                                baseCollisionShapeIndex=col_shape,
                                                baseVisualShapeIndex=vis_shape,
                                                basePosition=pos)
                    self.obstacle_ids.append(body_id)

        # Start position
        random_start = self.cell_size * random.randint(-3, 3)
        start_pos = [0, random_start, 0.3]

        if random_goal:
            # Goal in a random direction around the start
            radius = self.cell_size * 5.0
            theta = random.uniform(0, 2 * pi)
            goal_x = start_pos[0] + radius * cos(theta)
            goal_y = start_pos[1] + radius * sin(theta)
        else:
            # Goal in front of the robot
            random_end = self.cell_size * random.randint(-3, 3)
            goal_x = self.cell_size * 7.5
            goal_y = random_end

        goal_pos = [goal_x, goal_y, 0.3]

        return start_pos, goal_pos, self.obstacle_ids
