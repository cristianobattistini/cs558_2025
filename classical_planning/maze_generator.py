import pybullet as p
import random
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


    def create_simplified_maze(self):
        """
        Create a denser maze by placing smaller block obstacles in multiple rows along the y-direction.
        Obstacles are scattered along the x-axis and multiple y rows, making a denser corridor-like passage.
        """
        self.obstacle_ids = []

        # Use a denser y-grid: rows from -2.5 to 2.5 in steps of cell_size
        num_y_layers = 10  # number of rows in y-direction
        y_offsets = [self.cell_size * (i - num_y_layers // 2) for i in range(num_y_layers)]

        # Same x-range as before, not extending in x
        possible_x = [self.cell_size * i for i in range(2, 8)]

        for y in y_offsets:
            # For each y-row, randomly choose more x positions to place small obstacles
            num_obs = random.randint(2, 6)  # Obstacles per row
            x_positions = random.sample(possible_x, num_obs)
            for x in x_positions:
                pos = [x, y, self.wall_height / 2]
                size = [self.cell_size / 4.5, self.cell_size / 4.5, self.wall_height / 2]  # smaller blocks
                col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
                vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.7, 0.7, 0.7, 1])
                body_id = p.createMultiBody(baseMass=0,
                                            baseCollisionShapeIndex=col_shape,
                                            baseVisualShapeIndex=vis_shape,
                                            basePosition=pos)
                self.obstacle_ids.append(body_id)

        # Keep same start and goal positions (robot needs to zigzag through the denser field)
        start_pos = [0, 0, 0.3]
        goal_pos  = [self.cell_size * 7.5, 0, 0.3]

        return start_pos, goal_pos, self.obstacle_ids    