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

    def find_first_free_cell(self, grid):
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    return (i, j)
        raise ValueError("No free cell found in the maze!")

    def find_last_free_cell(self, grid):
        for i in reversed(range(len(grid))):
            for j in reversed(range(len(grid[0]))):
                if grid[i][j] == 0:
                    return (i, j)
        raise ValueError("No free cell found in the maze!")

    def generate_maze_grid(self):
        """Generates a random maze using DFS with backtracking. Ensures multiple paths."""
        grid = [[1 for _ in range(self.cols)] for _ in range(self.rows)]  # 1 = wall, 0 = path

        def carve_passages_from(cx, cy):
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols and grid[nx][ny] == 1:
                    grid[cx + dx // 2][cy + dy // 2] = 0
                    grid[nx][ny] = 0
                    carve_passages_from(nx, ny)

        # Start carving from top-left
        start_x, start_y = 0, 0
        grid[start_x][start_y] = 0
        carve_passages_from(start_x, start_y)

        # Add extra random openings for multiple paths
        for _ in range(int(self.rows * self.cols * 0.1)):
            x, y = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            grid[x][y] = 0

        # Ensure start and goal are on valid cells
        start_cell = self.find_first_free_cell(grid)
        end_cell = self.find_last_free_cell(grid)

        return grid, start_cell, end_cell


    def create_maze_in_simulation(self, grid):
        """
        Create the maze in PyBullet using boxes for walls.

        Args:
            grid (List[List[int]]): 2D grid where 1 = wall, 0 = path.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if grid[i][j] == 1:
                    pos = [j * self.cell_size, -i * self.cell_size, self.wall_height / 2]
                    size = [self.cell_size / 2, self.cell_size / 2, self.wall_height / 2]
                    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
                    vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.7, 0.7, 0.7, 1])
                    body_id = p.createMultiBody(baseMass=0,
                                                baseCollisionShapeIndex=col_shape,
                                                baseVisualShapeIndex=vis_shape,
                                                basePosition=pos)
                    self.obstacle_ids.append(body_id)

    def get_world_position(self, cell):
        """
        Convert (row, col) grid cell to PyBullet world coordinates.

        Args:
            cell (Tuple[int, int]): Grid position as (row, col).

        Returns:
            List[float]: World position [x, y, z].
        """
        row, col = cell
        return [col * self.cell_size, -row * self.cell_size, 0.3]

    def create_maze(self):
        """
        Generate the maze grid, place it in simulation, and return start/end world positions.

        Returns:
            Tuple[List[float], List[float]]: Start and end positions in world coordinates.
        """
        grid, start_cell, end_cell = self.generate_maze_grid()
        self.create_maze_in_simulation(grid)
        return self.get_world_position(start_cell), self.get_world_position(end_cell), self.obstacle_ids
    

    def create_simplified_maze(self):
        """
        Create a simplified maze by placing multiple block obstacles.
        The obstacles are arranged in two parallel rows, creating a corridor 
        in the middle that the robot must navigate through or around.
        
        Returns:
            Tuple[List[float], List[float], List[int]]: 
                (start_pos, goal_pos, obstacle_ids)
        """
        # Clear any previously stored obstacles
        self.obstacle_ids = []

        # For best results, pick a cell_size that’s wide enough for your robot
        # (e.g., 1.0 or 1.2 if your robot trunk is ~0.2 m wide).
        # We'll define two rows of blocks: top row (y = +1 * cell_size), bottom row (y = -1 * cell_size).
        # That leaves a corridor in the middle (y ~ 0).

        # You can adjust how many blocks and how far in x they go.
        block_positions = [
            # Row +2
            [self.cell_size * 2,  self.cell_size * 2, self.wall_height / 2],
            [self.cell_size * 5,  self.cell_size * 2, self.wall_height / 2],

            [self.cell_size * 2,  0, self.wall_height / 2],
            [self.cell_size * 4,  0, self.wall_height / 2],
            [self.cell_size * 6,  0, self.wall_height / 2],

            [self.cell_size * 2, -self.cell_size * 2, self.wall_height / 2],
            [self.cell_size * 4, -self.cell_size * 2, self.wall_height / 2],
        ]



        for pos in block_positions:
            size = [self.cell_size / 2, self.cell_size / 2, self.wall_height / 2]
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.7, 0.7, 0.7, 1])
            body_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=col_shape,
                                        baseVisualShapeIndex=vis_shape,
                                        basePosition=pos)
            self.obstacle_ids.append(body_id)

        # Place the start on the left side of the corridor and the goal on the right.
        # Adjust if you want them closer/farther from the blocks.
        start_pos = [0, 0, 0.3]
        goal_pos  = [self.cell_size * 7.5, 0, 0.3]  # a bit beyond the last blocks

        return start_pos, goal_pos, self.obstacle_ids




'''

        block_positions = [
            # Row +2
            [self.cell_size * 2,  self.cell_size * 2, self.wall_height / 2],
            [self.cell_size * 5,  self.cell_size * 2, self.wall_height / 2],

            # Row +1
            # [self.cell_size * 3,  self.cell_size * 1, self.wall_height / 2],
            # [self.cell_size * 6,  self.cell_size * 1, self.wall_height / 2],

            # Row 0 (center line)
            [self.cell_size * 2,  0, self.wall_height / 2],
            [self.cell_size * 4,  0, self.wall_height / 2],
            [self.cell_size * 6,  0, self.wall_height / 2],

            # Row -1
            # [self.cell_size * 3, -self.cell_size * 1, self.wall_height / 2],
            # [self.cell_size * 5, -self.cell_size * 1, self.wall_height / 2],

            # Row -2
            [self.cell_size * 2, -self.cell_size * 2, self.wall_height / 2],
            [self.cell_size * 4, -self.cell_size * 2, self.wall_height / 2],
        ]


'''