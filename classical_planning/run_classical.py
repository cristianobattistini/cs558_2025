from environment import A1Simulation
from rtt_path_planner import run_rrt
import pybullet as p

def main():
    # Maze configuration
    rows = 10
    cols = 10
    cell_size = 1.2

    # Initialize simulation
    sim = A1Simulation(gui=True, time_step=1. / 50.)
    start_pos, goal_pos, obstacles = sim.create_maze(rows=rows, cols=cols, cell_size=cell_size)

    # Define start and goal configurations
    start_conf = (start_pos[0], start_pos[1], 0)
    goal_conf = (goal_pos[0], goal_pos[1], 0)


    # Visualize start (green sphere) and goal (red sphere)
    start_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1])
    goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0, 0, 1])

    p.createMultiBody(baseVisualShapeIndex=start_vis, basePosition=[start_pos[0], start_pos[1], 0.2])
    p.createMultiBody(baseVisualShapeIndex=goal_vis, basePosition=[goal_pos[0], goal_pos[1], 0.2])

    # Define maze bounds for RRT sampling
    min_x = 0
    max_x = cols * cell_size
    min_y = -rows * cell_size
    max_y = 0
    maze_bounds = (min_x, max_x, min_y, max_y)

    # Plan path with RRT
    print("Planning path using RRT...")
    path = run_rrt(start_conf, goal_conf, obstacles, maze_bounds)

    # Execute path by teleporting the robot along waypoints
    if path:
        print("Executing path...")
        for conf in path:
            sim.teleport_robot([conf[0], conf[1], 0.3])
            sim.step_simulation(steps=50)
    else:
        print("No valid path found.")

    # Zoom control loop
    print("Ready for keyboard control (Z to zoom in, X to zoom out)")
    while True:
        sim.handle_keyboard_zoom()
        sim.step_simulation(steps=1)

if __name__ == "__main__":
    main()
