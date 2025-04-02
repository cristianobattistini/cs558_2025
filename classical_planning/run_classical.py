from environment import A1Simulation
from rtt_path_planner import run_rrt 
from rtt_star_path_planner import run_rrt_star 
import pybullet as p
import time

def main():
    # Maze configuration
    rows = 10
    cols = 10
    cell_size = 1.2

    # Initialize simulation
    sim = A1Simulation(gui=True, time_step=1. / 50.)
    start_pos, goal_pos, obstacles = sim.create_maze(rows=rows, cols=cols, cell_size=cell_size)
    # for obs in obstacles:
    #     aabb_min, aabb_max = p.getAABB(obs)
    #     print("Obstacle", obs, "AABB:", aabb_min, aabb_max)
        
    # Define start and goal configurations as (x, y, theta)
    start_conf = (start_pos[0], start_pos[1], 0)
    goal_conf  = (goal_pos[0], goal_pos[1], 0)

    # Visualize start (green sphere) and goal (red sphere)
    start_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1])
    goal_vis  = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseVisualShapeIndex=start_vis, basePosition=[start_pos[0], start_pos[1], 0.2])
    p.createMultiBody(baseVisualShapeIndex=goal_vis,  basePosition=[goal_pos[0], goal_pos[1], 0.2])

    # Define maze bounds for RRT sampling (min_x, max_x, min_y, max_y)
    min_x = 0
    max_x = cols * cell_size
    min_y = -rows * cell_size
    max_y = 0
    maze_bounds = (min_x, max_x, min_y, max_y)

    # Define the robot's box dimensions (length, width, height)
    # (Adding some margin to the trunk collision box dimensions.)
    robot_dims = (0.267 + 0.100, 0.194 + 0.100, 0.114 + 0.100)

    # --- Run RRT ---
    print("Planning path using RRT...")
    start_time_rrt = time.time()

    path_rrt = run_rrt(start_conf, goal_conf, obstacles, maze_bounds, robot_dims,
                    max_iter=10000, goal_threshold=0.6)

    end_time_rrt = time.time()
    rrt_time = end_time_rrt - start_time_rrt
    rrt_length = sim.compute_path_length(path_rrt)

    print(f"RRT length: {rrt_length:.2f}")
    print(f"RRT planning time: {rrt_time:.2f} seconds")


    if path_rrt:
        print("RRT Path found:")
        # for conf in path_rrt:
        #     print(conf)
        # Optionally visualize RRT path with a red line:
        sim.visualize_path(path_rrt, color=[1, 0, 0])
        print("Executing RRT* path...")
        # for i in range(len(path_rrt) - 1):
        #     sim.move_robot_along_segment(path_rrt[i], path_rrt[i+1])
    else:
        print("No valid RRT path found.")

    # --- Run RRT* (RRT STAR) ---
    print("Planning path using RRT*...")
    start_time_rrt_star = time.time()

    path_rrt_star = run_rrt_star(start_conf, goal_conf, obstacles, maze_bounds, robot_dims,
                                max_iter=10000, goal_threshold=0.6, step_size=0.3)

    end_time_rrt_star = time.time()
    rrt_star_time = end_time_rrt_star - start_time_rrt_star
    rrt_star_length = sim.compute_path_length(path_rrt_star)

    print(f"RRT* length: {rrt_star_length:.2f}")
    print(f"RRT* planning time: {rrt_star_time:.2f} seconds")
    if path_rrt_star:
        print("RRT* Path found:")
        #for conf in path_rrt_star:
            #print(conf)
        # Visualize the RRT* path with a green line.
        sim.visualize_path(path_rrt_star, color=[0, 1, 0])
        # Execute path by moving the robot along each segment.
        print("Executing RRT* path...")
        for i in range(len(path_rrt_star) - 1):
            sim.move_robot_along_segment(path_rrt_star[i], path_rrt_star[i+1])
    else:
        print("No valid RRT* path found.")

    # Zoom control loop for visualization
    print("Ready for keyboard control (Z to zoom in, X to zoom out)")
    while True:
        sim.handle_keyboard_zoom()
        sim.step_simulation(steps=1)

if __name__ == "__main__":
    main()
