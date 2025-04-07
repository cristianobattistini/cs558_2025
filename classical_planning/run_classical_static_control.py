from environment import *
from controller import *
from rtt_star_path_planner import run_rrt_star 
import matplotlib.pyplot as plt

def main():
    # Maze configuration
    rows = 10
    cols = 10
    cell_size = 1.2

    # Initialize simulation
    step = 1./50.
    sim = A1Simulation(gui=True, time_step=step)
    start_pos, goal_pos, obstacles = sim.create_maze(rows=rows, cols=cols, cell_size=cell_size)
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
    robot_dims = (0.5, 0.4, 0.3)

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
        # Visualize the RRT* path with a green line.
        sim.visualize_path(path_rrt_star, color=[0, 1, 0])
        # Execute path by moving the robot along each segment.
        print("Executing RRT* path...")
        for i in range(len(path_rrt_star) - 1):
            path_rrt_star[i] = path_rrt_star[i][0:2]

    else:
        print("No valid RRT* path found.")

    # Setup video recording  
    #out = sim.setup_video_recording(video_path="simulation.mp4", fps=1/step)                                                 

    sim.teleport_robot([0,0,0.5])    
    print("\n---------------------------------------------\n")
    print("Robot teleported to initial position, stepping simulation and starting controller...")
    print("\n---------------------------------------------")
    sim.step_simulation(10)


    # INITIALIZE CONTROLLER
    # Cost function weights (State cost Q, Input cost R)
    Q = np.diag([1e5, 1e5, 2e5,    # x, y, z
             1e6, 1e6, 1e6, 1e6,  # quaternion
             1e4, 1e4, 1e4,      # lin_vel
             5e3, 5e3, 5e3])     # ang_vel
    R = 0.1 * np.eye(12)                # Penalty on high GRFs

    cost_weights = {  'Q': Q, 'R': R  }
    horizon = 5
    dt = step 
    find_foot_postions_func = sim.compute_foot_positions

    #mpc = MPCController(horizon, cost_weights, dt, find_foot_postions_func)  # Options :horizon, cost_weights, dt, find_foot_postions_func
    pid = Controller(dt, control_type='pid',) 
    print("\n---------------------------------------------\n")
    print("Controller initialized, starting simulation...")
    print("\n---------------------------------------------")
    i = 0
    # Simulation/Control loop
     
    x_vals = []
    y_vals = []
    t_vals = []
    desired_x_vals = []
    desired_y_vals = []
    step_n = 0
    for i in path_rrt_star:
        desired_state = np.array([i[0], i[1]]) 
        close_to_goal = False
        while close_to_goal == False:
            
            #frame = sim.get_video_frame()  # Video recording
            #out.write(frame)                                                           # Write to video file
        
            # Get current state 
            x, y, theta, left_wheel_vel, right_wheel_vel = sim.get_base_state()
            current_state = np.array([x, y, theta, left_wheel_vel, right_wheel_vel])

            #Apply PID control
            left_wheel_vel, right_wheel_vel = pid.compute_control(current_state, desired_state)

            # Logging data
            x_vals.append(x)
            y_vals.append(y)
            t_vals.append(step_n * dt)
            desired_x_vals.append(desired_state[0])
            desired_y_vals.append(desired_state[1])
            
            # Apply control inputs
            sim.set_wheel_velocities(left_wheel_vel, right_wheel_vel)
            if (x - i[0])**2 + (y - i[1])**2 < 0.1:
                close_to_goal = True
            step_n+=1
            sim.step_simulation()

    #out.release()       # Release video file
    plt.figure()
    plt.plot(t_vals, x_vals, label='x')
    plt.plot(t_vals, y_vals, label='y')
    plt.plot(t_vals, desired_x_vals, label='desired x', linestyle='--')
    plt.plot(t_vals, desired_y_vals, label='desired y', linestyle='--')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('x and y over time')
    plt.legend()
    plt.grid()
    plt.show()
    sim.disconnect()    # Disconnect from simulation


if __name__ == "__main__":
    main()
