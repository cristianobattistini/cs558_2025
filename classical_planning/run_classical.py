from environment import *
from controller import *
from rtt_star_path_planner import run_rrt_star 
import matplotlib.pyplot as plt
import argparse

def main():

    # Parse command line arguments
    # Create the parser
    parser = argparse.ArgumentParser(description='Controller simulation')
    
    # Add arguments 
    parser.add_argument('--visualization', dest='visual',action='store_false',help="Disable visualization")
    parser.add_argument('--sim_step', type=float,default=1./100.,help="Simulation step size (default: 0.01)")
    parser.add_argument('--robot', choices=['roomba', 'a1'], default='roomba',help="Robot type: 'roomba' or 'a1' (default: roomba)")
                        
    # Parse and save the arguments
    args = parser.parse_args()
    if not hasattr(args, 'visual'):
        args.visualization = True

    visual = args.visual
    step = args.sim_step
    robot = args.robot

    # Initialize simulation
    dt = step
    sim = A1Simulation(gui=visual, time_step=step, robot=robot)

    # Create obstacles
    rows=10
    cols=10
    cell_size=1.2
    start_pos, goal_pos, obstacles = sim.create_maze()

    # Define start and goal configurations as (x, y, theta)
    start_conf = (start_pos[0], start_pos[1], 0)
    goal_conf  = (goal_pos[0], goal_pos[1], 0)

    # Define maze bounds for RRT sampling (min_x, max_x, min_y, max_y)
    min_x = 0
    max_y = 0
    max_x = cols * cell_size
    min_y = -rows * cell_size
    maze_bounds = (min_x, max_x, min_y, max_y)

    # Define the robot's box dimensions (length, width, height)
    robot_dims = (0.5, 0.4, 0.3)

    # Run RRT* 
    print("\n---------------------------------------------\n")
    print("Planning path using RRT*...")
    print("\n---------------------------------------------")
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
        for i in range(len(path_rrt_star) - 1):
            path_rrt_star[i] = path_rrt_star[i][0:2]  # Angles are not needed
    else:
        print("No valid RRT* path found.")

   
    sim.reset_robot()
    sim.teleport_robot([0,0,0.5])    
    print("\n---------------------------------------------\n")
    print("Robot teleported to initial position, stepping simulation and starting controller...")
    print("\n---------------------------------------------")
    sim.step_simulation(10)

    # Initialize controller
    controller = Controller(dt, robot=robot) 
    print("\n---------------------------------------------\n")
    print("Controller initialized, starting simulation...")
    print("\n---------------------------------------------")
    i = 0
    
    # Initialize lists for logging data
    x_vals = []
    y_vals = []
    t_vals = []
    desired_x_vals = []
    desired_y_vals = []
    step_n = 0

    # Simulation/Control loop
    for i in path_rrt_star:
        desired_state = np.array([i[0], i[1]]) 
        close_to_goal = False
        while close_to_goal == False:
            
            # Get current state 
            x, y, theta, left_wheel_vel, right_wheel_vel = sim.get_base_state()
            current_state = np.array([x, y, theta, left_wheel_vel, right_wheel_vel])

            #Apply PID control
            left_wheel_vel, right_wheel_vel = controller.compute_control(current_state, desired_state)

            # Log data
            x_vals.append(x)
            y_vals.append(y)
            t_vals.append(step_n * dt)
            desired_x_vals.append(desired_state[0])
            desired_y_vals.append(desired_state[1])
            
            # Apply control inputs
            sim.set_wheel_velocities(left_wheel_vel, right_wheel_vel)

            # Check if the robot is close to the goal
            if (x - i[0])**2 + (y - i[1])**2 < 0.1:
                close_to_goal = True

            # Step simulation    
            step_n+=1
            sim.step_simulation()


    # Plot the results
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
