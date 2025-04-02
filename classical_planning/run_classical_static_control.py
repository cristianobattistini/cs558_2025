from environment import *
from controller import *

def main():

    # Initialize simulation
    step = 1./240.
    sim = A1Simulation(gui=True, time_step=step)
    # Start recording video (Disabled for now)
    """video_path = "simulation.mp4"
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
    if log_id < 0:
        print("Failed to start video recording!")
    else:
        print("Video recording started successfully!")    "
        """
    sim.teleport_robot([0,0,0.4])
    print("\n---------------------------------------------\n")
    print("Robot teleported to initial position, stepping simulation and starting controller...")
    print("\n---------------------------------------------")
    sim.step_simulation(10)


    # INITIALIZE CONTROLLER
    # Cost function weights (State cost Q, Input cost R)
    Q = np.diag([10, 10, 100,        # x, y, z
                10, 10, 10, 10, # quaternion (penalize tilt)
                1, 1, 1,          # lin_vel
                1, 1, 1])         # ang_vel

    R = 1 * np.eye(12)                # Penalty on high GRFs

    cost_weights = {  'Q': Q, 'R': R  }
    horizon = 10
    dt = step 
    find_foot_postions_func = sim.compute_foot_positions

    mpc = MPCController(horizon, cost_weights, 10*dt, find_foot_postions_func)  # Options :horizon, cost_weights, dt, find_foot_postions_func
    print("\n---------------------------------------------\n")
    print("MPC controller initialized, starting simulation...")
    print("\n---------------------------------------------")
    i = 0
    # Simulation/Control loop
    while True:
        # Get current state and contacts
        pos, orn, vel, ang_vel = sim.get_base_state()
        contacts = sim.get_contact_points()
        joint_angles = sim.get_joint_angles() 

        # Solve MPC
        grf_opt = mpc.solve(x0=np.concatenate([pos, orn, vel, ang_vel]),
                            contact_mask=contacts)
        # Use only the first control input from the sequence
        grf_opt = grf_opt[0]
        print("\n\nFirst control input: ",grf_opt)

        # Convert GRFs to torques 
        torques = grf_to_torques(grf_opt, contacts, joint_angles, sim)
       

        # Apply torques
        sim.set_joint_torques(torques)
        print("Iteration number ", i," done")
        i+=1
        sim.step_simulation(10)

    #p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
    sim.disconnect()


if __name__ == "__main__":
    main()
