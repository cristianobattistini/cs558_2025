from environment import A1Simulation

def main():
    # Initialize the simulation with GUI enabled
    sim = A1Simulation(gui=True, time_step=1. / 50.)

    # Step the simulation for 100 steps
    print("Stepping the simulation...")
    sim.step_simulation(steps=100)

    # Retrieve and print joint states
    print("Retrieving joint states...")
    joint_states = sim.get_joint_states()
    for i, (position, velocity) in enumerate(joint_states):
        print(f"Joint {i}: Position={position}, Velocity={velocity}")

    # Reset the robot to its initial state
    print("Resetting the robot...")
    sim.reset_robot()

    # Step the simulation again after reset
    print("Stepping the simulation after reset...")
    sim.step_simulation(steps=50)

    # Disconnect from the simulation
    print("Disconnecting from the simulation...")
    sim.disconnect()

if __name__ == "__main__":
    main()