import pybullet as p
import pybullet_data
import time
import os

class A1Simulation:
    def __init__(self, gui=True, plane="plane.urdf", time_step=1. / 240. ):
        """Initialize PyBullet simulation with or without GUI."""

        # Start PyBullet in GUI (Visuals on) or Direct mode (Visuals off)
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT) 

        # Set search path for default URDFs and assets
        current_dir = os.path.dirname(__file__) # Get the directory of this script
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.setAdditionalSearchPath(os.path.join(current_dir, '../assets/a1')) 

        self.plane_id = self.load_plane(plane)          # Load a plane (Currently it is the most basic)
        self.robot_id = self.load_robot(current_dir)    # Load the A1 quadruped robot 

        # Set the simulation parameters
        p.setGravity(0, 0, -9.81)   # Gravity
        self.time_step = time_step  # Simulation time step
        p.setTimeStep(self.time_step)

        # Robot parameters
        self.num_joints = p.getNumJoints(self.robot_id) # Number of joints in the robot

    
    def load_plane(self, plane):
        """Load the plane in to the simulation."""
        plane_urdf = os.path.join(pybullet_data.getDataPath(), plane)
        return p.loadURDF(plane_urdf)
    
    def load_robot(self, current_dir, start_pos=[0, 0, 0.3]):
        """Load the A1 quadruped robot."""
        a1_urdf = os.path.join(current_dir, '../assets/a1/urdf/a1.urdf')  # Path to the A1 URDF
        return p.loadURDF(a1_urdf, basePosition=start_pos)

    def step_simulation(self, steps=1):
        """Step the simulation forward a certain number of steps."""
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.time_step)  # Only sleep if using GUI to sync real-time movement

    def get_joint_states(self):
        """Retrieve all joint positions and velocities."""
        joint_states = []
        for i in range(self.num_joints):
            joint_state = p.getJointState(self.robot_id, i)[:2] # The full joint state: (position, velocity, forces and torques, applied_motor_torque)
            joint_states.append(joint_state)
        return joint_states  # Returns (position, velocity) tuples

    def reset_robot(self):
        """Reset robot position and velocity."""
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.3], [0, 0, 0, 1])
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            p.resetJointState(self.robot_id, i, targetValue=0, targetVelocity=0)

    def disconnect(self):
        """Disconnect PyBullet simulation."""
        p.disconnect()