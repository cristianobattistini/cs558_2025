import pybullet as p
import pybullet_data
import time
import os
from maze_generator import MazeGenerator
import numpy as np
import math


class A1Simulation:
    def __init__(self, gui=True, plane="plane.urdf", time_step=1. / 240. ):
        """Initialize PyBullet simulation with or without GUI."""

        # Start PyBullet in GUI (Visuals on) or Direct mode (Visuals off)
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT) 
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)              # Hide side debug sliders
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)


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
        self.joint_indices = self.get_joint_indices()

    
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
    
    def zoom_camera(self, direction, step=0.5):
        cam = p.getDebugVisualizerCamera()
        current_distance = cam[10]
        yaw = cam[8]
        pitch = cam[9]
        target = cam[11]

        if direction == "in":
            new_distance = max(1.0, current_distance - step)
        elif direction == "out":
            new_distance = current_distance + step
        else:
            return  # unknown direction, do nothing

        p.resetDebugVisualizerCamera(cameraDistance=new_distance, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target)

    def handle_keyboard_zoom(self):
            keys = p.getKeyboardEvents()

            if ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
                self.zoom_camera("in")
            if ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
                self.zoom_camera("out")
            if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                print("Exiting simulation...")
                p.disconnect()
                exit()

    def teleport_robot(self, position, orientation=[0, 0, 0, 1]):
        """Teleport the robot base to a specific position and orientation."""
        p.resetBasePositionAndOrientation(self.robot_id, position, orientation)

    def create_maze(self, rows=10, cols=10, cell_size=1.2):
        """Use MazeGenerator to build maze and reset robot at start position."""
        maze = MazeGenerator(rows=rows, cols=cols, cell_size=cell_size)
        start_pos, end_pos, obstacles = maze.create_simplified_maze()

        self.reset_robot()
        self.teleport_robot([start_pos[0], start_pos[1], 0.3])

        return start_pos, end_pos, obstacles
    

    def move_robot_along_segment(self, start_conf, end_conf, steps=20):
        """
        Move the robot smoothly from start_conf to end_conf by interpolating 
        position and orientation.
        Each conf is (x, y, theta), where theta is the rotation about z.
        """
        xs = np.linspace(start_conf[0], end_conf[0], steps)
        ys = np.linspace(start_conf[1], end_conf[1], steps)
        thetas = np.linspace(start_conf[2], end_conf[2], steps)
        
        for x, y, theta in zip(xs, ys, thetas):
            # Compute quaternion from theta (assuming no roll or pitch)
            orn = p.getQuaternionFromEuler([0, 0, theta])
            # Pass both position and orientation to teleport_robot.
            # If your teleport_robot doesn't currently accept orientation,
            # you can modify it accordingly:
            # def teleport_robot(self, pos, orn):
            #     p.resetBasePositionAndOrientation(self.robot_id, pos, orn)
            self.teleport_robot([x, y, 0.3], orn)
            self.step_simulation(steps=5)


    def visualize_path(self, path, color=[1, 0, 0], line_width=2.0, z_offset=0.02):
        """
        Draws a series of red line segments connecting each waypoint in 'path'.

        path: list of configurations (x, y, theta) or (x, y) in order
        color: (R, G, B) color of the debug line
        line_width: thickness of the line
        z_offset: how high above the ground to draw the line (to avoid z-fighting)
        """
        for i in range(len(path) - 1):
            (x1, y1, *_) = path[i]
            (x2, y2, *_) = path[i + 1]
            start = [x1, y1, z_offset]
            end = [x2, y2, z_offset]
            p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=line_width, lifeTime=0)


    def compute_path_length(self, path):
        total_length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i][0], path[i][1]
            x2, y2 = path[i+1][0], path[i+1][1]
            segment_length = math.dist((x1, y1), (x2, y2))
            total_length += segment_length
        return total_length

    def disconnect(self):
        """Disconnect PyBullet simulation."""
        p.disconnect()

    
    def get_base_state(self):
        """Get robot base (torso) position, orientation, and velocity."""
        # Position and Orientation of the torso. Position of CoM in Cartesian world coordinates, orientation in [x,y,z,w] from the world frame to the object's local frame.
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)  
        # Linear and Angular velocity. Linear velocity of the CoM is in the world frame. Angular velocity is in ref to the torso.
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        return pos, orn, vel, ang_vel
    
    def get_foot_link_indices(self):
        """Find and return the link indices for the robot's feet."""
        foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        foot_indices = {}

        for foot_name in foot_names:
            foot_index = -1
            for i in range(p.getNumJoints(self.robot_id)):
                info = p.getJointInfo(self.robot_id, i)
                link_name = info[12].decode('utf-8')
                if link_name == foot_name:
                    foot_index = info[0]
                    break

            if foot_index == -1:
                print(f"Warning: Foot link '{foot_name}' not found in URDF.")
            else:
                foot_indices[foot_name] = foot_index

        return foot_indices

    def get_contact_points(self, threshold=0.01):
        """Check if feet are in contact with the ground (force > threshold)."""
        foot_indices = self.get_foot_link_indices()
        contacts = []

        for foot_name, foot_index in foot_indices.items():
            if foot_index == -1:
                contacts.append(False)
                continue

            contact = p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=foot_index)

            if contact and len(contact) > 0 and contact[0][9] > threshold:
                contacts.append(True)
            else:
                contacts.append(False)

        return np.array(contacts)  # [FR, FL, RR, RL]
    
    def compute_foot_positions(self):
        """Return the 3D positions of the robot's feet in the world's frame."""
        foot_indices = self.get_foot_link_indices()
        foot_positions = []
        # Order of feet: FR, FL, RR, RL
        for foot_name in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            if foot_name not in foot_indices:
                print(f"Error: {foot_name} index not found.")
                return []
            foot_index = foot_indices[foot_name]
            foot_state = p.getLinkState(self.robot_id, foot_index)
            foot_pos = foot_state[0]  # World position of the foot link's CoM
    
            foot_positions.append(foot_pos)
        
        return foot_positions
    
    def get_joint_indices(self):
        """Map joint names to indices."""
        joint_indices = []
        joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        for name in joint_names:
            for i in range(self.num_joints):
                info = p.getJointInfo(self.robot_id, i)
                if info[1].decode("utf-8") == name:
                    joint_indices.append(i)
                    break
        return joint_indices

    def get_joint_angles(self):
        """Returns a 12D np.ndarray of joint angles in radians, ordered as:
        [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]."""
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        return np.array([state[0] for state in joint_states])
    
    def set_joint_torques(self, torques):
        """Applies torques to all 12 joints."""
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.TORQUE_CONTROL,
                force=torques[i]
            )    
