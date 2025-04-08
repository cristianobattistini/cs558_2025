import scipy.optimize as opt
import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import block_diag
import os

class Controller:
    def __init__(self, dt, robot='roomba', horizon=10, cost_weights=None):
        """
        Initialize the controller.
        
        Args:
            robot (str): 'roomba' or 'a1'.
            horizon (int): Prediction horizon for MPC (default=10).
            dt (float): Simulation time step
            cost_weights (dict): Cost weights for MPC 
        """
        self.robot = robot.lower()
        self.dt = dt
        self.horizon = horizon if robot == 'a1' else 0
        
        #ROOMBA
        # Robot parameters (from URDF)
        self.wheel_radius = 0.03  
        self.wheel_separation = 0.28 
        
        # PID gains
        self.Kp = np.array([1.2, 1.2])  # [distance, theta]
        self.Ki = np.array([0, 0])
        self.Kd = np.array([0.4, 0.4])
        self.prev_error = np.zeros(2)
        self.integral_error = np.zeros(2)
        
        # A1
        # MPC cost weights 
        if cost_weights is None and robot == 'a1':
            print("No cost weights provided, cannot initialize MPC.")
            os._exit(1)           
        else:
            self.cost_weights = cost_weights 
        
    def roomba_inverse_kinematics(self, v, omega):
        """
        Convert linear (v) and angular (omega) velocities to wheel velocities.
        
        Args:
            v (float): Linear velocity (m/s).
            omega (float): Angular velocity (rad/s).
            
        Returns:
            tuple: (left_wheel_vel, right_wheel_vel) in rad/s.
        """
        # v = 1 # Constant linear velocity
        left_wheel_vel = (v - omega * self.wheel_separation / 2) / self.wheel_radius
        right_wheel_vel = (v + omega * self.wheel_separation / 2) / self.wheel_radius
        return left_wheel_vel, right_wheel_vel
    
    
    def pid_control(self, desired_state, current_state):
        """
        PID control for trajectory tracking.
        
        Args:
            desired_state (np.array): [x_desired, y_desired].
            current_state (np.array): [x, y, theta, left_wheel_omega, right_wheel_omega].
            
        Returns:
            tuple: (left_wheel_omega, right_wheel_omega) in rad/s.
        """
        x, y, theta = current_state[:3]
        x_d, y_d = desired_state[:2]
        left_omega, right_omega = current_state[3:5]

        # Position error
        error_x = x_d - x
        error_y = y_d - y
        distance_error = np.sqrt(error_x**2 + error_y**2)
        target_angle = np.arctan2(error_y, error_x)
        
        # Orientation error (wrap to [-π, π])
        angle_error = (target_angle - theta + np.pi) % (2 * np.pi) - np.pi
        
        # PID terms
        self.integral_error[0] += distance_error * self.dt
        self.integral_error[1] += angle_error * self.dt
        
        derivative_error = np.array([
            distance_error - self.prev_error[0],
            angle_error - self.prev_error[1]
        ]) / self.dt
        self.prev_error = np.array([distance_error, angle_error])
        
        # Control output
        v = self.Kp[0]*distance_error + self.Ki[0]*self.integral_error[0] + self.Kd[0]*derivative_error[0]
        omega = self.Kp[1]*angle_error + self.Ki[1]*self.integral_error[1] + self.Kd[1]*derivative_error[1]
        
        # Add URDF-consistent limits
        max_vel = 5  
        v = np.clip(v, -max_vel, max_vel)

        # Convert to wheel velocities
        left_wheel_vel, right_wheel_vel = self.roomba_inverse_kinematics(v, omega)

        return (left_wheel_vel, right_wheel_vel)
    
    def compute_control(self, current_state, desired_state):
        """
        Compute control signals based on the selected controller type.
        
        Args:
            desired_state (np.array): [x_desired, y_desired, theta_desired].
            current_state (np.array): [x, y, theta, left_wheel_vel, right_wheel_vel].
            
        Returns:
            tuple: Control signals (velocities or torques).
        """
        
        if self.robot == 'roomba':
            return self.pid_control(desired_state, current_state)
        elif self.robot == 'a1':
            # For MPC, desired_state should be a trajectory (horizon x 3)
            if desired_state.ndim == 1:
                desired_trajectory = np.tile(desired_state, (self.horizon, 1))
            else:
                desired_trajectory = desired_state
            return self.mpc_control(desired_trajectory, current_state)
        else:
            raise ValueError("Invalid control_type. Use 'pid' or 'mpc'.")

    