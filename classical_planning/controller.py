import scipy.optimize as opt
import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import block_diag
#import cvxpy as cp  # For MPC optimization
import os

class Controller:
    def __init__(self, dt, control_type='pid', horizon=10, cost_weights=None):
        """
        Initialize the controller.
        
        Args:
            control_type (str): 'pid' or 'mpc'.
            horizon (int): Prediction horizon for MPC (default=10).
            dt (float): Simulation time step
            cost_weights (dict): Cost weights for MPC 
        """
        self.control_type = control_type.lower()
        self.dt = dt
        self.horizon = horizon if control_type == 'mpc' else 0
        
        # Robot parameters (from URDF)
        self.wheel_radius = 0.03  
        self.wheel_separation = 0.28 
        
        # PID gains
        self.Kp = np.array([0.4, 0.4])  # [distance, theta]
        self.Ki = np.array([0, 0])
        self.Kd = np.array([0.2, 0.1])
        self.prev_error = np.zeros(2)
        self.integral_error = np.zeros(2)
        
        # MPC cost weights 
        if cost_weights is None and control_type == 'mpc':
            print("No cost weights provided, cannot initialize MPC.")
            os._exit(1)           
        else:
            self.cost_weights = cost_weights 
        
    def inverse_kinematics(self, v, omega):
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
        left_wheel_vel, right_wheel_vel = self.inverse_kinematics(v, omega)
        
       

        return (left_wheel_vel, right_wheel_vel)
    """
    def mpc_control(self, desired_trajectory, current_state):
        
        MPC control for trajectory tracking.
        
        Args:
            desired_trajectory (np.array): N x 3 array of [x_d, y_d, theta_d] over horizon.
            current_state (np.array): [x, y, theta, left_wheel_vel, right_wheel_vel].
            
        Returns:
            tuple: (left_wheel_force, right_wheel_force) in N·m.
        
        # State: [x, y, theta, v_left, v_right]
        # Control: [torque_left, torque_right]
        
        # Discretized dynamics (simplified)
        def dynamics(x, u):
            v = self.wheel_radius * (x[3] + x[4]) / 2
            omega = self.wheel_radius * (x[4] - x[3]) / self.wheel_separation
            return np.array([
                v * np.cos(x[2]),  # dx/dt
                v * np.sin(x[2]),  # dy/dt
                omega,             # dtheta/dt
                u[0],              # dv_left/dt (simplified)
                u[1]               # dv_right/dt
            ]) * self.dt + x
        
        # MPC setup
        n_states = 5
        n_controls = 2
        x = cp.Variable((self.horizon + 1, n_states))
        u = cp.Variable((self.horizon, n_controls))
        
        # Cost and constraints
        cost = 0
        constraints = []
        
        for k in range(self.horizon):
            # Tracking cost (state error)
            cost += self.cost_weights['state'] * cp.sum_squares(x[k, :3] - desired_trajectory[k])
            # Control effort cost
            cost += self.cost_weights['control'] * cp.sum_squares(u[k])
            
            # Dynamics constraint
            constraints += [x[k+1] == dynamics(x[k], u[k])]
            
            # Torque limits (example: ±10 N·m)
            constraints += [cp.abs(u[k]) <= 10]
        
        # Initial condition constraint
        constraints += [x[0] == current_state]
        
        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)
        
        if prob.status != cp.OPTIMAL:
            print("MPC failed to find optimal solution!")
            return 0.0, 0.0
        
        # Return first control input
        return u.value[0, 0], u.value[0, 1]
    """
    def compute_control(self, current_state, desired_state):
        """
        Compute control signals based on the selected controller type.
        
        Args:
            desired_state (np.array): [x_desired, y_desired, theta_desired].
            current_state (np.array): [x, y, theta, left_wheel_vel, right_wheel_vel].
            
        Returns:
            tuple: Control signals (velocities or torques).
        """
        
        if self.control_type == 'pid':
            return self.pid_control(desired_state, current_state)
        elif self.control_type == 'mpc':
            # For MPC, desired_state should be a trajectory (horizon x 3)
            if desired_state.ndim == 1:
                desired_trajectory = np.tile(desired_state, (self.horizon, 1))
            else:
                desired_trajectory = desired_state
            return self.mpc_control(desired_trajectory, current_state)
        else:
            raise ValueError("Invalid control_type. Use 'pid' or 'mpc'.")

        """ def execute_control(self, robot_id, control_signals):
        
        Apply control signals to the robot in PyBullet.
        
        Args:
            robot_id (int): PyBullet robot ID.
            control_signals (tuple): (left_signal, right_signal).
    
        left_signal, right_signal = control_signals
        
        if self.control_type == 'pid':
            # Velocity control (rad/s)
            p.setJointMotorControl2(
                robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=left_signal)
            p.setJointMotorControl2(
                robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=right_signal)
        elif self.control_type == 'mpc':
            # Torque control (N·m)
            p.setJointMotorControl2(
                robot_id, 0, p.TORQUE_CONTROL, force=left_signal)
            p.setJointMotorControl2(
                robot_id, 1, p.TORQUE_CONTROL, force=right_signal)"""