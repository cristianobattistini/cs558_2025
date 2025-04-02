import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


# ROBOT PARAMETERS
# Inertia of the robot   (From the URDF)
ixx = 0.01683993
ixy = 8.3902e-05
ixz = 0.000597679
iyy = 0.056579028
iyz = 2.5134e-05
izz = 0.064713601

TORSO_MASS = 4.713          # Mass of the trunk (From the URDF)
FRICTION_U = 0.6            # Friction coefficient

STATIC_POSITION = [0, 0, 0.4]  # For static control where you want the robot to be


class MPCController:
    def __init__(self, horizon, cost_weights, dt, find_foot_postions_func, n_states=13, m_inputs=12, u_bounds=None, x_bounds=None):
        """
        Initialize the MPC controller.
        x = [base_pos (x, y, z), base_quat (qw, qx, qy, qz), base_lin_vel (vx, vy, vz), base_ang_vel (ωx, ωy, ωz)]  → 13-dimensions (13D).
        u = [GRF_x1, GRF_y1, GRF_z1, GRF_x2, ..., GRF_z4]  # 3D Ground Reaction Forces for 4 feet → 12D.
        """
        # Controller parameters
        self.n = n_states
        self.m = m_inputs
        self.horizon = horizon
        self.Q = cost_weights['Q']  # State cost matrix
        self.R = cost_weights['R']  # Input cost matrix
        self.u_bounds = u_bounds if u_bounds is not None else (-np.inf, np.inf)  # Default: no bounds
        self.x_bounds = x_bounds

        # Robot parameters
        self.mass = TORSO_MASS          # Mass of the trunk 
        self.mu = FRICTION_U            # Friction coefficient
        self.dt = dt                    # Time step for dynamics calculation, it has to be the same as the simulation time step
        self.inertia = np.array([       # Inertia of the robot 
            [ixx,  ixy,  ixz],
            [ixy,  iyy,  iyz],
            [ixz,  iyz,  izz]
        ])                    
        self.find_foot_positions = find_foot_postions_func    # Find the foot positions in the world frame


    def cost_function(self, u_flat, x0, contact_mask):
        """
        x: [pos (3), quat (4), lin_vel (3), ang_vel (3)] → 13D
        u: [GRF_x1, GRF_y1, GRF_z1, ..., GRF_z4] → 12D
        contact_mask: [4] boolean array (True = foot in contact)
        """
        # Reshape the full control sequence into (horizon, m)
        u = u_flat.reshape(self.horizon, self.m)
        
        x = np.zeros((self.n, self.horizon + 1))
        x[:, 0] = x0
        cost = 0

        for t in range(self.horizon):
            # For each time step, reshape the control to (4,3)
            u_t = u[t].reshape(4, 3)
            x[:, t+1] = self.quadruped_dynamics(x[:, t], u_t, contact_mask)

            # Target state for static control (standing still)
            target_pos = np.array(STATIC_POSITION)  
            target_quat = np.array([0, 0, 0, 1])    # Upright
            target_vel = np.zeros(3)                # No velocity
            target_ang_vel = np.zeros(3)            # No angular velocity
            x_target = np.concatenate([target_pos, target_quat, target_vel, target_ang_vel])

            # State error cost
            error = x[:, t+1] - x_target
            cost += error.T @ self.Q @ error 

            # Input cost (penalize large GRFs)
            cost += u[t].T @ self.R @ u[t]  # u[t] is already the correct shape (12,)

        return cost

    def control_constraints(self, contact_mask):
        """
        Generate constraints for the optimization problem based on contact_mask.
        Returns a list of constraint dictionaries compatible with scipy.optimize.minimize.
        """
        constraints = []
        m_per_step = self.m     

        for t in range(self.horizon):
            for foot_idx in range(4):
                # Index range for the current foot's GRF in the flattened control array
                start_idx = t * m_per_step + foot_idx * 3
                end_idx = start_idx + 3

                if contact_mask[foot_idx]:
                    # Foot is in contact: enforce friction pyramid constraints and normal force >= 0
                    def friction_inequality(u, idx=start_idx):
                        fx, fy, fz = u[idx], u[idx+1], u[idx+2]
                        return [
                            fz,  # fz >= 0
                            self.mu * fz - fx,  # fx <= μ * fz
                            self.mu * fz + fx,  # fx >= -μ * fz
                            self.mu * fz - fy,  # fy <= μ * fz
                            self.mu * fz + fy   # fy >= -μ * fz
                        ]

                    constraints.append({
                        'type': 'ineq',
                        'fun': friction_inequality
                    })
                else:
                    # Foot is not in contact: GRF must be zero
                    def zero_force(u, idx=start_idx):
                        return u[idx:idx+3]

                    constraints.append({
                        'type': 'eq',
                        'fun': zero_force
                    })

        return constraints

    def solve(self, x0, contact_mask):
        """
        Solve the MPC problem to find optimal control inputs over the horizon.
        Returns the optimal control sequence (horizon x m_inputs).
        """
        # Initial guess (flattened)
        u0 = np.zeros(self.m * self.horizon)
        # Better initial guess: Distribute robot weight equally among contacting feet
        num_contacts = np.sum(contact_mask)
        grf_z = (self.mass * 9.81) / (num_contacts + 1e-6)
    
        # Initialize vertical forces for contacting feet
        for t in range(self.horizon):
            for foot in range(4):
                if contact_mask[foot]:
                    u0[t*self.m + foot*3 + 2] = grf_z  # Set z-component

        # Generate constraints
        constraints = self.control_constraints(contact_mask)

        # Bounds for each control input (if any, else defaults to (-inf, inf))
        bounds = [self.u_bounds] * (self.m * self.horizon)

        # Solve the optimization problem
        result = opt.minimize(self.cost_function, u0, args=(x0, contact_mask),
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            options={'maxiter': 100})

        if not result.success:
            print("\n\n\nWarning: MPC optimization failed - ", result.message, "\n\n\n")
            # Returning zeros as a fallback but this should be handled better in practice
            return np.zeros((self.horizon, self.m))

        # Reshape the solution into (horizon x m_inputs)
        return result.x.reshape((self.horizon, self.m))

    # Define system dynamics
    def quadruped_dynamics(self,x, u, contact_mask):
        """
        x: [pos (3), quat (4), lin_vel (3), ang_vel (3)] → 13D
        u: [GRF_x1, GRF_y1, GRF_z1, ..., GRF_z4] → 12D
        contact_mask: [4] boolean array (True = foot in contact)
        """
        # Extract states
        pos = x[0:3]
        quat = x[3:7]       # [qx, qy, qz, qw]
        lin_vel = x[7:10]
        ang_vel = x[10:13]

        # Convert quaternion to rotation matrix (world to body)
        R = quat_to_rot(quat)  

        # Total force/torque from GRFs (in world frame)
        GRF = u.reshape(4, 3) * contact_mask[:, np.newaxis]  # Mask non-contacting feet
        total_force = np.sum(GRF, axis=0) + self.mass * np.array([0, 0, -9.81])  # Gravity

        # Use Single Rigid Body (SRB) dynamics (Treat the robot as a single rigid body)
        new_lin_vel = lin_vel + (total_force / self.mass) * self.dt     # Basic F/m = a and v = a*t
        new_pos = pos + lin_vel * self.dt                               # Basic x = v*t

        # Compute the foot positions relative to the robot's CoM
        foot_positions = self.find_foot_positions()  

        # Angular dynamics
        inertia_inv = np.linalg.inv(self.inertia)                                   # Inverse inertia matrix
        total_torque = np.sum([np.cross((R.T @ (foot_pos - pos)), (R.T @ GRF[i]))   # Compute the total torque of the torso from the contribution of each foot 
                      for i, foot_pos in enumerate(foot_positions)], axis=0)        # Later the torque is needed in body frame so R.T is used 
        new_ang_vel = ang_vel + inertia_inv @ (total_torque - np.cross(ang_vel,self.inertia @ ang_vel)) * self.dt  # Basic τ = I * α and integration rearranged 

        # Obtain new quaternion based on angular velocity and the time step
        new_quat = new_quaternion(quat, ang_vel, self.dt) 

        return np.concatenate([new_pos, new_quat, new_lin_vel, new_ang_vel])



# UTILS
def quat_to_rot(q):
    """Convert PyBullet quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,   2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2,   2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,       2*qy*qz + 2*qx*qw,    1 - 2*qx**2 - 2*qy**2]
    ])


def new_quaternion(q_pybullet, w_body, dt):
    """
    Compute the new robot's orientation over a time step dt based on its angular velocity.
    q: [qx, qy, qz, qw] (current orientation)
    ω: [ωx, ωy, ωz] (angular velocity in body frame)
    dt: time step
    """
    # Convert to Hamilton form [w, x, y, z] for calculations
    x, y, z, w = q_pybullet
    q_hamilton = np.array([w, x, y, z])
    
    # Compute quaternion derivative
    w_quat = np.array([0, w_body[0], w_body[1], w_body[2]])  # Body-frame ω
    q_dot = 0.5 * quaternion_multiply(q_hamilton, w_quat)    # Quaternion derivative equation (q_dot = 1/2 * q * ω)
    
    # Euler integration
    q_new_hamilton = q_hamilton + q_dot * dt            
    q_new_hamilton /= np.linalg.norm(q_new_hamilton)    # Normalize to obtain a unit quaternion (PyBullet requires unit quaternions)
    
    # Convert back to PyBullet [x, y, z, w]
    w_new, x_new, y_new, z_new = q_new_hamilton
    return np.array([x_new, y_new, z_new, w_new])


def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])







def grf_to_torques(grf_opt, contact_mask, joint_angles, sim):
    """
    Convert ground reaction forces (GRFs) to joint torques using the Jacobian transpose method.
    
    Args:
        grf_opt (np.ndarray): Optimized GRFs (12-dimensional: 4 legs * 3D forces).
        contact_mask (list): Boolean list indicating if each leg is in contact.
        joint_angles (np.ndarray): Current joint angles (12-dimensional).
        sim (A1Simulation): Simulation object to compute foot positions.
        
    Returns:
        np.ndarray: Joint torques (12-dimensional).
    """
    torques = np.zeros(12)
    
    # Order of legs: FR, FL, RR, RL
    for leg_idx in range(4):
        if contact_mask[leg_idx]:
            # Extract GRF for this leg (3D vector)
            F = grf_opt[leg_idx*3 : (leg_idx+1)*3]
            
            # Get joint angles for hip, thigh, calf of this leg
            q_hip = joint_angles[leg_idx*3 + 0]
            q_thigh = joint_angles[leg_idx*3 + 1]
            q_calf = joint_angles[leg_idx*3 + 2]
            
            # Compute foot position in world frame
            foot_pos = sim.compute_foot_positions()[leg_idx]
            
            # Compute Jacobian for the leg
            J = compute_leg_jacobian(q_hip, q_thigh, q_calf, leg_idx, foot_pos)
            
            # Torque = J^T * F
            tau = J.T @ F
            torques[leg_idx*3 : (leg_idx+1)*3] = tau
            
    return torques






def compute_leg_jacobian(q_hip, q_thigh, q_calf, leg_idx, foot_pos):
    """
    Compute the Jacobian for a leg based on joint angles and URDF parameters.
    """
    # Convert foot_pos to numpy array if it isn't already
    foot_pos = np.asarray(foot_pos)

    # Joint axes (from URDF, in local frames)
    axes = [
        np.array([1, 0, 0]),   # Hip joint (x-axis)
        np.array([0, 1, 0]),   # Thigh joint (y-axis)
        np.array([0, 1, 0])    # Calf joint (y-axis)
    ]
    
    # Positions of joints in base frame (Using simplified forward kinematics aproximations from URDF)
    hip_pos = get_hip_position(leg_idx)  
    thigh_pos = hip_pos + rotate_vector([0, -0.08505, 0], q_hip, axes[0])
    calf_pos = thigh_pos + rotate_vector([0, 0, -0.2], q_thigh, axes[1])
    
    # Vectors from joints to foot
    r_hip = foot_pos - hip_pos
    r_thigh = foot_pos - thigh_pos
    r_calf = foot_pos - calf_pos
    
    # Compute Jacobian columns
    J = np.zeros((3, 3))
    J[:, 0] = np.cross(axes[0], r_hip)    # Hip column
    J[:, 1] = np.cross(axes[1], r_thigh)  # Thigh column
    J[:, 2] = np.cross(axes[2], r_calf)   # Calf column
    
    return J

def get_hip_position(leg_idx):
    """Get hip joint position relative to base (from URDF)."""
    positions = [
        [0.183, -0.047, 0],   # FR
        [0.183, 0.047, 0],     # FL
        [-0.183, -0.047, 0],   # RR
        [-0.183, 0.047, 0]     # RL
    ]
    return np.array(positions[leg_idx])
    
def rotate_vector(vec, angle, axis):
    """Rotate a vector by an angle around an axis using Rodrigues' formula."""
    # Convert inputs to numpy arrays
    vec = np.asarray(vec, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    
    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    
    cos = np.cos(angle)
    sin = np.sin(angle)
    return vec * cos + np.cross(axis, vec) * sin + axis * np.dot(axis, vec) * (1 - cos)

