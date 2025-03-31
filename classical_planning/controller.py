import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

amplitude = 4  
frequency = 0.3

class MPCController:
    def __init__(self, n_states, m_inputs, horizon, system_dynamics, cost_weights, u_bounds=None, x_bounds=None, moving_target=None):
        """
        Initialize the MPC controller with flexible parameters.
        """
        self.n = n_states
        self.m = m_inputs
        self.horizon = horizon
        self.system_dynamics = system_dynamics
        self.Q = cost_weights['Q']  # State cost matrix
        self.R = cost_weights['R']  # Input cost matrix
        self.u_bounds = u_bounds if u_bounds is not None else (-np.inf, np.inf)  # Default: no bounds
        self.x_bounds = x_bounds
        self.moving_target = moving_target  # Function to compute the moving target

    def cost_function(self, u, x0, time_step):
        """
        The cost function to minimize, based on state tracking and input effort.
        """
        u = u.reshape((self.m, self.horizon))  # Reshape u to match the number of control inputs per time step
        x = np.zeros((self.n, self.horizon+1))  # State trajectory
        x[:, 0] = x0  # Initial state
        
        cost = 0
        for t in range(self.horizon):
            # Update the target state based on the moving target function
            x_target = self.moving_target(time_step + t + 1)
            
            # State update using the system dynamics
            x[:, t+1] = self.system_dynamics(x[:, t], u[:, t])
            
            # Add cost (state tracking and control effort)
            cost += np.dot((x[:, t+1] - x_target).T, np.dot(self.Q, (x[:, t+1] - x_target)))  # State tracking cost
            cost += np.dot(u[:, t].T, np.dot(self.R, u[:, t]))  # Control effort cost
        
        return cost

    def control_constraints(self, u):
        """
        Control input constraints (bounds on control inputs).
        """
        u = u.reshape((self.m, self.horizon))
        constraints = []
        for i in range(self.m):
            constraints.append(u[i, :] - self.u_bounds[0])  # Lower bound constraint
            constraints.append(self.u_bounds[1] - u[i, :])  # Upper bound constraint
        return np.concatenate(constraints)

    def solve(self, x0, time_step):
        """
        Solve the MPC optimization problem to find the optimal control inputs.
        """
        # Start with linearly increasing control inputs as a better guess
        u_initial = np.linspace(0, 1, self.horizon * self.m)  # Simple increasing control guess
        
        # Define bounds for control inputs
        bounds = [self.u_bounds for _ in range(self.m * self.horizon)]
        
        # Minimize the cost function
        result = opt.minimize(
            self.cost_function, 
            u_initial, 
            args=(x0, time_step), 
            bounds=bounds, 
            constraints={'type': 'ineq', 'fun': self.control_constraints}
        )
        
        # Extract optimal control inputs
        u_opt = result.x.reshape((self.m, self.horizon))
        
        # Simulate the system with the optimal control inputs
        x_opt = np.zeros((self.n, self.horizon+1))
        x_opt[:, 0] = x0
        for t in range(self.horizon):
            x_opt[:, t+1] = self.system_dynamics(x_opt[:, t], u_opt[:, t])
        
        return u_opt, x_opt

# Define a simple system dynamics (1D example)
def system_dynamics(x, u):
    """
    Simple system dynamics: x_{k+1} = x_k + u_k
    - x: state vector at time step k
    - u: control input at time step k
    """
    return x + u

# Define a moving goal (e.g., a linear moving target)
def moving_target(t):
    """
    A function that returns the moving target state at time t.
    """
    # Example: a sinusoidal target that moves over time
    return np.array([amplitude * np.sin(2 * np.pi * frequency * t)])
    

# Define MPC parameters
n_states = 1  # 1D state
m_inputs = 1  # 1 control input
horizon = 10  # 10 time steps
x0 = np.array([0])  # Initial state

# Cost function weights (State cost Q, Input cost R)
cost_weights = {
    'Q': 100 *np.eye(n_states),  # Identity matrix for state cost
    'R': 0.1 *np.eye(m_inputs)   # Identity matrix for input cost
}

# Control input bounds
u_bounds = (-10, 10)  

# Initialize the MPC controller with a moving goal
mpc = MPCController(
    n_states=n_states, 
    m_inputs=m_inputs, 
    horizon=horizon, 
    system_dynamics=system_dynamics, 
    cost_weights=cost_weights, 
    u_bounds=u_bounds,
    moving_target=moving_target  # Pass the moving target function
)

# Time steps for the simulation
time_steps = 50  # Number of time steps to simulate
x0 = np.array([0])  # Initial state

# Lists to store the results for plotting
states = []
controls = []
states.append(x0.copy())
# Initialize the current state
x_current = x0.copy()

# Simulate the MPC over time
for t in range(time_steps):
    # Solve the MPC problem at the current time step
    u_opt, x_opt = mpc.solve(x_current, t)
    
    # Apply only the first control input
    u_applied = u_opt[:, 0]
    x_next = system_dynamics(x_current, u_applied)
    
    # Store results
    states.append(x_next)
    controls.append(u_applied)
    
    # Update state
    x_current = x_next

# Convert lists to arrays for plotting
states = np.array(states)
controls = np.array(controls)

# Plot the results
time = np.arange(time_steps+1)

plt.figure(figsize=(10, 6))

# Plot states
plt.subplot(2, 1, 1)
plt.plot(time, states, label="State (x)")
plt.plot(time, amplitude * np.sin(2 * np.pi * frequency * time), 'r--', label="Moving target")  # The moving target trajectory
plt.xlabel("Time step")
plt.ylabel("State")
plt.legend()

time = np.arange(time_steps)

# Plot controls
plt.subplot(2, 1, 2)
plt.step(time, controls, label="Control input (u)", where="post")
plt.xlabel("Time step")
plt.ylabel("Control input")
plt.tight_layout()
plt.legend()

plt.show()
