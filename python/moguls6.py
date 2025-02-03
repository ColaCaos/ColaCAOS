import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

# =============================================================================
# Simulation Parameters
# =============================================================================

# Height of the moguls (meters)
b_default = 0.5  # Set to 0.5 meters as per your request

# Inclination angle range (degrees)
theta_values = np.linspace(10, 45, 100)  # 10° to 45° with 100 increments

# Number of sleds per theta
N_sleds = 1000

# Simulation time parameters
dt = 0.01  # Time step size (seconds)
N_t = 3000  # Number of time steps
x_end = 20.0  # End position (meters)

# =============================================================================
# Define the Moguls and Board Functions
# =============================================================================

# Constants for mogul shape
a = 0.25
q = (2 * np.pi) / 4.0
p = (2 * np.pi) / 10.0

def H_func(x, y, b=b_default):
    """
    Function that parametrizes the moguls.
    Returns the height of the slope at position (x, y).
    """
    return -a * x - b * np.cos(p * x) * np.cos(q * y)

def board(t, X_0, b=b_default, theta=0.0):
    """
    Right-hand-side of the equations for a board going down a slope with moguls.
    
    Parameters:
    - t: time variable (unused, kept for compatibility)
    - X_0: array containing [x, y, u, v], initial conditions
    - b: height of the moguls (default is 0.5)
    - theta: inclination angle in degrees (default is 0.0)
    """
    x0, y0, u0, v0 = X_0.copy()
    
    # Convert theta to radians
    theta_rad = np.deg2rad(theta)
    
    # Gravitational acceleration components
    g_parallel = 9.81 * np.sin(theta_rad)
    g_perpendicular = 9.81 * np.cos(theta_rad)
    
    c = 0.5  # Damping coefficient
    
    # Compute derivatives of H
    H = -a * x0 - b * np.cos(p * x0) * np.cos(q * y0)
    H_x = -a + b * p * np.sin(p * x0) * np.cos(q * y0)
    H_y = b * q * np.cos(p * x0) * np.sin(q * y0)
    
    # Force calculation
    F = (g_parallel) / (1 + H_x**2 + H_y**2)
    
    # Acceleration calculations
    dU = -F * H_x - c * u0
    dV = -F * H_y - c * v0
    
    return np.array([u0, v0, dU, dV])

def runge_kutta_step(f, X_0, dt, **kwargs):
    """
    Computes a single Runge-Kutta 4th order step.
    
    Parameters:
    - f: RHS function
    - X_0: current state vector
    - dt: time step
    - **kwargs: additional parameters for the RHS function
    
    Returns:
    - X_new: updated state vector after time dt
    """
    k1 = f(0, X_0, **kwargs) * dt
    k2 = f(0, X_0 + k1 / 2, **kwargs) * dt
    k3 = f(0, X_0 + k2 / 2, **kwargs) * dt
    k4 = f(0, X_0 + k3, **kwargs) * dt
    X_new = X_0 + (k1 + 2*k2 + 2*k3 + k4) / 6
    return X_new

def solver(f, x0, y0, v0, u0, dt, N_t, N, b=b_default, theta=0.0):
    """
    Solver that iterates the Runge-Kutta steps for N sleds over N_t time steps.
    
    Parameters:
    - f: RHS function
    - x0, y0, v0, u0: initial conditions arrays (size N)
    - dt: time step size
    - N_t: number of time steps
    - N: number of sleds
    - b: mogul height
    - theta: inclination angle in degrees
    
    Returns:
    - solution: array of shape (4, N_t+1, N)
                Contains [x, y, u, v] for each sled over time
    """
    solution = np.zeros((4, N_t + 1, N))
    solution[0, 0, :] = x0
    solution[1, 0, :] = y0
    solution[2, 0, :] = u0
    solution[3, 0, :] = v0
    
    for i in range(1, N_t + 1):
        for k in range(N):
            X_0 = solution[:, i - 1, k]
            solution[:, i, k] = runge_kutta_step(f, X_0, dt, b=b, theta=theta)
    
    return solution

# =============================================================================
# Functions to Compute Travel Times and Bifurcation Diagram
# =============================================================================

def compute_travel_times_sorted(solution, dt, x_end=20.0):
    """
    Computes and returns sorted travel times for sleds to reach or exceed x_end.
    
    Parameters:
    - solution: array of shape (4, N_t+1, N)
    - dt: time step size
    - x_end: end position (meters)
    
    Returns:
    - sorted_travel_times: sorted list of travel times (seconds)
    """
    N = solution.shape[2]
    travel_times = []
    for k in range(N):
        x_traj = solution[0, :, k]
        reached = np.where(x_traj >= x_end)[0]
        if reached.size > 0:
            first_reach = reached[0]
            time_reached = first_reach * dt
            travel_times.append(time_reached)
    sorted_travel_times = sorted(travel_times)
    return sorted_travel_times

def bifurcation_travel_time_parallel(theta_range, dt=0.01, N_t=3000, N=1000, x_end=20.0):
    """
    Generates bifurcation data by simulating travel times across varying theta angles.
    
    Parameters:
    - theta_range: array-like, range of inclination angles (degrees)
    - dt: time step size
    - N_t: number of time steps
    - N: number of sleds per theta
    - x_end: end position (meters)
    
    Returns:
    - theta_vals: list of theta values corresponding to each travel time
    - travel_times: list of travel times (seconds)
    """
    def simulate_theta(theta):
        # Initial conditions
        x_init = np.zeros(N)
        y_init = np.random.uniform(-5, 5, N)
        v_init = np.zeros(N)
        u_init = np.full(N, 3.5)
        
        # Run the solver
        sol = solver(board, x_init, y_init, v_init, u_init, dt, N_t, N, b=b_default, theta=theta)
        
        # Compute sorted travel times
        sorted_travel_times = compute_travel_times_sorted(sol, dt, x_end)
        
        # Associate each travel time with the current theta
        theta_repeated = [theta] * len(sorted_travel_times)
        
        return (theta_repeated, sorted_travel_times)
    
    # Parallel execution
    results = Parallel(n_jobs=10)(
        delayed(simulate_theta)(theta) for theta in tqdm(theta_range, desc="Simulating theta values")
    )
    
    # Aggregate results
    theta_vals = []
    travel_times = []
    for theta_repeated, t_times in results:
        theta_vals.extend(theta_repeated)
        travel_times.extend(t_times)
    
    return theta_vals, travel_times

def plot_bifurcation_travel_time(theta_vals, travel_times, title='Bifurcation Diagram: Travel Time vs Inclination Angle'):
    """
    Plots the bifurcation diagram of travel time vs inclination angle.
    
    Parameters:
    - theta_vals: list of theta values (degrees)
    - travel_times: list of travel times (seconds)
    - title: title of the plot
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(theta_vals, travel_times, s=1, color='blue', alpha=0.3)
    plt.xlabel('Theta - Inclination Angle (degrees)', fontsize=14)
    plt.ylabel('Travel Time (seconds)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bifurcation_travel_time_vs_theta.png')
    plt.show()
    
    # Print statistics
    print(f"\nTotal Travel Times: {len(travel_times)}")
    print(f"Minimum Travel Time: {np.min(travel_times):.2f} seconds")
    print(f"Maximum Travel Time: {np.max(travel_times):.2f} seconds")
    print(f"Average Travel Time: {np.mean(travel_times):.2f} seconds")
    print(f"Median Travel Time: {np.median(travel_times):.2f} seconds")

# =============================================================================
# Run the Simulation and Plot Bifurcation Diagram
# =============================================================================

if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)
    
    # Generate bifurcation data
    print("Starting bifurcation simulation: Travel Time vs Inclination Angle...")
    theta_vals, travel_times = bifurcation_travel_time_parallel(
        theta_range=theta_values,
        dt=dt,
        N_t=N_t,
        N=N_sleds,
        x_end=x_end
    )
    print("Simulation completed.")
    
    # Plot the bifurcation diagram
    plot_bifurcation_travel_time(theta_vals, travel_times, title='Bifurcation Diagram: Travel Time vs Inclination Angle')
    
    # Optionally, save the data to CSV for future analysis
    df_bifurcation = pd.DataFrame({
        'theta': theta_vals,
        'travel_time': travel_times
    })
    df_bifurcation.to_csv('bifurcation_travel_time_vs_theta.csv', index=False)
    print("Bifurcation data saved to 'bifurcation_travel_time_vs_theta.csv'.")
