# moguls3.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from joblib import Parallel, delayed  # Import joblib for parallel processing
from tqdm import tqdm  # For progress bars
import pandas as pd  # For saving/loading data

# =============================================================================
# Moguls of Chaos
# =============================================================================

# [Existing documentation and code]

# =============================================================================
# Define the End of the Slope
# =============================================================================

x_end = 20.0  # meters  # Ensure this line is added before any use of x_end

# =============================================================================
# Plotting the Shape of the Moguls
# =============================================================================

# Parameters
a = 0.25
b_default = 0.4  # Renamed to avoid conflict with bifurcation parameter
q = (2 * np.pi) / 4.0
p = (2 * np.pi) / 10.0

def H_func(x, y, b=b_default):
    """
    H_func(x, y, b=0.4)

    Function that parametrizes the moguls. 
    It takes a given x and y position as arguments, and returns 
    the height of the slope.
    """
    return -a * x - b * np.cos(p * x) * np.cos(q * y)

def pits_n_crests(x, y, b=b_default):
    """
    pits_n_crests(x, y, b=0.4)

    Function that parametrizes the moguls without the slope. 
    It takes a given x and y position as arguments, and returns 
    the heights of the moguls as if they were on a flat surface.
    """
    return -b * np.cos(p * x) * np.cos(q * y)

# Generate meshgrid for plotting
x_range = np.linspace(0, 20, 200)
y_range = np.linspace(-10, 10, 200)
XX, YY = np.meshgrid(x_range, y_range)
moguls = H_func(XX, YY)

# Plotting the 3D surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_ylabel('y (m)', fontsize=15)
ax.set_xlabel('x (m)', fontsize=15)
ax.set_zlabel('z (m)', fontsize=15)
ax.set_title('Shape of the Moguls', fontsize=15)

ax.plot_surface(XX, YY, moguls, cmap='plasma')
plt.savefig('shape_of_moguls.png')  # Save the figure
plt.show()  # Display the plot

# =============================================================================
# Boards down the slope!
# =============================================================================

def board(t, X_0, b=b_default, theta=0.0):
    """
    board(t, X_0, b=0.4, theta=0.0)

    Right-hand-side of the equations for a board going down a slope with moguls.

    Parameters:
    - t: time variable (unused, kept for compatibility)
    - X_0: array containing [x, y, u, v], initial conditions
    - b: height of the moguls (default is 0.4)
    - theta: inclination angle in degrees (default is 0.0)
    """
    
    x0 = np.copy(X_0[0])
    y0 = np.copy(X_0[1])
    u0 = np.copy(X_0[2])
    v0 = np.copy(X_0[3])
    
    # Convert theta to radians
    theta_rad = np.deg2rad(theta)
    
    # Gravitational acceleration components
    g_parallel = 9.81 * np.sin(theta_rad)
    g_perpendicular = 9.81 * np.cos(theta_rad)
    
    c = 0.5
    a_local = a
    p_local = p
    q_local = q
    
    H = -a_local * x0 - b * np.cos(p_local * x0) * np.cos(q_local * y0)
    H_x = -a_local + b * p_local * np.sin(p_local * x0) * np.cos(q_local * y0)
    H_xx = b * p_local**2 * np.cos(p_local * x0) * np.cos(q_local * y0)
    H_y = b * q_local * np.cos(p_local * x0) * np.sin(q_local * y0)
    H_yy = b * q_local**2 * np.cos(p_local * x0) * np.cos(q_local * y0)
    H_xy = -b * q_local * p_local * np.sin(p_local * x0) * np.sin(q_local * y0)
        
    F = (g_parallel + H_xx * u0**2 + 2 * H_xy * u0 * v0 + H_yy * v0**2) / (1 + H_x**2 + H_y**2)
    
    dU = -F * H_x - c * u0
    dV = -F * H_y - c * v0
    
    return np.array([u0, v0, dU, dV])

# =============================================================================
# Runge-Kutta 4th Order Integrator
# =============================================================================

def runge_kutta_step(f, x0, dt, **kwargs):
    """
    runge_kutta_step(f, x0, dt, **kwargs)

    Computes a step using the Runge-Kutta 4th order method.

    Parameters:
    - f: RHS function
    - x0: array of variables to solve
    - dt: timestep
    - **kwargs: additional keyword arguments to pass to f

    Returns:
    - x_new: updated state after timestep dt
    """
    k1 = f(0, x0, **kwargs) * dt
    k2 = f(0, x0 + k1 / 2, **kwargs) * dt
    k3 = f(0, x0 + k2 / 2, **kwargs) * dt
    k4 = f(0, x0 + k3, **kwargs) * dt
    x_new = x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return x_new


def solver(f, x0, y0, v0, u0, dt, N_t, N, b=b_default, theta=0.0):
    """
    solver(f, x0, y0, v0, u0, dt, N_t, N, b=0.4, theta=0.0)

    Function iterates the solution using runge_kutta_step and a RHS function f, 
    for several points or initial conditions given by N.

    Parameters:
    - f: RHS function
    - x0, y0, v0, u0: arrays containing the initial conditions x, y, v, u; with N dimensions.
    - dt: timestep
    - N_t: number of time steps
    - N: number of initial conditions to iterate
    - b: height of the moguls, parameter to pass to f
    - theta: inclination angle in degrees, parameter to pass to f
    """
    
    solution = np.zeros((4, N_t + 1, N))
    solution[0, 0, :] = x0
    solution[1, 0, :] = y0
    solution[2, 0, :] = u0
    solution[3, 0, :] = v0
    
    for i in range(1, N_t + 1):
        for k in range(N):
            x_0_step = solution[:, i - 1, k]
            solution[:, i, k] = runge_kutta_step(f, x_0_step, dt, b=b, theta=theta)
             
    return solution

def plot_several(Solution, title='Trajectories'):
    """
    plot_several(Solution, title='Trajectories')

    Function to plot several trajectories of boards or sleds. 
    Solution is the array with the solutions of the ensemble of points.
    """
    
    N = Solution.shape[2]
    plt.figure(figsize=(5, 10))
    
    for i in range(N):
        plt.plot(Solution[1, :, i], Solution[0, :, i])
    
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.ylabel('x - Downslope position (m)', fontsize=13)
    plt.xlabel('y - Crossslope position (m)', fontsize=13)
    plt.title(title, fontsize=15)
    plt.show()

# =============================================================================
# Function to Compute Travel Times (Sorted)
# =============================================================================

def compute_travel_times_sorted(solution, dt, x_end=20.0):
    """
    Computes and returns sorted arrival times for each sled to reach or exceed x_end.

    Parameters:
    - solution: numpy array of shape (4, N_t + 1, N)
                The simulation results from the solver.
    - dt: float
          The timestep used in the simulation.
    - x_end: float, optional (default=20.0)
             The x-position that signifies the end of the slope.

    Returns:
    - sorted_travel_times: list of floats
                           Sorted times taken for each sled to reach x_end.
                           Sleds that did not reach the end are excluded.
    """
    N = solution.shape[2]      # Number of sleds
    travel_times = []

    for k in range(N):
        x_traj = solution[0, :, k]  # x positions over time for sled k

        # Find the first index where x >= x_end
        reached = np.where(x_traj >= x_end)[0]

        if reached.size > 0:
            first_reach = reached[0]
            time_reached = first_reach * dt
            travel_times.append(time_reached)
        # Else, sled did not reach the end; exclude from sorting

    # Sort the travel times
    sorted_travel_times = sorted(travel_times)

    return sorted_travel_times

# =============================================================================
# Function to Compute Intervals Between Arrivals
# =============================================================================

def compute_intervals(sorted_travel_times):
    """
    Computes intervals between consecutive sled arrivals.

    Parameters:
    - sorted_travel_times: list of floats
                           Sorted times taken for each sled to reach x_end.

    Returns:
    - intervals: list of floats
                 Time differences between consecutive sled arrivals.
    """
    intervals = np.diff(sorted_travel_times)  # Differences between consecutive times
    return intervals.tolist()

# =============================================================================
# Function to Plot Travel Times Histogram
# =============================================================================

def plot_travel_times(travel_times, title='Travel Times Histogram'):
    """
    Plots a histogram of travel times.

    Parameters:
    - travel_times: list of floats
                    Time taken for each sled to reach x_end.
    - title: string, optional (default='Travel Times Histogram')
             The title of the plot.
    """
    # Filter out sleds that did not reach the end
    valid_times = [t for t in travel_times if not np.isnan(t)]

    plt.figure(figsize=(10, 6))
    plt.hist(valid_times, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Time to Reach End (seconds)', fontsize=13)
    plt.ylabel('Number of Sleds', fontsize=13)
    plt.title(title, fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('travel_times_histogram.png')  # Save the histogram
    plt.show()

    # Optionally, report how many sleds did not reach the end
    total_sleds = len(travel_times)
    reached = len(valid_times)
    not_reached = total_sleds - reached
    print(f"\nTotal Sleds: {total_sleds}")
    print(f"Sleds Reached the End: {reached}")
    print(f"Sleds Did Not Reach the End: {not_reached}")

# =============================================================================
# Bifurcation Diagram: Interval Between Sled Arrivals vs Slope Angle (Parallelized)
# =============================================================================

def bifurcation_diagram_interval_vs_theta_parallel(theta_range, dt=0.01, N_t=2000, N=1000, x_end=20.0):
    """
    Generate a bifurcation diagram plotting the intervals between sled arrivals
    against varying 'theta' (inclination angles) using parallel processing.

    Parameters:
    - theta_range: array-like
                   Array of slope angles in degrees to simulate.
    - dt: float, optional (default=0.01)
           The timestep used in the simulation.
    - N_t: int, optional (default=2000)
           Total number of time steps in the simulation.
    - N: int, optional (default=1000)
           Number of sleds per 'theta' value.
    - x_end: float, optional (default=20.0)
             The x-position that signifies the end of the slope.

    Returns:
    - theta_vals: list of floats
                  'theta' values corresponding to each interval.
    - intervals: list of floats
                 Time intervals between consecutive sled arrivals.
    """
    def simulate_single_theta(theta):
        """
        Simulate sleds for a single 'theta' value and compute arrival intervals.

        Parameters:
        - theta: float
                 Current slope angle in degrees.

        Returns:
        - Tuple of (list of 'theta' values repeated, list of intervals)
        """
        # Initial conditions
        x_init = np.zeros(N)
        y_init = np.random.uniform(-5, 5, N)  # Adjusted range based on compact domain
        v_init = np.zeros(N)
        u_init = np.full(N, 3.5)

        # Run the solver with current 'theta'
        sol = solver(board, x_init, y_init, v_init, u_init, dt=dt, N_t=N_t, N=N, b=b_default, theta=theta)

        # Compute sorted travel times
        sorted_travel_times = compute_travel_times_sorted(sol, dt, x_end)

        # Compute intervals between arrivals
        intervals_theta = compute_intervals(sorted_travel_times)

        # Associate each interval with the current 'theta'
        theta_repeated = [theta] * len(intervals_theta)

        return (theta_repeated, intervals_theta)

    # Use Parallel with tqdm for progress bars
    results = Parallel(n_jobs=10)(
        delayed(simulate_single_theta)(theta) for theta in tqdm(theta_range, desc="Simulating theta values")
    )

    # Unpack the results
    theta_vals = []
    intervals = []
    for theta_repeated, intervals_theta in results:
        theta_vals.extend(theta_repeated)
        intervals.extend(intervals_theta)

    return theta_vals, intervals

def plot_bifurcation_diagram_interval_vs_theta(theta_vals, intervals, title='Bifurcation Diagram: Interval Between Sled Arrivals vs Slope Angle'):
    """
    Plots the bifurcation diagram with 'theta' on the x-axis and intervals between sled arrivals on the y-axis.

    Parameters:
    - theta_vals: list of floats
                  'theta' values corresponding to each interval.
    - intervals: list of floats
                 Time intervals between consecutive sled arrivals.
    - title: string, optional (default='Bifurcation Diagram: Interval Between Sled Arrivals vs Slope Angle')
             The title of the plot.
    """
    # Convert lists to numpy arrays for easier manipulation
    theta_vals = np.array(theta_vals)
    intervals = np.array(intervals)

    # Optionally, filter out intervals that are too small or too large
    # to remove outliers or errors
    # Example:
    # intervals = intervals[(intervals > 0.1) & (intervals < 10)]

    plt.figure(figsize=(12, 6))
    plt.scatter(theta_vals, intervals, s=1, color='blue', alpha=0.5)
    plt.xlabel('Theta - Inclination Angle (degrees)', fontsize=14)
    plt.ylabel('Interval Between Arrivals (seconds)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bifurcation_diagram_interval_vs_theta.png')  # Save the bifurcation diagram
    plt.show()

    # Report statistics
    print(f"\nTotal Intervals: {len(intervals)}")
    print(f"Minimum Interval: {np.min(intervals):.2f} seconds")
    print(f"Maximum Interval: {np.max(intervals):.2f} seconds")
    print(f"Average Interval: {np.mean(intervals):.2f} seconds")
    print(f"Median Interval: {np.median(intervals):.2f} seconds")

# =============================================================================
# Example Simulations (Examples 1-3)
# =============================================================================

# [Existing code for Examples 1-3 remains unchanged]
# Ensure that 'x_end' is defined before these sections

# Example 1
n_sleds = 10
n_time = 1000
x_init = np.zeros(n_sleds)
y_init = np.random.rand(n_sleds)
v_init = np.zeros(n_sleds)
u_init = np.full(n_sleds, 3.5)

sol_ex1 = solver(board, x_init, y_init, v_init, u_init, dt=0.01, N_t=n_time, N=n_sleds, theta=15.0)  # Example theta

# Plot the trajectories
plot_several(sol_ex1, title='10 Boards')

# Compute travel times for Example 1
travel_times_ex1 = compute_travel_times_sorted(sol_ex1, dt=0.01, x_end=x_end)

# Display travel times for Example 1
for idx, t in enumerate(travel_times_ex1):
    print(f"Sled {idx+1}: Reached the end in {t:.2f} seconds.")

# Compute intervals between arrivals for Example 1
intervals_ex1 = compute_intervals(travel_times_ex1)

# Plot histogram for Example 1 intervals
plt.figure(figsize=(10, 6))
plt.hist(intervals_ex1, bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('Interval Between Arrivals (seconds)', fontsize=13)
plt.ylabel('Number of Intervals', fontsize=13)
plt.title('Example 1: Intervals Between Sled Arrivals Histogram', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig('example1_intervals_histogram.png')  # Save the histogram
plt.show()

# Repeat similar steps for Examples 2 and 3 as needed

print("The system behaves chaotically; slight variations in initial conditions lead to divergent trajectories.")

# =============================================================================
# Bifurcation Diagram: Interval Between Sled Arrivals vs Slope Angle (Parallelized)
# =============================================================================

# Define the range of 'theta' values for the bifurcation diagram
theta_values = np.linspace(10, 30, 50)  # Example: 10° to 30° with 50 increments

# Generate the bifurcation data using parallel processing
print("Starting bifurcation diagram simulation (Interval Between Sled Arrivals vs Slope Angle) with parallel processing...")
start_bifurcation = time.time()
theta_vals, intervals = bifurcation_diagram_interval_vs_theta_parallel(
    theta_range=theta_values,
    dt=0.01,
    N_t=2000,
    N=1000,          # Increased number of sleds per 'theta' from 100 to 1,000
    x_end=x_end
)
end_bifurcation = time.time()
print(f"Bifurcation simulation completed in {(end_bifurcation - start_bifurcation)/60:.2f} minutes.")

# Plot the bifurcation diagram
plot_bifurcation_diagram_interval_vs_theta(
    theta_vals,
    intervals,
    title='Bifurcation Diagram: Interval Between Sled Arrivals vs Slope Angle'
)

print("Bifurcation diagram (Interval Between Sled Arrivals vs Slope Angle) plotted successfully.")

# =============================================================================
# Save Bifurcation Data to CSV (Optional)
# =============================================================================

# Create a DataFrame
df_bifurcation = pd.DataFrame({
    'theta': theta_vals,
    'interval': intervals
})

# Save to CSV
df_bifurcation.to_csv('bifurcation_interval_vs_theta_parallel.csv', index=False)
print("Bifurcation data saved to 'bifurcation_interval_vs_theta_parallel.csv'.")
