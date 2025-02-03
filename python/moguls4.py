# moguls3.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from joblib import Parallel, delayed  # Import joblib for parallel processing
from tqdm import tqdm  # Optional: For progress bars
import pandas as pd  # Optional: For saving/loading data

# =============================================================================
# Moguls of Chaos
# =============================================================================

# [Existing documentation and code]

# =============================================================================
# Define the End of the Slope
# =============================================================================

x_end = 20.0  # meters  # <-- Ensure this line is added before any use of x_end

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

def board(t, X_0, b=b_default):
    """
    board(t, X_0, b=0.4)

    Right-hand-side of the equations for a board going down a slope with moguls.

    Parameters:
    - t: time variable (unused, kept for compatibility)
    - X_0: array containing [x, y, u, v], initial conditions
    - b: height of the moguls (default is 0.4)
    """
    
    x0 = np.copy(X_0[0])
    y0 = np.copy(X_0[1])
    u0 = np.copy(X_0[2])
    v0 = np.copy(X_0[3])
    
    g = 9.81
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
        
    F = (g + H_xx * u0**2 + 2 * H_xy * u0 * v0 + H_yy * v0**2) / (1 + H_x**2 + H_y**2)
    
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


def solver(f, x0, y0, v0, u0, dt, N_t, N, b=b_default):
    """
    solver(f, x0, y0, v0, u0, dt, N_t, N, b=0.4)

    Function iterates the solution using runge_kutta_step and a RHS function f, 
    for several points or initial conditions given by N.

    Parameters:
    - f: RHS function
    - x0, y0, v0, u0: arrays containing the initial conditions x, y, v, u; with N dimensions.
    - dt: timestep
    - N_t: number of time steps
    - N: number of initial conditions to iterate
    - b: height of the moguls, parameter to pass to f
    """
    
    solution = np.zeros((4, N_t + 1, N))
    solution[0, 0, :] = x0
    solution[1, 0, :] = y0
    solution[2, 0, :] = u0
    solution[3, 0, :] = v0
    
    for i in range(1, N_t + 1):
        for k in range(N):
            x_0_step = solution[:, i - 1, k]
            solution[:, i, k] = runge_kutta_step(f, x_0_step, dt, b=b)
             
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
# Function to Compute Travel Times
# =============================================================================

def compute_travel_times(solution, dt, x_end=20.0):
    """
    Computes the time taken for each sled to reach or exceed x_end.

    Parameters:
    - solution: numpy array of shape (4, N_t + 1, N)
                The simulation results from the solver.
    - dt: float
          The timestep used in the simulation.
    - x_end: float, optional (default=20.0)
             The x-position that signifies the end of the slope.

    Returns:
    - travel_times: list of floats
                    Time taken for each sled to reach x_end.
                    If a sled doesn't reach x_end, np.nan is recorded.
    """
    N = solution.shape[2]      # Number of sleds
    N_t = solution.shape[1]    # Number of time steps
    travel_times = []

    for k in range(N):
        x_traj = solution[0, :, k]  # x positions over time for sled k

        # Find the first index where x >= x_end
        reached = np.where(x_traj >= x_end)[0]

        if reached.size > 0:
            first_reach = reached[0]
            time_reached = first_reach * dt
            travel_times.append(time_reached)
        else:
            travel_times.append(np.nan)  # Sled did not reach the end

    return travel_times

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
# Bifurcation Diagram: Travel Time vs Height of Moguls (Parallelized)
# =============================================================================

def bifurcation_diagram_travel_time_parallel(b_range, dt=0.01, N_t=2000, N=100, x_end=20.0):
    """
    Generate a bifurcation diagram plotting the travel time for each sled
    against varying 'b' (height of moguls) using parallel processing.
    """
    def simulate_single_b(b):
        """
        Simulate sleds for a single 'b' value and compute travel times.
        """
        # Initial conditions
        x_init = np.zeros(N)
        y_init = np.random.uniform(-5, 5, N)
        v_init = np.zeros(N)
        u_init = np.full(N, 3.5)

        # Run the solver with current 'b'
        sol = solver(board, x_init, y_init, v_init, u_init, dt=dt, N_t=N_t, N=N, b=b)

        # Compute travel times
        t_times = compute_travel_times(sol, dt, x_end)

        # Repeat 'b' for each sled
        b_repeated = [b] * N

        return (b_repeated, t_times)

    # Use Parallel with tqdm for progress bars
    results = Parallel(n_jobs=10)(
        delayed(simulate_single_b)(b) for b in tqdm(b_range, desc="Simulating b values")
    )

    # Unpack the results
    b_vals = []
    travel_times = []
    for b_repeated, t_times in results:
        b_vals.extend(b_repeated)
        travel_times.extend(t_times)

    return b_vals, travel_times

def plot_bifurcation_diagram_travel_time(b_vals, travel_times, title='Bifurcation Diagram: Travel Time vs Height of Moguls'):
    """
    Plots the bifurcation diagram with 'b' on the x-axis and travel times on the y-axis.

    Parameters:
    - b_vals: list of floats
              'b' values corresponding to each travel time.
    - travel_times: list of floats
                    Travel times for each sled.
    - title: string, optional (default='Bifurcation Diagram: Travel Time vs Height of Moguls')
             The title of the plot.
    """
    # Convert lists to numpy arrays for easier manipulation
    b_vals = np.array(b_vals)
    travel_times = np.array(travel_times)

    # Filter out sleds that did not reach the end (i.e., travel_times == NaN)
    valid_indices = ~np.isnan(travel_times)
    b_vals_valid = b_vals[valid_indices]
    travel_times_valid = travel_times[valid_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(b_vals_valid, travel_times_valid, s=10, color='green', alpha=0.6, edgecolors='none')
    plt.xlabel('b - Height of Moguls', fontsize=13)
    plt.ylabel('Travel Time (seconds)', fontsize=13)
    plt.title(title, fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bifurcation_diagram_travel_time.png')  # Save the bifurcation diagram
    plt.show()

    # Report how many sleds did not reach the end
    total_sleds = len(travel_times)
    reached = len(travel_times_valid)
    not_reached = total_sleds - reached
    print(f"\nTotal Sleds: {total_sleds}")
    print(f"Sleds Reached the End: {reached}")
    print(f"Sleds Did Not Reach the End: {not_reached}")

# =============================================================================
# Example Simulations (Examples 1-3)
# =============================================================================

# [Existing code for Examples 1-3 remains unchanged]
# Ensure that 'x_end' is defined before these sections

# =============================================================================
# Generate and Plot the Bifurcation Diagram
# =============================================================================

# Define the range of 'b' values for the bifurcation diagram
b_values = np.linspace(0.1, 1.0, 100)  # Adjust the range and number of points as needed

# Generate the bifurcation data using parallel processing
print("Starting bifurcation diagram simulation (Travel Time) with parallel processing...")
start_bifurcation = time.time()
b_vals, travel_times = bifurcation_diagram_travel_time_parallel(
    b_range=b_values,
    dt=0.01,
    N_t=2000,
    N=100,          # Increased number of sleds per 'b' from 10 to 100
    x_end=x_end
)
end_bifurcation = time.time()
print(f"Bifurcation simulation completed in {(end_bifurcation - start_bifurcation)/60:.2f} minutes.")

# Plot the bifurcation diagram
plot_bifurcation_diagram_travel_time(
    b_vals,
    travel_times,
    title='Bifurcation Diagram: Travel Time vs Height of Moguls'
)

print("Bifurcation diagram (Travel Time) plotted successfully.")

# =============================================================================
# Save Bifurcation Data to CSV (Optional)
# =============================================================================

# Create a DataFrame
df = pd.DataFrame({
    'b': b_vals,
    'travel_time': travel_times
})

# Save to CSV
df.to_csv('bifurcation_travel_times_parallel.csv', index=False)
print("Bifurcation data saved to 'bifurcation_travel_times_parallel.csv'.")
