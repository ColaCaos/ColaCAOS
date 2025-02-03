# moguls3.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

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
# Bifurcation Diagram: Travel Time vs Height of Moguls
# =============================================================================

def bifurcation_diagram_travel_time(b_range, dt=0.01, N_t=2000, N=10, transient=1000, x_end=20.0):
    """
    Generate a bifurcation diagram plotting the travel time for each sled
    against varying 'b' (height of moguls).

    Parameters:
    - b_range: array-like
               Array of 'b' values to simulate.
    - dt: float, optional (default=0.01)
           The timestep used in the simulation.
    - N_t: int, optional (default=2000)
           Total number of time steps in the simulation.
    - N: int, optional (default=10)
        Number of sleds per 'b' value.
    - transient: int, optional (default=1000)
                Number of initial time steps to discard to eliminate transients.
    - x_end: float, optional (default=20.0)
             The x-position that signifies the end of the slope.

    Returns:
    - b_vals: list of floats
              'b' values corresponding to each travel time.
    - travel_times: list of floats
                    Travel times for each sled.
    """
    b_vals = []
    travel_times = []

    for idx, b in enumerate(b_range):
        print(f"Simulating for b = {b:.3f} ({idx+1}/{len(b_range)})")

        # Initial conditions
        x_init = np.zeros(N)
        y_init = np.random.uniform(-5, 5, N)  # Adjusted range based on compact domain
        v_init = np.zeros(N)
        u_init = np.full(N, 3.5)

        # Run the solver with current 'b'
        sol = solver(board, x_init, y_init, v_init, u_init, dt=dt, N_t=N_t, N=N, b=b)

        # Compute travel times
        t_times = compute_travel_times(sol, dt, x_end)

        # Append 'b' and travel times
        b_vals.extend([b] * N)
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

# Example 1
n_sleds = 10
n_time = 1000
x_init = np.zeros(n_sleds)
y_init = np.random.rand(n_sleds)
v_init = np.zeros(n_sleds)
u_init = np.full(n_sleds, 3.5)

sol_ex1 = solver(board, x_init, y_init, v_init, u_init, dt=0.01, N_t=n_time, N=n_sleds)

# Plot the trajectories
plot_several(sol_ex1, title='10 Boards')

# Compute travel times for Example 1
travel_times_ex1 = compute_travel_times(sol_ex1, dt=0.01, x_end=x_end)

# Display travel times for Example 1
for idx, t in enumerate(travel_times_ex1):
    if not np.isnan(t):
        print(f"Sled {idx+1}: Reached the end in {t:.2f} seconds.")
    else:
        print(f"Sled {idx+1}: Did not reach the end within the simulation time.")

# Plot histogram for Example 1
plot_travel_times(travel_times_ex1, title='Example 1: Travel Times Histogram')

# =============================================================================
# Example 2
# =============================================================================

n_sleds = 10
n_time = 1000
x_init = np.zeros(n_sleds)
y_init = np.linspace(0.1, 1, n_sleds)
v_init = np.full(n_sleds, 2.0)
u_init = np.full(n_sleds, 4.0)

sol_ex2 = solver(board, x_init, y_init, v_init, u_init, dt=0.01, N_t=n_time, N=n_sleds)

# Plot the trajectories
plot_several(sol_ex2, title='1 cm Between Boards')

# Compute travel times for Example 2
travel_times_ex2 = compute_travel_times(sol_ex2, dt=0.01, x_end=x_end)

# Display travel times for Example 2
for idx, t in enumerate(travel_times_ex2):
    if not np.isnan(t):
        print(f"Sled {idx+1}: Reached the end in {t:.2f} seconds.")
    else:
        print(f"Sled {idx+1}: Did not reach the end within the simulation time.")

# Plot histogram for Example 2
plot_travel_times(travel_times_ex2, title='Example 2: Travel Times Histogram')

# =============================================================================
# Example 3
# =============================================================================

n_sleds = 7
n_time = 2200
x_init = np.zeros(n_sleds)
y_init = np.linspace(0.497, 0.503, n_sleds)
v_init = np.full(n_sleds, 2.0)
u_init = np.full(n_sleds, 4.0)

sol_ex3 = solver(board, x_init, y_init, v_init, u_init, dt=0.01, N_t=n_time, N=n_sleds)

# Plot the trajectories
plot_several(sol_ex3, title='1 mm Between Boards')

# Compute travel times for Example 3
travel_times_ex3 = compute_travel_times(sol_ex3, dt=0.01, x_end=x_end)

# Display travel times for Example 3
for idx, t in enumerate(travel_times_ex3):
    if not np.isnan(t):
        print(f"Sled {idx+1}: Reached the end in {t:.2f} seconds.")
    else:
        print(f"Sled {idx+1}: Did not reach the end within the simulation time.")

# Plot histogram for Example 3
plot_travel_times(travel_times_ex3, title='Example 3: Travel Times Histogram')

print("The system behaves chaotically; slight variations in initial conditions lead to divergent trajectories.")

# =============================================================================
# Sleds down the slope!
# =============================================================================

def sled(t, X_0, b=b_default):
    """
    sled(t, X_0, b=0.4)

    Right-hand-side of the equations for a sled, with constant downward velocity, 
    going down a slope with moguls.

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
    
    dU = 0  # Constant downward velocity
    dV = -F * H_y - c * v0
    
    return np.array([u0, v0, dU, dV])

# =============================================================================
# Solver for Compact Domain
# =============================================================================

def compactor(x, lower_bound, upper_bound):
    """
    compactor(x, lower_bound, upper_bound)

    Function that maps any point outside the interval [lower_bound, upper_bound] 
    to a point in the interval, preserving length. If the point is in the interval,
    it does not change.

    Parameters:
    - x: point to be compacted
    - lower_bound: lower bound of the interval
    - upper_bound: upper bound of the interval
    """
    range_width = upper_bound - lower_bound
    return ((x - lower_bound) % range_width) + lower_bound

def solver_compact(f, x0, y0, v0, u0, dt, N_t, N, b=b_default):
    """
    solver_compact(f, x0, y0, v0, u0, dt, N_t, N, b=0.4)

    Function iterates the solution using runge_kutta_step and a RHS function f, 
    for several points or initial conditions given by N, in a compact domain. 

    Parameters:
    - f: RHS function
    - x0, y0, v0, u0: arrays containing the initial conditions x, y, v, u; with N dimensions.
    - dt: timestep
    - N_t: number of time steps
    - N: number of initial conditions to iterate
    - b: height of the moguls, parameter fixed
    """
    
    solution = np.zeros((3, N_t + 1, N))
    solution[0, 0, :] = x0
    solution[1, 0, :] = y0
    solution[2, 0, :] = v0
    
    for i in range(1, N_t + 1):
        for k in range(N):
            # Insert u0 into the state vector for the sled function
            x_0_step = np.insert(solution[:, i - 1, k], 2, u0[k])
            
            aux = runge_kutta_step(f, x_0_step, dt, b=b)
            # Apply periodic boundary conditions
            solution[0, i, k] = compactor(aux[0], -5, 5)
            solution[1, i, k] = compactor(aux[1], -2, 2)
            solution[2, i, k] = aux[3]
            
    return solution

# =============================================================================
# Plotting Compact Domain
# =============================================================================

x_range_compact = np.linspace(-5, 5, 50)
y_range_compact = np.linspace(-2, 2, 50)
XX_c, YY_c = np.meshgrid(x_range_compact, y_range_compact)

moguls_compact = pits_n_crests(XX_c, YY_c)

plt.figure(figsize=(5, 4))
contour = plt.contourf(YY_c, XX_c, moguls_compact, cmap='viridis')
plt.colorbar(contour, label='Height of Bumps (m)')
plt.ylabel('x - Downslope position (m)', fontsize=13)
plt.xlabel('y - Crossslope position (m)', fontsize=13)
plt.title('Shape of Moguls (Compact Domain)', fontsize=15)
plt.savefig('shape_of_moguls_compact.png')  # Save the figure
plt.show()

# =============================================================================
# Example: Sleds in Compact Domain
# =============================================================================

n_sleds = 1
n_time = 1000
x_init = np.zeros(n_sleds)
y_init = np.random.uniform(-2, 2, n_sleds)
v_init = np.random.uniform(-5, 5, n_sleds)
u_constant = 3.5  # Constant downslope velocity

sled_compact_1 = solver_compact(
    f=sled,
    x0=x_init,
    y0=y_init,
    v0=v_init,
    u0=np.full(n_sleds, u_constant),
    dt=0.01,
    N_t=n_time,
    N=n_sleds
)

plot_several(sled_compact_1, title='1 Sled in Compact Domain')

# Compute travel times for Sleds in Compact Domain
travel_times_sled_compact_1 = compute_travel_times(sled_compact_1, dt=0.01, x_end=x_end)

# Display travel times for Sleds in Compact Domain
for idx, t in enumerate(travel_times_sled_compact_1):
    if not np.isnan(t):
        print(f"Sled {idx+1}: Reached the end in {t:.2f} seconds.")
    else:
        print(f"Sled {idx+1}: Did not reach the end within the simulation time.")

# Plot histogram for Sleds in Compact Domain
plot_travel_times(travel_times_sled_compact_1, title='Sleds in Compact Domain: Travel Times Histogram')

# =============================================================================
# Building a Strange Attractor
# =============================================================================

n_sleds = 1000
n_time = 1000
x_init = np.zeros(n_sleds)
y_init = np.random.uniform(-2, 2, n_sleds)
v_init = np.random.uniform(-5, 5, n_sleds)
u_constant = 3.5  # Constant downslope velocity

start = time.time()
sled_compact_1000 = solver_compact(
    f=sled,
    x0=x_init,
    y0=y_init,
    v0=v_init,
    u0=np.full(n_sleds, u_constant),
    dt=0.01,
    N_t=n_time,
    N=n_sleds
)
end = time.time()
print(f"Elapsed time: {(end - start)/60:.2f} minutes...")

# Plotting Phase Space
time_steps = range(0, 1000, 180)  # Selecting specific time steps

fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(6, 8))
axes = axes.flatten()

for idx, j in enumerate(time_steps):
    dist = round(j * (u_constant * 0.01), 2)
    axes[idx].scatter(
        sled_compact_1000[1, j, :],  # y positions
        sled_compact_1000[2, j, :],  # v velocities
        s=0.5,
        color='k'
    )
    axes[idx].text(1.8, -4.9, f'{dist} m', ha='center')
    axes[idx].set_title(f'Time Step: {j}')

plt.tight_layout()
fig.text(0.5, -0.01, 'y - Cross slope position (m)', ha='center', fontsize=13)
fig.text(-0.01, 0.5, 'v - Cross slope velocity (m/s)', va='center', rotation='vertical', fontsize=13)
plt.show()

print("A strange attractor has been formed, indicating chaotic behavior.")

# =============================================================================
# Bifurcation Diagram: Travel Time vs Height of Moguls
# =============================================================================

def bifurcation_diagram_travel_time(b_range, dt=0.01, N_t=2000, N=10, transient=1000, x_end=20.0):
    """
    Generate a bifurcation diagram plotting the travel time for each sled
    against varying 'b' (height of moguls).

    Parameters:
    - b_range: array-like
               Array of 'b' values to simulate.
    - dt: float, optional (default=0.01)
           The timestep used in the simulation.
    - N_t: int, optional (default=2000)
           Total number of time steps in the simulation.
    - N: int, optional (default=10)
        Number of sleds per 'b' value.
    - transient: int, optional (default=1000)
                Number of initial time steps to discard to eliminate transients.
    - x_end: float, optional (default=20.0)
             The x-position that signifies the end of the slope.

    Returns:
    - b_vals: list of floats
              'b' values corresponding to each travel time.
    - travel_times: list of floats
                    Travel times for each sled.
    """
    b_vals = []
    travel_times = []

    for idx, b in enumerate(b_range):
        print(f"Simulating for b = {b:.3f} ({idx+1}/{len(b_range)})")

        # Initial conditions
        x_init = np.zeros(N)
        y_init = np.random.uniform(-5, 5, N)  # Adjusted range based on compact domain
        v_init = np.zeros(N)
        u_init = np.full(N, 3.5)

        # Run the solver with current 'b'
        sol = solver(board, x_init, y_init, v_init, u_init, dt=dt, N_t=N_t, N=N, b=b)

        # Compute travel times
        t_times = compute_travel_times(sol, dt, x_end)

        # Append 'b' and travel times
        b_vals.extend([b] * N)
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
# Generate and Plot the Bifurcation Diagram
# =============================================================================

# Define the range of 'b' values for the bifurcation diagram
b_values = np.linspace(0.1, 1.0, 100)  # Adjust the range and number of points as needed

# Generate the bifurcation data
print("Starting bifurcation diagram simulation (Travel Time)...")
start_bifurcation = time.time()
b_vals, travel_times = bifurcation_diagram_travel_time(
    b_range=b_values,
    dt=0.01,
    N_t=2000,
    N=10,
    transient=1000,
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
