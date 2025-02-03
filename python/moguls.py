# moguls.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Note: Removed `%matplotlib inline` as it's a Jupyter magic command.

# =============================================================================
# Moguls of Chaos
# 
# This example was taken from the book **The Essence of Chaos** by *Edward Lorenz*. 
# The aim of the model he presented in that chapter is to show how we can construct 
# parsimonious models from real case phenomena which exhibit chaos. 
# 
# The model consists of a board that goes down a regularly-bumpy slope or moguls 
# without any control of the direction whatsoever. The dynamical system of this board 
# is described by the system of ODE shown below
# 
# $$\frac{dx}{dt} = U, \quad \frac{dy}{dt} = V, \quad \frac{dz}{dt} = W.$$
# 
# \begin{align}
# \frac{du}{dt} &= -F \partial_x H - cu, \\
# \frac{dv}{dt} &= -F \partial_y H - cv, \\
# \frac{dw}{dt} &= -g + F - cw.
# \end{align}
# 
# $\vec{x} = (x,y,z)$ are the spatial coordinates and $\vec{u} = (u,v,w)$ are the 
# components of the velocity of the board. And $H$ is the shape of the slope that 
# is parameterized by the following expression: 
# 
# $$H(x,y) = -ax - b \cos(px) \cos(qy).$$
# 
# $a$ is the angle of the slope, $b$ is the height of the bumps, and $q$ and $p$ 
# are spatial frequencies.
# =============================================================================

# =============================================================================
# Plotting the Shape of the Moguls
# 
# We can plot the shape of the moguls to help us understand better the problem. 
# For this, we use the parameters that Lorenz suggest in his book, in which $a$ is 
# the angle of the slope, $b$ is the depth of the moguls, and $p$ and $q$ are the 
# spatial frequencies of the moguls in the $x$ and $y$ directions.
# =============================================================================

# Parameters
a = 0.25
b = 0.4
q = (2 * np.pi) / 4.0
p = (2 * np.pi) / 10.0

def H_func(x, y):
    """
    H_func(x, y)

    Function that parametrizes the moguls. 
    It takes a given x and y position as arguments, and returns 
    the height of the slope.
    """
    return -a * x - b * np.cos(p * x) * np.cos(q * y)

def pits_n_crests(x, y):
    """
    pits_n_crests(x, y)
    
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
# 
# Our first goal is to solve the ODE system for a given initial condition using 
# a numerical integration method. For this, we declare a function containing 
# the right-hand side terms of the ODE system that we want to solve.
# =============================================================================

def board(t, X_0):
    """
    board(t, X_0)

    Right-hand-side of the equations for a board going down a slope with moguls.

    X_0 is the set of initial conditions containing [x, y, u, v], in that order.
    t is the optional parameter.
    """
    
    x0 = np.copy(X_0[0])
    y0 = np.copy(X_0[1])
    u0 = np.copy(X_0[2])
    v0 = np.copy(X_0[3])
    
    g = 9.81
    c = 0.5
    a_local = 0.25
    b_local = 0.5
    p_local = (2 * np.pi) / 10.0
    q_local = (2 * np.pi) / 4.0
    
    H = -a_local * x0 - b_local * np.cos(p_local * x0) * np.cos(q_local * y0) 
    H_x = -a_local + b_local * p_local * np.sin(p_local * x0) * np.cos(q_local * y0)
    H_xx = b_local * p_local**2 * np.cos(p_local * x0) * np.cos(q_local * y0)
    H_y = b_local * q_local * np.cos(p_local * x0) * np.sin(q_local * y0)
    H_yy = b_local * q_local**2 * np.cos(p_local * x0) * np.cos(q_local * y0)
    H_xy = -b_local * q_local * p_local * np.sin(p_local * x0) * np.sin(q_local * y0)
        
    F = (g + H_xx * u0**2 + 2 * H_xy * u0 * v0 + H_yy * v0**2) / (1 + H_x**2 + H_y**2)
    
    dU = -F * H_x - c * u0
    dV = -F * H_y - c * v0
    
    return np.array([u0, v0, dU, dV])

# =============================================================================
# Runge-Kutta 4th Order Integrator
# =============================================================================

def runge_kutta_step(f, x0, dt, t=None):
    """
    runge_kutta_step(f, x0, dt, t=None)

    Computes a step using the Runge-Kutta 4th order method.

    f is a RHS function, x0 is an array of variables to solve, 
    dt is the timestep, and t corresponds to an extra parameter of
    the RHS.
    """

    k1 = f(t, x0) * dt
    k2 = f(t, x0 + k1 / 2) * dt
    k3 = f(t, x0 + k2 / 2) * dt
    k4 = f(t, x0 + k3) * dt
    x_new = x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return x_new

def solver(f, x0, y0, v0, u0, dt, N_t, N, b=0.5):
    """
    solver(f, x0, y0, v0, u0, dt, N_t, N, b=0.5)

    Function iterates the solution using runge_kutta_step and a RHS function f, 
    for several points or initial conditions given by N.

    Parameters:
    - f: RHS function
    - x0, y0, v0, u0: arrays containing the initial conditions x, y, v, u; with N dimensions.
    - dt: timestep
    - N_t: number of time steps
    - N: number of initial conditions to iterate
    - b: height of the moguls, parameter fixed
    """
    
    solution = np.zeros((4, N_t + 1, N))
    solution[0, 0, :] = x0
    solution[1, 0, :] = y0
    solution[2, 0, :] = u0
    solution[3, 0, :] = v0
    
    for i in range(1, N_t + 1):
        for k in range(N):
            x_0_step = solution[:, i - 1, k]
            solution[:, i, k] = runge_kutta_step(f, x_0_step, dt, b)
             
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
# Example 1
# 
# Initial conditions: x = 0.0, y = [0,1] (randomly chosen), U = 3.5 and V = 0.
# =============================================================================

n_sleds = 10
n_time = 1000
x_init = np.zeros(n_sleds)
y_init = np.random.rand(n_sleds)
v_init = np.zeros(n_sleds)
u_init = np.full(n_sleds, 3.5)

sol_ex1 = solver(board, x_init, y_init, v_init, u_init, dt=0.01, N_t=n_time, N=n_sleds)

plot_several(sol_ex1, title='10 Boards')

# =============================================================================
# Example 2
# 
# Initial conditions: x = 0.0, y = [0,1] scattered evenly, U = 4.0 and V = 2.0.
# =============================================================================

n_sleds = 10
n_time = 1000
x_init = np.zeros(n_sleds)
y_init = np.linspace(0.1, 1, n_sleds)
v_init = np.full(n_sleds, 2.0)
u_init = np.full(n_sleds, 4.0)

sol_ex2 = solver(board, x_init, y_init, v_init, u_init, dt=0.01, N_t=n_time, N=n_sleds)

plot_several(sol_ex2, title='1 cm Between Boards')

# =============================================================================
# Example 3
# 
# Initial conditions just spaced 1mm apart from 0.497m to 0.503m.
# =============================================================================

n_sleds = 7
n_time = 2200
x_init = np.zeros(n_sleds)
y_init = np.linspace(0.497, 0.503, n_sleds)
v_init = np.full(n_sleds, 2.0)
u_init = np.full(n_sleds, 4.0)

sol_ex3 = solver(board, x_init, y_init, v_init, u_init, dt=0.01, N_t=n_time, N=n_sleds)

plot_several(sol_ex3, title='1 mm Between Boards')

print("The system behaves chaotically; slight variations in initial conditions lead to divergent trajectories.")

# =============================================================================
# Sleds down the slope!
# 
# To reduce the dimensions of our system, we can make the downslope velocity 
# constants, or in the real world, by equipping our board with some brakes and 
# an engine so it can maintain a constant velocity while going down the pits or 
# up the bumps. In a mathematical sense, we say that ∂t U = 0.
# =============================================================================

def sled(t, X_0):
    """
    sled(t, X_0)

    Right-hand-side of the equations for a sled, with constant downward velocity, 
    going down a slope with moguls.

    X_0 is the set of initial conditions containing [x, y, u, v], in that order.
    t is the optional parameter.
    """

    x0 = np.copy(X_0[0])
    y0 = np.copy(X_0[1])
    u0 = np.copy(X_0[2])
    v0 = np.copy(X_0[3])
    
    g = 9.81
    c = 0.5
    a_local = 0.25
    b_local = 0.5  # Ensure 'b' is defined
    p_local = (2 * np.pi) / 10.0
    q_local = (2 * np.pi) / 4.0
    
    H = -a_local * x0 - b_local * np.cos(p_local * x0) * np.cos(q_local * y0) 
    H_x = -a_local + b_local * p_local * np.sin(p_local * x0) * np.cos(q_local * y0)
    H_xx = b_local * p_local**2 * np.cos(p_local * x0) * np.cos(q_local * y0)
    H_y = b_local * q_local * np.cos(p_local * x0) * np.sin(q_local * y0)
    H_yy = b_local * q_local**2 * np.cos(p_local * x0) * np.cos(q_local * y0)
    H_xy = -b_local * q_local * p_local * np.sin(p_local * x0) * np.sin(q_local * y0)
        
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

def solver_compact(f, x0, y0, v0, u0, dt, N_t, N, b=0.5):
    """
    solver_compact(f, x0, y0, v0, u0, dt, N_t, N, b=0.5)

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
            
            aux = runge_kutta_step(f, x_0_step, dt, b)
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

# =============================================================================
# Plotting Phase Space
# =============================================================================

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
# References
# 
# - Lorenz, Edward N. *The Essence of Chaos*. Univ. of Washington Press, 1993.
# 
# ----
# Code by Claudio Pierard. *Master Environmental Fluid Mechanics*, Université Grenoble Alpes. January 2020.
# =============================================================================
