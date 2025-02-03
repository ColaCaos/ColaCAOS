import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import linregress

# Pendulum parameters
L = 1       # Length of simple pendulum (m)
m = 1       # Mass (kg)
g = 9.81    # Gravitational acceleration (m/s^2)

# Time parameters
tmax, dt = 100, 0.01
t = np.arange(0, tmax + dt, dt)

# Equations of motion for a simple pendulum with damping
def deriv_simple_pendulum(y, t, L, m, g, b):
    theta, omega = y
    theta_dot = omega
    omega_dot = -(g / L) * np.sin(theta) - (b / m) * omega
    return theta_dot, omega_dot

# Initial conditions
y0_simple = [np.pi / 4, 0]  # Initial angle (rad) and angular velocity (rad/s)
b = 0.05  # Friction coefficient

# Number of simulations
num_simulations = 1000
variation = 0.0001  # 0.01% variation

# Arrays to store results
initial_energies_simple = []
total_traveled_distances_simple = []

# Function to calculate the total mechanical energy of a simple pendulum
def calc_total_energy_simple(theta, omega, L, m, g):
    T = 0.5 * m * (L * omega)**2  # Kinetic energy
    V = -m * g * L * np.cos(theta)  # Potential energy
    return T + V

# Simulate for a larger number of initial conditions
for i in range(num_simulations):
    varied_y0 = y0_simple.copy()
    varied_y0[0] += i * variation  # Slight variation in theta

    # Solve the equations of motion with damping
    y_sim_simple = odeint(deriv_simple_pendulum, varied_y0, t, args=(L, m, g, b))
    theta_sim, omega_sim = y_sim_simple[:, 0], y_sim_simple[:, 1]

    # Convert to Cartesian coordinates
    x_sim = L * np.sin(theta_sim)
    y_sim = -L * np.cos(theta_sim)

    # Compute initial energy
    initial_energy = calc_total_energy_simple(varied_y0[0], varied_y0[1], L, m, g)
    initial_energies_simple.append(initial_energy)

    # Compute total traveled distance
    total_distance = np.sum(np.sqrt(np.diff(x_sim)**2 + np.diff(y_sim)**2))
    total_traveled_distances_simple.append(total_distance)

# Perform linear regression on the new data
slope, intercept, r_value, p_value, std_err = linregress(initial_energies_simple, total_traveled_distances_simple)

# Scatter plot with linear fit
plt.figure(figsize=(10, 6))
plt.scatter(initial_energies_simple, total_traveled_distances_simple, c='b', s=1, label='Simulations')
plt.plot(initial_energies_simple, intercept + slope * np.array(initial_energies_simple), 'r--',
         label=f'Linear Fit (R²={r_value**2:.4f})')
plt.title('Traveled Distance vs Initial Energy (Simple Pendulum, 1000 Runs)')
plt.xlabel('Initial Energy (J)')
plt.ylabel('Total Traveled Distance (m)')
plt.grid()
plt.legend()
plt.show()

# Plot center of mass position vs time for simple pendulum
plt.figure(figsize=(10, 6))
x_com = x_sim  # For a simple pendulum, x_com is x_sim
y_com = y_sim  # For a simple pendulum, y_com is y_sim

plt.plot(t, x_com, label='X Center of Mass', color='blue')
plt.plot(t, y_com, label='Y Center of Mass', color='green')

plt.title('Center of Mass Position vs Time (Simple Pendulum)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid()
plt.legend()
plt.show()
