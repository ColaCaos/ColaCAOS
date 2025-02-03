import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import linregress

# Pendulum parameters
L1, L2 = 1, 1  # Lengths (m)
m1, m2 = 1, 1  # Masses (kg)
g = 9.81       # Gravitational acceleration (m/s^2)

# Time parameters
tmax, dt = 100, 0.01
t = np.arange(0, tmax + dt, dt)

# Function to calculate the total mechanical energy of the system
def calc_total_energy(theta1, z1, theta2, z2, L1, L2, m1, m2, g):
    """Calculate the total energy (kinetic + potential) of the pendulum."""
    # Kinetic energy
    T = 0.5 * m1 * (L1 * z1)**2 + \
        0.5 * m2 * ((L1 * z1)**2 + (L2 * z2)**2 + 2 * L1 * L2 * z1 * z2 * np.cos(theta1 - theta2))
    # Potential energy
    V = -(m1 + m2) * L1 * g * np.cos(theta1) - m2 * L2 * g * np.cos(theta2)
    return T + V

# Equations of motion with damping
def deriv_with_damping(y, t, L1, L2, m1, m2, b):
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1dot = z1
    z1dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1**2 * c + L2 * z2**2) -
             (m1 + m2) * g * np.sin(theta1) - b * z1) / L1 / (m1 + m2 * s**2)
    theta2dot = z2
    z2dot = ((m1 + m2) * (L1 * z1**2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
             m2 * L2 * z2**2 * s * c - b * z2) / L2 / (m1 + m2 * s**2)
    return theta1dot, z1dot, theta2dot, z2dot

# Initial conditions
y0 = [3 * np.pi / 7, 0, 3 * np.pi / 4, 0]
b = 0.05  # Friction coefficient

# Number of simulations
num_simulations = 1000
variation = 0.0001  # 0.01% variation

# Arrays to store results
initial_energies = []
total_traveled_distances = []

# Simulate for a larger number of initial conditions
for i in range(num_simulations):
    varied_y0 = y0.copy()
    varied_y0[0] += i * variation  # Slight variation in theta1

    # Solve the equations of motion with damping
    y_sim_damped = odeint(deriv_with_damping, varied_y0, t, args=(L1, L2, m1, m2, b))
    theta1_sim, theta2_sim = y_sim_damped[:, 0], y_sim_damped[:, 2]

    # Convert to Cartesian coordinates
    x1_sim = L1 * np.sin(theta1_sim)
    y1_sim = -L1 * np.cos(theta1_sim)
    x2_sim = x1_sim + L2 * np.sin(theta2_sim)
    y2_sim = y1_sim - L2 * np.cos(theta2_sim)

    # Compute initial energy
    initial_energy = calc_total_energy(varied_y0[0], varied_y0[1], varied_y0[2], varied_y0[3], L1, L2, m1, m2, g)
    initial_energies.append(initial_energy)

    # Compute total traveled distance
    total_distance = np.sum(np.sqrt(np.diff(x2_sim)**2 + np.diff(y2_sim)**2))
    total_traveled_distances.append(total_distance)

# Perform linear regression on the new data
slope, intercept, r_value, p_value, std_err = linregress(initial_energies, total_traveled_distances)

# Scatter plot with linear fit
plt.figure(figsize=(10, 6))
plt.scatter(initial_energies, total_traveled_distances, c='b', s=1, label='Simulations')
plt.plot(initial_energies, intercept + slope * np.array(initial_energies), 'r--',
         label=f'Linear Fit (R²={r_value**2:.4f})')
plt.title('Traveled Distance vs Initial Energy (1000 Runs)')
plt.xlabel('Initial Energy (J)')
plt.ylabel('Total Traveled Distance (m)')
plt.grid()
plt.legend()
plt.show()

# Display statistical analysis results
print("Linear Regression Results:")
print(f"  Slope: {slope}")
print(f"  Intercept: {intercept}")
print(f"  R-squared: {r_value**2}")
print(f"  P-value: {p_value}")
print(f"  Standard Error: {std_err}")
