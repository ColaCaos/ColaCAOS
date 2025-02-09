import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
g = 9.81  # Gravity (m/s^2)
L1, L2 = 1.0, 1.0  # Lengths of pendulum arms (m)
m1, m2 = 1.0, 1.0  # Masses of pendulums (kg)
dt = 0.01  # Time step (s)
num_steps = 300  # Number of simulation steps
grid_size = 360  # Grid resolution

def derivatives(state):
    theta1, z1, theta2, z2 = state
    delta = theta2 - theta1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    den2 = (L2 / L1) * den1
    dz1 = (
        (m2 * L1 * z1 ** 2 * np.sin(delta) * np.cos(delta) +
         m2 * g * np.sin(theta2) * np.cos(delta) +
         m2 * L2 * z2 ** 2 * np.sin(delta) -
         (m1 + m2) * g * np.sin(theta1)) / den1
    )
    dz2 = (
        (- L2 / L1 * m2 * L2 * z2 ** 2 * np.sin(delta) * np.cos(delta) +
         (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
         (m1 + m2) * L1 * z1 ** 2 * np.sin(delta) -
         (m1 + m2) * g * np.sin(theta2)) / den2
    )
    return np.array([z1, dz1, z2, dz2])

# Initialize the state grid
theta1_range = np.linspace(-np.pi, np.pi, grid_size)
theta2_range = np.linspace(-np.pi, np.pi, grid_size)
state_grid = np.zeros((grid_size, grid_size, 4))
state_grid[:, :, 0] = theta1_range[:, None]  # Theta1
state_grid[:, :, 2] = theta2_range[None, :]  # Theta2

def get_color(theta1, theta2):
    if theta1 >= 0 and theta2 >= 0:
        return [255, 255, 0]  # Yellow
    elif theta1 < 0 and theta2 >= 0:
        return [0, 0, 255]  # Blue
    elif theta1 < 0 and theta2 < 0:
        return [0, 255, 0]  # Green
    else:
        return [255, 0, 0]  # Red

# Initialize the image
image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
for i in range(grid_size):
    for j in range(grid_size):
        image[i, j] = get_color(state_grid[i, j, 0], state_grid[i, j, 2])

# Create animation
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(image, origin='lower', extent=[-180, 180, -180, 180])
ax.set_xlabel(r'$\theta_1$ (degrees)')
ax.set_ylabel(r'$\theta_2$ (degrees)')
ax.set_title('Double Pendulum Phase Space')

def update(frame):
    global state_grid
    for i in range(grid_size):
        for j in range(grid_size):
            k1 = derivatives(state_grid[i, j])
            k2 = derivatives(state_grid[i, j] + 0.5 * dt * k1)
            k3 = derivatives(state_grid[i, j] + 0.5 * dt * k2)
            k4 = derivatives(state_grid[i, j] + dt * k3)
            state_grid[i, j] += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            image[i, j] = get_color(state_grid[i, j, 0], state_grid[i, j, 2])
    im.set_data(image)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=50, blit=False)
plt.show()
