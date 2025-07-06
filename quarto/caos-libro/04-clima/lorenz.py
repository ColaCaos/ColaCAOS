import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Parámetros del sistema de Lorenz
sigma, rho, beta = 10.0, 28.0, 8/3
dt = 0.01
steps = 10000
transient = 1000

def lorenz_deriv(state):
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])

def integrate_lorenz(state0, dt, steps):
    traj = np.empty((steps, 3))
    state = np.array(state0, float)
    for i in range(steps):
        k1 = lorenz_deriv(state)
        k2 = lorenz_deriv(state + dt*k1/2)
        k3 = lorenz_deriv(state + dt*k2/2)
        k4 = lorenz_deriv(state + dt*k3)
        state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        traj[i] = state
    return traj

# Integrar trayectoria
traj = integrate_lorenz([1, 1, 1], dt, steps)
# Construir segmentos y colores
points = traj[transient:]
segments = np.stack([points[:-1], points[1:]], axis=1)
total_segs = len(segments)
cmap = plt.cm.rainbow
colors = [cmap(i / total_segs) for i in range(total_segs)]

# Crear figura 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Atractor de Lorenz (líneas coloreadas)')

# Colección de líneas
lc = Line3DCollection([], linewidths=1.0)
ax.add_collection(lc)

# Texto de tiempo
time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

def init():
    lc.set_segments([])
    lc.set_color([])
    time_text.set_text('')
    return lc, time_text

def animate(i):
    segs = segments[:i]
    lc.set_segments(segs)
    lc.set_color(colors[:i])
    current_time = i * dt
    time_text.set_text(f"t = {current_time:.2f}")
    return lc, time_text

ani = FuncAnimation(fig, animate, frames=total_segs, init_func=init,
                    interval=20, blit=True)

# Guardar GIF
writer = PillowWriter(fps=30)
ani.save("lorenz_lines_rainbow_time.gif", writer=writer)
plt.close(fig)

print("Animación guardada como 'lorenz_lines_rainbow_time.gif'")
