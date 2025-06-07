import numpy as np
import matplotlib.pyplot as plt

def derivs(state, t):
    g = 9.81
    m1 = m2 = l1 = l2 = 1.0
    theta1, z1, theta2, z2 = state
    delta = theta2 - theta1
    denom1 = (m1 + m2)*l1 - m2*l1*np.cos(delta)**2
    denom2 = (l2/l1)*denom1
    # angular accelerations
    a1 = (m2*l1*z1**2*np.sin(delta)*np.cos(delta)
          + m2*g*np.sin(theta2)*np.cos(delta)
          + m2*l2*z2**2*np.sin(delta)
          - (m1+m2)*g*np.sin(theta1)) / denom1
    a2 = (-m2*l2*z2**2*np.sin(delta)*np.cos(delta)
          + (m1+m2)*(g*np.sin(theta1)*np.cos(delta)
                     - l1*z1**2*np.sin(delta)
                     - g*np.sin(theta2))) / denom2
    return np.array([z1, a1, z2, a2])

def rk4_step(func, y, t, dt):
    k1 = func(y, t)
    k2 = func(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = func(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = func(y + dt*k3, t + dt)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6

# simulation parameters
t_max = 20.0
dt = 0.01
t = np.arange(0, t_max+dt, dt)
n_trajectories = 20
distances = np.zeros((n_trajectories, len(t)))

# simulate each trajectory
for i in range(n_trajectories):
    theta1_0 = np.deg2rad(170.0)
    theta2_0 = np.deg2rad(170.0 + i*0.0005)
    state = np.array([theta1_0, 0.0, theta2_0, 0.0])
    # initial position of tip
    l1 = l2 = 1.0
    x_prev = l1*np.sin(theta1_0) + l2*np.sin(theta2_0)
    y_prev = -l1*np.cos(theta1_0) - l2*np.cos(theta2_0)
    cum_dist = 0.0
    distances[i,0] = cum_dist
    for j in range(1, len(t)):
        state = rk4_step(derivs, state, t[j-1], dt)
        theta1, _, theta2, _ = state
        x = l1*np.sin(theta1) + l2*np.sin(theta2)
        y = -l1*np.cos(theta1) - l2*np.cos(theta2)
        cum_dist += np.hypot(x - x_prev, y - y_prev)
        distances[i,j] = cum_dist
        x_prev, y_prev = x, y

# plot
plt.figure()
for i in range(n_trajectories):
    plt.plot(t, distances[i])
plt.xlabel('Tiempo (s)')
plt.ylabel('Distancia recorrida (m)')
plt.title('Distancia recorrida por el extremo vs tiempo para 20 trayectorias')
plt.show()
