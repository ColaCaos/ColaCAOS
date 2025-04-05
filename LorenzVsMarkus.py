import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------
# Water wheel simulation with parameters chosen to yield an effective Lorenz r ~ 28
# ---------------------
def water_wheel(t, state, K, q1, nu, g, r_wheel, I):
    a1, b1, omega = state
    da1_dt = omega * b1 - K * a1
    db1_dt = -omega * a1 - K * b1 + q1
    domega_dt = (-nu * omega + np.pi * g * r_wheel * a1) / I
    return [da1_dt, db1_dt, domega_dt]

# Simulation parameters (extended simulation to view the attractor)
t_span_wheel = (0, 2000)
t_eval_wheel = np.linspace(t_span_wheel[0], t_span_wheel[1], 400000)

# Water wheel parameters:
K = 10.0
q1 = 106.0      # Adjusted to obtain an effective Rayleigh number ~28
nu = 0.25
g = 6.0
r_wheel = 0.35  # r is used both as a parameter in the waterwheel and in the Rayleigh number expression; here we denote it r_wheel
I = 1.0

# Initial condition:
# The fixed point for no rotation is (a1, b1, ω) = (0, q1/K, 0) = (0, 10.6, 0).
# We use a small perturbation from that fixed point.
initial_state_wheel = [0.1, 10.6, 0.1]

sol_wheel = solve_ivp(water_wheel, t_span_wheel, initial_state_wheel, t_eval=t_eval_wheel,
                      args=(K, q1, nu, g, r_wheel, I))

# ---------------------
# Plotting the results
# ---------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(sol_wheel.y[0], sol_wheel.y[1], sol_wheel.y[2], lw=0.5)
ax.set_title("Chaotic Water Wheel Attractor (Effective Lorenz r ~ 28)")
ax.set_xlabel("a₁")
ax.set_ylabel("b₁")
ax.set_zlabel("ω")

plt.tight_layout()
plt.show()
