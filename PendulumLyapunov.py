import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# ----------------------------------
# Parámetros generales de la simulación
# ----------------------------------
dt             = 0.01
n_steps        = 1000       # total de iteraciones para bifurcación
burn_in        = 100        # descartamos los primeros 1000
n_alpha        = 1000       # número de ángulos iniciales
max_per_alpha  = 500         # tope de picos por hilo

# Ángulos iniciales de −π a +π
alphas = np.linspace(-math.pi, math.pi, n_alpha).astype(np.float64)

# Reservamos en GPU contadores y valores para máximos y mínimos
max_counts = cuda.to_device(np.zeros(n_alpha, dtype=np.int32))
min_counts = cuda.to_device(np.zeros(n_alpha, dtype=np.int32))
max_values = cuda.to_device(np.zeros((n_alpha, max_per_alpha), dtype=np.float64))
min_values = cuda.to_device(np.zeros((n_alpha, max_per_alpha), dtype=np.float64))

@cuda.jit
def simulate_bifurcation(alphas, dt, n_steps, burn_in, max_per_alpha,
                         max_counts, max_values, min_counts, min_values):
    tid = cuda.grid(1)
    if tid >= alphas.size:
        return

    theta1 = alphas[tid]
    omega1 = 0.0
    theta2 = alphas[tid]
    omega2 = 0.0
    prev_omega2 = omega2
    cnt_max = 0
    cnt_min = 0

    def derivs(t1, w1, t2, w2):
        δ = t1 - t2
        denom = 2.0 + 1.0 - math.cos(2.0 * δ)
        dω1 = (-9.81*(2.0+1.0)*math.sin(t1)
               - 9.81*math.sin(t1 - 2.0*t2)
               - 2.0*math.sin(δ)*(w2*w2 + w1*w1*math.cos(δ))) / denom
        dω2 = (2.0*math.sin(δ)*(w1*w1*(1.0+1.0)
               + 9.81*(1.0+1.0)*math.cos(t1)
               + w2*w2*math.cos(δ))) / denom
        return dω1, dω2

    for step in range(n_steps):
        # RK4
        k1ω1, k1ω2 = derivs(theta1, omega1, theta2, omega2)
        k1θ1, k1θ2 = omega1, omega2

        t1_2 = theta1 + 0.5*dt*k1θ1
        w1_2 = omega1 + 0.5*dt*k1ω1
        t2_2 = theta2 + 0.5*dt*k1θ2
        w2_2 = omega2 + 0.5*dt*k1ω2
        k2ω1, k2ω2 = derivs(t1_2, w1_2, t2_2, w2_2)
        k2θ1, k2θ2 = w1_2, w2_2

        t1_3 = theta1 + 0.5*dt*k2θ1
        w1_3 = omega1 + 0.5*dt*k2ω1
        t2_3 = theta2 + 0.5*dt*k2θ2
        w2_3 = omega2 + 0.5*dt*k2ω2
        k3ω1, k3ω2 = derivs(t1_3, w1_3, t2_3, w2_3)
        k3θ1, k3θ2 = w1_3, w2_3

        t1_4 = theta1 + dt*k3θ1
        w1_4 = omega1 + dt*k3ω1
        t2_4 = theta2 + dt*k3θ2
        w2_4 = omega2 + dt*k3ω2
        k4ω1, k4ω2 = derivs(t1_4, w1_4, t2_4, w2_4)
        k4θ1, k4θ2 = w1_4, w2_4

        omega1 += (dt/6.0)*(k1ω1 + 2*k2ω1 + 2*k3ω1 + k4ω1)
        omega2 += (dt/6.0)*(k1ω2 + 2*k2ω2 + 2*k3ω2 + k4ω2)
        theta1 += (dt/6.0)*(k1θ1 + 2*k2θ1 + 2*k3θ1 + k4θ1)
        theta2 += (dt/6.0)*(k1θ2 + 2*k2θ2 + 2*k3θ2 + k4θ2)

        # envolver theta2 en [-π,π]
        theta2 = theta2 - 2*math.pi * math.floor((theta2 + math.pi)/(2*math.pi))

        if step >= burn_in:
            # máximos
            if prev_omega2 > 0.0 and omega2 < 0.0 and theta2 > 0.0:
                if cnt_max < max_per_alpha:
                    max_values[tid, cnt_max] = theta2
                cnt_max += 1
            # mínimos
            if prev_omega2 < 0.0 and omega2 > 0.0 and theta2 < 0.0:
                if cnt_min < max_per_alpha:
                    min_values[tid, cnt_min] = theta2
                cnt_min += 1

        prev_omega2 = omega2

    max_counts[tid] = cnt_max
    min_counts[tid] = cnt_min

# Lanzamos el kernel de bifurcación
threads_per_block = 128
blocks_per_grid   = (n_alpha + threads_per_block - 1) // threads_per_block
simulate_bifurcation[blocks_per_grid, threads_per_block](
    alphas, dt, n_steps, burn_in, max_per_alpha,
    max_counts, max_values, min_counts, min_values
)
cuda.synchronize()

# Recuperar datos de bifurcación
h_max_counts = max_counts.copy_to_host()
h_max_vals   = max_values.copy_to_host()

alpha_bif  = []
theta_bif  = []
for i in range(n_alpha):
    cnt = min(h_max_counts[i], max_per_alpha)
    alpha_deg = alphas[i] * 180.0 / math.pi
    for j in range(cnt):
        alpha_bif.append(alpha_deg)
        theta_bif.append(h_max_vals[i, j] * 180.0 / math.pi)

# ----------------------------------------------------
# Cálculo del exponente de Lyapunov (método de Benettin)
# ----------------------------------------------------
epsilon   = 1e-8
steps_bt  = 10               # integration steps per renormalization
Delta_t   = steps_bt * dt
N         = 2000             # number of renormalizations

def derivs_cpu(state):
    θ1, ω1, θ2, ω2 = state
    δ = θ1 - θ2
    denom = 2 + 1 - math.cos(2*δ)
    dω1 = (-9.81*(2+1)*math.sin(θ1)
           - 9.81*math.sin(θ1-2*θ2)
           - 2*math.sin(δ)*(ω2*ω2 + ω1*ω1*math.cos(δ))) / denom
    dω2 = (2*math.sin(δ)*(ω1*ω1*2 + 9.81*2*math.cos(θ1) + ω2*ω2*math.cos(δ))) / denom
    return np.array([ω1, dω1, ω2, dω2])

def rk4_step(state):
    k1 = derivs_cpu(state)
    k2 = derivs_cpu(state + 0.5*dt*k1)
    k3 = derivs_cpu(state + 0.5*dt*k2)
    k4 = derivs_cpu(state +     dt*k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

lambdas = np.zeros(n_alpha)
for i in tqdm(range(n_alpha), desc="Lyapunov"):
    α = alphas[i]
    # initial states
    x  = np.array([α, 0.0, α, 0.0])
    dx = np.random.randn(4)
    dx *= epsilon/np.linalg.norm(dx)
    xp = x + dx
    S  = 0.0
    for _ in range(N):
        # integrate Δt = steps_bt*dt
        for _ in range(steps_bt):
            x  = rk4_step(x)
            xp = rk4_step(xp)
        # compute divergence
        dx = xp - x
        d  = np.linalg.norm(dx)
        S += math.log(d/epsilon)
        # renormalize
        dx *= (epsilon/d)
        xp = x + dx
    lambdas[i] = S/(N*Delta_t)

alpha_deg = alphas * 180.0 / math.pi

# -------------------
# Dibujar el gráfico
# -------------------
fig, ax1 = plt.subplots(figsize=(10,6))

# bifurcación
ax1.scatter(alpha_bif, theta_bif, s=0.1, color='C0', label='bifurcación')
ax1.set_xlabel("Ángulo inicial α (º)")
ax1.set_ylabel("θ₂ máximos (º)", color='C0')
ax1.tick_params(axis='y', labelcolor='C0')

# exponente de Lyapunov en eje secundario
ax2 = ax1.twinx()
ax2.plot(alpha_deg, lambdas, 'k-', lw=1, label='λ máximo')
ax2.set_ylabel("λ máximo (unidad de tiempo)", color='k')
ax2.tick_params(axis='y', labelcolor='k')

# leyenda conjunta
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title("Diagrama de bifurcación y exponente de Lyapunov para el péndulo doble")
plt.tight_layout()
plt.show()
