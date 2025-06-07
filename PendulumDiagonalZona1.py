import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import math

# ----------------------------------
# Parámetros generales de la simulación
# ----------------------------------
dt             = 0.01
n_steps        = 10000       # total de iteraciones
burn_in        = 1000        # descartamos los primeros 1000
n_alpha        = 10000       # número de ángulos iniciales
max_per_alpha  = 500         # tope de picos por hilo

# Ángulos iniciales de −π a +π
alphas = np.linspace(-math.pi, math.pi, n_alpha).astype(np.float64)

# Reservamos en GPU contadores y valores para máximos y mínimos
max_counts = cuda.to_device(np.zeros(n_alpha, dtype=np.int32))
min_counts = cuda.to_device(np.zeros(n_alpha, dtype=np.int32))
max_values = cuda.to_device(np.zeros((n_alpha, max_per_alpha), dtype=np.float64))
min_values = cuda.to_device(np.zeros((n_alpha, max_per_alpha), dtype=np.float64))

@cuda.jit
def simulate_bifurcation(alphas,
                         dt, n_steps, burn_in, max_per_alpha,
                         max_counts, max_values,
                         min_counts, min_values):
    tid = cuda.grid(1)
    if tid >= alphas.size:
        return

    # condiciones iniciales
    theta1 = alphas[tid]
    omega1 = 0.0
    theta2 = alphas[tid]
    omega2 = 0.0

    prev_omega2 = omega2
    cnt_max = 0
    cnt_min = 0

    # derivadas inline
    def derivs(t1, w1, t2, w2):
        δ = t1 - t2
        denom = 3.0 - math.cos(2.0*δ)  # 2*m1+m2 - m2*cos(2δ) con m1=m2=1
        dω1 = (
            -9.81*3.0*math.sin(t1)
            - 9.81*math.sin(t1 - 2.0*t2)
            - 2.0*math.sin(δ)*(w2*w2 + w1*w1*math.cos(δ))
        ) / denom
        dω2 = (
            2.0*math.sin(δ)*(
                w1*w1*2.0
                + 9.81*2.0*math.cos(t1)
                + w2*w2*math.cos(δ)
            )
        ) / denom
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

        # actualizamos estado
        omega1 += (dt/6.0)*(k1ω1 + 2*k2ω1 + 2*k3ω1 + k4ω1)
        omega2 += (dt/6.0)*(k1ω2 + 2*k2ω2 + 2*k3ω2 + k4ω2)
        theta1 += (dt/6.0)*(k1θ1 + 2*k2θ1 + 2*k3θ1 + k4θ1)
        theta2 += (dt/6.0)*(k1θ2 + 2*k2θ2 + 2*k3θ2 + k4θ2)

        # envolver theta2 en [-π,π]
        theta2 = theta2 - 2*math.pi * math.floor((theta2 + math.pi)/(2*math.pi))

        # tras 'burn_in', buscamos cruces de ω2 para picos
        if step >= burn_in:
            # máximo positivo: ω2 cruza +→− y θ2>0
            if prev_omega2 > 0.0 and omega2 < 0.0 and theta2 > 0.0:
                if cnt_max < max_per_alpha:
                    max_values[tid, cnt_max] = theta2
                cnt_max += 1
            # mínimo negativo: ω2 cruza −→+ y θ2<0
            if prev_omega2 < 0.0 and omega2 > 0.0 and theta2 < 0.0:
                if cnt_min < max_per_alpha:
                    min_values[tid, cnt_min] = theta2
                cnt_min += 1

        prev_omega2 = omega2

    max_counts[tid] = cnt_max
    min_counts[tid] = cnt_min

# configuramos y lanzamos el kernel
threads_per_block = 128
blocks_per_grid   = (n_alpha + threads_per_block - 1) // threads_per_block
simulate_bifurcation[blocks_per_grid, threads_per_block](
    alphas, dt, n_steps, burn_in, max_per_alpha,
    max_counts, max_values,
    min_counts, min_values
)
cuda.synchronize()

# traemos resultados a CPU
h_max_counts = max_counts.copy_to_host()
h_max_vals   = max_values.copy_to_host()

# reconstruimos lista de máximos para graficar
alpha_max = []
theta2_max = []

for i in range(n_alpha):
    cnt = min(h_max_counts[i], max_per_alpha)
    for j in range(cnt):
        # convertir alpha a grados
        alpha_deg = alphas[i] * 180/math.pi
        # solo almacenar si alpha entre 0 y 40 grados
        if 0 <= alpha_deg <= 40:
            alpha_max.append(alpha_deg)
            theta2_max.append(h_max_vals[i, j] * 180/math.pi)

# graficar solo máximos en [0°,40°]
plt.figure(figsize=(8,6))
plt.scatter(alpha_max, theta2_max, s=0.1, edgecolor='none')
plt.xlabel("Ángulo inicial α (grados)")
plt.ylabel("Máximos positivos de θ₂ (grados)")
plt.title("Máximos para α ∈ [0°, 40°]")
plt.xlim(0, 40)
plt.tight_layout()
plt.show()
