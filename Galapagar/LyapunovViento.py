import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# 1. Cargar serie de velocidad del viento (km/h) desde 1950
df = pd.read_csv('Galapagar.csv', skiprows=3, parse_dates=['time'])
df = df.rename(columns={'wind_speed_10m_mean (km/h)': 'wind_speed'})
df = df[df['time'] >= '1950-01-01'].reset_index(drop=True)
wind = df['wind_speed'].values
N = len(wind)

# 2. Parámetros de embedding
m, tau = 12, 3
W = m * tau  # ventana de Theiler

# 3. Reconstrucción del espacio de fases
M = N - (m - 1) * tau
Y = np.empty((M, m))
for i in range(M):
    Y[i] = wind[i + np.arange(m) * tau]

# 4. Construir KD-tree para búsqueda de vecinos
tree = cKDTree(Y)

# 5. Cálculo de distancia inicial media para el umbral
d0_list = []
for i in range(M):
    dists, idxs = tree.query(Y[i], k=5)
    for d, j in zip(dists, idxs):
        if j != i and abs(j - i) > W:
            d0_list.append(d)
            break
d0_mean = np.mean(d0_list)
threshold = 2 * d0_mean  # umbral para renormalización

# 6. Algoritmo de Wolf para cada punto de referencia
local_lambdas = []
ref_indices = np.linspace(0, M - W - 1, num=300, dtype=int)
for i in ref_indices:
    # encontrar primer vecino válido
    dists, idxs = tree.query(Y[i], k=5)
    for d0, j0 in zip(dists, idxs):
        if j0 != i and abs(j0 - i) > W:
            break
    else:
        continue

    dist_seq = []
    # evolución hasta cruce de umbral
    for k in range(1, M - max(i, j0)):
        d_k = norm(Y[i + k] - Y[j0 + k])
        dist_seq.append(d_k)
        if d_k > threshold:
            break

    if len(dist_seq) > 1:
        ln_terms = np.log(np.array(dist_seq) / d0)
        lambda_local = np.mean(ln_terms / np.arange(1, len(dist_seq) + 1))
        local_lambdas.append(lambda_local)

# 7. Cálculo del exponente máximo
lambda_max = np.mean(local_lambdas)

# 8. Mostrar histograma de exponentes locales
plt.figure(figsize=(6,4))
plt.hist(local_lambdas, bins=20, edgecolor='black')
plt.axvline(lambda_max, color='red', linestyle='--', label=f'λ ≈ {lambda_max:.3f}')
plt.title('Distribución de exponentes locales (Wolf) para viento')
plt.xlabel('λ local (día⁻¹)')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

print(f"Exponente de Lyapunov máximo λ_max ≈ {lambda_max:.4f} día⁻¹")
