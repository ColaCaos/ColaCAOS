import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree

# 1. Cargar datos
df = pd.read_csv('Galapagar.csv', skiprows=3, parse_dates=['time'])
df = df.rename(columns={'temperature_2m_mean (°C)': 'temperature'})
df = df[df['time'] >= '1950-01-01'].reset_index(drop=True)
temps = df['temperature'].values
N = len(temps)

# 2. Parámetros de embedding
m, tau = 12, 3
W = m * tau  # ventana de Theiler

# 3. Reconstrucción del espacio de fases
M = N - (m - 1) * tau
Y = np.empty((M, m))
for i in range(M):
    Y[i] = temps[i + np.arange(m) * tau]

# 4. Búsqueda de vecinos con KD-tree
tree = cKDTree(Y)

# 5. Calcular distancia inicial media para definir umbral
d0_list = []
for i in range(M):
    dists, idxs = tree.query(Y[i], k=5)
    for d, j in zip(dists, idxs):
        if j != i and abs(j - i) > W:
            d0_list.append(d)
            break
d0_mean = np.mean(d0_list)
threshold = 2 * d0_mean  # umbral de renormalización

# 6. Método de Wolf
local_lambdas = []
ref_indices = np.linspace(0, M - W - 1, num=300, dtype=int)
for i in ref_indices:
    # vecino inicial válido
    dists, idxs = tree.query(Y[i], k=5)
    for d0, j0 in zip(dists, idxs):
        if j0 != i and abs(j0 - i) > W:
            break
    else:
        continue

    # evolucionar hasta cruce de umbral
    dist_seq = []
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
print(f"Exponente de Lyapunov máximo λ_max ≈ {lambda_max:.4f} día⁻¹")
