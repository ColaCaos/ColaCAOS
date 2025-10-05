import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========= Config =========
INPUT_PATTERN = 'coordenadas_cm_ps{}.csv'  # ps1..ps5
N_TRIALS = 5
TIME_CANDIDATES = ['t_s','time_s','tiempo_s','t']  # nombres posibles de tiempo
X_CANDIDATES = ['red_x_cm','x_cm','x']
Y_CANDIDATES = ['red_y_cm','y_cm','y']

T_FIRST_SECONDS = 15.0    # ventana de plot
FS_RESAMPLE = 200.0       # Hz de la rejilla común
MAX_SHIFT_S = 2.0         # búsqueda de desfase ±2 s
NEON = ['#FF073A','#04D9FF','#39FF14','#FFF700','#CC00FF']
OUTPUT_PNG = 'theta_alineadas_sin_wraps.png'
# =========================

def find_col(df, candidates):
    # búsqueda sencilla por nombre exacto (ajusta si lo necesitas)
    for name in candidates:
        if name in df.columns:
            return name
    return None

def compute_theta_from_xy(x, y):
    # θ crudo
    theta = np.arctan2(y, x)            # (-π, π]
    # desenrolla (quita saltos de ±π)
    theta = np.unwrap(theta)            # continuidad temporal
    # reubica a (-3π/2, π/2]
    lo = -1.5*np.pi
    span = 2*np.pi
    z = (theta - lo) % span + lo
    z[z <= lo] += span
    return z

def zscore(x):
    m = np.nanmean(x); s = np.nanstd(x)
    if not np.isfinite(s) or s == 0: return np.zeros_like(x)
    return (x - m)/s

def best_shift_seconds(ref, cur, dt, max_shift_s):
    mask = np.isfinite(ref) & np.isfinite(cur)
    if mask.sum() < 10: return 0.0
    a, b = zscore(ref[mask]), zscore(cur[mask])
    n = len(a)
    max_lag = int(round(max_shift_s/dt))
    best_lag, best_val = 0, -np.inf
    for L in range(-max_lag, max_lag+1):
        if L >= 0:
            aL, bL = a[L:], b[:n-L]
        else:
            aL, bL = a[:n+L], b[-L:]
        if len(aL) < 10: continue
        val = np.dot(aL, bL)/len(aL)
        if val > best_val:
            best_val, best_lag = val, L
    return best_lag*dt

# 1) Cargar y preparar (t, θ) de cada tirada
dfs = []
for i in range(1, N_TRIALS+1):
    path = Path(INPUT_PATTERN.format(i))
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    df = pd.read_csv(path)

    tcol = find_col(df, TIME_CANDIDATES)
    xcol = find_col(df, X_CANDIDATES)
    ycol = find_col(df, Y_CANDIDATES)
    if tcol is None or xcol is None or ycol is None:
        raise KeyError(
            f"Tirada {i}: faltan columnas (tiempo: {TIME_CANDIDATES}, "
            f"X: {X_CANDIDATES}, Y: {Y_CANDIDATES}). Columnas={list(df.columns)}"
        )

    t = pd.to_numeric(df[tcol], errors='coerce')
    t = t - t.dropna().iloc[0]  # tiempo relativo desde el inicio
    x = pd.to_numeric(df[xcol], errors='coerce').to_numpy()
    y = pd.to_numeric(df[ycol], errors='coerce').to_numpy()
    theta = compute_theta_from_xy(x, y)

    dfi = pd.DataFrame({'t': t.to_numpy(), 'theta': theta}).dropna()
    dfs.append(dfi)

# 2) Rejilla temporal común
dt = 1.0/FS_RESAMPLE
t_grid = np.arange(0.0, T_FIRST_SECONDS + 2*MAX_SHIFT_S + 5*dt, dt)
def to_grid(df):
    return np.interp(t_grid, df['t'].values, df['theta'].values, left=np.nan, right=np.nan)
series = [to_grid(d) for d in dfs]

# 3) Alinear por correlación (vs tirada 1)
ref = series[0]
shifts = [0.0]
for i in range(1, N_TRIALS):
    shifts.append(best_shift_seconds(ref, series[i], dt, MAX_SHIFT_S))
print("Desplazamientos (s):", [round(s,4) for s in shifts])

def shift_series(y, s):
    t_shifted = t_grid + s
    return np.interp(t_grid, t_shifted, y, left=np.nan, right=np.nan)
series_aligned = [shift_series(y, s) for y, s in zip(series, shifts)]

# 4) Recorte y plot (primeros 15 s)
imax = int(np.floor(T_FIRST_SECONDS/dt))
t_plot = t_grid[:imax+1]
series_plot = [y[:imax+1] for y in series_aligned]

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(9,6))
for i, (y, c) in enumerate(zip(series_plot, NEON), start=1):
    ax.plot(t_plot, y, lw=2, color=c, label=f'Tirada {i}')
ax.set_xlim(0, T_FIRST_SECONDS)
ax.set_ylim(-1.5*np.pi, 0.5*np.pi)  # (-3π/2, π/2]
ax.set_xlabel('Tiempo [ s ]', color='white')
ax.set_ylabel(r'Ángulo $\theta$ [ rad ]', color='white')
ax.set_title(r'$\theta(t)$ sin wraps — tiradas alineadas (primeros %.0f s)' % T_FIRST_SECONDS, color='white')
ax.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='upper right')
ax.grid(color='gray', alpha=0.3)
ax.tick_params(colors='white')
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=160)
plt.close()
