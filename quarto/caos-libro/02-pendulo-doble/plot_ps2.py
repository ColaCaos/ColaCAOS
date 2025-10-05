import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ================== Configuración ==================
INPUT_PATTERN = 'coordenadas_cm_ps{}.csv'
N_TRIALS = 5
TIME_CANDIDATES  = ['t_s', 'time_s', 'tiempo_s', 't']  # detecta en este orden
ANGLE_CANDIDATES = ['theta', 'theta_rad', 'ang', 'angulo', 'angle']
FORCE_ANGLE_RANGE = True              # fuerza ángulo a (-3π/2, π/2]
T_FIRST_SECONDS = 15.0                # ventana de trazado tras alinear
FS_RESAMPLE = 200.0                   # Hz para la rejilla común de correlación/plot
MAX_SHIFT_S = 2.0                     # búsqueda de desfase ±MAX_SHIFT_S
OUTPUT_PNG = 'theta_alineadas_sin_wraps.png'
NEON = ['#FF073A','#04D9FF','#39FF14','#FFF700','#CC00FF']
# ===================================================

def find_col(df, candidates):
    for name in candidates:
        if name in df.columns: 
            return name
        # insensible a espacios y mayúsculas
        patt = re.sub(r'\s+', '', name.lower())
        for c in df.columns:
            if re.sub(r'\s+', '', c.lower()) == patt:
                return c
    return None

def normalize_angle(theta):
    # Lleva a (-3π/2, π/2] ≡ (-(1.5π), 0.5π]
    lo, hi = -1.5*np.pi, 0.5*np.pi
    span = 2*np.pi
    # map to (lo, lo+2π]
    wrapped = (theta - lo) % span + lo
    # for exact lo -> move to (lo, hi] properly
    wrapped[wrapped <= lo] += span
    # Ahora rango es (lo, lo+2π]; forzamos límites visuales si hace falta
    # (ya cumple lo pedido: por abajo ~π/2-2π = -3π/2 y por arriba < π/2)
    return wrapped

def zscore(x):
    x = np.asarray(x, float)
    m = np.nanmean(x); s = np.nanstd(x)
    if s == 0 or np.isnan(s): 
        return x*0
    return (x - m)/s

def best_shift_seconds(ref, cur, dt, max_shift_s):
    """
    ref, cur: arrays ya en la misma rejilla temporal y con NaN posibles.
    dt: paso temporal de la rejilla.
    Busca el lag que maximiza la correlación cruzada (normalizada) en ±max_shift_s.
    """
    # Enmascara NaN comunes
    mask = np.isfinite(ref) & np.isfinite(cur)
    if mask.sum() < 10:
        return 0.0
    a = zscore(ref[mask])
    b = zscore(cur[mask])
    n = len(a)
    # límite en muestras
    max_lag = int(round(max_shift_s/dt))
    # correlación por lags discretos
    lags = np.arange(-max_lag, max_lag+1)
    best_lag = 0
    best_val = -np.inf
    for L in lags:
        if L >= 0:
            aL = a[L:]; bL = b[:n-L]
        else:
            aL = a[:n+L]; bL = b[-L:]
        if len(aL) < 10: 
            continue
        val = np.dot(aL, bL)/len(aL)
        if val > best_val:
            best_val = val
            best_lag = L
    return best_lag * dt

# 1) Cargar tiradas y detectar columnas
dfs = []
for i in range(1, N_TRIALS+1):
    path = Path(INPUT_PATTERN.format(i))
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    df = pd.read_csv(path)
    tcol = find_col(df, TIME_CANDIDATES)
    acol = find_col(df, ANGLE_CANDIDATES)
    if tcol is None or acol is None:
        raise KeyError(f"Tirada {i}: columnas no encontradas (tiempo: {TIME_CANDIDATES}, ángulo: {ANGLE_CANDIDATES}). "
                       f"Columnas={list(df.columns)}")
    # tiempo relativo desde el primer valor válido
    t = pd.to_numeric(df[tcol], errors='coerce')
    t = t - t.dropna().iloc[0]
    th = pd.to_numeric(df[acol], errors='coerce')
    if FORCE_ANGLE_RANGE:
        th = normalize_angle(th)
    dfs.append(pd.DataFrame({'t': t, 'theta': th}).dropna())

# 2) Rejilla temporal común
dt = 1.0/FS_RESAMPLE
tmin = 0.0
# para cubrir al menos los primeros T_FIRST_SECONDS después de alinear
tmax_base = T_FIRST_SECONDS + 2*MAX_SHIFT_S
t_grid = np.arange(tmin, tmax_base + 5*dt, dt)

def interp_to_grid(df, tgrid):
    return np.interp(tgrid, df['t'].values, df['theta'].values, left=np.nan, right=np.nan)

series = [interp_to_grid(df, t_grid) for df in dfs]

# 3) Calcular desfases vs referencia (tirada 1)
ref = series[0]
shifts = [0.0]
for i in range(1, N_TRIALS):
    s = best_shift_seconds(ref, series[i], dt, MAX_SHIFT_S)
    shifts.append(s)

print("Desplazamientos encontrados (s):", [round(s, 4) for s in shifts])

# 4) Aplicar desplazamientos y recortar a primeros T_FIRST_SECONDS
def shift_series(y, shift_s, dt):
    # desplaza en tiempo usando interpolación: y(t) -> y(t+shift)
    # equivalente a mover la serie hacia la izquierda si shift>0
    t_shifted = t_grid + shift_s
    return np.interp(t_grid, t_shifted, y, left=np.nan, right=np.nan)

series_aligned = [shift_series(y, s, dt) for y, s in zip(series, shifts)]

# recorte exacto de 0..T_FIRST_SECONDS
imax = int(np.floor(T_FIRST_SECONDS/dt))
t_plot = t_grid[:imax+1]
series_plot = [y[:imax+1] for y in series_aligned]

# 5) Plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(9, 6))
for i, (y, c) in enumerate(zip(series_plot, NEON), start=1):
    ax.plot(t_plot, y, lw=2, color=c, label=f'Tirada {i}')

ax.set_xlim(0, T_FIRST_SECONDS)
ax.set_xlabel('Tiempo [ s ]', color='white')
ax.set_ylabel(r'Ángulo $\theta$ [ rad ]', color='white')
ax.set_title(r'$\theta(t)$ sin wraps — tiradas alineadas (primeros %.0f s)' % T_FIRST_SECONDS, color='white')

# límites verticales solicitados: (-3π/2, π/2)
ax.set_ylim(-1.5*np.pi, 0.5*np.pi)
ax.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='upper right')
ax.grid(color='gray', alpha=0.3)
ax.tick_params(colors='white')
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=160)
plt.close()
