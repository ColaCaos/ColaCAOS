# build_rueda_figs.py
# Genera figuras del sistema caótico (atractor, Poincaré, histograma, recurrencia)
# y empaqueta todo en un ZIP local.
#
# Uso:
#   1) Coloca este script junto a los 3 CSV de entrada.
#   2) python build_rueda_figs.py
#
# Salida:
#   - Carpeta: figs_rueda_caotica/
#   - ZIP: rueda_caotica_figs.zip

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------- Config ----------
IN_Y = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_Estadio_Inicial_plot_0001.csv"
IN_X = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_Estadio_Inicial_plot_0002.csv"
IN_W = "Velocidadangular1.csv"
OUT_DIR = "figs_rueda_caotica"
OUT_ZIP = "rueda_caotica_figs.zip"

# ---------- Util ----------
def load_or_fail(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No encuentro el archivo: {path}")
    return pd.read_csv(path)

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def cumulative_trapezoid(y, x):
    """Integral acumulada por trapecios. theta[0]=0."""
    y = np.asarray(y)
    x = np.asarray(x)
    out = np.zeros_like(y, dtype=float)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    return out

# ---------- Carga ----------
print("Cargando CSV...")
df_y = load_or_fail(IN_Y).rename(columns={"Position (y)": "y"})
df_x = load_or_fail(IN_X).rename(columns={"Position (x)": "x"})
df_w = load_or_fail(IN_W).rename(columns={"Angular velocity": "omega"})

for df in (df_x, df_y, df_w):
    if "Time" not in df.columns:
        raise ValueError("Falta columna 'Time' en alguno de los CSV.")

# Merge por tiempo
df_xy = pd.merge(df_x[["Time", "x"]], df_y[["Time", "y"]], on="Time", how="inner")
df = pd.merge(df_xy, df_w[["Time", "omega"]], on="Time", how="inner")
df = df.sort_values("Time").drop_duplicates(subset=["Time"]).reset_index(drop=True)

t = df["Time"].to_numpy()
x = df["x"].to_numpy()
y = df["y"].to_numpy()
w = df["omega"].to_numpy()

# Integra theta(t) a partir de omega
theta = cumulative_trapezoid(w, t)
df["theta"] = theta

# ---------- Resumen ----------
summary = {
    "n_samples": int(len(df)),
    "t_start": float(df["Time"].iloc[0]),
    "t_end": float(df["Time"].iloc[-1]),
    "dt_mean": float(np.mean(np.diff(t)) if len(t) > 1 else float("nan")),
    "omega_min": float(np.min(w)),
    "omega_max": float(np.max(w)),
    "x_min": float(np.min(x)),
    "x_max": float(np.max(x)),
    "y_min": float(np.min(y)),
    "y_max": float(np.max(y)),
}
print("Resumen:", summary)

# ---------- Figuras ----------
ensure_outdir(OUT_DIR)

# 1) ω(t)
plt.figure(figsize=(9,4.5))
plt.plot(t, w, linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad angular ω [rad/s]")
plt.title("Evolución temporal de ω(t)")
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_omega_vs_time.png"), dpi=150)
plt.close()

# 2) 3D (x, y, ω)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, w, linewidth=0.6)
ax.set_xlabel("x_CM [m]")
ax.set_ylabel("y_CM [m]")
ax.set_zlabel("ω [rad/s]")
ax.set_title("Espacio de fases 3D: (x_CM, y_CM, ω)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_phase_space_3d.png"), dpi=150)
plt.close(fig)

# 3) Vista superior x–y
plt.figure(figsize=(6,6))
plt.plot(x, y, linewidth=0.6)
plt.xlabel("x_CM [m]")
plt.ylabel("y_CM [m]")
plt.title("Atracción proyectada (vista superior): x_CM vs y_CM")
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_top_view_xy.png"), dpi=150)
plt.close()

# 4) Proyección ω vs x
plt.figure(figsize=(7,5))
plt.plot(x, w, linewidth=0.4)
plt.xlabel("x_CM [m]")
plt.ylabel("ω [rad/s]")
plt.title("Proyección: ω vs x_CM")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_side_proj_omega_x.png"), dpi=150)
plt.close()

# 5) Sección de Poincaré en θ = k·π/2 (cruces ascendentes de sin(2θ)=0)
s = np.sin(2.0 * theta)
cross_idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0] + 1
px = x[cross_idx]
py = y[cross_idx]
pw = w[cross_idx]
pt = t[cross_idx]

# (x, ω)
plt.figure(figsize=(6.5,5.2))
plt.scatter(px, pw, s=6, alpha=0.8)
plt.xlabel("x_CM [m]  (sección θ = k·π/2)")
plt.ylabel("ω [rad/s]")
plt.title("Sección de Poincaré: (x_CM, ω)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_poincare_x_omega.png"), dpi=160)
plt.close()

# (x, y)
plt.figure(figsize=(6.2,5.8))
plt.scatter(px, py, s=6, alpha=0.8)
plt.xlabel("x_CM [m]  (sección θ = k·π/2)")
plt.ylabel("y_CM [m]")
plt.title("Sección de Poincaré: (x_CM, y_CM)")
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_poincare_x_y.png"), dpi=160)
plt.close()

# Guarda CSV con los puntos de Poincaré
poincare_df = pd.DataFrame({"Time": pt, "x_CM": px, "y_CM": py, "omega": pw, "theta": theta[cross_idx]})
poincare_df.to_csv(os.path.join(OUT_DIR, "poincare_section_theta_k_pi_over_2.csv"), index=False)

# 6) Histograma de intervalos entre cruces ω=0 (cambios de sentido)
sign = np.sign(w)
zc_idx = np.where(sign[:-1] * sign[1:] < 0)[0] + 1
zc_times = t[zc_idx]
intervals = np.diff(zc_times) if len(zc_times) > 1 else np.array([])

plt.figure(figsize=(7,4.5))
if len(intervals) > 0:
    plt.hist(intervals, bins=20, edgecolor='k')
plt.xlabel("Intervalo entre cruces de ω=0 [s]")
plt.ylabel("Frecuencia")
plt.title("Distribución de intervalos entre cambios de sentido")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_hist_intervals_zero_cross.png"), dpi=160)
plt.close()

# Guarda estadísticos
intervals_stats = pd.DataFrame({
    "n_zero_crossings": [len(zc_idx)],
    "mean_interval_s": [float(np.mean(intervals)) if len(intervals) else float("nan")],
    "median_interval_s": [float(np.median(intervals)) if len(intervals) else float("nan")],
    "std_interval_s": [float(np.std(intervals, ddof=1)) if len(intervals) > 1 else float("nan")]
})
intervals_stats.to_csv(os.path.join(OUT_DIR, "zero_crossings_intervals_stats.csv"), index=False)

# 7) Mapa de recurrencia de ω(t)
# Para que no pese mucho, reducimos muestras si es largo
step = max(1, len(w) // 1200)
w_ds = w[::step]
w_norm = (w_ds - np.mean(w_ds)) / (np.std(w_ds) + 1e-12)
D = np.abs(w_norm[:, None] - w_norm[None, :])
eps = np.quantile(D, 0.1)  # umbral en el percentil 10
R = (D < eps).astype(float)

plt.figure(figsize=(6,6))
plt.imshow(R, origin='lower', interpolation='nearest')
plt.title("Mapa de recurrencia de ω(t)")
plt.xlabel("índice temporal")
plt.ylabel("índice temporal")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_recurrence_plot_omega.png"), dpi=160)
plt.close()

# Empaqueta todo en ZIP
with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(os.listdir(OUT_DIR)):
        zf.write(os.path.join(OUT_DIR, f), arcname=f)

print("\nListo ✅")
print(f"Carpeta con figuras: {OUT_DIR}")
print(f"ZIP creado: {OUT_ZIP}")
