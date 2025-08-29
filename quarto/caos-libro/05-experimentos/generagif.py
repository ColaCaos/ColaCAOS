# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter

# Cargar datos (suavizados si existen)
smoothed_path = "phase_space_xyz_omega_smoothed.csv"
merged_path = "phase_space_xyz_omega_merged.csv"

if os.path.exists(smoothed_path):
    df = pd.read_csv(smoothed_path)
elif os.path.exists(merged_path):
    df = pd.read_csv(merged_path)
else:
    p_w = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0001.csv"
    p_x = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0002.csv"
    p_y = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0003.csv"
    df_w = pd.read_csv(p_w).rename(columns={"Angular velocity": "omega"})
    df_x = pd.read_csv(p_x).rename(columns={"Position (x)": "x"})
    df_y = pd.read_csv(p_y).rename(columns={"Position (y)": "y"})
    df = df_x.merge(df_y, on="Time", how="inner").merge(df_w, on="Time", how="inner")

# Orden y limpieza
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Time","x","y","omega"]).sort_values("Time")

t = df["Time"].to_numpy().astype(float)
x = df["x"].to_numpy().astype(float)
y = df["y"].to_numpy().astype(float)
w = df["omega"].to_numpy().astype(float)

# Garantizar tiempos estrictamente crecientes
keep = np.concatenate(([True], np.diff(t) > 0))
t = t[keep]; x = x[keep]; y = y[keep]; w = w[keep]

t_min, t_max = float(t[0]), float(t[-1])
sim_duration = t_max - t_min

# Parámetros de reproducción: 10× => cada segundo de GIF recorre 10 s de simulación
gif_fps = 1  # mantener bajo para evitar timeouts
gif_duration = sim_duration / 10.0  # segundos de GIF
n_frames = max(2, int(np.ceil(gif_fps * gif_duration)))

# Umbrales temporales de cada frame
t_cuts = np.linspace(t_min, t_max, n_frames)

# Precalcular límites
xlim = (float(np.min(x)), float(np.max(x)))
ylim = (float(np.min(y)), float(np.max(y)))
zlim = (float(np.min(w)), float(np.max(w)))

# Figura
fig = plt.figure(figsize=(7, 5.5), dpi=96)
ax = fig.add_subplot(111, projection="3d")

# Elementos gráficos
line, = ax.plot([], [], [], linewidth=0.5, alpha=0.95)  # color por defecto
head = ax.scatter([], [], [], s=10)

# Vista ISO_NW
ax.view_init(elev=30, azim=135)
ax.set_xlabel("x (posición)")
ax.set_ylabel("y (posición)")
ax.set_zlabel("ω (velocidad angular)")
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
ax.set_title("Diagrama de fases 3D — ISO_NW (avance temporal 10×)")

time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
cmap = plt.cm.viridis  # cambio progresivo de color del punto "cabeza"

# Para eficiencia, precomputamos los índices máximos por frame con searchsorted
indices = np.searchsorted(t, t_cuts, side="right")  # incluye todos <= t_cut

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    head._offsets3d = (np.array([]), np.array([]), np.array([]))
    time_text.set_text("")
    return line, head, time_text

def update(i):
    k = indices[i]
    if k < 1:
        k = 1  # para poder mostrar al menos un punto/segmento
    xi, yi, wi = x[:k], y[:k], w[:k]

    # Trazo acumulado hasta t_cut (SIN saltos: usa TODAS las muestras reales)
    line.set_data(xi, yi)
    line.set_3d_properties(wi)

    # Punto de cabeza con color según progreso
    frac = i / (n_frames - 1) if n_frames > 1 else 1.0
    head._offsets3d = (np.array([xi[-1]]), np.array([yi[-1]]), np.array([wi[-1]]))
    head.set_facecolor(cmap(frac))
    head.set_edgecolor(cmap(frac))

    # Texto de tiempo
    time_text.set_text(f"t = {t_cuts[i]:.1f} s")
    return line, head, time_text

anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=1000 / gif_fps, blit=False)

gif_path = "phase_space_isoNW_anim_curva.gif"
writer = PillowWriter(fps=gif_fps)
anim.save(gif_path, writer=writer)
plt.close(fig)

gif_path
