# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from zipfile import ZipFile

# Cargar datos suavizados si existen; si no, combinados originales
smoothed = "phase_space_xyz_omega_smoothed.csv"
merged = "phase_space_xyz_omega_merged.csv"

if os.path.exists(smoothed):
    df = pd.read_csv(smoothed)
elif os.path.exists(merged):
    df = pd.read_csv(merged)
else:
    # Reconstrucción de emergencia
    p_w = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0008.csv"
    p_x = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0009.csv"
    p_y = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0010.csv"
    df_w = pd.read_csv(p_w).rename(columns={"Angular velocity": "omega"})
    df_x = pd.read_csv(p_x).rename(columns={"Position (x)": "x"})
    df_y = pd.read_csv(p_y).rename(columns={"Position (y)": "y"})
    df = df_x.merge(df_y, on="Time", how="inner").merge(df_w, on="Time", how="inner").sort_values("Time")

# Limpieza y orden
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y", "omega"]).sort_values("Time")

# Reducir puntos si es enorme para mantener agilidad de render
MAX_POINTS = 80000
if len(df) > MAX_POINTS:
    idx = np.linspace(0, len(df)-1, MAX_POINTS).astype(int)
    dfp = df.iloc[idx]
else:
    dfp = df

# Definir vistas (nombre, elev, azim)
views = [
    ("iso_NE", 30, 45),
    ("iso_NW", 30, 135),
    ("iso_SW", 30, 225),
    ("iso_SE", 30, 315),
    ("top", 80, 45),
    ("side_X", 20, 0),
    ("side_Y", 20, 90),
]

out_files = []




def render_view(name, elev, azim):
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib as mpl

    # --- Figura clara, sin tight ---
    fig = plt.figure(figsize=(9, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # === 1) Trazo con DEGRADADO temporal (más impactante que un color plano) ===
    x = dfp["x"].to_numpy()
    y = dfp["y"].to_numpy()
    z = dfp["omega"].to_numpy()
    t = dfp["Time"].to_numpy() if "Time" in dfp.columns else np.arange(len(x))

    # Segmentos consecutivos para colorear cada tramo según el tiempo
    pts = np.vstack([x, y, z]).T
    segs = np.stack([pts[:-1], pts[1:]], axis=1)              # (N-1, 2, 3)

    # Colormap claro y “vivo”; puedes probar "plasma", "viridis" o "turbo"
    cmap = mpl.cm.turbo
    norm = mpl.colors.Normalize(vmin=t.min(), vmax=t.max())
    colors = cmap(norm(t[:-1]))

    lc = Line3DCollection(segs, linewidths=0.45, colors=colors, alpha=0.95, antialiased=True)
    ax.add_collection(lc)

    # Marcadores de inicio/fin (sutiles)
    ax.scatter([x[0]], [y[0]], [z[0]], s=12, color=cmap(norm(t[0])), edgecolors="none")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], s=12, color=cmap(norm(t[-1])), edgecolors="none")

    # Límites automáticos con pequeño margen (evitar cortes)
    def lim(a, pad=0.04):
        mn, mx = float(np.min(a)), float(np.max(a))
        span = mx - mn if mx > mn else 1.0
        return mn - pad*span, mx + pad*span
    ax.set_xlim(*lim(x)); ax.set_ylim(*lim(y)); ax.set_zlim(*lim(z))

    # Aspecto cúbico (los 3 ejes ocupan lo mismo en pantalla)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # === 2) Sin cuadrícula / sin “panes” ===
    ax.grid(False)
    # Oculta relleno de los planos y líneas de rejilla
    try:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
        # Apaga la “grid” interna (compatibilidad con algunas versiones)
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    except Exception:
        pass

    # === 3) Etiquetas y estilo claro ===
    ax.set_xlabel("x (posición)", labelpad=12)
    ax.set_ylabel("y (posición)", labelpad=12)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("ω (velocidad angular)", rotation=90, labelpad=20)
    ax.set_title(f"Diagrama de fases 3D — vista: {name} (elev={elev}°, azim={azim}°)", pad=10)

    # Ticks discretos y limpios
    ax.tick_params(length=3, width=0.8)

    # === 4) Cámara y margen para NO cortar el z-label ===
    ax.view_init(elev=elev, azim=azim)
    fig.canvas.draw()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.86, box.height])  # deja ~14% a la derecha
    ax.zaxis.set_label_coords(1.06, 0.5)

    # === 5) Barra de color con el tiempo (opcional, muy informativa) ===
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Tiempo (s)")

    # Guardado
    out_path = f"phase_space_3d_{name}.png"
    plt.savefig(out_path, dpi=150)  # sin 'tight' para evitar recortes
    plt.close(fig)
    return out_path



for name, elev, azim in views:
    out_files.append(render_view(name, elev, azim))

# Empaquetar en un ZIP
zip_path = "phase_space_3d_views.zip"
with ZipFile(zip_path, "w") as zf:
    for f in out_files:
        zf.write(f, arcname=os.path.basename(f))

# Crear también una imagen "ficha técnica" con miniaturas (aunque la instrucción recomienda no subplots,
# no la usaremos como gráfico principal; la omitimos para respetar estrictamente la norma).
# Devolver rutas de salida
(out_files, zip_path)
