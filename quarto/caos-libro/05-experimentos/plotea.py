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
    p_w = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0001.csv"
    p_x = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0002.csv"
    p_y = "NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0003.csv"
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
    # 1) figura sin 'tight' (3D no se lleva bien con tight/constrained)
    fig = plt.figure(figsize=(9, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # 2) dibuja
    ax.plot(dfp["x"].to_numpy(), dfp["y"].to_numpy(), dfp["omega"].to_numpy(),
            linewidth=0.35, alpha=0.95, antialiased=True)

    ax.set_xlabel("x (posición)", labelpad=12)
    ax.set_ylabel("y (posición)", labelpad=12)

    # 3) etiqueta Z: sin rotación automática y con más separación
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("ω (velocidad angular)", rotation=90, labelpad=20)

    ax.set_title(f"Diagrama de fases 3D — vista: {name} (elev={elev}°, azim={azim}°)", pad=10)

    # 4) aspecto cúbico (mismo tamaño visual de ejes)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.view_init(elev=elev, azim=azim)

    # 5) **Reserva margen derecho**: encoge ligeramente el área del eje
    #    Esto es mucho más fiable que 'tight' con 3D.
    fig.canvas.draw()  # asegura que el eje tenga posición válida
    box = ax.get_position()  # [x0, y0, width, height] en coords de figura
    ax.set_position([box.x0, box.y0, box.width * 0.86, box.height])  # deja ~14% de margen a la derecha

    # 6) Coloca la etiqueta Z dentro del margen (x,y en coords de *ejes*)
    ax.zaxis.set_label_coords(1.02, 0.5)  # muévela un poco a la derecha (ajusta 1.02→1.06 si fuese necesario)

    out_path = f"phase_space_3d_{name}.png"
    # 7) Guardado SIN bbox_inches="tight" (evita recortes en 3D)
    plt.savefig(out_path, dpi=150)
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
