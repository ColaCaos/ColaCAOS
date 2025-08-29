#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animación 3D estilo 'Lorenz' para tus datos (x, y, ω) desde 3 CSV.

Incluye:
- Suavizado opcional (--smooth_sec) con media móvil centrada.
- Límite temporal (--t_max), submuestreo (--stride) y transitorio (--transient).
- Barra de progreso (tqdm si está instalado; si no, logs cada 5%).
- Márgenes independientes por eje (--pad_x, --pad_y, --pad_z) y límites Y opcionales (--ymin, --ymax).
- Control del aspecto de ejes: --aspect {cube,data}.
- Exporta GIF o MP4 (si la salida termina en .mp4 requiere ffmpeg en el PATH).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ---------------------------------------------
# Barra de progreso (tqdm si está disponible)
try:
    from tqdm.auto import tqdm
    def progress_iter(it, total):
        return tqdm(it, total=total, desc="Generando", unit="frame")
except Exception:
    tqdm = None
    def progress_iter(it, total):
        def gen():
            last = -1
            for i, v in enumerate(it, 1):
                pct = int(100 * i / total)
                if pct % 5 == 0 and pct != last:
                    print(f"Progreso: {pct}% ({i}/{total})")
                    last = pct
                yield v
        return gen()

# ---------------------------------------------
def limits_with_pad(a, pad):
    """Devuelve (min, max) con margen relativo 'pad' (por ej. 0.05 = 5%)."""
    mn, mx = float(np.min(a)), float(np.max(a))
    span = mx - mn if mx > mn else 1.0
    return mn - pad*span, mx + pad*span

def smooth_series_by_seconds(t, v, smooth_sec):
    """
    Media móvil centrada con ventana temporal ~smooth_sec.
    Calcula el nº de muestras a partir del dt mediano.
    """
    if smooth_sec is None or smooth_sec <= 0:
        return v
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return v
    w = int(round(smooth_sec / float(np.median(dt))))
    w = max(w, 3)
    if w % 2 == 0:
        w += 1
    return pd.Series(v).rolling(window=w, center=True, min_periods=1).mean().to_numpy()

def main():
    ap = argparse.ArgumentParser(description="Animación 3D estilo Lorenz para (x, y, ω) desde 3 CSV, con suavizado y control de aspecto.")
    ap.add_argument("--w_csv", required=True, help="CSV con columnas: Time, Angular velocity")
    ap.add_argument("--x_csv", required=True, help="CSV con columnas: Time, Position (x)")
    ap.add_argument("--y_csv", required=True, help="CSV con columnas: Time, Position (y)")
    ap.add_argument("--out", default="phase_space_lines_rainbow_time.gif",
                    help="Ruta de salida (.gif o .mp4). Si termina en .mp4, se usa ffmpeg.")
    ap.add_argument("--fps", type=int, default=30, help="FPS del vídeo/GIF")
    ap.add_argument("--t_max", type=float, default=None, help="Tiempo máximo a animar en segundos (ej. 50).")
    ap.add_argument("--stride", type=int, default=1, help="Submuestreo (1 = sin submuestreo).")
    ap.add_argument("--transient", type=int, default=0, help="Número de muestras iniciales a descartar.")
    ap.add_argument("--linewidth", type=float, default=1.0, help="Grosor de línea.")
    ap.add_argument("--elev", type=float, default=30.0, help="Elevación de cámara.")
    ap.add_argument("--azim", type=float, default=135.0, help="Azimut de cámara.")
    ap.add_argument("--cmap", default="rainbow", help="Colormap para el trazo progresivo.")
    ap.add_argument("--dpi", type=int, default=110, help="DPI de exportación.")
    ap.add_argument("--smooth_sec", type=float, default=0.0, help="Suavizado temporal (segundos). 0 = sin suavizado.")
    # Márgenes por eje y límites manuales para Y
    ap.add_argument("--pad_x", type=float, default=0.05, help="Margen relativo en X (p. ej., 0.05 = 5%).")
    ap.add_argument("--pad_y", type=float, default=0.20, help="Margen relativo en Y (por defecto más ancho).")
    ap.add_argument("--pad_z", type=float, default=0.05, help="Margen relativo en Z.")
    ap.add_argument("--ymin", type=float, default=None, help="Límite inferior Y (opcional).")
    ap.add_argument("--ymax", type=float, default=None, help="Límite superior Y (opcional).")
    # NUEVO: control de aspecto
    ap.add_argument("--aspect", choices=["cube", "data"], default="cube",
                    help="cube: los tres ejes ocupan lo mismo en pantalla; data: proporcional a los rangos de datos.")
    args = ap.parse_args()

    # ================== LECTURA Y PREPARACIÓN ==================
    df_w = pd.read_csv(args.w_csv).rename(columns={"Angular velocity": "omega"})
    df_x = pd.read_csv(args.x_csv).rename(columns={"Position (x)": "x"})
    df_y = pd.read_csv(args.y_csv).rename(columns={"Position (y)": "y"})
    df = df_x.merge(df_y, on="Time", how="inner").merge(df_w, on="Time", how="inner")
    df = df.sort_values("Time").dropna(subset=["Time", "x", "y", "omega"])

    t = df["Time"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    w = df["omega"].to_numpy(dtype=float)

    # Asegurar tiempos estrictamente crecientes
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, x, y, w = t[keep], x[keep], y[keep], w[keep]

    # Tope temporal (opcional)
    if args.t_max is not None:
        idx_max = np.searchsorted(t, args.t_max, side="right")
        t, x, y, w = t[:idx_max], x[:idx_max], y[:idx_max], w[:idx_max]

    # Suavizado opcional
    if args.smooth_sec and args.smooth_sec > 0:
        x = smooth_series_by_seconds(t, x, args.smooth_sec)
        y = smooth_series_by_seconds(t, y, args.smooth_sec)
        w = smooth_series_by_seconds(t, w, args.smooth_sec)

    # Transitorio y submuestreo (opcional)
    if args.transient > 0:
        t, x, y, w = t[args.transient:], x[args.transient:], y[args.transient:], w[args.transient:]
    s = max(1, int(args.stride))
    if s > 1:
        t, x, y, w = t[::s], x[::s], y[::s], w[::s]

    # Trayectoria -> segmentos y colores (idéntico patrón Lorenz)
    points = np.stack([x, y, w], axis=1)           # (N, 3)
    if len(points) < 2:
        raise RuntimeError("No hay suficientes muestras para animar tras filtros aplicados.")
    segments = np.stack([points[:-1], points[1:]], axis=1)  # (N-1, 2, 3)
    total_segs = len(segments)

    cmap = plt.get_cmap(args.cmap)
    colors = [cmap(i / total_segs) for i in range(total_segs)]
    t_seg = t[1:]  # tiempo del extremo final de cada segmento

    # ================== FIGURA Y ARTISTA ==================
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Márgenes por eje y posibles overrides en Y
    xlim = limits_with_pad(x, args.pad_x)
    ylim = limits_with_pad(y, args.pad_y)
    zlim = limits_with_pad(w, args.pad_z)
    if args.ymin is not None:
        ylim = (args.ymin, ylim[1])
    if args.ymax is not None:
        ylim = (ylim[0], args.ymax)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    # Control del aspecto:
    # - "cube": ejes con igual tamaño visual (lo que solicitas).
    # - "data": proporcional a los rangos de datos.
    try:
        if args.aspect == "cube":
            ax.set_box_aspect((1, 1, 1))
        else:
            ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))
    except Exception:
        # set_box_aspect no disponible en versiones antiguas; se ignora.
        pass

    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_xlabel('x (posición)')
    ax.set_ylabel('y (posición)')
    ax.set_zlabel('ω (velocidad angular)')
    ax.set_title('Diagrama de fases (líneas coloreadas)')

    lc = Line3DCollection([], linewidths=args.linewidth)
    ax.add_collection(lc)
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

    # ================== EXPORTACIÓN CON PROGRESO ==================
    out_lower = args.out.lower()
    if out_lower.endswith(".mp4"):
        writer = FFMpegWriter(fps=args.fps, bitrate=4000, metadata={"title": "Phase Space 3D"})
    else:
        writer = PillowWriter(fps=args.fps)

    with writer.saving(fig, args.out, dpi=args.dpi):
        # Primer frame vacío
        lc.set_segments([]); lc.set_color([]); time_text.set_text('')
        writer.grab_frame()

        # Bucle con barra de progreso
        for i in progress_iter(range(1, total_segs + 1), total=total_segs):
            lc.set_segments(segments[:i])
            lc.set_color(colors[:i])
            time_text.set_text(f"t = {t_seg[i-1]:.2f} s")
            writer.grab_frame()

    plt.close(fig)
    print(f"Animación guardada como '{args.out}'")

if __name__ == "__main__":
    main()
