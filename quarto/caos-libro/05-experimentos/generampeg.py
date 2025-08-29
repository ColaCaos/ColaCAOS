#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FFMpegWriter, writers
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def load_data(w_csv, x_csv, y_csv):
    df_w = pd.read_csv(w_csv).rename(columns={"Angular velocity": "omega"})
    df_x = pd.read_csv(x_csv).rename(columns={"Position (x)": "x"})
    df_y = pd.read_csv(y_csv).rename(columns={"Position (y)": "y"})
    df = df_x.merge(df_y, on="Time", how="inner").merge(df_w, on="Time", how="inner")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Time","x","y","omega"]).sort_values("Time")
    t = df["Time"].to_numpy()
    keep = np.concatenate(([True], np.diff(t) > 0))
    return df.loc[keep].reset_index(drop=True)

def optional_smooth(df, smooth_sec=0.0):
    if smooth_sec <= 0:
        return df
    t = df["Time"].to_numpy().astype(float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return df
    dt_med = float(np.median(dt))
    w = max(3, int(round(smooth_sec / dt_med)))
    if w % 2 == 0: w += 1
    df2 = df.copy()
    for c in ["x","y","omega"]:
        df2[c] = df[c].rolling(window=w, center=True, min_periods=1).mean()
    return df2

def build_segments(x, y, z):
    pts = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)

def main():
    ap = argparse.ArgumentParser(description="MP4 3D (x,y,ω) en tiempo real con color progresivo y barra de progreso (vista ISO_NW).")
    ap.add_argument("--w_csv", default="NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0001.csv")
    ap.add_argument("--x_csv", default="NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0002.csv")
    ap.add_argument("--y_csv", default="NuevaRuedaCaoticaSimetricaAnilloConSinCanicasAceroFuenteMejoradaMasAgua_Transfer_plot_0003.csv")
    ap.add_argument("--out", default="phase_space_isoNW_realtime.mp4")
    ap.add_argument("--fps", type=float, default=None, help="FPS deseados. Si se omite, se ajustan con --max-frames manteniendo 1×.")
    ap.add_argument("--max-frames", type=int, default=6000, help="Máximo de frames para evitar bloqueos.")
    ap.add_argument("--elev", type=float, default=30.0)
    ap.add_argument("--azim", type=float, default=135.0)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--bitrate", type=int, default=4000)
    ap.add_argument("--smooth-sec", type=float, default=0.0, help="Suavizado temporal (s). 0 = sin suavizado.")
    args = ap.parse_args()

    if not writers.is_available('ffmpeg'):
        raise RuntimeError("No se encontró ffmpeg en el PATH.")

    # Datos
    df = load_data(args.w_csv, args.x_csv, args.y_csv)
    df = optional_smooth(df, args.smooth_sec)
    t = df["Time"].to_numpy().astype(float)
    x = df["x"].to_numpy().astype(float)
    y = df["y"].to_numpy().astype(float)
    w = df["omega"].to_numpy().astype(float)

    t_min, t_max = float(t[0]), float(t[-1])
    duration = t_max - t_min
    if duration <= 0: raise RuntimeError("Duración temporal nula.")

    # FPS y nº de frames manteniendo 1×
    if args.fps is None or args.fps <= 0:
        fps = max(1, int(np.floor(args.max_frames / duration)))
    else:
        fps = float(args.fps)
        tot = int(np.round(duration * fps)) + 1
        if tot > args.max_frames:
            fps = max(1, np.floor(args.max_frames / duration))
    fps = float(fps)
    n_frames = int(np.round(duration * fps)) + 1
    print(f"Duración (1×): {duration:.2f} s | FPS={fps:.2f} | Frames={n_frames} | Salida: {args.out}")

    # Tiempos de frame y prefijos acumulados (sin saltos)
    t_frames = np.linspace(t_min, t_max, n_frames)
    idx_end = np.searchsorted(t, t_frames, side="right")

    # Segmentos + colores progresivos (por tiempo medio de segmento)
    segs_full = build_segments(x, y, w)              # (N-1, 2, 3)
    t_seg = 0.5 * (t[1:] + t[:-1])
    norm = Normalize(vmin=t_min, vmax=t_max)
    cmap = plt.cm.viridis

    # Figura
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_xlabel("x (posición)"); ax.set_ylabel("y (posición)"); ax.set_zlabel("ω (velocidad angular)")
    ax.set_xlim(float(np.min(x))); ax.set_ylim(float(np.min(y))); ax.set_zlim(float(np.min(w)))
    ax.set_title("Diagrama de fases 3D — ISO_NW (1× tiempo real)")
    lc = Line3DCollection([], cmap=cmap, linewidth=0.6); lc.set_norm(norm); ax.add_collection(lc)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    # Progreso: tqdm si está disponible; si no, logs cada 5 %
    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=n_frames, desc="Renderizando", unit="frame", leave=True)
        use_tqdm = True
    except Exception:
        pbar = None
        use_tqdm = False
        print("Progreso: 0%")

    writer = FFMpegWriter(fps=fps, metadata={"title": "Phase Space ISO_NW realtime"}, bitrate=args.bitrate)
    with writer.saving(fig, args.out, args.dpi):
        last_logged_pct = -1
        for i in range(n_frames):
            k = idx_end[i]
            if k >= 2:
                lc.set_segments(segs_full[:k-1])
                lc.set_array(t_seg[:k-1])
            else:
                lc.set_segments([])
                lc.set_array(np.array([]))
            time_text.set_text(f"t = {t_frames[i]:.2f} s")
            writer.grab_frame()

            if use_tqdm:
                pbar.update(1)
            else:
                pct = int(100 * (i + 1) / n_frames)
                if pct % 5 == 0 and pct != last_logged_pct:
                    print(f"Progreso: {pct}% ({i+1}/{n_frames})")
                    last_logged_pct = pct

    if use_tqdm: pbar.close()
    print(f"Vídeo exportado en: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
