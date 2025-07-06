import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')
colors = {
    1: '#FF5733',
    2: '#04D9FF',
    3: '#39FF14',
    4: '#FFF700',
    5: '#FF33EC'
}

# 1) Cargar y suavizar datos
dfs = {}
for i in range(1, 6):
    df = pd.read_csv(f'coordenadas_cm_ps{i}.csv')
    t0 = df['t_s'].iloc[0]
    df['t_rel'] = df['t_s'] - t0
    df['x_s'] = df['red_x_cm'].rolling(7, center=True, min_periods=1).mean()
    df['y_s'] = df['red_y_cm'].rolling(7, center=True, min_periods=1).mean()
    dfs[i] = df

# 2) Definir tiempo común y truncar a primeros 5 s
max_loop = 5.0
dt_data = dfs[1]['t_rel'].diff().median()
factor_interp = 5
max_t_rel = max(df['t_rel'].max() for df in dfs.values())
t_common_full = np.linspace(0, max_t_rel, int(np.ceil((max_t_rel / dt_data) * factor_interp)))
t_common = t_common_full[t_common_full <= max_loop]
n_frames = len(t_common)

# 3) Interpolación de trayectorias suavizadas
interp = {}
for i, df in dfs.items():
    xi = np.interp(t_common, df['t_rel'], df['x_s'])
    yi = np.interp(t_common, df['t_rel'], df['y_s'])
    interp[i] = (xi, yi)

# 4) Límites con margen
all_x = np.hstack([coords[0] for coords in interp.values()])
all_y = np.hstack([coords[1] for coords in interp.values()])
dx = (all_x.max() - all_x.min()) * 0.05
dy = (all_y.max() - all_y.min()) * 0.05

# 5) Configurar figura
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('X [cm]', color='white')
ax.set_ylabel('Y [cm]', color='white')
ax.set_title('Loop primeros 5 s: tiradas 1 a 5', color='white')
ax.grid(color='gray', alpha=0.3)
ax.set_aspect('equal', 'box')
ax.tick_params(colors='white')
ax.set_xlim(all_x.min() - dx, all_x.max() + dx)
ax.set_ylim(all_y.min() - dy, all_y.max() + dy)

lines = {}
for i in range(1, 6):
    ln, = ax.plot([], [], color=colors[i], lw=2, label=f'Tirada {i}')
    lines[i] = ln
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')
ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

def init():
    for ln in lines.values():
        ln.set_data([], [])
    time_text.set_text('')
    return list(lines.values()) + [time_text]

interval_ms = dt_data * 1000 * 3 / factor_interp

def update(frame):
    t_now = t_common[frame]
    for i, (xi, yi) in interp.items():
        lines[i].set_data(xi[:frame+1], yi[:frame+1])
    time_text.set_text(f'Time = {t_now:.2f} s')
    return list(lines.values()) + [time_text]

anim = FuncAnimation(
    fig, update, frames=n_frames, init_func=init,
    blit=True, interval=interval_ms, repeat=True
)

anim.save("trayectorias_1_5.gif", writer="imagemagick", fps=30)
plt.show()
