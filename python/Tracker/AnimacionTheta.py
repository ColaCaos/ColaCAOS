import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')
colors = {
    1: '#FF5733',
    2: '#04D9FF',
    3: '#39FF14',
    4: '#FFF700',
    5: '#FF33EC'
}

# 1) Cargar, suavizar y calcular theta sin saltos
dfs = {}
for i in range(1, 6):
    df = pd.read_csv(f'coordenadas_cm{i}.csv')
    t0 = df['t_s'].iloc[0]
    df['t_rel'] = df['t_s'] - t0
    df['x_s'] = df['red_x_cm'].rolling(7, center=True, min_periods=1).mean()
    df['y_s'] = df['red_y_cm'].rolling(7, center=True, min_periods=1).mean()
    df['theta'] = np.arctan2(df['y_s'], df['x_s'])
    # Unwrap para quitar los wraps de ±π
    df['theta'] = np.unwrap(df['theta'])
    dfs[i] = df

# 2) Tiempo común hasta 5 s
max_loop = 15.0
dt_data = dfs[1]['t_rel'].diff().median()
factor_interp = 5
max_t_rel = max(df['t_rel'].max() for df in dfs.values())
t_common_full = np.linspace(0, max_t_rel, int(np.ceil((max_t_rel / dt_data) * factor_interp)))
t_common = t_common_full[t_common_full <= max_loop]
n_frames = len(t_common)

# 3) Interpolar θ continuo
interp_theta = {}
for i, df in dfs.items():
    theta_i = np.interp(t_common, df['t_rel'], df['theta'])
    interp_theta[i] = theta_i

# 4) Límites verticales
all_theta = np.hstack(list(interp_theta.values()))
dtheta = (all_theta.max() - all_theta.min()) * 0.1

# 5) Figura
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('Tiempo $t$ [s]', color='white')
ax.set_ylabel('Ángulo $\\theta$ [rad]', color='white')
ax.set_title('$\\theta(t)$ sin wraps (primeros 15 s)', color='white')
ax.grid(color='gray', alpha=0.3)
ax.tick_params(colors='white')
ax.set_xlim(0, max_loop)
ax.set_ylim(all_theta.min() - dtheta, all_theta.max() + dtheta)

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
    for i, theta_i in interp_theta.items():
        lines[i].set_data(t_common[:frame+1], theta_i[:frame+1])
    time_text.set_text(f'Tiempo = {t_now:.2f} s')
    return list(lines.values()) + [time_text]

anim = FuncAnimation(
    fig, update, frames=n_frames, init_func=init,
    blit=True, interval=interval_ms, repeat=True
)

anim.save("theta_vs_time_unwrapped.gif", writer="imagemagick", fps=30)
plt.show()
