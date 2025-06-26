import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Estilo oscuro y colores neón
plt.style.use('dark_background')
neon_colors = ['#FF073A','#04D9FF','#39FF14','#FFF700','#CC00FF']

# Instantes relativos máximos: 0.5s, 1.0s, …, 5.0s
tiempos = np.arange(0.5, 5.01, 0.5)

# Cuadrícula de subplots (2 filas x 5 columnas)
n_cols = 5
n_rows = int(np.ceil(len(tiempos) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, t_max in zip(axes, tiempos):
    for i, color in enumerate(neon_colors, start=1):
        # 1) Leemos el CSV y calculamos tiempo relativo
        df = pd.read_csv(f'coordenadas_cm{i}.csv')
        t0 = df['t_s'].iloc[0]                      # primer instante
        df['t_rel'] = df['t_s'] - t0                # ecuación: t_rel = t_s - t0

        # 2) Filtramos hasta el tiempo relativo t_max
        df_rec = df[df['t_rel'] <= t_max]

        # 3) Dibujamos la trayectoria recortada
        ax.plot(df_rec['red_x_cm'], df_rec['red_y_cm'],
                color=color, linewidth=1, label=f'Tirada {i}')

    # Título con LaTeX para indicar el máximo de cada subplot
    ax.set_title(f'$t_{{rel}} \\leq {t_max:.1f}\\,$s', color='white')
    ax.grid(color='gray', alpha=0.3)
    ax.tick_params(colors='white')

# Etiquetas de ejes compartidas
fig.text(0.5, 0.04, 'X [cm]', ha='center', color='white', fontsize=14)
fig.text(0.04, 0.5, 'Y [cm]', va='center', rotation='vertical', color='white', fontsize=14)

# Una sola leyenda
axes[0].legend(facecolor='black', edgecolor='white', labelcolor='white', loc='upper right')

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
fig.suptitle('Trayectorias del punto rojo hasta distintos tiempos relativos', 
             color='white', fontsize=16, y=1.02)
plt.show()
