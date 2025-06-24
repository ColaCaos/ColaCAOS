import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Leer datos
df = pd.read_csv('coordenadas_cm5.csv')

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8,6))

neon_red   = '#FF073A'
neon_green = '#39FF14'
neon_blue  = '#04D9FF'

# helper para líneas con glow más pequeño
def neon_plot(x, y, color, label):
    line, = ax.plot(x, y,
        color=color,
        linewidth=1.0,
        label=label,
        zorder=2)
    # Glow reducido: trazo más fino
    line.set_path_effects([
        pe.Stroke(linewidth=4, foreground=color, alpha=0.2),
        pe.Normal()
    ])
    return line

neon_plot(df['red_x_cm'],   df['red_y_cm'],   neon_red,   'Rojo')
neon_plot(df['green_x_cm'], df['green_y_cm'], neon_green, 'Verde')
neon_plot(df['blue_x_cm'],  df['blue_y_cm'],  neon_blue,  'Azul')

ax.set_xlabel('X [cm]', color='white')
ax.set_ylabel('Y [cm]', color='white')
ax.set_title('Trayectoria XY con glow reducido', color='white')
ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
ax.grid(color='gray', alpha=0.3)
ax.tick_params(colors='white')

plt.show()
