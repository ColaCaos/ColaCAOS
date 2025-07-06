import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 6))

# Colores neón para las 5 corridas
neon_colors = [
    '#FF073A',  # rojo
    '#04D9FF',  # azul
    '#39FF14',  # verde
    '#FFF700',  # amarillo
    '#CC00FF',  # violeta
]

# Leer y plotear cada archivo sin marcadores
for i, color in enumerate(neon_colors, start=1):
    df = pd.read_csv(f'coordenadas_cm_ps{i}.csv')
    ax.plot(df['red_x_cm'], df['red_y_cm'],
            color=color, linewidth=1, label=f'Tirada {i}')

# Etiquetas y leyenda
ax.set_xlabel('X [cm]', color='white')
ax.set_ylabel('Y [cm]', color='white')
ax.set_title('Superposición de trayectorias del punto rojo', color='white')
ax.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='upper right')
ax.grid(color='gray', alpha=0.3)
ax.tick_params(colors='white')

plt.show()
