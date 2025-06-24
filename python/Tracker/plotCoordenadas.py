import pandas as pd
import matplotlib.pyplot as plt

# Lee el CSV generado previamente
df = pd.read_csv('coordenadas_cm.csv')

# Crea la figura y los ejes
fig, ax = plt.subplots()

# Dibuja cada trayectoria con su color
ax.plot(df['red_x_cm'],   - df['red_y_cm'],   'r-', label='Rojo')
ax.plot(df['green_x_cm'], - df['green_y_cm'], 'g-', label='Verde')
ax.plot(df['blue_x_cm'],  - df['blue_y_cm'],  'b-', label='Azul')

# Etiquetas y leyenda
ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
ax.set_title('Trayectoria XY de los tres puntos')
ax.legend()
ax.grid(True)

# Para que el eje Y crezca hacia abajo como en la visualización original:
ax.invert_yaxis()

plt.show()
