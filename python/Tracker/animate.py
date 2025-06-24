import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_csv(filename):
    # Carga de datos
    df = pd.read_csv(filename)
    t = df['t_s'].values
    x = df['red_x_cm'].values
    y = df['red_y_cm'].values

    # Prepara figura
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot([], [], 'r-', linewidth=1)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')
    coord_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='white')

    ax.set_xlim(x.min() - 5, x.max() + 5)
    ax.set_ylim(y.min() - 5, y.max() + 5)
    ax.set_xlabel('X [cm]', color='white')
    ax.set_ylabel('Y [cm]', color='white')
    ax.set_title('Animación de la trayectoria del punto rojo', color='white')
    ax.grid(color='gray', alpha=0.3)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        coord_text.set_text('')
        return line, time_text, coord_text

    def update(i):
        # actualiza línea
        line.set_data(x[:i], y[:i])
        # actualiza texto
        time_text.set_text(f'Tiempo: {t[i]:.2f} s')
        coord_text.set_text(f'X: {x[i]:.2f} cm  Y: {y[i]:.2f} cm')
        return line, time_text, coord_text

    ani = FuncAnimation(
        fig, update, frames=len(df), init_func=init,
        interval=50, blit=True, repeat=False
    )
    plt.show()


if __name__ == "__main__":
    # Cambia 'coordenadas_cm1.csv' por el archivo que quieras animar
    animate_csv('coordenadas_cm3.csv')
