import tkinter as tk
from tkinter import ttk
import numpy as np
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

GRAVITY = 9.81  # m/s^2

class ProjectileSimulation:
    def __init__(self, mass, speed, angle_deg):
        self.mass = mass
        self.v0 = speed
        self.angle = math.radians(angle_deg)
        self.v0x = self.v0 * math.cos(self.angle)
        self.v0y = self.v0 * math.sin(self.angle)
        self.t = 0.0
        self.dt = 0.02  # paso temporal en segundos
        self.x = 0.0
        self.y = 0.0
        self.trajectory = [(self.x, self.y)]
        self.running = False

        # Precomputados para escalado de ejes
        self.max_range = (self.v0 ** 2) * math.sin(2 * self.angle) / GRAVITY if self.v0 > 0 else 0
        self.max_height = (self.v0y ** 2) / (2 * GRAVITY) if self.v0 > 0 else 0

    def step(self):
        """Avanza un paso de simulación; retorna True si aún está en aire, False si tocó el suelo."""
        if not self.running:
            return False
        self.t += self.dt
        self.x = self.v0x * self.t
        self.y = self.v0y * self.t - 0.5 * GRAVITY * self.t ** 2
        if self.y < 0:
            self.y = 0
            self.trajectory.append((self.x, self.y))
            self.running = False
            return False
        self.trajectory.append((self.x, self.y))
        return True

    def reset(self, mass, speed, angle_deg):
        self.__init__(mass, speed, angle_deg)


class SimulationApp:
    def __init__(self, root):
        self.root = root
        root.title("Simulación de Tiro Parabólico 2D")

        # Variables vinculadas
        self.mass_var = tk.DoubleVar(value=1.0)           # kg
        self.speed_var = tk.DoubleVar(value=20.0)         # m/s
        self.angle_var = tk.DoubleVar(value=45.0)         # grados

        # Panel de controles
        control_frame = ttk.Frame(root, padding=10)
        control_frame.grid(row=0, column=0, sticky="nw")

        # Masa
        ttk.Label(control_frame, text="Masa (kg):").grid(row=0, column=0, sticky="w")
        self.mass_slider = ttk.Scale(control_frame, from_=0.1, to=10.0, variable=self.mass_var, orient="horizontal")
        self.mass_slider.grid(row=0, column=1, sticky="ew", padx=5)
        self.mass_label = ttk.Label(control_frame, textvariable=self._formatted(self.mass_var, "{:.2f}"))
        self.mass_label.grid(row=0, column=2, sticky="w", padx=5)

        # Fuerza (velocidad inicial)
        ttk.Label(control_frame, text="Velocidad inicial (m/s):").grid(row=1, column=0, sticky="w")
        self.speed_slider = ttk.Scale(control_frame, from_=1.0, to=100.0, variable=self.speed_var, orient="horizontal")
        self.speed_slider.grid(row=1, column=1, sticky="ew", padx=5)
        self.speed_label = ttk.Label(control_frame, textvariable=self._formatted(self.speed_var, "{:.2f}"))
        self.speed_label.grid(row=1, column=2, sticky="w", padx=5)

        # Ángulo
        ttk.Label(control_frame, text="Ángulo (°):").grid(row=2, column=0, sticky="w")
        self.angle_slider = ttk.Scale(control_frame, from_=0.0, to=90.0, variable=self.angle_var, orient="horizontal")
        self.angle_slider.grid(row=2, column=1, sticky="ew", padx=5)
        self.angle_label = ttk.Label(control_frame, textvariable=self._formatted(self.angle_var, "{:.1f}"))
        self.angle_label.grid(row=2, column=2, sticky="w", padx=5)

        # Botones
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(10,0))
        self.start_button = ttk.Button(button_frame, text="Iniciar", command=self.start_simulation)
        self.start_button.grid(row=0, column=0, padx=5)
        self.end_button = ttk.Button(button_frame, text="Finalizar", command=self.end_simulation, state="disabled")
        self.end_button.grid(row=0, column=1, padx=5)

        # Display de estado
        status_frame = ttk.Labelframe(control_frame, text="Estado de la simulación", padding=5)
        status_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky="ew")
        ttk.Label(status_frame, text="Tiempo (s):").grid(row=0, column=0, sticky="e")
        self.time_var = tk.StringVar(value="0.00")
        ttk.Label(status_frame, textvariable=self.time_var, width=10).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="x (m):").grid(row=1, column=0, sticky="e")
        self.x_var = tk.StringVar(value="0.00")
        ttk.Label(status_frame, textvariable=self.x_var, width=10).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(status_frame, text="y (m):").grid(row=2, column=0, sticky="e")
        self.y_var = tk.StringVar(value="0.00")
        ttk.Label(status_frame, textvariable=self.y_var, width=10).grid(row=2, column=1, sticky="w", padx=5)

        # Figura matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.ax.set_title("Tiro parabólico")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.line_traj, = self.ax.plot([], [], linestyle='-', marker='', label="Trayectoria")
        self.point, = self.ax.plot([], [], marker='o', label="Posición actual")
        self.ax.grid(True)
        self.ax.legend(loc="upper right")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

        self.sim = None
        self.animation_id = None

    def _formatted(self, var, fmt):
        """Devuelve StringVar formateado desde una DoubleVar para mostrar valor con formato."""
        out = tk.StringVar()
        def update(*args):
            try:
                out.set(fmt.format(var.get()))
            except tk.TclError:
                out.set("0")
        var.trace_add("write", update)
        update()
        return out

    def start_simulation(self):
        # Inicializa la simulación con los parámetros actuales
        mass = self.mass_var.get()
        speed = self.speed_var.get()
        angle = self.angle_var.get()

        self.sim = ProjectileSimulation(mass, speed, angle)
        self.sim.running = True
        self.start_button.config(state="disabled")
        self.end_button.config(state="normal")

        # Ajuste de ejes según alcance máximo estimado
        margin = 0.1
        xmax = max(1.0, self.sim.max_range * (1 + margin))
        ymax = max(1.0, self.sim.max_height * (1 + margin))
        self.ax.set_xlim(0, xmax)
        self.ax.set_ylim(0, ymax)

        # Limpiar trazado previo
        self.line_traj.set_data([], [])
        self.point.set_data([], [])
        self.canvas.draw()

        # Iniciar bucle
        self._update_loop()

    def _update_loop(self):
        if self.sim is None or not self.sim.running:
            self._on_simulation_end()
            return

        alive = self.sim.step()
        # Actualizar display
        self.time_var.set(f"{self.sim.t:.2f}")
        self.x_var.set(f"{self.sim.x:.2f}")
        self.y_var.set(f"{self.sim.y:.2f}")

        # Actualizar gráfico
        xs, ys = zip(*self.sim.trajectory)
        self.line_traj.set_data(xs, ys)
        # CORRECCIÓN: pasar secuencias para la posición actual
        self.point.set_data([self.sim.x], [self.sim.y])
        self.canvas.draw_idle()

        if alive:
            self.animation_id = self.root.after(int(self.sim.dt * 1000), self._update_loop)
        else:
            self._on_simulation_end()

    def end_simulation(self):
        if self.sim:
            self.sim.running = False
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
        self._on_simulation_end()

    def _on_simulation_end(self):
        self.start_button.config(state="normal")
        self.end_button.config(state="disabled")
        # Si terminó por tocar suelo, se mantiene el último punto y trayectoria.

def main():
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
