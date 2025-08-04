import tkinter as tk
from tkinter import ttk
import math
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

G = 9.81  # gravedad m/s^2

class PendulumSimulation:
    def __init__(self, length, mass, theta_deg):
        self.length = length  # en metros
        self.mass = mass      # en kg (no afecta la dinámica sin amortiguamiento)
        self.theta = math.radians(theta_deg)  # ángulo desde la vertical
        self.omega = 0.0  # velocidad angular
        self.t = 0.0
        self.dt = 0.02  # paso temporal
        self.running = False

    def derivatives(self, theta, omega):
        dtheta_dt = omega
        domega_dt = - (G / self.length) * math.sin(theta)
        return dtheta_dt, domega_dt

    def step(self):
        """Avanza un paso con RK4. Devuelve True si sigue en simulación."""
        if not self.running:
            return False
        h = self.dt
        theta0 = self.theta
        omega0 = self.omega

        k1_theta, k1_omega = self.derivatives(theta0, omega0)
        k2_theta, k2_omega = self.derivatives(theta0 + 0.5 * h * k1_theta,
                                              omega0 + 0.5 * h * k1_omega)
        k3_theta, k3_omega = self.derivatives(theta0 + 0.5 * h * k2_theta,
                                              omega0 + 0.5 * h * k2_omega)
        k4_theta, k4_omega = self.derivatives(theta0 + h * k3_theta,
                                              omega0 + h * k3_omega)

        self.theta += (h / 6.0) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
        self.omega += (h / 6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        self.t += h
        return True

    def reset(self, length, mass, theta_deg):
        self.__init__(length, mass, theta_deg)


class PendulumApp:
    def __init__(self, root):
        self.root = root
        root.title("Simulación de Péndulo Simple")

        # Variables
        self.length_var = tk.DoubleVar(value=1.0)  # metros
        self.mass_var = tk.DoubleVar(value=1.0)    # kg
        self.angle_var = tk.DoubleVar(value=30.0)  # grados desde vertical

        # Marco de controles
        control = ttk.Frame(root, padding=10)
        control.grid(row=0, column=0, sticky="nw")

        # Longitud
        ttk.Label(control, text="Longitud (m):").grid(row=0, column=0, sticky="w")
        ttk.Scale(control, from_=0.1, to=5.0, variable=self.length_var, orient="horizontal").grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(control, textvariable=self._formatted(self.length_var, "{:.2f}")).grid(row=0, column=2, padx=5)

        # Masa
        ttk.Label(control, text="Masa (kg):").grid(row=1, column=0, sticky="w")
        ttk.Scale(control, from_=0.1, to=10.0, variable=self.mass_var, orient="horizontal").grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Label(control, textvariable=self._formatted(self.mass_var, "{:.2f}")).grid(row=1, column=2, padx=5)

        # Ángulo inicial
        ttk.Label(control, text="Ángulo inicial (°):").grid(row=2, column=0, sticky="w")
        ttk.Scale(control, from_=-179.0, to=179.0, variable=self.angle_var, orient="horizontal").grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Label(control, textvariable=self._formatted(self.angle_var, "{:.1f}")).grid(row=2, column=2, padx=5)

        # Botones
        btn_frame = ttk.Frame(control)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=(10,0))
        self.start_btn = ttk.Button(btn_frame, text="Iniciar", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Finalizar", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        self.reset_btn = ttk.Button(btn_frame, text="Reiniciar", command=self.reset_simulation)
        self.reset_btn.grid(row=0, column=2, padx=5)

        # Estado
        status = ttk.Labelframe(control, text="Estado", padding=5)
        status.grid(row=4, column=0, columnspan=3, pady=10, sticky="ew")
        ttk.Label(status, text="Tiempo (s):").grid(row=0, column=0, sticky="e")
        self.time_var = tk.StringVar(value="0.00")
        ttk.Label(status, textvariable=self.time_var, width=10).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(status, text="Ángulo (°):").grid(row=1, column=0, sticky="e")
        self.theta_display = tk.StringVar(value="0.00")
        ttk.Label(status, textvariable=self.theta_display, width=10).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(status, text="Vel. angular (rad/s):").grid(row=2, column=0, sticky="e")
        self.omega_display = tk.StringVar(value="0.00")
        ttk.Label(status, textvariable=self.omega_display, width=10).grid(row=2, column=1, sticky="w", padx=5)

        # Figura del péndulo
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.ax.set_aspect("equal", "box")
        self.ax.set_xlim(-1.2 * self.length_var.get(), 1.2 * self.length_var.get())
        self.ax.set_ylim(-1.2 * self.length_var.get(), 0.2 * self.length_var.get())
        self.ax.set_title("Péndulo simple")
        self.line, = self.ax.plot([], [], lw=3, marker='', label="Brazo")
        self.bob, = self.ax.plot([], [], marker='o', markersize=15, label="Masa")
        self.ax.axhline(0, color='k', linewidth=0.5)
        self.ax.grid(True)
        self.ax.legend(loc="upper right")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

        self.sim = None
        self.animation_id = None

    def _formatted(self, var, fmt):
        out = tk.StringVar()
        def update(*args):
            try:
                out.set(fmt.format(var.get()))
            except tk.TclError:
                out.set("0")
        var.trace_add("write", update)
        update()
        return out

    def start(self):
        length = self.length_var.get()
        mass = self.mass_var.get()
        angle = self.angle_var.get()
        self.sim = PendulumSimulation(length, mass, angle)
        self.sim.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        # Ajustar ejes dinámicamente según longitud
        L = self.sim.length
        margin = 0.2
        self.ax.set_xlim(- (1 + margin)*L, (1 + margin)*L)
        self.ax.set_ylim(- (1 + margin)*L, 0.1 * L)
        self.update_plot(force=True)
        self._loop()

    def stop(self):
        if self.sim:
            self.sim.running = False
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def reset_simulation(self):
        # Detener, y limpiar visual
        self.stop()
        self.time_var.set("0.00")
        self.theta_display.set("0.00")
        self.omega_display.set("0.00")
        self.line.set_data([], [])
        self.bob.set_data([], [])
        self.canvas.draw_idle()

    def _loop(self):
        if self.sim is None or not self.sim.running:
            self._on_end()
            return
        self.sim.step()
        self.update_plot()
        # programar siguiente paso
        self.animation_id = self.root.after(int(self.sim.dt * 1000), self._loop)

    def update_plot(self, force=False):
        theta = self.sim.theta
        omega = self.sim.omega
        L = self.sim.length

        # Coordenadas del bob (pivot en (0,0), hacia abajo es y negativo)
        x = L * math.sin(theta)
        y = -L * math.cos(theta)

        # Actualizar línea y bob
        self.line.set_data([0, x], [0, y])
        self.bob.set_data([x], [y])

        # Estado textual
        self.time_var.set(f"{self.sim.t:.2f}")
        self.theta_display.set(f"{math.degrees(theta):.2f}")
        self.omega_display.set(f"{omega:.4f}")

        self.canvas.draw_idle()

    def _on_end(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

def main():
    root = tk.Tk()
    app = PendulumApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
