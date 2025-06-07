import numpy as np
import cv2
from numba import cuda
import math

# =============================================================================
# Configuración global de la simulación y variables de selección con el ratón
# =============================================================================

WIDTH = 720
HEIGHT = 720

current_t1_min = -math.pi
current_t1_max = math.pi
current_t2_min = -math.pi
current_t2_max = math.pi

selecting = False
selection_start = None
selection_end = None
simulation_needs_reinit = [False]

# =============================================================================
# Kernels CUDA
# =============================================================================

@cuda.jit
def initPendulums(theta1, theta2, v1, v2, t1_min, t1_max, t2_min, t2_max):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < WIDTH and j < HEIGHT:
        idx = j * WIDTH + i
        theta1[idx] = t1_min + i * ((t1_max - t1_min) / (WIDTH - 1))
        theta2[idx] = t2_min + j * ((t2_max - t2_min) / (HEIGHT - 1))
        v1[idx] = 0.0
        v2[idx] = 0.0

@cuda.jit
def updatePendulums(theta1, theta2, v1, v2, dt):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < WIDTH and j < HEIGHT:
        idx = j * WIDTH + i

        m1 = 1.0
        m2 = 1.0
        l1 = 1.0
        l2 = 1.0
        g = 9.81

        t1 = theta1[idx]
        t2 = theta2[idx]
        w1 = v1[idx]
        w2 = v2[idx]

        delta = t1 - t2
        denom = 2 * m1 + m2 - m2 * math.cos(2 * delta)

        a1 = (-g * (2 * m1 + m2) * math.sin(t1)
              - m2 * g * math.sin(t1 - 2 * t2)
              - 2 * math.sin(delta) * m2 * (w2*w2*l2 + w1*w1*l1*math.cos(delta))
             ) / (l1 * denom)
        a2 = (2 * math.sin(delta) * (w1*w1*l1*(m1 + m2)
              + g*(m1 + m2)*math.cos(t1)
              + w2*w2*l2*m2*math.cos(delta))
             ) / (l2 * denom)

        w1 += a1 * dt
        w2 += a2 * dt
        t1 += w1 * dt
        t2 += w2 * dt

        theta1[idx] = t1
        theta2[idx] = t2
        v1[idx] = w1
        v2[idx] = w2

@cuda.jit
def updateColors(theta1, theta2, img):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < WIDTH and j < HEIGHT:
        idx = j * WIDTH + i
        norm1 = (math.sin(theta1[idx]) + 1.0) / 2.0
        norm2 = (math.sin(theta2[idx]) + 1.0) / 2.0
        gray_val = int(((norm1 + norm2) / 2.0) * 255)
        img[idx] = gray_val

# =============================================================================
# Callback del ratón para zoom
# =============================================================================

def mouse_callback(event, x, y, flags, param):
    global selecting, selection_start, selection_end
    global current_t1_min, current_t1_max, current_t2_min, current_t2_max
    global simulation_needs_reinit

    if event == cv2.EVENT_LBUTTONDOWN:
        selection_start = (x, y)
        selection_end = (x, y)
        selecting = True
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        selection_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and selecting:
        selection_end = (x, y)
        selecting = False
        x1, y1 = selection_start
        x2, y2 = selection_end
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        new_t1_min = current_t1_min + (left / (WIDTH - 1)) * (current_t1_max - current_t1_min)
        new_t1_max = current_t1_min + (right / (WIDTH - 1)) * (current_t1_max - current_t1_min)
        new_t2_min = current_t2_min + (top / (HEIGHT - 1)) * (current_t2_max - current_t2_min)
        new_t2_max = current_t2_min + (bottom / (HEIGHT - 1)) * (current_t2_max - current_t2_min)
        current_t1_min, current_t1_max = new_t1_min, new_t1_max
        current_t2_min, current_t2_max = new_t2_min, new_t2_max
        simulation_needs_reinit[0] = True

# =============================================================================
# Función principal
# =============================================================================

def main():
    global current_t1_min, current_t1_max, current_t2_min, current_t2_max
    global simulation_needs_reinit, selecting, selection_start, selection_end

    size = WIDTH * HEIGHT
    theta1 = cuda.device_array(size, dtype=np.float64)
    theta2 = cuda.device_array(size, dtype=np.float64)
    v1     = cuda.device_array(size, dtype=np.float64)
    v2     = cuda.device_array(size, dtype=np.float64)
    img    = cuda.device_array(size, dtype=np.uint8)

    threads = (16, 16)
    blocks = ((WIDTH + threads[0] - 1)//threads[0],
              (HEIGHT + threads[1] - 1)//threads[1])

    initPendulums[blocks, threads](theta1, theta2, v1, v2,
                                   current_t1_min, current_t1_max,
                                   current_t2_min, current_t2_max)
    cuda.synchronize()

    dt = 0.01
    simulation_time = 0.0

    window_name = "Double Pendulum"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        updatePendulums[blocks, threads](theta1, theta2, v1, v2, dt)
        updateColors[blocks, threads](theta1, theta2, img)
        cuda.synchronize()

        if simulation_needs_reinit[0]:
            initPendulums[blocks, threads](theta1, theta2, v1, v2,
                                           current_t1_min, current_t1_max,
                                           current_t2_min, current_t2_max)
            cuda.synchronize()
            simulation_needs_reinit[0] = False
            selection_start = selection_end = None

        host_img = img.copy_to_host().reshape((HEIGHT, WIDTH))
        colored = cv2.applyColorMap(host_img, cv2.COLORMAP_JET)

        if selecting and selection_start and selection_end:
            cv2.rectangle(colored, selection_start, selection_end, (255,255,255), 1)

        simulation_time += dt

        angle_text = (
            "Theta1: [{:.2f},{:.2f}] deg  Theta2: [{:.2f},{:.2f}] deg"
            .format(current_t1_min*180/math.pi,
                    current_t1_max*180/math.pi,
                    current_t2_min*180/math.pi,
                    current_t2_max*180/math.pi)
        )
        cv2.putText(colored, angle_text, (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        time_text = "Time: {:.2f} s".format(simulation_time)
        cv2.putText(colored, time_text, (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(window_name, colored)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
