import numpy as np
import cv2
from numba import cuda
import math

# =============================================================================
# Simulation / Drawing Parameters
# =============================================================================
N = 10000  # number of pendulums

# Image dimensions (all pendulums drawn onto the same image)
IMG_WIDTH = 720
IMG_HEIGHT = 720

# Pivot (insertion point) for all pendulums (center of the image)
pivot_x = IMG_WIDTH // 2  # 360
pivot_y = IMG_HEIGHT // 2  # 360

# Scale factor to convert simulation units to pixels (for drawing pendulum arms)
scale = 150  # adjust as needed

# =============================================================================
# CUDA Kernels for 1D (vector of pendulums)
# =============================================================================
@cuda.jit
def initPendulums1D(theta1, theta2, v1, v2, N, init_theta1, init_theta2_array):
    """
    Initialize each pendulum’s state:
      - theta1 is set to a constant (init_theta1 in radians) for all pendulums.
      - theta2 is set from the provided array (init_theta2_array, in radians).
      - Angular velocities (v1 and v2) are set to zero.
    """
    i = cuda.grid(1)
    if i < N:
        theta1[i] = init_theta1
        theta2[i] = init_theta2_array[i]
        v1[i] = 0.0
        v2[i] = 0.0

@cuda.jit
def updatePendulums1DMulti(theta1, theta2, v1, v2, dt, num_steps, N):
    """
    Advance each pendulum’s state using Euler integration over num_steps iterations.
    Standard double pendulum equations are used with parameters:
       m1 = m2 = 1, l1 = l2 = 1, and g = 9.81.
    """
    i = cuda.grid(1)
    if i < N:
        for s in range(num_steps):
            m1 = 1.0
            m2 = 1.0
            l1 = 1.0
            l2 = 1.0
            g = 9.81

            t1 = theta1[i]
            t2 = theta2[i]
            w1 = v1[i]
            w2 = v2[i]
            delta = t1 - t2
            denom = 2 * m1 + m2 - m2 * math.cos(2 * delta)
            a1 = (-g * (2*m1 + m2) * math.sin(t1)
                  - m2 * g * math.sin(t1 - 2*t2)
                  - 2 * math.sin(delta) * m2 * (w2*w2*l2 + w1*w1*l1*math.cos(delta))
                 ) / (l1 * denom)
            a2 = (2 * math.sin(delta) * (w1*w1*l1*(m1+m2)
                  + g * (m1+m2) * math.cos(t1)
                  + w2*w2*l2*m2*math.cos(delta))
                 ) / (l2 * denom)
            w1 = w1 + a1 * dt
            w2 = w2 + a2 * dt
            t1 = t1 + w1 * dt
            t2 = t2 + w2 * dt
            theta1[i] = t1
            theta2[i] = t2
            v1[i] = w1
            v2[i] = w2

# =============================================================================
# Generate a 10,000-color scale (each color is a unique BGR triple)
# =============================================================================
def generate_color_scale(n):
    colors = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        # Map i in [0, n-1] to a hue value in [0, 179] (OpenCV hue range)
        hue = int(179 * i / (n - 1))
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colors[i] = bgr[0, 0]
    return colors

colors = generate_color_scale(N)

# =============================================================================
# Main Simulation and Drawing Loop (with real-time elapsed seconds)
# =============================================================================
def main():
    # Initial conditions:
    #   First angle: fixed at 170° (converted to radians)
    #   Second angle: linearly spaced from 170° to 170.1° (converted to radians)
    init_theta1_deg = 170.0
    init_theta2_deg = np.linspace(170.0, 170.1, N)
    init_theta1 = init_theta1_deg * math.pi / 180.0
    init_theta2 = init_theta2_deg * math.pi / 180.0

    # Allocate device arrays for simulation state (length N)
    d_theta1 = cuda.device_array(N, dtype=np.float64)
    d_theta2 = cuda.device_array(N, dtype=np.float64)
    d_v1 = cuda.device_array(N, dtype=np.float64)
    d_v2 = cuda.device_array(N, dtype=np.float64)

    # Kernel launch configuration for a 1D grid
    threads_per_block = 128
    blocks = (N + threads_per_block - 1) // threads_per_block

    # Initialize simulation state on the GPU.
    initPendulums1D[blocks, threads_per_block](d_theta1, d_theta2, d_v1, d_v2, N, init_theta1, init_theta2)
    cuda.synchronize()

    # Simulation parameters:
    dt = np.float64(0.001)  # time step (adjust as needed)
    num_steps = 5           # integration steps per frame (adjust to slow down or speed up evolution)

    # Variable to accumulate elapsed simulation time
    elapsed_time = 0.0

    while True:
        # Advance simulation by num_steps iterations per frame.
        updatePendulums1DMulti[blocks, threads_per_block](d_theta1, d_theta2, d_v1, d_v2, dt, num_steps, N)
        cuda.synchronize()

        # Update elapsed time (num_steps * dt per frame)
        elapsed_time += float(num_steps * dt)

        # Copy simulation state from device to host.
        h_theta1 = d_theta1.copy_to_host()
        h_theta2 = d_theta2.copy_to_host()

        # Create a blank image (black, 720x720).
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        # Overlay the elapsed time (in seconds) as text in the top-left corner.
        # Use white color, font scale 1, and thickness 2.
        time_text = f"Tiempo: {elapsed_time:.2f} s"
        cv2.putText(
            img,
            time_text,
            (10, 30),  # position: 10 px right, 30 px down
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Draw each pendulum (all overlaid on the same image) using its unique color.
        # Note: Iterating over 10,000 pendulums in Python with drawing calls can be slow.
        for i in range(N):
            t1 = h_theta1[i]
            t2 = h_theta2[i]
            if math.isnan(t1) or math.isnan(t2):
                continue
            # Compute the first mass’s position.
            x1 = int(pivot_x + scale * math.sin(t1))
            y1 = int(pivot_y + scale * math.cos(t1))
            # Compute the second mass’s position.
            x2 = int(x1 + scale * math.sin(t2))
            y2 = int(y1 + scale * math.cos(t2))
            col = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
            cv2.line(img, (pivot_x, pivot_y), (x1, y1), col, 1)
            cv2.line(img, (x1, y1), (x2, y2), col, 1)
            cv2.circle(img, (pivot_x, pivot_y), 2, col, -1)
            cv2.circle(img, (x1, y1), 2, col, -1)
            cv2.circle(img, (x2, y2), 2, col, -1)

        cv2.imshow("10k Pendulums Overlay", img)
        key = cv2.waitKey(1)
        if key == 27:  # Exit if Esc is pressed.
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
