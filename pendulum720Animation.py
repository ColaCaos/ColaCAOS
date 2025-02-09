import numpy as np
import cv2
from numba import cuda
import math

# Grid dimensions: 720x720 (approx. 0.5° steps from -180° to 180°)
WIDTH = 720
HEIGHT = 720

@cuda.jit
def initPendulums(theta1, theta2, v1, v2):
    """
    Initialize each double pendulum simulation on a 720x720 grid.
    The angles are mapped linearly from -pi to pi.
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < WIDTH and j < HEIGHT:
        idx = j * WIDTH + i
        # Map horizontal coordinate to theta1 in [-pi, pi]
        theta1[idx] = -math.pi + i * (2 * math.pi / (WIDTH - 1))
        # Map vertical coordinate to theta2 in [-pi, pi]
        theta2[idx] = -math.pi + j * (2 * math.pi / (HEIGHT - 1))
        v1[idx] = 0.0
        v2[idx] = 0.0

@cuda.jit
def updatePendulums(theta1, theta2, v1, v2, dt):
    """
    Update the state of each double pendulum using Euler integration.
    Uses standard double pendulum equations with:
      m1 = m2 = 1, l1 = l2 = 1, and g = 9.81.
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < WIDTH and j < HEIGHT:
        idx = j * WIDTH + i

        # Physical parameters
        m1 = 1.0
        m2 = 1.0
        l1 = 1.0
        l2 = 1.0
        g = 9.81

        # Read current state.
        t1 = theta1[idx]
        t2 = theta2[idx]
        w1 = v1[idx]
        w2 = v2[idx]

        delta = t1 - t2
        denom = 2 * m1 + m2 - m2 * math.cos(2 * delta)

        # Compute angular accelerations.
        a1 = (-g * (2 * m1 + m2) * math.sin(t1)
              - m2 * g * math.sin(t1 - 2 * t2)
              - 2 * math.sin(delta) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * math.cos(delta))
             ) / (l1 * denom)
        a2 = (2 * math.sin(delta) * (w1 * w1 * l1 * (m1 + m2)
              + g * (m1 + m2) * math.cos(t1)
              + w2 * w2 * l2 * m2 * math.cos(delta))
             ) / (l2 * denom)

        # Euler integration update.
        w1 = w1 + a1 * dt
        w2 = w2 + a2 * dt
        t1 = t1 + w1 * dt
        t2 = t2 + w2 * dt

        # Write back the updated state.
        theta1[idx] = t1
        theta2[idx] = t2
        v1[idx] = w1
        v2[idx] = w2

@cuda.jit
def updateColors(theta1, theta2, img):
    """
    Compute a grayscale value (0–255) for each pixel based on the pendulum angles.
    In this example the value is derived from the average of the normalized sine values
    of the two angles. (Normalization maps sine outputs from [-1, 1] to [0, 1].)
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < WIDTH and j < HEIGHT:
        idx = j * WIDTH + i
        # Normalize the sine values from [-1,1] to [0,1]
        norm1 = (math.sin(theta1[idx]) + 1.0) / 2.0
        norm2 = (math.sin(theta2[idx]) + 1.0) / 2.0
        # Average the two normalized values and scale to 0-255
        gray_val = int(((norm1 + norm2) / 2.0) * 255)
        img[idx] = gray_val

def main():
    # Total number of grid points.
    size = WIDTH * HEIGHT

    # Allocate device arrays for simulation state (angles and angular velocities).
    theta1 = cuda.device_array(size, dtype=np.float64)
    theta2 = cuda.device_array(size, dtype=np.float64)
    v1 = cuda.device_array(size, dtype=np.float64)
    v2 = cuda.device_array(size, dtype=np.float64)
    # Allocate a single-channel image buffer on the device (grayscale).
    img = cuda.device_array(size, dtype=np.uint8)

    # CUDA grid configuration.
    threads_per_block = (16, 16)
    blocks_per_grid_x = (WIDTH + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (HEIGHT + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Initialize the pendulums.
    initPendulums[blocks_per_grid, threads_per_block](theta1, theta2, v1, v2)
    cuda.synchronize()

    dt = 0.01  # Time step

    # Create an OpenCV window.
    cv2.namedWindow("Double Pendulum", cv2.WINDOW_AUTOSIZE)

    while True:
        # Update simulation state.
        updatePendulums[blocks_per_grid, threads_per_block](theta1, theta2, v1, v2, dt)
        # Update the grayscale image based on current angles.
        updateColors[blocks_per_grid, threads_per_block](theta1, theta2, img)
        cuda.synchronize()

        # Copy the image from device to host and reshape it to (HEIGHT, WIDTH).
        host_img = img.copy_to_host().reshape((HEIGHT, WIDTH))
        # Apply a colormap (e.g., JET) to convert grayscale to a colorful image.
        colored_img = cv2.applyColorMap(host_img, cv2.COLORMAP_JET)

        # Display the image.
        cv2.imshow("Double Pendulum", colored_img)
        # Exit on pressing the 'Esc' key (ASCII 27)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
