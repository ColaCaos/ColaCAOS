import numpy as np
import cv2
from numba import cuda
import math

# Grid dimensions
WIDTH = 360
HEIGHT = 360

@cuda.jit
def initPendulums(theta1, theta2, v1, v2):
    """
    Initialize each simulation on a 360x360 grid.
    - theta1 (in radians) is mapped from x coordinate: [-180, 180] degrees.
    - theta2 (in radians) is mapped from y coordinate: [-180, 180] degrees.
    - Angular velocities are initialized to 0.
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < WIDTH and j < HEIGHT:
        idx = j * WIDTH + i
        theta1[idx] = (i - 180.0) * math.pi / 180.0
        theta2[idx] = (j - 180.0) * math.pi / 180.0
        v1[idx] = 0.0
        v2[idx] = 0.0

@cuda.jit
def updatePendulums(theta1, theta2, v1, v2, dt):
    """
    Update the state of each double pendulum using Euler integration.
    Uses the standard equations for a double pendulum with:
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

        # Compute angular accelerations
        a1 = (-g * (2 * m1 + m2) * math.sin(t1)
              - m2 * g * math.sin(t1 - 2 * t2)
              - 2 * math.sin(delta) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * math.cos(delta))
             ) / (l1 * denom)

        a2 = (2 * math.sin(delta) * (w1 * w1 * l1 * (m1 + m2)
              + g * (m1 + m2) * math.cos(t1)
              + w2 * w2 * l2 * m2 * math.cos(delta))
             ) / (l2 * denom)

        # Euler integration
        w1 = w1 + a1 * dt
        w2 = w2 + a2 * dt
        t1 = t1 + w1 * dt
        t2 = t2 + w2 * dt

        # Write back updated state.
        theta1[idx] = t1
        theta2[idx] = t2
        v1[idx] = w1
        v2[idx] = w2

@cuda.jit
def updateColors(theta1, theta2, img):
    """
    Update the color of each pixel in the image buffer based on the signs of theta1 and theta2.
    The image is a flat array of size WIDTH*HEIGHT*3 (BGR order).
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < WIDTH and j < HEIGHT:
        idx = j * WIDTH + i
        pix = idx * 3  # each pixel has 3 channels

        t1 = theta1[idx]
        t2 = theta2[idx]

        # Color based on sign of angles (BGR order)
        if t1 >= 0.0 and t2 >= 0.0:
            # Yellow: (B, G, R) = (0, 255, 255)
            img[pix]     = 0
            img[pix + 1] = 255
            img[pix + 2] = 255
        elif t1 < 0.0 and t2 >= 0.0:
            # Blue: (255, 0, 0)
            img[pix]     = 255
            img[pix + 1] = 0
            img[pix + 2] = 0
        elif t1 < 0.0 and t2 < 0.0:
            # Green: (0, 255, 0)
            img[pix]     = 0
            img[pix + 1] = 255
            img[pix + 2] = 0
        elif t1 >= 0.0 and t2 < 0.0:
            # Red: (0, 0, 255)
            img[pix]     = 0
            img[pix + 1] = 0
            img[pix + 2] = 255

def main():
    # Allocate simulation state arrays on the device.
    size = WIDTH * HEIGHT
    theta1 = cuda.device_array(size, dtype=np.float64)
    theta2 = cuda.device_array(size, dtype=np.float64)
    v1     = cuda.device_array(size, dtype=np.float64)
    v2     = cuda.device_array(size, dtype=np.float64)
    # Allocate an image buffer (flat array of 8-bit values, 3 channels per pixel).
    img = cuda.device_array(size * 3, dtype=np.uint8)

    # Configure CUDA grid and block dimensions.
    threads_per_block = (16, 16)
    blocks_per_grid_x = (WIDTH + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (HEIGHT + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Initialize pendulums.
    initPendulums[blocks_per_grid, threads_per_block](theta1, theta2, v1, v2)
    cuda.synchronize()

    dt = 0.01  # Time step

    # Create a window for display.
    cv2.namedWindow("Double Pendulum", cv2.WINDOW_AUTOSIZE)

    while True:
        # Update the simulation state.
        updatePendulums[blocks_per_grid, threads_per_block](theta1, theta2, v1, v2, dt)
        # Update the color image based on current angles.
        updateColors[blocks_per_grid, threads_per_block](theta1, theta2, img)
        cuda.synchronize()

        # Copy image from device to host.
        host_img = img.copy_to_host()
        # Reshape the flat image array into (HEIGHT, WIDTH, 3).
        host_img = host_img.reshape((HEIGHT, WIDTH, 3))

        # Display the image.
        cv2.imshow("Double Pendulum", host_img)
        if cv2.waitKey(1) == 27:  # Exit if 'Esc' key is pressed.
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
