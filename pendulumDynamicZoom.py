import numpy as np
import cv2
from numba import cuda
import math

# =============================================================================
# Global simulation settings and mouse selection globals
# =============================================================================

# Display grid dimensions (always 720x720)
WIDTH = 720
HEIGHT = 720

# Current angular bounds for both theta1 and theta2.
# (Initially, these correspond to -pi to pi, i.e. -180° to 180°.)
current_t1_min = -math.pi
current_t1_max = math.pi
current_t2_min = -math.pi
current_t2_max = math.pi

# Mouse selection state (for zooming)
selecting = False
selection_start = None  # (x, y)
selection_end = None    # (x, y)
# A mutable flag (using a list so it can be modified inside the callback)
simulation_needs_reinit = [False]

# =============================================================================
# CUDA Kernels
# =============================================================================

@cuda.jit
def initPendulums(theta1, theta2, v1, v2, t1_min, t1_max, t2_min, t2_max):
    """
    Initialize the simulation grid.
    Each thread sets the initial conditions based on its grid position.
    The horizontal axis (i) is mapped to theta1 in [t1_min, t1_max]
    and the vertical axis (j) is mapped to theta2 in [t2_min, t2_max].
    Angular velocities (v1, v2) are set to zero.
    """
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
    """
    Update the state of each double pendulum using Euler integration.
    Uses the standard double pendulum equations (with m1 = m2 = 1, l1 = l2 = 1, g = 9.81).
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

        # Current state
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

        # Euler integration update
        w1 += a1 * dt
        w2 += a2 * dt
        t1 += w1 * dt
        t2 += w2 * dt

        # Write back updated state
        theta1[idx] = t1
        theta2[idx] = t2
        v1[idx] = w1
        v2[idx] = w2

@cuda.jit
def updateColors(theta1, theta2, img):
    """
    Compute a grayscale value (0–255) for each pixel based on the pendulum angles.
    The mapping is based on the average of the normalized sine values of theta1 and theta2.
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < WIDTH and j < HEIGHT:
        idx = j * WIDTH + i
        norm1 = (math.sin(theta1[idx]) + 1.0) / 2.0
        norm2 = (math.sin(theta2[idx]) + 1.0) / 2.0
        gray_val = int(((norm1 + norm2) / 2.0) * 255)
        img[idx] = gray_val

# =============================================================================
# Mouse Callback for Zoom Selection
# =============================================================================

def mouse_callback(event, x, y, flags, param):
    global selecting, selection_start, selection_end
    global current_t1_min, current_t1_max, current_t2_min, current_t2_max
    # Use the mutable flag list so we can trigger reinit
    global simulation_needs_reinit

    if event == cv2.EVENT_LBUTTONDOWN:
        selection_start = (x, y)
        selection_end = (x, y)
        selecting = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            selection_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if selecting:
            selection_end = (x, y)
            selecting = False
            # Compute the selected rectangle (ensure proper ordering)
            x1, y1 = selection_start
            x2, y2 = selection_end
            left = min(x1, x2)
            right = max(x1, x2)
            top = min(y1, y2)
            bottom = max(y1, y2)
            # Convert pixel coordinates to angles based on the current bounds.
            new_t1_min = current_t1_min + (left / (WIDTH - 1)) * (current_t1_max - current_t1_min)
            new_t1_max = current_t1_min + (right / (WIDTH - 1)) * (current_t1_max - current_t1_min)
            new_t2_min = current_t2_min + (top / (HEIGHT - 1)) * (current_t2_max - current_t2_min)
            new_t2_max = current_t2_min + (bottom / (HEIGHT - 1)) * (current_t2_max - current_t2_min)
            # Update the global angular boundaries
            current_t1_min = new_t1_min
            current_t1_max = new_t1_max
            current_t2_min = new_t2_min
            current_t2_max = new_t2_max

            simulation_needs_reinit[0] = True
            print("Zooming to:")
            print("  Theta1: [{:.5f}, {:.5f}] radians".format(new_t1_min, new_t1_max))
            print("  Theta2: [{:.5f}, {:.5f}] radians".format(new_t2_min, new_t2_max))

# =============================================================================
# Main Function
# =============================================================================

def main():
    global current_t1_min, current_t1_max, current_t2_min, current_t2_max
    global simulation_needs_reinit, selecting, selection_start, selection_end

    # Allocate device arrays for simulation state.
    size = WIDTH * HEIGHT
    theta1 = cuda.device_array(size, dtype=np.float64)
    theta2 = cuda.device_array(size, dtype=np.float64)
    v1     = cuda.device_array(size, dtype=np.float64)
    v2     = cuda.device_array(size, dtype=np.float64)
    # Allocate a single-channel image buffer on the device.
    img = cuda.device_array(size, dtype=np.uint8)

    # CUDA grid configuration.
    threads_per_block = (16, 16)
    blocks_per_grid_x = (WIDTH + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (HEIGHT + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Initialize the simulation using the current angular bounds.
    initPendulums[blocks_per_grid, threads_per_block](
        theta1, theta2, v1, v2,
        current_t1_min, current_t1_max, current_t2_min, current_t2_max
    )
    cuda.synchronize()

    dt = 0.01  # Time step

    # Create an OpenCV window and set the mouse callback.
    window_name = "Double Pendulum"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        # Update the simulation.
        updatePendulums[blocks_per_grid, threads_per_block](theta1, theta2, v1, v2, dt)
        # Update the grayscale image.
        updateColors[blocks_per_grid, threads_per_block](theta1, theta2, img)
        cuda.synchronize()

        # If a zoom selection was made, reinitialize the simulation with new bounds.
        if simulation_needs_reinit[0]:
            initPendulums[blocks_per_grid, threads_per_block](
                theta1, theta2, v1, v2,
                current_t1_min, current_t1_max, current_t2_min, current_t2_max
            )
            cuda.synchronize()
            simulation_needs_reinit[0] = False
            # Reset the selection variables so the rectangle is no longer displayed.
            selection_start = None
            selection_end = None

        # Copy image from device to host and reshape.
        host_img = img.copy_to_host().reshape((HEIGHT, WIDTH))
        # Apply a colormap to create a colorful display.
        colored_img = cv2.applyColorMap(host_img, cv2.COLORMAP_JET)

        # If a selection is in progress, draw the selection rectangle.
        if selecting and selection_start is not None and selection_end is not None:
            cv2.rectangle(colored_img, selection_start, selection_end, (255, 255, 255), 1)

        # Prepare a text string that indicates the current angular bounds (in degrees).
        text = ("Theta1: [{:.2f}, {:.2f}] deg, Theta2: [{:.2f}, {:.2f}] deg"
                .format(current_t1_min * 180/math.pi, current_t1_max * 180/math.pi,
                        current_t2_min * 180/math.pi, current_t2_max * 180/math.pi))
        # Overlay the text onto the image.
        cv2.putText(colored_img, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, colored_img)
        key = cv2.waitKey(1)
        if key == 27:  # Esc key to exit.
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
