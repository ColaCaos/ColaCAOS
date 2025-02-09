import numpy as np
import cv2
from numba import cuda
import math

# -------------------------------------------------------------------
# Simulation and Drawing Parameters
# -------------------------------------------------------------------
SIM_WIDTH = 36    # number of pendulums horizontally
SIM_HEIGHT = 36   # number of pendulums vertically
BOX_SIZE = 20     # each simulation is drawn in a 20x20 pixel box
IMG_WIDTH = SIM_WIDTH * BOX_SIZE   # overall image width (720 pixels)
IMG_HEIGHT = SIM_HEIGHT * BOX_SIZE   # overall image height (720 pixels)

# -------------------------------------------------------------------
# CUDA Kernels
# -------------------------------------------------------------------
@cuda.jit
def initPendulums(theta1, theta2, v1, v2):
    """
    Initialize each pendulum simulation with:
      theta1 = (-180 + i*10)° in radians, 
      theta2 = (-180 + j*10)° in radians,
    where (i,j) is the cell index.
    Angular velocities are set to zero.
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < SIM_WIDTH and j < SIM_HEIGHT:
        idx = j * SIM_WIDTH + i
        theta1[idx] = (-180.0 + i * 10.0) * math.pi / 180.0
        theta2[idx] = (-180.0 + j * 10.0) * math.pi / 180.0
        v1[idx] = 0.0
        v2[idx] = 0.0

@cuda.jit
def updatePendulumsMulti(theta1, theta2, v1, v2, dt, num_steps):
    """
    Update each pendulum simulation using Euler integration and the standard double pendulum equations.
    Instead of doing one update per kernel launch, this kernel runs a loop of 'num_steps' iterations.
    (Parameters: m1 = m2 = 1, l1 = l2 = 1, g = 9.81)
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if i < SIM_WIDTH and j < SIM_HEIGHT:
        idx = j * SIM_WIDTH + i
        
        # Run multiple simulation steps in a single kernel launch
        for s in range(num_steps):
            # Physical parameters
            m1 = 1.0
            m2 = 1.0
            l1 = 1.0
            l2 = 1.0
            g = 9.81
            
            # Retrieve current state
            t1 = theta1[idx]
            t2 = theta2[idx]
            w1 = v1[idx]
            w2 = v2[idx]
            
            delta = t1 - t2
            denom = 2 * m1 + m2 - m2 * math.cos(2 * delta)
            
            a1 = (-g * (2 * m1 + m2) * math.sin(t1)
                  - m2 * g * math.sin(t1 - 2 * t2)
                  - 2 * math.sin(delta) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * math.cos(delta))
                 ) / (l1 * denom)
            a2 = (2 * math.sin(delta) * (w1 * w1 * l1 * (m1 + m2)
                  + g * (m1 + m2) * math.cos(t1)
                  + w2 * w2 * l2 * m2 * math.cos(delta))
                 ) / (l2 * denom)
            
            # Euler integration update
            w1 = w1 + a1 * dt
            w2 = w2 + a2 * dt
            t1 = t1 + w1 * dt
            t2 = t2 + w2 * dt
            
            # Write back updated state
            theta1[idx] = t1
            theta2[idx] = t2
            v1[idx] = w1
            v2[idx] = w2

# -------------------------------------------------------------------
# Main Simulation and Drawing Routine
# -------------------------------------------------------------------
def main():
    size = SIM_WIDTH * SIM_HEIGHT
    
    # Allocate device arrays for state variables (angles and angular velocities)
    d_theta1 = cuda.device_array(size, dtype=np.float64)
    d_theta2 = cuda.device_array(size, dtype=np.float64)
    d_v1 = cuda.device_array(size, dtype=np.float64)
    d_v2 = cuda.device_array(size, dtype=np.float64)
    
    # Configure CUDA kernel launch parameters.
    threads_per_block = (16, 16)
    blocks_per_grid_x = (SIM_WIDTH + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (SIM_HEIGHT + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Initialize the pendulum states on the GPU
    initPendulums[blocks_per_grid, threads_per_block](d_theta1, d_theta2, d_v1, d_v2)
    cuda.synchronize()
    
    # Set dt (time step) and the number of simulation steps per frame.
    dt = np.float64(0.001)  # try a value between 0.005 and 0.01 for visible motion
    num_steps = 10         # number of simulation steps per frame (adjust for speed)
    
    # Drawing parameters: arm lengths (in pixels) for drawing each pendulum in a 20x20 box.
    L1 = 6  # length of first arm (in pixels)
    L2 = 6  # length of second arm (in pixels)
    
    while True:
        # Update the simulation state on the GPU with many steps per kernel launch.
        updatePendulumsMulti[blocks_per_grid, threads_per_block](d_theta1, d_theta2, d_v1, d_v2, dt, num_steps)
        cuda.synchronize()
        
        # Copy the simulation state from GPU to host.
        h_theta1 = d_theta1.copy_to_host()
        h_theta2 = d_theta2.copy_to_host()
        
        # Create a blank image for drawing (RGB, 720x720).
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        
        # For each simulation cell, compute and draw the pendulum.
        for j in range(SIM_HEIGHT):
            for i in range(SIM_WIDTH):
                idx = j * SIM_WIDTH + i
                cell_x = i * BOX_SIZE
                cell_y = j * BOX_SIZE
                # Define the pivot point (top center of the cell, with a slight margin at the top)
                pivot_x = cell_x + BOX_SIZE // 2
                pivot_y = cell_y + 2
                
                t1 = h_theta1[idx]
                t2 = h_theta2[idx]
                
                # Skip drawing if state is invalid.
                if math.isnan(t1) or math.isnan(t2):
                    continue
                
                # Compute the position of the first mass.
                x1 = int(pivot_x + L1 * math.sin(t1))
                y1 = int(pivot_y + L1 * math.cos(t1))
                
                # Compute the position of the second mass.
                x2 = int(x1 + L2 * math.sin(t2))
                y2 = int(y1 + L2 * math.cos(t2))
                
                # Draw the arms (lines) and joints (small circles).
                cv2.line(img, (pivot_x, pivot_y), (x1, y1), (255, 255, 255), 1)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.circle(img, (pivot_x, pivot_y), 1, (0, 0, 255), -1)
                cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
                cv2.circle(img, (x2, y2), 1, (255, 0, 0), -1)
        
        cv2.imshow("Grid of Pendulums", img)
        key = cv2.waitKey(1)
        if key == 27:  # Exit on pressing Esc.
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
