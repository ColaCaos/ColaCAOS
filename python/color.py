import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def main():
    # Open video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Attempt a higher FPS

    # HSV range for green color (adjust if needed)
    lower_h, lower_s, lower_v = 35, 80, 50
    upper_h, upper_s, upper_v = 85, 255, 255

    # Create a window and make it resizable
    cv2.namedWindow("GPU Accelerated Green Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GPU Accelerated Green Tracking", 1280, 720)

    # Prepare GPU Mats
    gpu_frame = cv2.cuda_GpuMat()

    # Time measurement for FPS
    prev_time = time.time()

    # Initialize lists to store the last 100 x,y positions
    x_positions = []
    y_positions = []

    # Setup matplotlib figure for live plotting (y vs x)
    plt.ion()
    fig, ax = plt.subplots()
    # Initialize an empty line; 'b-' means blue line
    line, = ax.plot([], [], 'b-', linewidth=1)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Object Trajectory (Y vs X) - Last 100 Points')

    # Set plot limits (adjust based on your frame size)
    ax.set_xlim(0, 640)  # Assuming frame width is 640
    ax.set_ylim(360, 0)  # Assuming frame height is 360 (invert y-axis for image coordinates)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally, do not resize the frame to retain maximum resolution
        # frame = cv2.resize(frame, (640, 360))  # Commented out

        # Upload frame to GPU
        gpu_frame.upload(frame)

        # Convert to HSV on GPU
        gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)

        # Split HSV channels
        gpu_h_channel, gpu_s_channel, gpu_v_channel = cv2.cuda.split(gpu_hsv)

        # Get size and type for GpuMat creation
        size = gpu_h_channel.size()
        type_ = gpu_h_channel.type()

        # Create GPU Mats for scalar bounds
        gpu_h_lower_bound = cv2.cuda_GpuMat(size, type_)
        gpu_h_lower_bound.setTo((lower_h,))
        gpu_h_upper_bound = cv2.cuda_GpuMat(size, type_)
        gpu_h_upper_bound.setTo((upper_h,))

        gpu_s_lower_bound = cv2.cuda_GpuMat(size, type_)
        gpu_s_lower_bound.setTo((lower_s,))
        gpu_s_upper_bound = cv2.cuda_GpuMat(size, type_)
        gpu_s_upper_bound.setTo((upper_s,))

        gpu_v_lower_bound = cv2.cuda_GpuMat(size, type_)
        gpu_v_lower_bound.setTo((lower_v,))
        gpu_v_upper_bound = cv2.cuda_GpuMat(size, type_)
        gpu_v_upper_bound.setTo((upper_v,))

        # Compare operations on GPU for H channel
        gpu_h_lower_cmp = cv2.cuda.compare(gpu_h_channel, gpu_h_lower_bound, cv2.CMP_GE)
        gpu_h_upper_cmp = cv2.cuda.compare(gpu_h_channel, gpu_h_upper_bound, cv2.CMP_LE)
        gpu_h_mask = cv2.cuda.bitwise_and(gpu_h_lower_cmp, gpu_h_upper_cmp)

        # S channel
        gpu_s_lower_cmp = cv2.cuda.compare(gpu_s_channel, gpu_s_lower_bound, cv2.CMP_GE)
        gpu_s_upper_cmp = cv2.cuda.compare(gpu_s_channel, gpu_s_upper_bound, cv2.CMP_LE)
        gpu_s_mask = cv2.cuda.bitwise_and(gpu_s_lower_cmp, gpu_s_upper_cmp)

        # V channel
        gpu_v_lower_cmp = cv2.cuda.compare(gpu_v_channel, gpu_v_lower_bound, cv2.CMP_GE)
        gpu_v_upper_cmp = cv2.cuda.compare(gpu_v_channel, gpu_v_upper_bound, cv2.CMP_LE)
        gpu_v_mask = cv2.cuda.bitwise_and(gpu_v_lower_cmp, gpu_v_upper_cmp)

        # Combine H, S, and V masks
        gpu_hs_mask = cv2.cuda.bitwise_and(gpu_h_mask, gpu_s_mask)
        gpu_mask = cv2.cuda.bitwise_and(gpu_hs_mask, gpu_v_mask)

        # Download mask for contour detection
        mask = gpu_mask.download()

        # Morphological operations (adjust if object is too small)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        # Find contours on CPU
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None
        if contours:
            # Select the largest contour
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Draw the center on the frame
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("GPU Accelerated Green Tracking", frame)

        # Compute FPS and print to console
        current_time = time.time()
        elapsed = current_time - prev_time
        prev_time = current_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        print(f"FPS: {fps:.2f}")

        # Update plotting data if a new point is detected
        if cx is not None and cy is not None:
            x_positions.append(cx)
            y_positions.append(cy)

            # Keep only the last 100 points
            if len(x_positions) > 100:
                x_positions = x_positions[-100:]
                y_positions = y_positions[-100:]

            # Perform spline interpolation for a smooth curve
            if len(x_positions) >= 4:  # Minimum number of points for splprep
                try:
                    # Parametrize the points
                    tck, u = splprep([x_positions, y_positions], s=5)
                    # Evaluate the spline for smoothness
                    u_new = np.linspace(0, 1, 100)
                    x_new, y_new = splev(u_new, tck, der=0)
                except Exception as e:
                    print(f"Interpolation error: {e}")
                    x_new, y_new = x_positions, y_positions
            else:
                x_new, y_new = x_positions, y_positions

            # Update the plot data (y vs x)
            line.set_xdata(x_new)
            line.set_ydata(y_new)
            ax.relim()
            ax.autoscale_view()

            # Redraw the plot
            plt.draw()
            plt.pause(0.001)  # Short pause to allow the plot to update

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()  # Show final plot when done

if __name__ == "__main__":
    main()
