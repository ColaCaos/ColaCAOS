import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def create_kalman_filter():
    """
    Creates and initializes a Kalman filter for tracking 2D position.
    """
    kf = cv2.KalmanFilter(4, 2)  # 4 states (x, y, vx, vy), 2 measurements (x, y)

    # State transition matrix (A)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],  # x = x + vx
                                     [0, 1, 0, 1],  # y = y + vy
                                     [0, 0, 1, 0],  # vx = vx
                                     [0, 0, 0, 1]], dtype=np.float32)

    # Measurement matrix (H)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],  # We measure x
                                     [0, 1, 0, 0]], dtype=np.float32)  # We measure y

    # Process noise covariance (Q)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4

    # Measurement noise covariance (R)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2

    # Error covariance matrix (P)
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    # Initial state (x, y, vx, vy)
    kf.statePost = np.array([0, 0, 0, 0], dtype=np.float32)

    return kf

def main():
    # Open video capture (0 for default camera)
    cap = cv2.VideoCapture(1)

    # Attempt to set a higher frame rate (may not be honored by all cameras)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Define HSV range for green color tracking
    lower_h, lower_s, lower_v = 35, 80, 50
    upper_h, upper_s, upper_v = 85, 255, 255

    # Create a window and make it resizable
    window_name = "Kalman Filtered Green Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Prepare GPU Mats
    gpu_frame = cv2.cuda_GpuMat()

    # Initialize Kalman filter
    kalman = create_kalman_filter()

    # For FPS and Points/sec calculation
    prev_time = time.time()
    last_measure_time = time.time()
    frame_count = 0
    points_count = 0

    # Initialize lists to store the last 100 x,y positions for plotting
    x_positions = []
    y_positions = []

    # Setup matplotlib figure for live plotting (y vs x)
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-', linewidth=1)  # 'b-' for blue line
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Object Trajectory (Y vs X) - Last 100 Points')

    # Set plot limits based on expected frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)  # Invert y-axis to match image coordinates

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Optionally, do not resize the frame to retain maximum resolution
        # Uncomment the next line if you want to resize
        # frame = cv2.resize(frame, (640, 360))  # Adjust as needed

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

        # S channel comparison
        gpu_s_lower_cmp = cv2.cuda.compare(gpu_s_channel, gpu_s_lower_bound, cv2.CMP_GE)
        gpu_s_upper_cmp = cv2.cuda.compare(gpu_s_channel, gpu_s_upper_bound, cv2.CMP_LE)
        gpu_s_mask = cv2.cuda.bitwise_and(gpu_s_lower_cmp, gpu_s_upper_cmp)

        # V channel comparison
        gpu_v_lower_cmp = cv2.cuda.compare(gpu_v_channel, gpu_v_lower_bound, cv2.CMP_GE)
        gpu_v_upper_cmp = cv2.cuda.compare(gpu_v_channel, gpu_v_upper_bound, cv2.CMP_LE)
        gpu_v_mask = cv2.cuda.bitwise_and(gpu_v_lower_cmp, gpu_v_upper_cmp)

        # Combine H, S, and V masks
        gpu_hs_mask = cv2.cuda.bitwise_and(gpu_h_mask, gpu_s_mask)
        gpu_mask = cv2.cuda.bitwise_and(gpu_hs_mask, gpu_v_mask)

        # Download mask for contour detection
        mask = gpu_mask.download()

        # Morphological operations to reduce noise (adjust iterations as needed)
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

        if cx is not None and cy is not None:
            # Kalman filter update step
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            kalman.correct(measurement)

        # Kalman filter prediction step
        predicted = kalman.predict()
        predicted_x, predicted_y = int(predicted[0]), int(predicted[1])

        # Get the estimated position from the filter's statePost
        estimated_x = int(kalman.statePost[0])
        estimated_y = int(kalman.statePost[1])

        # Add the estimated position to the plotting data
        x_positions.append(estimated_x)
        y_positions.append(estimated_y)

        if len(x_positions) > 100:
            x_positions = x_positions[-100:]
            y_positions = y_positions[-100:]

        # Update the plot
        line.set_xdata(x_positions)
        line.set_ydata(y_positions)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        # Draw the estimated position on the frame
        cv2.circle(frame, (estimated_x, estimated_y), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"({estimated_x}, {estimated_y})", (estimated_x + 10, estimated_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow(window_name, frame)

        # Compute FPS
        current_time = time.time()
        elapsed = current_time - prev_time
        prev_time = current_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        frame_count += 1

        # Update Points/sec if a new measurement was detected
        if cx is not None and cy is not None:
            # Check if the new point is sufficiently different from the last one
            if len(x_positions) == 0 or (abs(cx - x_positions[-1]) > 2 or abs(cy - y_positions[-1]) > 2):
                points_count += 1

        # Check if one second has passed to update FPS and Points/sec
        if current_time - last_measure_time >= 1.0:
            print(f"FPS: {frame_count} Points/sec: {points_count}", end='\r', flush=True)
            frame_count = 0
            points_count = 0
            last_measure_time = current_time

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
