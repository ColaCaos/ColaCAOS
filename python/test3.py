import cv2
import numpy as np

def main():
    # Open video capture
    cap = cv2.VideoCapture(0)
    
    # Try to set a higher frame rate (may not always work)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Define HSV range for blue color tracking (adjust as needed)
    # H: 100 to 140, S: 150 to 255, V: 0 to 255 is a common range for blue
    lower_h, lower_s, lower_v = 100, 150, 0
    upper_h, upper_s, upper_v = 140, 255, 255
    
    # Create a window and make it resizable
    cv2.namedWindow("GPU Accelerated Blue Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GPU Accelerated Blue Tracking", 1280, 720)

    # Initialize GPU Mats
    gpu_frame = cv2.cuda_GpuMat()
    gpu_hsv = cv2.cuda_GpuMat()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reduce resolution to increase processing speed
        frame = cv2.resize(frame, (640, 360))

        # Upload the frame to GPU
        gpu_frame.upload(frame)

        # Convert BGR to HSV on GPU
        gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)

        # Split into channels (H, S, V)
        gpu_h_channel, gpu_s_channel, gpu_v_channel = cv2.cuda.split(gpu_hsv)
        
        # Threshold each channel using compare() on GPU
        # For Hue
        gpu_h_lower = cv2.cuda.compare(gpu_h_channel, lower_h, cv2.CMP_GE)
        gpu_h_upper = cv2.cuda.compare(gpu_h_channel, upper_h, cv2.CMP_LE)
        gpu_h_mask = cv2.cuda.bitwise_and(gpu_h_lower, gpu_h_upper)

        # For Saturation
        gpu_s_lower = cv2.cuda.compare(gpu_s_channel, lower_s, cv2.CMP_GE)
        gpu_s_upper = cv2.cuda.compare(gpu_s_channel, upper_s, cv2.CMP_LE)
        gpu_s_mask = cv2.cuda.bitwise_and(gpu_s_lower, gpu_s_upper)

        # For Value
        gpu_v_lower = cv2.cuda.compare(gpu_v_channel, lower_v, cv2.CMP_GE)
        gpu_v_upper = cv2.cuda.compare(gpu_v_channel, upper_v, cv2.CMP_LE)
        gpu_v_mask = cv2.cuda.bitwise_and(gpu_v_lower, gpu_v_upper)

        # Combine H, S, and V masks
        gpu_hs_mask = cv2.cuda.bitwise_and(gpu_h_mask, gpu_s_mask)
        gpu_mask = cv2.cuda.bitwise_and(gpu_hs_mask, gpu_v_mask)

        # Download mask to CPU for contour detection
        mask = gpu_mask.download()

        # Optional morphological operations on CPU (can also be done on GPU)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        # Find contours on CPU
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Choose the largest contour as the marker
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw the center on the frame
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("GPU Accelerated Blue Tracking", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
