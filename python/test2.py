import cv2
import numpy as np

def check_gpu_support():
    # Check if OpenCV is built with CUDA
    build_info = cv2.getBuildInformation()
    if "CUDA" in build_info and "cuDNN" in build_info:
        print("[INFO] OpenCV is built with CUDA and cuDNN support.")
    else:
        print("[WARNING] OpenCV is not built with CUDA/cuDNN support.")
        return False

    # Check available GPUs
    num_devices = cv2.cuda.getCudaEnabledDeviceCount()
    if num_devices > 0:
        print(f"[INFO] {num_devices} GPU(s) available for OpenCV.")
        return True
    else:
        print("[WARNING] No GPU devices found for OpenCV.")
        return False
def process_frame_on_gpu(frame, bg_subtractor):
    try:
        # Convert to grayscale on CPU
        gray_cpu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Upload grayscale image to GPU
        gpu_gray = cv2.cuda_GpuMat()
        gpu_gray.upload(gray_cpu)

        # Create a CUDA stream
        stream = cv2.cuda.Stream()

        # Call apply() with minimal arguments first
        # Try just the image: fgmask = bg_subtractor.apply(gpu_gray)
        fgmask = bg_subtractor.apply(gpu_gray, None, 0.01, stream)
        
        # Wait for GPU operations to complete
        stream.waitForCompletion()

        # fgmask should now be a GpuMat; download to CPU
        cpu_fgmask = fgmask.download()

        # Find contours on CPU
        contours, _ = cv2.findContours(cpu_fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, cpu_fgmask
    except Exception as e:
        print(f"[ERROR] GPU processing failed: {e}")
        return [], frame

def main():
    # Check GPU support
    if not check_gpu_support():
        return

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open the camera.")
        return

    # Create a CUDA-based background subtractor
    bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2()

    print("[INFO] Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame from the camera.")
            break

        # Process the frame on GPU
        contours, fgmask = process_frame_on_gpu(frame, bg_subtractor)

        # Draw bounding boxes around detected objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ignore small objects
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the original frame and the foreground mask
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Foreground Mask", fgmask)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("===== OpenCV Real-Time GPU Object Detection =====")
    main()
