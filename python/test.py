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

def process_frame_on_gpu(frame):
    try:
        # Upload frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Convert to grayscale
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (15, 15), 0)
        gpu_blurred = gaussian_filter.apply(gpu_gray)

        # Apply Canny edge detection
        gpu_canny = cv2.cuda.createCannyEdgeDetector(50, 150)
        gpu_edges = gpu_canny.detect(gpu_blurred)

        # Download results back to CPU
        edges = gpu_edges.download()

        return edges
    except Exception as e:
        print(f"[ERROR] GPU processing failed: {e}")
        return frame

def main():
    # Check GPU support
    if not check_gpu_support():
        return

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open the camera.")
        return

    print("[INFO] Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame from the camera.")
            break

        # Process the frame on GPU
        processed_frame = process_frame_on_gpu(frame)

        # Display the original and processed frames
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Processed Frame (Edges)", processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("===== OpenCV Real-Time GPU Processing =====")
    main()
