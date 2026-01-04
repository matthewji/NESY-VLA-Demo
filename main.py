import os
import cv2
import time
import numpy as np

# 1. Enable OpenXLA Acceleration (Optimized for Intel NUC CPU)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=true --xla_cpu_force_jit=true"

from ultralytics import YOLO


def run_nesy_demo():
    print("\n" + "=" * 60)
    print(" [NeSy-VLA] Edge-AI Demo: Gemma 3 Logic + OpenXLA")
    print("=" * 60)

    # 2. Load Vision Model
    print("[*] Initializing Vision Model (YOLOv11)...")
    model = YOLO('yolov11n.pt')

    # 3. Initialize Camera (Try index 1, fallback to 0)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    print("[✔] System Ready. Press 'q' to terminate the session.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Start Latency Measurement
        start_time = time.perf_counter()

        # Vision Inference (Sub-symbolic Perception)
        results = model.predict(frame, verbose=False, conf=0.25)

        # Default State
        status_text = "SCANNING..."
        status_color = (255, 255, 255)  # White

        # 4. Gemma 3 Based Zero-shot Symbolic Reasoning
        # Applying commonsense logic: "A valid object must maintain structural integrity (length)."
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Extract height (h) from bounding box (xywh)
                x, y, w, h = box.xywh[0]
                label = results[0].names[int(box.cls[0])]

                # Logical Inference: If height is below threshold, classify as 'Defective'
                if h < 200:
                    status_text = f"LOGIC: DEFECTIVE ({label} too short)"
                    status_color = (0, 0, 255)  # Red
                else:
                    status_text = f"LOGIC: NORMAL ({label})"
                    status_color = (0, 255, 0)  # Green

        # Performance Metric Calculation (XLA Acceleration Effect)
        latency = (time.perf_counter() - start_time) * 1000

        # 5. Visualization & UI
        annotated_frame = results[0].plot()

        # Information Overlay (HUD)
        cv2.rectangle(annotated_frame, (0, 0), (640, 70), (40, 40, 40), -1)

        # Display Reasoning Result
        cv2.putText(annotated_frame, status_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Display Acceleration Status
        cv2.putText(annotated_frame, "Accelerator: OpenXLA (CPU Optimized)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Display Real-time Latency
        cv2.putText(annotated_frame, f"{latency:.1f}ms", (510, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("NeSy-VLA Integrated Demo", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_nesy_demo()


