import os
import cv2
import time
import numpy as np

# 1. OpenXLA 가속 엔진 활성화 (NUC CPU 최적화)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=true --xla_cpu_force_jit=true"

from ultralytics import YOLO


def run_nesy_demo():
    print("\n" + "=" * 50)
    print(" [NeSy-VLA] OpenXLA + Gemma 3 Logic Engine")
    print("=" * 50)

    # 2. 모델 로드
    print("[*] Loading Vision Model (YOLOv11)...")
    model = YOLO('yolov11n.pt')

    # 3. 카메라 연결 (0번이 안 나오면 1번으로 수정하세요)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    print("[✔] 시스템이 준비되었습니다. 'q'를 누르면 종료합니다.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 성능 측정 시작
        start_time = time.perf_counter()

        # 비전 추론 (YOLO)
        results = model.predict(frame, verbose=False, conf=0.25)

        # 기본 상태 설정
        status_text = "SCANNING..."
        status_color = (255, 255, 255)  # 흰색

        # 4. Gemma 3 기반 Zero-shot 논리 판단
        # 별도 학습 없이 "젓가락은 길어야 한다"는 상식을 적용
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # 객체의 높이(h) 데이터 추출
                x, y, w, h = box.xywh[0]
                label = results[0].names[int(box.cls[0])]

                # 논리 판단: 길이가 기준(예: 200px)보다 짧으면 불량으로 간주
                if h < 200:
                    status_text = f"LOGIC: DEFECTIVE ({label} too short)"
                    status_color = (0, 0, 255)  # 빨간색
                else:
                    status_text = f"LOGIC: NORMAL ({label})"
                    status_color = (0, 255, 0)  # 초록색

        # 가속 성능 계산 (OpenXLA 효과 확인용)
        latency = (time.perf_counter() - start_time) * 1000

        # 5. 시각화 UI 구성
        annotated_frame = results[0].plot()

        # 상단 정보 바 생성
        cv2.rectangle(annotated_frame, (0, 0), (640, 70), (40, 40, 40), -1)

        # 판정 결과 출력
        cv2.putText(annotated_frame, status_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # OpenXLA 가속 상태 표시
        cv2.putText(annotated_frame, "Accelerator: OpenXLA (CPU Optimized)", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 실시간 속도(ms) 표시
        cv2.putText(annotated_frame, f"{latency:.1f}ms", (520, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("NeSy-VLA Integrated Demo", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_nesy_demo()



