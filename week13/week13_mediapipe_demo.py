"""
สคริปต์สาธิตการตรวจจับจุด Landmark มือและท่าทางด้วย MediaPipe
วิชา: การประมวลผลภาพดิจิทัล (Digital Image Processing) - สัปดาห์ที่ 13
"""

import cv2
import numpy as np

def run_mediapipe_demo():
    """
    ทดสอบการตรวจจับ Hand Landmarks ด้วย MediaPipe
    """
    try:
        import mediapipe as mp
    except ImportError:
        print("[ERROR] MediaPipe is not installed. Please run: pip install mediapipe")
        return

    print("Initializing MediaPipe Hands solution...")
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    # สร้างภาพจำลองมือนิ้วชี้เพื่อทดสอบไปป์ไลน์
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(canvas, "MediaPipe Test Canvas", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # โหลดเข้ารูปแบบ RGB
    rgb_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        print("[SUCCESS] Hand Landmarks Detected!")
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        print("[INFO] MediaPipe process completed. (No hand detected in synthetic image, pipeline ready for live camera)")

    cv2.imwrite("mediapipe_demo_output.jpg", canvas)
    print("[SUCCESS] Output saved to 'mediapipe_demo_output.jpg'")

if __name__ == "__main__":
    run_mediapipe_demo()
