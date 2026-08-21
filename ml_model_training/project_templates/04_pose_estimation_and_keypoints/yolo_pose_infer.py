"""
สคริปต์สกัดพิกัดข้อต่อ 17 จุด (17 Keypoints) และวาด Skeleton ด้วย YOLO-Pose
"""

import sys
import io
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import numpy as np
from ultralytics import YOLO

def infer_pose_and_skeleton(image_path=None, model_path="yolo11n-pose.pt"):
    print("=" * 65)
    print(" 🦴 RUNNING YOLO-POSE KEYPOINT EXTRACTION")
    print("=" * 65)

    # 1. โหลดโมเดล
    model = YOLO(model_path)

    # 2. เตรียมภาพ
    if image_path is None or not cv2.haveImageReader(image_path):
        print("🖼️ Generating sample image for pose testing...")
        img = np.ones((640, 640, 3), dtype=np.uint8) * 235
    else:
        img = cv2.imread(image_path)

    # 3. รัน Inference
    results = model.predict(source=img, verbose=False)
    result = results[0]

    # 4. สกัด Keypoints
    if result.keypoints is not None:
        kpts_xy = result.keypoints.xy.cpu().numpy()
        print(f"\n👤 Detected {len(kpts_xy)} person(s):")
        for i, person_kpts in enumerate(kpts_xy):
            print(f"   Person {i+1}: Total {len(person_kpts)} Keypoints Extracted")
            # จุดสำคัญ เช่น จมูก(0), ไหล่ซ้าย(5), ไหล่ขวา(6)
            if len(person_kpts) > 6:
                print(f"      - Nose       (0): {person_kpts[0]}")
                print(f"      - LeftShould (5): {person_kpts[5]}")
                print(f"      - RightShould(6): {person_kpts[6]}")

    # 5. พล็อตภาพผลลัพธ์
    annotated = result.plot()
    out_file = "pose_skeleton_result.jpg"
    cv2.imwrite(out_file, annotated)
    print(f"\n💾 Saved pose skeleton result to: '{out_file}'")

if __name__ == '__main__':
    infer_pose_and_skeleton()
