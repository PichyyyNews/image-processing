"""
สคริปต์ AI Fitness Tracker: ระบบนับจำนวนครั้งการทำ Squat อัตโนมัติด้วย YOLO-Pose
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

def calculate_joint_angle(a, b, c):
    """คำนวณมุมระหว่าง 3 จุดข้อต่อ A - B - C (องศา)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def run_squat_counter_simulation():
    print("=" * 65)
    print(" 🏋️ RUNNING AI SQUAT FITNESS COUNTER")
    print("=" * 65)

    # 1. โหลดโมเดล YOLO-Pose
    model = YOLO("yolo11n-pose.pt")

    # 2. จำลองลำดับมุมเข่าของคนทำ Squat (ยืน -> ย่อ -> ยืน)
    knee_angle_simulation = [170.0, 140.0, 110.0, 85.0, 80.0, 110.0, 150.0, 168.0]
    
    count = 0
    stage = "up"
    
    print("\n⏱️ Simulating Repetition Detection:")
    for step, angle in enumerate(knee_angle_simulation, 1):
        # ตรรกะตรวจจับท่า Squat
        if angle < 90.0:
            stage = "down"
        if angle > 160.0 and stage == "down":
            stage = "up"
            count += 1
            print(f"   🎉 Step {step}: Repetition Completed! Total Count = {count} (Knee Angle: {angle:.1f}°)")
        else:
            print(f"   Step {step}: Knee Angle = {angle:5.1f}° | Current Stage: {stage:5s} | Count = {count}")
            
    print("\n" + "=" * 65)
    print(f" ✅ WORKOUT SESSION FINISHED! Total Squats Done: {count}")
    print("=" * 65)

if __name__ == '__main__':
    run_squat_counter_simulation()
