"""
สคริปต์ Multi-Object Tracking & Line-Crossing Counter
สำหรับนับจำนวนคน/รถยนต์ที่เดินข้ามเส้นตรวจจับ
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

def is_intersecting(p1, p2, q1, q2):
    """ฟังก์ชันเช็คการตัดกันของ 2 เส้นตรง (Line Intersection Math)"""
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def run_tracking_counter_simulation():
    print("=" * 65)
    print(" 🚦 RUNNING MULTI-OBJECT TRACKING & LINE COUNTER")
    print("=" * 65)

    # 1. โหลดโมเดล YOLO
    model = YOLO("yolo11n.pt")

    # 2. กำหนดพิกัดเส้นตรวจจับ (Line Virtual Gate)
    line_start = (100, 300)
    line_end   = (540, 300)

    # 3. จำลองการเคลื่อนที่ของรถยนต์ 2 คันข้ามเส้น (y=300)
    # รถคันที่ 1 (ID: 101) เคลื่อนที่ลงล่าง (In)
    # รถคันที่ 2 (ID: 102) เคลื่อนที่ขึ้นบน (Out)
    car_1_track = [(200, 260), (200, 290), (200, 320), (200, 350)]
    car_2_track = [(400, 340), (400, 310), (400, 280), (400, 250)]

    in_count = 0
    out_count = 0
    counted_ids = set()

    print("\n🚗 Simulating Object Trajectory Tracking:")
    
    # ทดสอบรถคันที่ 1
    car_id = 101
    for i in range(len(car_1_track)-1):
        p_prev = car_1_track[i]
        p_curr = car_1_track[i+1]
        
        if is_intersecting(line_start, line_end, p_prev, p_curr) and car_id not in counted_ids:
            in_count += 1
            counted_ids.add(car_id)
            print(f"   🟢 [Event] Object ID #{car_id} crossed line downwards! (IN: {in_count})")

    # ทดสอบรถคันที่ 2
    car_id = 102
    for i in range(len(car_2_track)-1):
        p_prev = car_2_track[i]
        p_curr = car_2_track[i+1]
        
        if is_intersecting(line_start, line_end, p_prev, p_curr) and car_id not in counted_ids:
            out_count += 1
            counted_ids.add(car_id)
            print(f"   🔴 [Event] Object ID #{car_id} crossed line upwards!   (OUT: {out_count})")

    print("\n" + "=" * 65)
    print(f" 📊 FINAL TRAFFIC STATISTICS:")
    print(f"    - Total Vehicles Entered (IN)  : {in_count}")
    print(f"    - Total Vehicles Exited  (OUT) : {out_count}")
    print(f"    - Net Vehicles Remaining      : {in_count - out_count}")
    print("=" * 65)

if __name__ == '__main__':
    run_tracking_counter_simulation()
