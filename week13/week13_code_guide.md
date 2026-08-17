# คำแนะนำซอร์สโค้ด - สัปดาห์ที่ 13 (Code Guide)
## การทำงานกับ MediaPipe Hands & Pose Landmark Estimation

---

## 1. คำอธิบายไฟล์โค้ดประจำสัปดาห์

| ชื่อไฟล์ | วัตถุประสงค์หลัก | คำสั่งสำหรับรัน |
|---|---|---|
| **[`mediapipe_demo.py`](mediapipe_demo.py)** | สคริปต์ตรวจจับพิกัดมือ คำนวณเวกเตอร์ และจำลองระบบ Air Canvas บน OpenCV | `python mediapipe_demo.py` |

---

## 2. โครงสร้างโค้ดหลักใน `mediapipe_demo.py`

```python
import cv2
import numpy as np

def calculate_distance(p1, p2):
    """คำนวณระยะทาง Euclidean Distance ระหว่าง 2 จุด"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ตัวอย่างตรรกะ Pinch Gesture
# landmark 8 = index tip, landmark 4 = thumb tip
dist = calculate_distance(index_tip, thumb_tip)
if dist < 30:
    print("Pinch / Click Event Triggered!")
```
