# ปฏิบัติการการเรียนรู้ (Lab Tutorial) - สัปดาห์ที่ 13
## โมเดลการหาจุดข้อต่อร่างกายมนุษย์ด้วย MediaPipe (Hand & Pose Landmarks)

---

## ขั้นตอนที่ 1: การติดตั้งแพ็คเกจ MediaPipe
เปิด Terminal ใน VS Code และรันคำสั่ง:

```bash
pip install mediapipe
```

---

## ขั้นตอนที่ 2: การสั่งรันสคริปต์ตรวจจับพิกัดนิ้วและท่าทาง (`mediapipe_demo.py`)

รันสคริปต์ [mediapipe_demo.py](mediapipe_demo.py) ผ่าน VS Code Terminal:

```bash
python mediapipe_demo.py
```

---

## สรุปผลสัมฤทธิ์ปฏิบัติการ
1. สกัดพิกัดปลายนิ้วชี้และปลายนิ้วโป้ง
2. คำนวณระยะทาง Euclidean Distance เพื่อทำ Pinch Gesture
3. สร้างเส้นวาดภาพตามรอยเคลื่อนที่ของปลายนิ้วชี้ลงบนเฟรมวิดีโอ
