# ปฏิบัติการการเรียนรู้ (Lab Tutorial) - สัปดาห์ที่ 11
## การตรวจจับวัตถุเป้าหมายด้วยแบบจำลอง YOLO (Ultralytics YOLOv8 & OpenCV)

---

## ขั้นตอนที่ 1: การติดตั้งแพ็คเกจ Ultralytics
เปิด Terminal ใน VS Code และตรวจสอบความพร้อมของแพ็คเกจ `ultralytics`:

```bash
pip install ultralytics
```

---

## ขั้นตอนที่ 2: การรัน Inference ด้วยคำสั่ง CLI
ทดลองรันคำสั่ง CLI ของ YOLOv8 ผ่าน Terminal เพื่อดูผลลัพธ์การตีกรอบอัตโนมัติ:

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

---

## ขั้นตอนที่ 3: การเขียนสคริปต์ Python รวมกับ OpenCV (`yolo_inference_demo.py`)

เปิดไฟล์ [yolo_inference_demo.py](yolo_inference_demo.py) และสั่งรัน:

```bash
python yolo_inference_demo.py
```

---

## สรุปผลสัมฤทธิ์ปฏิบัติการ
1. เข้าใจโครงสร้างวัตถุ `Results` และการดึง Bounding Box ($x_1, y_1, x_2, y_2$)
2. สามารถวาด Overlay คำนวณพื้นที่ปลอดภัยและแสดงกรอบแจ้งเตือนบน OpenCV Window ได้
