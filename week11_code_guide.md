# คำแนะนำซอร์สโค้ด - สัปดาห์ที่ 11 (Code Guide)
## การทำงานกับ Ultralytics YOLOv8 Inference & OpenCV Overlay

---

## 1. คำอธิบายไฟล์โค้ดประจำสัปดาห์

| ชื่อไฟล์ | วัตถุประสงค์หลัก | คำสั่งสำหรับรัน |
|---|---|---|
| **[`yolo_inference_demo.py`](yolo_inference_demo.py)** | สคริปต์ตรวจจับวัตถุ สกัดพิกัด Bounding Box และตีกรอบพร้อมแจ้งเตือนใน OpenCV | `python yolo_inference_demo.py` |

---

## 2. โครงสร้างโค้ดหลักใน `yolo_inference_demo.py`

```python
from ultralytics import YOLO
import cv2

# โหลด Weights สำเร็จรูป YOLOv8 Nano
model = YOLO("yolov8n.pt")

# รัน Inference กับไฟล์ภาพ หรือกล้องสด (source=0)
results = model("image.jpg", conf=0.5)

for r in results:
    for box in r.boxes:
        # พิกัด 4 จุดของกรอบ (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{model.names[cls_id]} {confidence:.2f}"
        
        # แสดงผลผ่าน OpenCV
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```
