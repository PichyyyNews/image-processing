# โครงร่างสไลด์นำเสนอ - สัปดาห์ที่ 11
## การตรวจจับวัตถุเป้าหมายด้วยแบบจำลอง YOLO (Object Detection Inference)

---

### Slide 1: หน้าปก (Title Slide)
* **หัวข้อ:** Object Detection with YOLO Architecture & OpenCV Integration
* **วิชา:** การประมวลผลภาพดิจิทัล (Digital Image Processing)
* **สัปดาห์ที่ 11:** ตรวจจับและระบุตำแหน่งวัตถุในระดับเรียลไทม์

---

### Slide 2: Classification vs Localization vs Object Detection
* **Classification:** บอกว่าในรูปมีวัตถุอะไรบ้าง (Single Label)
* **Localization:** บอกคลาสพร้อมตีกรอบ 1 วัตถุในภาพ
* **Object Detection:** ตรวจจับวัตถุ **หลายชิ้นต่างประเภทกัน** ในภาพเดียว พร้อมบอกพิกัด Bounding Box ทุกตัว

---

### Slide 3: ทำไมต้อง YOLO? (You Only Look Once)
* **Two-Stage Detectors (e.g. Faster R-CNN):** แยกขั้นตอนการเสนอ Region (RPN) และทำ Classify $\rightarrow$ แม่นยำแต่ช้า (~5-15 FPS)
* **One-Stage Detectors (YOLO):** ประมวลผลจากภาพดิบผ่าน Neural Network ครั้งเดียวจบ $\rightarrow$ เร็วมาก (30-150 FPS) เหมาะกับงาน Real-time Surveillance

---

### Slide 4: มโนทัศน์สำคัญใน YOLO
* **Intersection over Union (IoU):**
  $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
* **Non-Maximum Suppression (NMS):** คัดกรอง Bounding Box ที่ทับซ้อนกัน ซ้ำซ้อน ซ่อนกรอบที่มีค่า Confidence ต่ำกว่า Threshold

---

### Slide 5: โครงสร้างคำสั่ง Ultralytics YOLOv8
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt") # nano model lightweight
results = model("image.jpg")

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
```

---

### Slide 6: สรุปปฏิบัติการ LAB 11
* เขียนโปรแกรมสแกนตรวจจับบุคคล/ยานพาหนะจากคลิปวิดีโอ
* กำหนดเส้นสีเหลืองจำลอง (Virtual Boundary Line) บนหน้าจอ OpenCV
* หากศูนย์กลาง Bounding Box ของวัตถุตัดผ่านเส้น ให้ทำการตีกรอบสีแดงเตือน "Intrusion Detected!"
