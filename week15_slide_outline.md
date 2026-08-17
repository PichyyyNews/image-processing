# โครงร่างสไลด์นำเสนอ - สัปดาห์ที่ 15
## สรุปภาพรวมรายวิชาและการเตรียมสอบปลายภาค (Final Examination Overview)

---

### Slide 1: หน้าปก (Title Slide)
* **หัวข้อ:** Comprehensive Synthesis & Final Examination Guide
* **วิชา:** การประมวลผลภาพดิจิทัล (Digital Image Processing)
* **สัปดาห์ที่ 15:** สรุปการเรียนรู้ตลอด 15 สัปดาห์

---

### Slide 2: สรุปวิวัฒนาการเทคโนโลยี (The Spectrum of Computer Vision)
```
Classical DIP (Weeks 1–7)                Deep Learning & AI (Weeks 9–13)
┌───────────────────────────────┐        ┌───────────────────────────────┐
│ Pixel Slicing, Gaussian Blur, │  ───>  │ CNNs, Transfer Learning, ONNX,│
│ CLAHE, Sobel, Canny, ORB      │        │ YOLOv8/v11, MediaPipe Pose    │
└───────────────────────────────┘        └───────────────────────────────┘
  (Hand-crafted Mathematical               (Data-driven Learned Features)
   Rule-based Processing)
```

---

### Slide 3: ประเด็นคำนวณที่ต้องทบทวนก่อนสอบ (Essential Equations)
* **Confusion Matrix & Precision-Recall:**
  $$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}$$
* **Intersection over Union (IoU):**
  $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
* **Joint Vector Angle:**
  $$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right)$$

---

### Slide 4: แนวทางข้อสอบปฏิบัติการปลายภาค (Practical Final Challenge)
* นักศึกษาจะได้รับการแจกไฟล์โมเดลสำเร็จรูป (เช่น `.onnx` หรือ `.pt`) และไฟล์วิดีโอทดสอบ
* **ภารกิจ:** เขียนสคริปต์ใน VS Code เพื่อ:
  1. โหลดโมเดลผ่าน OpenCV DNN หรือ Ultralytics YOLO
  2. รันประมวลผลวิดีโอเฟรมต่อเฟรม
  3. นับจำนวนวัตถุเฉพาะคลาส (เช่น นับจำนวนรถยนต์)
  4. แสดงผลกรอบ Bounding Box พร้อมค่า Confidence บนหน้าจอ OpenCV

---

### Slide 5: จริยธรรมและทิศทางอนาคต (AI Ethics & Privacy)
* ความเป็นส่วนตัวของข้อมูลภาพ (GDPR / PDPA): การเบลอใบหน้าและป้ายทะเบียน
* ข้อควรระวังเรื่อง Bias ในชุดข้อมูลภาพถ่าย
* การปรับใช้โมเดลในอุปกรณ์ขนาดเล็ก (Edge AI & Embedded Systems)
