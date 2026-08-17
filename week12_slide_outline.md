# โครงร่างสไลด์นำเสนอ - สัปดาห์ที่ 12
## การสร้างโมเดลตรวจจับวัตถุส่วนบุคคล (Custom YOLO Training & Evaluation)

---

### Slide 1: หน้าปก (Title Slide)
* **หัวข้อ:** Custom YOLO Model Training, Data Preparation & Performance Evaluation
* **วิชา:** การประมวลผลภาพดิจิทัล (Digital Image Processing)
* **สัปดาห์ที่ 12:** สร้างโมเดล AI ตรวจจับวัตถุเฉพาะทางด้วยตนเอง

---

### Slide 2: โครงสร้างไฟล์ชุดข้อมูลแบบ YOLO (YOLO Dataset Structure)
```
dataset/
├── data.yaml
├── train/
│   ├── images/ (img1.jpg, img2.jpg)
│   └── labels/ (img1.txt, img2.txt)
└── val/
    ├── images/
    └── labels/
```
* ไฟล์ Label `.txt` แต่ละบรรทัดบรรจุ:
  `<class_id> <x_center> <y_center> <width> <height>` (ค่าทุกตัวต้องถูก Normalize ให้อยู่ในช่วง 0.0 - 1.0)

---

### Slide 3: ไฟล์คอนฟิกูเรชัน `data.yaml`
```yaml
path: ../dataset
train: train/images
val: val/images

nc: 2
names: ['defect_scratch', 'defect_crack']
```

---

### Slide 4: คำสั่งฝึกฝนโมเดล (Training Command)
```python
from ultralytics import YOLO

# 1. โหลดโมเดลตั้งต้น
model = YOLO("yolov8n.pt")

# 2. สั่งเทรนด้วย custom dataset
results = model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    name="custom_detector"
)
```

---

### Slide 5: ตัวชี้วัดการประเมินผล (Evaluation Metrics)
* **Precision:** ความถูกต้องของการตีกรอบ $\rightarrow \frac{TP}{TP + FP}$
* **Recall:** ความสามารถในการเก็บวัตถุได้ครบ $\rightarrow \frac{TP}{TP + FN}$
* **mAP@50:** ค่า Average Precision เฉลี่ยทุกคลาสที่ IoU Threshold = 0.50
* **mAP@50-95:** ค่า Average Precision เฉลี่ยที่ IoU ตั้งแต่ 0.50 ถึง 0.95 (มาตรฐานความเนี้ยบสูงสุด)

---

### Slide 6: สรุปปฏิบัติการ LAB 12
* รวบรวม Dataset วัตถุในชีวิตประจำวัน หรือจุดบกพร่องชิ้นงาน
* สั่งรันเทรนและสังเกตไฟล์ผลลัพธ์ใน `runs/detect/custom_detector/`
* ทดสอบไฟล์ Weights `best.pt` กับรูปภาพใหม่เพื่อยืนยันประสิทธิภาพ
