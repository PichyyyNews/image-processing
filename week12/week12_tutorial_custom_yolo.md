# ปฏิบัติการการเรียนรู้ (Lab Tutorial) - สัปดาห์ที่ 12
## การสร้างโมเดลตรวจจับวัตถุส่วนบุคคล (Custom YOLO Training & Evaluation)

---

## ขั้นตอนที่ 1: การสร้างโครงสร้างไฟล์ Dataset
สร้างโฟลเดอร์สำหรับเก็บภาพและไฟล์ป้ายกำกับตามโครงสร้างมาตรฐาน YOLO:

```
custom_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

---

## ขั้นตอนที่ 2: การเขียนไฟล์ `data.yaml`
สร้างไฟล์ `custom_dataset/data.yaml` บรรจุเนื้อหา:

```yaml
path: custom_dataset
train: train/images
val: val/images

nc: 2
names: ['object_a', 'object_b']
```

---

## ขั้นตอนที่ 3: สคริปต์การสั่งเทรนโมเดล (`train_custom_yolo.py`)

รันสคริปต์ [train_custom_yolo.py](train_custom_yolo.py) ผ่าน VS Code Terminal:

```bash
python train_custom_yolo.py
```

---

## สรุปผลสัมฤทธิ์ปฏิบัติการ
1. โมเดลจะสร้างโฟลเดอร์เก็บค่าน้ำหนัก `runs/detect/custom_yolo_model/weights/best.pt`
2. กราฟ `results.png` จะแสดงแนวโน้ม Loss ที่ลดลงและค่า mAP50 ที่เพิ่มขึ้นตามจำนวน Epochs
