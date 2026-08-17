# ปฏิบัติการการเรียนรู้ (Lab Tutorial) - สัปดาห์ที่ 10
## การฝึกฝนโมเดล MobileNetV3 ด้วย Transfer Learning, ส่งออก ONNX และรันบน OpenCV DNN

---

## ขั้นตอนที่ 1: การจัดเตรียม Dataset
สร้างโครงสร้างโฟลเดอร์สำหรับ Dataset ในการจำแนกภาพ เช่น `dataset/cats` และ `dataset/dogs`:

```
dataset/
├── train/
│   ├── cats/
│   └── dogs/
└── val/
    ├── cats/
    └── dogs/
```

---

## ขั้นตอนที่ 2: สคริปต์การฝึกฝนด้วย PyTorch (`train_transfer_onnx.py`)

รันสคริปต์ [train_transfer_onnx.py](train_transfer_onnx.py) เพื่อทำ Fine-Tuning ค่าน้ำหนักและส่งออกเป็นไฟล์ `mobilenet_v3_cats_dogs.onnx`

```bash
python train_transfer_onnx.py
```

---

## ขั้นตอนที่ 3: สคริปต์การทำ Inference บน OpenCV (`infer_onnx.py`)

รันสคริปต์ [infer_onnx.py](infer_onnx.py) เพื่อทดสอบการอ่านโมเดล ONNX ผ่าน OpenCV DNN กับไฟล์ภาพ หรือกล้องวิดีโอเรียลไทม์

```bash
python infer_onnx.py --image test_cat.jpg
```

---

## สรุปผลสัมฤทธิ์ปฏิบัติการ
1. สังเกตว่ากระบวนการ Fine-Tuning ใช้เวลาไม่กี่ลูป (Epochs) แต่ได้ค่าความถูกต้องสูง >90%
2. ขนาดไฟล์ `.onnx` ของ MobileNetV3 Small มีขนาดเพียง ~9-12 MB
3. การทำ Inference บน OpenCV DNN ไม่จำเป็นต้องนำเข้าแพ็คเกจ `torch` ในสคริปต์ปลายทาง
