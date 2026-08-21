# 🏷️ Template โปรเจกต์ที่ 1: การจำแนกประเภทรูปภาพ (Image Classification with YOLO & PyTorch)

ยินดีต้อนรับสู่ Template สำหรับทำโปรเจกต์ **Image Classification (การจำแนกประเภทภาพ)** เหมาะสำหรับงานที่ต้องการให้ AI ตอบว่า *"รูปภาพนี้คือภาพอะไร?"* โดยผลลัพธ์จะเป็นชื่อคลาสและค่าความมั่นใจ (Confidence Score เช่น Cat 98.5%)

---

## 💡 ไอเดียหัวข้อโปรเจกต์ที่นักศึกษาสามารถนำไปทำได้
* 🌿 **ระบบคัดแยกโรคพืชและใบไม้ (Plant Disease Classification):** จำแนกใบข้าวโพดปกติ vs เป็นโรคราสนิม vs เป็นโรคใบจุด
* 🗑️ **ถังขยะอัจฉริยะคัดแยกประเภทขยะ (Smart Waste Classifier):** พลาสติก, ขวดแก้ว, กระป๋องอลูมิเนียม, ขยะอินทรีย์
* 🔬 **ระบบตรวจสอบตำหนิชิ้นส่วนอุตสาหกรรม (Industrial Defect Sorting):** แผงวงจรปกติ (Pass) vs บัดกรีหลุด (Defect)
* 🍎 **ระบบคัดเกรดผลไม้ (Fruit Ripeness & Grading):** กล้วยดิบ, กล้วยสุก, กล้วยงอม

---

## 📂 โครงสร้างโฟลเดอร์ Dataset มาตรฐาน (Dataset Directory Structure)

สำหรับงาน Classification โครงสร้างข้อมูลจะเรียบง่ายที่สุด โดยสร้างโฟลเดอร์ตามชื่อคลาส:

```
my_classification_dataset/
│
├── train/                        ← 70-80% ของข้อมูลทั้งหมด
│   ├── class_A/                  ← เช่น plastic_bottle/ (ใส่รูป .jpg, .png)
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── class_B/                  ← เช่น glass_bottle/
│       ├── img_101.jpg
│       └── img_102.jpg
│
├── val/                          ← 10-15% สำหรับประเมินผลระหว่างเทรน
│   ├── class_A/
│   └── class_B/
│
└── test/                         ← 10-15% ข้อมูลทดสอบที่ไม่เคยเห็นมาก่อน (Optional)
    ├── class_A/
    └── class_B/
```

> [!TIP]
> **คำแนะนำเรื่องจำนวนรูป:** ควรเตรียมภาพอย่างน้อย **50 - 200 รูปต่อ 1 คลาส** โดยถ่ายในสภาพแสง มุมมอง และพื้นหลังที่หลากหลาย

---

## 🚀 วิธีการฝึกสอนโมเดล (Model Training)

### วิธีที่ 1: ใช้คำสั่งผ่าน Terminal (CLI Command - ง่ายและเร็วที่สุด)
```bash
# เทรนด้วย YOLO11 Nano Classification Model
yolo classify train data=path/to/my_classification_dataset model=yolo11n-cls.pt epochs=50 imgsz=224 batch=16 device=cpu
```

### วิธีที่ 2: ใช้ Python Script (`train_classification.py`)
```bash
python ml_model_training/project_templates/01_image_classification/train_classification.py
```

---

## ⚙️ การปรับแต่ง Hyperparameters ที่สำคัญ

| พารามิเตอร์ | ค่าเริ่มต้น | คำแนะนำในการปรับแต่ง |
|---|:---:|---|
| `model` | `yolo11n-cls.pt` | เลือกรุ่น Nano (`n`) สำหรับรันบน CPU/Raspberry Pi หรือรุ่น Medium (`m`) เพื่อความแม่นยำสูง |
| `imgsz` | `224` | สำหรับงาน Classification ขนาด `224` หรือ `384` เหมาะสมที่สุด |
| `epochs` | `50` | หากข้อมูลน้อย (< 200 รูป) ปรับเป็น 30-50 Epochs; หากข้อมูลเยอะ ปรับเป็น 100 Epochs |
| `batch` | `16` | หากแรมไม่พอ (RAM/VRAM Out-of-Memory) ให้ลดเป็น `8` หรือ `4` |
| `lr0` | `0.001` | Learning Rate เริ่มต้น ปรับลดลงถ้า Loss สั่นสะเทือน |

---

## 🔮 การนำโมเดลไปใช้งานและแสดงผล (Inference & Prediction)

รันสคริปต์ทำนายผลรูปภาพ:
```bash
python ml_model_training/project_templates/01_image_classification/predict_classification.py
```
ผลลัพธ์จะแสดงรูปภาพพร้อม Overlay แถบเปอร์เซ็นต์ความน่าจะเป็นของแต่ละคลาส (Class Probability Bar)
