# 🎯 Template โปรเจกต์ที่ 2: การตรวจจับวัตถุด้วย YOLO (Object Detection with YOLOv8 / YOLO11)

ยินดีต้อนรับสู่ Template สำหรับทำโปรเจกต์ **Object Detection (การตรวจจับและระบุตำแหน่งวัตถุ)** เหมาะสำหรับงานที่ต้องการให้ AI ตอบว่า *"มีวัตถุอะไรบ้างในรูป อยู่ตรงไหน (Bounding Box) และมีกี่ชิ้น?"*

---

## 💡 ไอเดียหัวข้อโปรเจกต์ที่นักศึกษาสามารถนำไปทำได้
* 👷 **ระบบตรวจจับอุปกรณ์ความปลอดภัย PPE (Smart Safety Helmet & Vest Detection):** ตรวจจับคนสวม/ไม่สวมหมวกนิรภัยและเสื้อสะท้อนแสงในเขตก่อสร้าง
* 🚗 **ระบบตรวจนับยานพาหนะและการจราจร (Traffic Vehicle Detection & Counting):** ตรวจจับรถยนต์, รถจักรยานยนต์, รถบรรทุก, รถพยาบาล
* 🥬 **ระบบตรวจจับวัชพืชและศัตรูพืชในแปลงเกษตร (Weed & Pest Detection):** แยกต้นพืชหลักออกจากวัชพืชเพื่อสั่งงานหุ่นยนต์พ่นยา
* 📦 **ระบบตรวจนับสินค้าและบรรจุภัณฑ์บนสายพาน (Parcel & Packaging Inspection):** ตรวจจับกล่องสินค้าและเช็คฉลากบาร์โค้ด
* 🚫 **ระบบตรวจจับการบุกรุกพื้นที่หวงห้าม (Intrusion Detection):** ตรวจจับคนหรือสัตว์เดินข้ามรั้ว

---

## 📂 โครงสร้างโฟลเดอร์ Dataset มาตรฐานของ YOLO (YOLO Dataset Format)

YOLO กำหนดให้แยกโฟลเดอร์รูปภาพ (`images/`) และไฟล์พิกัด Label (`labels/`) ออกจากกัน โดยเชื่อมต่อด้วยไฟล์ตั้งค่า `data.yaml`:

```
my_detection_dataset/
│
├── data.yaml                     ← ไฟล์คอนฟิกบอกเส้นทางโฟลเดอร์และรายชื่อคลาส
│
├── images/
│   ├── train/                    ← รูปภาพชุดฝึกสอน (img_01.jpg, img_02.png, ...)
│   └── val/                      ← รูปภาพชุดประเมินผล
│
└── labels/
    ├── train/                    ← ไฟล์พิกัดกล่อง .txt ที่มีชื่อตรงกับชื่อรูปภาพ
    │   ├── img_01.txt            ← (เช่น img_01.jpg คู่กับ img_01.txt)
    │   └── img_02.txt
    └── val/
```

---

## 📝 รูปแบบข้อมูลพิกัดในไฟล์ `.txt` (Normalized Bounding Box Format)

แต่ละบรรทัดในไฟล์ `.txt` จะแทนวัตถุ 1 ชิ้นในภาพ โดยมีค่าตัวเลข 5 ค่า คั่นด้วยเว้นวรรค:

$$\text{Format:} \quad \langle\text{class\_id}\rangle \quad \langle x_{\text{center}}\rangle \quad \langle y_{\text{center}}\rangle \quad \langle\text{width}\rangle \quad \langle\text{height}\rangle$$

```
0 0.450000 0.520000 0.200000 0.350000
1 0.780000 0.610000 0.150000 0.220000
```

* ทุกค่าต้องทำการ Normalize ให้อยู่ในช่วง $[0.0, 1.0]$:
  * $x_{\text{center}} = \frac{\text{Center X}}{\text{Image Width}}$
  * $y_{\text{center}} = \frac{\text{Center Y}}{\text{Image Height}}$
  * $\text{width} = \frac{\text{Box Width}}{\text{Image Width}}$
  * $\text{height} = \frac{\text{Box Height}}{\text{Image Height}}$

---

## 📄 โครงสร้างไฟล์ `data.yaml` ตัวอย่าง

```yaml
path: ../my_detection_dataset  # root directory ของ dataset
train: images/train
val: images/val

# จำนวนคลาส
nc: 2

# รายชื่อคลาสเรียงตาม class_id (0, 1)
names:
  0: helmet
  1: no_helmet
```

> [!TIP]
> **เครื่องมือ Label ข้อมูลฟรีที่แนะนำ:**
> 1. [Roboflow](https://roboflow.com/) – วาดกล่องออนไลน์ สั่ง Export ออกมาเป็นฟอร์แมต **YOLOv8 PyTorch** ได้ทันที พร้อมสร้าง `data.yaml` ให้เสร็จสรรพ
> 2. [Labelme / LabelImg](https://github.com/HumanSignal/labelImg) – โปรแกรมวาดกล่องในเครื่องคอมพิวเตอร์

---

## 🚀 คำสั่งฝึกสอนโมเดล (Training Command)

### วิธีที่ 1: ผ่าน Terminal CLI
```bash
yolo detect train data=path/to/data.yaml model=yolo11n.pt epochs=50 imgsz=640 batch=16
```

### วิธีที่ 2: ผ่าน Python Script (`train_detection.py`)
```bash
python ml_model_training/project_templates/02_object_detection_yolo/train_detection.py
```

---

## 🔮 การทดสอบโมเดลและเปิดกล้อง Webcam (`predict_detection.py`)

```bash
# รันตรวจจับบนรูปภาพหรือเปิดกล้องเว็บแคม Real-time
python ml_model_training/project_templates/02_object_detection_yolo/predict_detection.py
```
