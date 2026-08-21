# ✂️ Template โปรเจกต์ที่ 3: การแบ่งส่วนวัตถุระดับพิกเซล (Instance Segmentation with YOLO-Seg)

ยินดีต้อนรับสู่ Template สำหรับทำโปรเจกต์ **Instance Segmentation (การแบ่งส่วนวัตถุระดับพิกเซล)** ซึ่งเหนือกว่า Object Detection ธรรมดาตรงที่สามารถสกัด **รูปร่างรูปหลายเหลี่ยม (Polygon Mask)** ของวัตถุออกมาได้อย่างแม่นยำ พร้อมคำนวณพื้นที่ (Area in pixels / $\text{cm}^2$)

---

## 💡 ไอเดียหัวข้อโปรเจกต์ที่นักศึกษาสามารถนำไปทำได้
* 🍃 **ระบบวัดเปอร์เซ็นต์ความเสียหายของใบไม้จากโรคพืช (Leaf Lesion Area Measurement):** คำนวณพื้นที่ใบไม้ทั้งหมด เทียบกับพื้นที่ที่เป็นแผลเน่า/รอยโรค
* 🚗 **ระบบตรวจจับช่องจราจรและยานพาหนะ (Lane Segmentation & Free Space Detection):** สกัดเส้นแบ่งเลนและพื้นที่ผิวถนนที่ว่างสำหรับรถยนต์ไร้คนขับ
* 🩸 **ระบบนับและวิเคราะห์เซลล์ทางการแพทย์ (Microscopic Cell & Tumor Segmentation):** ตีกรอบและวัดขนาดของเซลล์เม็ดเลือดหรือเนื้อร้าย
* 🥩 **ระบบตรวจวัดสัดส่วนไขมันในเนื้อสัตว์ (Meat Fat-to-Muscle Ratio Inspection):** วัดพื้นที่ไขมันแทรก (Marbling) ในเนื้อสเต็ก

---

## 📂 โครงสร้างโฟลเดอร์ Dataset และฟอร์แมต Polygon Label

```
my_segmentation_dataset/
│
├── data.yaml
├── images/ (train/, val/)
└── labels/ (train/, val/)
```

> **รูปแบบโครงสร้าง (Format):** `<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xn> <yn>`

```text
0 0.25 0.30 0.35 0.28 0.45 0.40 0.40 0.60 0.22 0.55
```

---

## 🚀 คำสั่งฝึกสอนโมเดล (Training Command)

```bash
# เทรนโมเดล YOLO-Seg ด้วย Terminal
yolo segment train data=path/to/data.yaml model=yolo11n-seg.pt epochs=50 imgsz=640 batch=8
```

หรือรันผ่าน Python Script:
```bash
python ml_model_training/project_templates/03_instance_segmentation/train_segmentation.py
```

---

## 🔮 การคำนวณพื้นที่ Mask และแสดงผล (`predict_segmentation.py`)

```bash
python ml_model_training/project_templates/03_instance_segmentation/predict_segmentation.py
```
ผลลัพธ์จะคำนวณพื้นที่ Mask ของแต่ละวัตถุ พร้อมวาด Mask สีโปร่งแสง (Transparent Mask Overlay) ลงบนภาพ
