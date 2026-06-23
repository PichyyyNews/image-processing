# 📷 Digital Image Processing (การประมวลผลภาพดิจิทัล)
### รหัสวิชา 31909-2007

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.6.0-green?logo=opencv&logoColor=white)
![Miniconda](https://img.shields.io/badge/Miniconda-Environment-orange?logo=anaconda&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-IDE-blue?logo=visualstudiocode&logoColor=white)

คลังเอกสารประกอบการเรียนวิชา **การประมวลผลภาพดิจิทัล (Digital Image Processing)** ระดับ ปวส./ปริญญาตรี สาขาเทคโนโลยีสารสนเทศ ครอบคลุมเนื้อหาตั้งแต่พื้นฐานการประมวลผลภาพด้วย OpenCV ไปจนถึงการนำโมเดล AI / Computer Vision มาประยุกต์ใช้งานจริง

---

## 📚 เนื้อหารายสัปดาห์

| สัปดาห์ | หัวข้อ | ไฟล์เนื้อหา | ไฟล์สไลด์ | โค้ด |
|:---:|---|---|---|---|
| 1 | บทนำ + ติดตั้งสภาพแวดล้อม Miniconda & VS Code | [week1_tutorial_basic_setup.md](week1_tutorial_basic_setup.md) | — | — |
| 2 | การประมวลผลพิกเซลและการดำเนินการเรขาคณิต | [week2_tutorial_image_manipulation.md](week2_tutorial_image_manipulation.md) | [week2_slide_guide.md](week2_slide_guide.md) | — |
| 3 | การจัดการแสง สี และการกรองภาพ | [week3_tutorial_contrast_filtering.md](week3_tutorial_contrast_filtering.md) | [week3_slide_outline.md](week3_slide_outline.md) | — |
| 4 | การสกัดเส้นขอบและ Morphological Operations | [week4_tutorial_edge_morphology.md](week4_tutorial_edge_morphology.md) | [week4_slide_outline.md](week4_slide_outline.md) | — |
| 5 | การตรวจจับโครงร่างและ Auto-Cropper | [week5_tutorial_contour_detection.md](week5_tutorial_contour_detection.md) | [week5_slide_outline.md](week5_slide_outline.md) | [codeweek5/](codeweek5/) |
| 6 | โดเมนความถี่ (DFT/FFT) และ Image Inpainting | [week6_detailed_guide.md](week6_detailed_guide.md) | [week6_slide_outline.md](week6_slide_outline.md) | [week6_code_guide.md](week6_code_guide.md) |

---

## 🗂️ โครงสร้างไฟล์ในโปรเจกต์

```
image-processing/
│
├── 📄 README.md                          ← ไฟล์นี้
├── 📄 course_syllabus_opencv.md          ← แผนการสอนรายวิชาทั้งหมด 15 สัปดาห์
├── 📄 TROUBLESHOOTING.md                 ← คู่มือแก้ปัญหาการติดตั้งและรันโค้ด
├── 📄 requirements.txt                   ← รายการ package ที่จำเป็น
├── 🐍 check_env.py                       ← สคริปต์ตรวจสอบเวอร์ชัน Environment
│
├── 📚 Week 1
│   └── week1_tutorial_basic_setup.md    ← ติดตั้ง Miniconda, VS Code, OpenCV
│
├── 📚 Week 2
│   ├── week2_tutorial_image_manipulation.md
│   └── week2_slide_guide.md
│
├── 📚 Week 3
│   ├── week3_tutorial_contrast_filtering.md
│   └── week3_slide_outline.md
│
├── 📚 Week 4
│   ├── week4_tutorial_edge_morphology.md
│   └── week4_slide_outline.md
│
├── 📚 Week 5
│   ├── week5_tutorial_contour_detection.md
│   ├── week5_slide_outline.md
│   ├── week5_course_outline.md
│   └── codeweek5/
│       ├── week.py                       ← โค้ดตัวอย่าง Lab 5
│       └── image.jpg                     ← ภาพทดสอบ
│
└── 📚 Week 6
    ├── week6_detailed_guide.md           ← เนื้อหาละเอียดฉบับเต็ม (อ่านเหมือนหนังสือเรียน)
    ├── week6_slide_outline.md            ← โครงสร้างสไลด์สำหรับนำเสนอ
    └── week6_code_guide.md              ← ตัวอย่างโค้ด Lab 6
```

---

## ⚙️ การติดตั้งสภาพแวดล้อม (Environment Setup)

โปรเจกต์นี้ใช้ **Miniconda** ในการจัดการ Environment เพื่อให้ทุกเครื่องทำงานได้เหมือนกัน 100%

### ขั้นที่ 1: ติดตั้ง Miniconda
ดาวน์โหลดและติดตั้งได้ที่ [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### ขั้นที่ 2: สร้าง Environment จากไฟล์ config
สร้างไฟล์ `environment.yml` ที่ root ของโปรเจกต์:

```yaml
name: dip_env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10.12
  - numpy=1.24.3
  - matplotlib=3.7.1
  - opencv=4.6.0
  - pytorch=2.0.1
  - torchvision=0.15.2
  - cpuonly
  - pip
  - pip:
    - ultralytics==8.0.196
    - mediapipe==0.10.7
```

จากนั้นรันคำสั่ง:

```bash
conda env create -f environment.yml
conda activate dip_env
```

### ขั้นที่ 3: ตรวจสอบการติดตั้ง
```bash
python check_env.py
```

---

## 🛠️ เครื่องมือหลักที่ใช้ในวิชา

| เครื่องมือ | วัตถุประสงค์ |
|---|---|
| **VS Code** | IDE หลักสำหรับเขียนและรันโค้ด Python |
| **Miniconda** | จัดการ Virtual Environment และ Package |
| **OpenCV (cv2)** | ไลบรารีประมวลผลภาพหลัก |
| **NumPy** | การคำนวณ Matrix และข้อมูลภาพ |
| **Matplotlib** | แสดงผลภาพและกราฟ |
| **PyTorch** | เฟรมเวิร์ก Deep Learning (สัปดาห์ 9+) |
| **Ultralytics YOLO** | โมเดล Object Detection (สัปดาห์ 11+) |
| **MediaPipe** | โมเดลตรวจจับ Pose / Hand (สัปดาห์ 13) |

---

## 📖 เนื้อหาสำคัญที่เรียนในวิชานี้

```
Week 1-2   │ พื้นฐาน: Pixel, Channel, Resize, Flip, Rotate
Week 3-4   │ Image Enhancement: Histogram, Filtering, Edge Detection, Morphology  
Week 5-6   │ Contour Detection, Frequency Domain (DFT/FFT), Inpainting
Week 7     │ Classical Feature Matching: SIFT, ORB, BFMatcher
Week 8     │ 🔬 สอบกลางภาค
Week 9-10  │ Deep Learning: CNN, Transfer Learning (MobileNetV3), ONNX Export
Week 11-12 │ Object Detection: YOLO Inference & Custom Training
Week 13    │ Pose Estimation: MediaPipe Hand / Face / Body
Week 14    │ 🏆 นำเสนอโครงงาน Mini-Project
Week 15    │ 🔬 สอบปลายภาค
```

---

## 🐛 พบปัญหา?

ดู [TROUBLESHOOTING.md](TROUBLESHOOTING.md) สำหรับวิธีแก้ปัญหาทั่วไป เช่น:
- ติดตั้ง Package ไม่ผ่าน
- OpenCV เปิดหน้าต่างไม่ได้
- Conda activate ไม่ทำงาน

---

> **หมายเหตุ:** เอกสารในคลังนี้อยู่ระหว่างการพัฒนาตามลำดับสัปดาห์ที่สอน จะมีการเพิ่มเนื้อหาใหม่ต่อเนื่องทุกสัปดาห์
