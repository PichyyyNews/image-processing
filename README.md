# 📷 Digital Image Processing (การประมวลผลภาพดิจิทัล)
### รหัสวิชา 31909-2007

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.6.0-green?logo=opencv&logoColor=white)
![Miniconda](https://img.shields.io/badge/Miniconda-Environment-orange?logo=anaconda&logoColor=white)
![VS Code](https://img.shields.io/badge/VS%20Code-IDE-blue?logo=visualstudiocode&logoColor=white)

คลังเอกสารประกอบการเรียนวิชา **การประมวลผลภาพดิจิทัล (Digital Image Processing)** ระดับ ปวส./ปริญญาตรี สาขาเทคโนโลยีสารสนเทศ ครอบคลุมเนื้อหาตั้งแต่พื้นฐานการประมวลผลภาพด้วย OpenCV ไปจนถึงการนำโมเดล AI / Computer Vision มาประยุกต์ใช้งานจริง

---

## 📚 เนื้อหารายสัปดาห์

| สัปดาห์ | หัวข้อ | ไฟล์เนื้อหา | ไฟล์สไลด์ | โค้ด / คำแนะนำ |
|:---:|---|---|---|---|
| 1 | บทนำ + ติดตั้งสภาพแวดล้อม Miniconda & VS Code | [week01_tutorial_basic_setup.md](week01/week01_tutorial_basic_setup.md) | — | [check_env.py](check_env.py) |
| 2 | การประมวลผลพิกเซลและการดำเนินการเรขาคณิต | [week02_tutorial_image_manipulation.md](week02/week02_tutorial_image_manipulation.md) | [week02_slide_guide.md](week02/week02_slide_guide.md) | — |
| 3 | การจัดการแสง สี และการกรองภาพ | [week03_tutorial_contrast_filtering.md](week03/week03_tutorial_contrast_filtering.md) | [week03_slide_outline.md](week03/week03_slide_outline.md) | — |
| 4 | การสกัดเส้นขอบและ Morphological Operations | [week04_tutorial_edge_morphology.md](week04/week04_tutorial_edge_morphology.md) | [week04_slide_outline.md](week04/week04_slide_outline.md) | [week04_resources.md](week04/week04_resources.md) |
| 5 | การตรวจจับโครงร่างและ Auto-Cropper | [week05_tutorial_contour_detection.md](week05/week05_tutorial_contour_detection.md) | [week05_slide_outline.md](week05/week05_slide_outline.md) | [codeweek5/](week05/codeweek5/) |
| 6 | โดเมนความถี่ (DFT/FFT) และ Image Inpainting | [week06_detailed_guide.md](week06/week06_detailed_guide.md) | [week06_slide_outline.md](week06/week06_slide_outline.md) | [week06_code_guide.md](week06/week06_code_guide.md) |
| 7 | การจับคู่จุดเด่นภาพ (SIFT, ORB, Homography) | [week07_detailed_guide.md](week07/week07_detailed_guide.md) | [week07_slide_outline.md](week07/week07_slide_outline.md) | [week07_code_guide.md](week07/week07_code_guide.md) |
| 8 | 🔬 การทดสอบกลางภาคเรียน (Midterm Review) | [week08_midterm_review_guide.md](week08/week08_midterm_review_guide.md) | [week08_slide_outline.md](week08/week08_slide_outline.md) | [week08_course_outline.md](week08/week08_course_outline.md) |
| 9 | Deep Learning & CNN ด้วย PyTorch | [week09_detailed_guide.md](week09/week09_detailed_guide.md) | [week09_slide_outline.md](week09/week09_slide_outline.md) | [week09_train_mnist.py](week09/week09_train_mnist.py), [week09_infer_mnist.py](week09/week09_infer_mnist.py) |
| 10 | Transfer Learning (MobileNetV3) & ONNX | [week10_detailed_guide.md](week10/week10_detailed_guide.md) | [week10_slide_outline.md](week10/week10_slide_outline.md) | [week10_train_transfer_onnx.py](week10/week10_train_transfer_onnx.py), [week10_infer_onnx.py](week10/week10_infer_onnx.py) |
| 11 | YOLO Object Detection Inference | [week11_detailed_guide.md](week11/week11_detailed_guide.md) | [week11_slide_outline.md](week11/week11_slide_outline.md) | [week11_yolo_inference_demo.py](week11/week11_yolo_inference_demo.py) |
| 12 | Custom YOLO Training & Evaluation | [week12_detailed_guide.md](week12/week12_detailed_guide.md) | [week12_slide_outline.md](week12/week12_slide_outline.md) | [week12_train_custom_yolo.py](week12/week12_train_custom_yolo.py) |
| 13 | MediaPipe Pose & Hand Landmark Tracking | [week13_detailed_guide.md](week13/week13_detailed_guide.md) | [week13_slide_outline.md](week13/week13_slide_outline.md) | [week13_mediapipe_demo.py](week13/week13_mediapipe_demo.py) |
| 14 | 🏆 การนำเสนอโครงงาน Mini-Project | [week14_detailed_guide.md](week14/week14_detailed_guide.md) | [week14_slide_outline.md](week14/week14_slide_outline.md) | [week14_mini_project_guide.md](week14/week14_mini_project_guide.md) |
| 15 | 🔬 การทดสอบปลายภาคเรียน (Final Exam Review) | [week15_detailed_guide.md](week15/week15_detailed_guide.md) | [week15_slide_outline.md](week15/week15_slide_outline.md) | [week15_final_exam_guide.md](week15/week15_final_exam_guide.md) |

---

## 🗂️ โครงสร้างไฟล์ในโปรเจกต์

```
image-processing/
│
├── 📄 README.md                          ← ดรรชนีหลักของรายวิชา
├── 📊 COURSE_SUMMARY_REPORT.md           ← รายงานสรุปหลักสูตรฉบับสมบูรณ์ 15 สัปดาห์
├── 📄 course_syllabus_opencv.md          ← แผนการสอนรายวิชา 15 สัปดาห์
├── 📄 TROUBLESHOOTING.md                 ← คู่มือแก้ปัญหาการติดตั้งและรันโค้ด
├── 📄 book.md                            ← แหล่งอ้างอิงและหนังสือเรียน 15 สัปดาห์
├── 📄 requirements.txt                   ← รายการ package ที่จำเป็น
├── 🐍 check_env.py                       ← สคริปต์ตรวจสอบเวอร์ชัน Environment
│
├── 👁️ ml_model_training/                 ← โมดูลหลักสูตรการฝึกฝนโมเดล Computer Vision, YOLO Object Detection, Image Classification & Vision Losses
│   ├── README.md                         ← สารบัญหลักสูตร Computer Vision Model Training
│   ├── 01_image_ml_pipeline_and_data_quality.md ← บทที่ 1: สถาปัตยกรรม Image ML Pipeline, การคัดกรองภาพเบลอ (Laplacian) & Letterbox Resize
│   ├── 02_vision_overfitting_and_augmentation.md ← บทที่ 2: Vision Overfitting, Spatial Dropout, MixUp, CutMix, Mosaic (YOLO) & Transfer Learning
│   ├── 03_cv_evaluation_metrics_classification_detection.md ← บทที่ 3: ตัวชี้วัดภาพ: Top-1/Top-5 Acc, IoU, Precision-Recall Curve, mAP@0.5, mAP@0.5:0.95 & NMS
│   ├── 04_vision_loss_functions_mastery.md ← บทที่ 4: เจาะลึก Loss โมเดลภาพ: Label Smoothing, Focal Loss, วิวัฒนาการ Box Loss สู่ YOLO CIoU Loss
│   └── 05_vision_optimizers_training_and_troubleshooting.md ← บทที่ 5: Vision Optimizers (SGD/AdamW), Mixed Precision (AMP), Gradient Accumulation & Vision Troubleshooting Matrix
│
├── 📚 Week 1 – Basic Setup & Image I/O
│   ├── (Digital Image Processing) Week 1 .pdf
│   └── week1_tutorial_basic_setup.md
│
├── 📚 Week 2 – Image Manipulation & Geometric Transform
│   ├── week2_tutorial_image_manipulation.md
│   └── week2_slide_guide.md
│
├── 📚 Week 3 – Contrast Enhancement & Filtering Pipeline
│   ├── week3_tutorial_contrast_filtering.md
│   └── week3_slide_outline.md
│
├── 📚 Week 4 – Edge Detection & Morphological Operations
│   ├── week4_tutorial_edge_morphology.md
│   ├── week4_slide_outline.md
│   └── resourceweek4.md
│
├── 📚 Week 5 – Contour Detection & Smart Auto-Cropper
│   ├── week5_tutorial_contour_detection.md
│   ├── week5_slide_outline.md
│   ├── week5_course_outline.md
│   └── codeweek5/
│
├── 📚 Week 6 – Frequency Domain (DFT/FFT) & Image Inpainting
│   ├── week6_detailed_guide.md
│   ├── week6_slide_outline.md
│   └── week6_code_guide.md
│
├── 📚 Week 7 – Classical Feature Matching (SIFT, ORB, Homography)
│   ├── week7_tutorial_feature_matching.md
│   ├── week7_detailed_guide.md
│   ├── week7_slide_outline.md
│   ├── week7_course_outline.md
│   └── week7_code_guide.md
│
├── 📚 Week 8 – Midterm Examination & Practice Review
│   ├── week8_midterm_review_guide.md
│   ├── week8_slide_outline.md
│   └── week8_course_outline.md
│
├── 📚 Week 9 – Deep Learning & CNN Architecture (PyTorch)
│   ├── week9_tutorial_cnn_pytorch.md
│   ├── week9_detailed_guide.md
│   ├── week9_slide_outline.md
│   ├── week9_course_outline.md
│   ├── week9_code_guide.md
│   ├── train_mnist.py
│   ├── infer_mnist.py
│   ├── perceptron_calculation.py
│   └── pipeline_decision_tree.py
│
├── 📚 Week 10 – Transfer Learning (MobileNetV3), ONNX & OpenCV DNN
│   ├── week10_detailed_guide.md
│   ├── week10_tutorial_transfer_learning_onnx.md
│   ├── week10_slide_outline.md
│   ├── week10_course_outline.md
│   ├── week10_code_guide.md
│   ├── train_transfer_onnx.py
│   └── infer_onnx.py
│
├── 📚 Week 11 – YOLO Object Detection Inference & OpenCV Overlay
│   ├── week11_detailed_guide.md
│   ├── week11_tutorial_yolo_inference.md
│   ├── week11_slide_outline.md
│   ├── week11_course_outline.md
│   ├── week11_code_guide.md
│   └── yolo_inference_demo.py
│
├── 📚 Week 12 – Custom YOLO Model Training & mAP Metrics
│   ├── week12_detailed_guide.md
│   ├── week12_tutorial_custom_yolo.md
│   ├── week12_slide_outline.md
│   ├── week12_course_outline.md
│   ├── week12_code_guide.md
│   └── train_custom_yolo.py
│
├── 📚 Week 13 – MediaPipe Pose, Hand & Face Landmark Tracking
│   ├── week13_detailed_guide.md
│   ├── week13_tutorial_mediapipe.md
│   ├── week13_slide_outline.md
│   ├── week13_course_outline.md
│   ├── week13_code_guide.md
│   └── mediapipe_demo.py
│
├── 📚 Week 14 – AI Computer Vision Mini-Project Showcase
│   ├── week14_detailed_guide.md
│   ├── week14_slide_outline.md
│   ├── week14_course_outline.md
│   └── week14_mini_project_guide.md
│
└── 📚 Week 15 – Final Examination & Course Synthesis
    ├── week15_detailed_guide.md
    ├── week15_slide_outline.md
    ├── week15_course_outline.md
    └── week15_final_exam_guide.md
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
