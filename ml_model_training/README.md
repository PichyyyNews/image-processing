# 👁️ คู่มือหลักสูตรการฝึกฝนโมเดล Computer Vision & Image Machine Learning (Vision ML Training Master Curriculum)

ยินดีต้อนรับสู่หลักสูตร **Computer Vision, Image Classification & YOLO Object Detection Model Training** คลังบทเรียนระดับตำราวิชาการและซอร์สโค้ดตัวอย่างภาษา Python / PyTorch / OpenCV ที่ออกแบบมาสำหรับงานประมวลผลภาพดิจิทัลและปัญญาประดิษฐ์เชิงลึก (Deep Learning for Computer Vision) โดยเฉพาะ

---

## 🗺️ แผนผังเส้นทางการเรียนรู้ (Computer Vision Roadmap)

```mermaid
flowchart LR
    CH1["<b>บทที่ 1</b><br>Image ML Pipeline & Quality"] --> CH2["<b>บทที่ 2</b><br>Vision Augmentation & Overfit"]
    CH2 --> CH3["<b>บทที่ 3</b><br>Detection & Classification Metrics"]
    CH3 --> CH4["<b>บทที่ 4</b><br>Vision Loss Mastery & CIoU"]
    CH4 --> CH5["<b>บทที่ 5</b><br>Vision Optimizers & Troubleshooting"]
```

---

## 📚 สารบัญบทเรียนฉบับสมบูรณ์ (Curriculum Table of Contents)

| บทที่ | ไฟล์คู่มือการเรียนรู้ (Markdown) | สาระสำคัญและองค์ความรู้หลัก | โค้ด & ไดอะแกรมในบทเรียน |
|:---:|---|---|:---:|
| **1** | [`01_image_ml_pipeline_and_data_quality.md`](01_image_ml_pipeline_and_data_quality.md) | **Image ML Pipeline & Data Quality**<br>• สถาปัตยกรรม 7 ขั้นตอนของ Image ML Pipeline<br>• การตรวจจับภาพเบลอด้วย Laplacian Variance ($\text{Var}(\Delta I) < 100$)<br>• การทำ Letterbox Resize เพื่อคงสัดส่วน Aspect Ratio<br>• ImageNet Normalization & PyTorch Custom Dataset | 🔹 ผังงาน 7-Stage Vision Pipeline<br>🔹 ไดอะแกรม Letterbox Padding<br>🔹 โค้ด PyTorch Custom Dataset & Loader |
| **2** | [`02_vision_overfitting_and_augmentation.md`](02_vision_overfitting_and_augmentation.md) | **Vision Overfitting & Data Augmentation**<br>• ปัญหา Shortcut Learning (จำ Texture พื้นหลังแทนวัตถุ)<br>• Spatial Dropout (2D Dropout) และ DropBlock<br>• Data Augmentation ขั้นสูง: **MixUp, CutMix, Mosaic (YOLO)**<br>• Transfer Learning: Frozen Backbone vs Layer-wise Fine-Tuning | 🔹 ไดอะแกรม Shortcut Learning<br>🔹 ภาพจำลอง CutMix & Mosaic<br>🔹 โค้ด PyTorch CutMix + MobileNetV3 |
| **3** | [`03_cv_evaluation_metrics_classification_detection.md`](03_cv_evaluation_metrics_classification_detection.md) | **Computer Vision Evaluation Metrics**<br>• Top-1 vs Top-5 Accuracy ใน Image Classification<br>• **Intersection over Union (IoU)** คำนวณความทับซ้อนของ Bounding Box<br>• Precision-Recall Curve, **mAP@0.5** และ **mAP@0.5:0.95 (COCO)**<br>• Non-Maximum Suppression (NMS) ตัดกล่องซ้ำซ้อน | 🔹 ไดอะแกรมกล่อง IoU & P-R Curve<br>🔹 ผังการทำงาน NMS<br>🔹 โค้ดคำนวณ IoU, NMS & mAP@0.5 |
| **4** | [`04_vision_loss_functions_mastery.md`](04_vision_loss_functions_mastery.md) | **Vision Loss Functions & YOLO CIoU Loss**<br>• สถาปัตยกรรม Multi-Task Loss ใน YOLO (Box + Cls + DFL)<br>• Cross-Entropy with Label Smoothing & Focal Loss ($\gamma=2.0$)<br>• วิวัฒนาการ Bounding Box Loss: Smooth L1 $\rightarrow$ IoU $\rightarrow$ GIoU $\rightarrow$ DIoU $\rightarrow$ **CIoU**<br>• เจาะลึกสูตร Complete IoU (CIoU) ควบคุม Aspect Ratio $v$ | 🔹 ผัง Multi-Task Loss ใน YOLO<br>🔹 กราฟพฤติกรรม Focal Loss<br>🔹 โค้ด PyTorch Vectorized CIoU Loss |
| **5** | [`05_vision_optimizers_training_and_troubleshooting.md`](05_vision_optimizers_training_and_troubleshooting.md) | **Vision Optimizers & Troubleshooting Matrix**<br>• การเลือก Optimizer: SGD with Momentum (CNNs) vs **AdamW** (ViTs / YOLO)<br>• Automatic Mixed Precision (**AMP / FP16**) เร่งความเร็ว 2 เท่า<br>• Gradient Accumulation จำลอง Batch Size ขนาดใหญ่บน GPU เล็ก<br>• **Vision Troubleshooting Matrix Table:** แก้ปัญหา CUDA OOM, Box Loss = NaN, Box Collapse | 🔹 ผัง GPU Acceleration Pipeline<br>🔹 ตารางคู่มือ Troubleshooting บัคโมเดลภาพ<br>🔹 โค้ด PyTorch AMP Training Loop |

---

## 🎯 จุดเด่นของหลักสูตรชุดนี้

1. **เชื่อมโยงกับเนื้อหา OpenCV / PyTorch / YOLO ในโปรเจกต์ 100%:** สอดคล้องกับแล็บตั้งแต่ Week 1 ถึง Week 15
2. **ครบถ้วนในตัว (Self-Contained):** ในแต่ละบทประกอบด้วยทฤษฎีระดับสากล, สูตรคณิตศาสตร์ ($\LaTeX$), ไดอะแกรม Mermaid, ภาพประกอบ ASCII, และโค้ด PyTorch ที่นำไปรันได้ทันที
3. **คู่มือแก้ปัญหาภาคสนาม (Vision Troubleshooting Matrix):** ช่วยให้นักศึกษาและนักพัฒนาแก้ปัญหาที่พบบ่อยในการเทรนโมเดลภาพ เช่น แรมการ์ดจอเต็ม (`CUDA OOM`), Loss กลายเป็น `NaN`, หรือโมเดลหาวัตถุไม่เจอได้อย่างตรงจุด
