# 🤖 คู่มือและตัวอย่างการฝึกฝนโมเดล Machine Learning (ML Model Training Guide)

ยินดีต้อนรับสู่โมดูล **ML Model Training & Evaluation Fundamentals** คลังบทเรียนและซอร์สโค้ดตัวอย่างภาษา Python สำหรับเรียนรู้กระบวนการฝึกฝนโมเดลการเรียนรู้ของเครื่อง (Machine Learning) ตั้งแต่ระดับพื้นฐานไปจนถึงการนำไปใช้งานจริงในภาคอุตสาหกรรม

---

## 📚 สารบัญเนื้อหาและสคริปต์ตัวอย่าง

| # | หัวข้อบทเรียน | ไฟล์เอกสารทฤษฎี | ไฟล์สคริปต์ตัวอย่าง Python |
|:---:|---|---|---|
| **1** | **ML Training Pipeline & Outlier Handling** | [`01_ml_training_pipeline.md`](01_ml_training_pipeline.md) | [`01_pipeline_and_outliers.py`](01_pipeline_and_outliers.py) |
| **2** | **Overfitting, Underfitting & Cross-Validation** | [`02_overfitting_underfitting.md`](02_overfitting_underfitting.md) | [`02_overfitting_cross_val.py`](02_overfitting_cross_val.py) |
| **3** | **Evaluation Metrics & Model Performance** | [`03_evaluation_metrics.md`](03_evaluation_metrics.md) | [`03_evaluation_metrics_demo.py`](03_evaluation_metrics_demo.py) |
| **4** | **Loss Functions vs Cost Functions** | [`04_loss_and_cost_functions.md`](04_loss_and_cost_functions.md) | — |
| **5** | **Optimizers, Gradient Descent & Schedulers** | [`05_optimizers_and_gradient_descent.md`](05_optimizers_and_gradient_descent.md) | — |
| **6** | **Training Monitoring & Troubleshooting Matrix** | [`06_training_monitoring_and_troubleshooting.md`](06_training_monitoring_and_troubleshooting.md) | — |

---

## 🎯 สรุปมโนทัศน์สำคัญในแต่ละบท

### 1. ML Training Pipeline & Outliers
* **7 ขั้นตอนสำคัญของ Pipeline:** Data Collection $\rightarrow$ Data Preprocessing $\rightarrow$ Feature Scaling $\rightarrow$ Train/Test Split $\rightarrow$ Model Training $\rightarrow$ Evaluation $\rightarrow$ Model Persistence (`joblib`/`pickle`)
* **การตรวจจับและจัดการ Outliers:** ใช้วิธี **IQR (Interquartile Range)** และ **Z-Score** เพื่อจัดการค่าสุดโต่งที่อาจส่งผลให้โมเดลคลาดเคลื่อน

### 2. Overfitting & Underfitting
* **Bias-Variance Tradeoff:**
  * **Underfitting (High Bias):** โมเดลเรียบง่ายเกินไป ทายผลลบทั้งชุด Train และ Test
  * **Overfitting (High Variance):** โมเดลจดจำข้อมูลมากเกินไป ผลลัพธ์ Train ดีเยี่ยม แต่ Test ต่ำ
* **วิธีแก้ไข:** การทำ **K-Fold / Stratified K-Fold Cross-Validation**, การใช้ **Regularization (L1 Lasso / L2 Ridge)**, และการสังเกต **Learning Curves**

### 3. Evaluation Metrics
* **Classification Metrics:** Confusion Matrix (TP, TN, FP, FN), Accuracy, Precision, Recall, F1-Score (Micro/Macro/Weighted), ROC-AUC Curve
* **Regression Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), $R^2$ Score

### 4. Loss Functions & Cost Functions
* **Loss vs Cost:** Loss คำนวณความผิดพลาดรายจุด $L(y_i, \hat{y}_i)$, Cost คือค่าเฉลี่ยความผิดพลาดรวมทั้งชุดข้อมูล $J(w, b)$
* **Classification Losses:** Binary Cross-Entropy (Log Loss), Categorical / Sparse Categorical Cross-Entropy, Focal Loss, Hinge Loss
* **Regression Losses:** MSE (L2), MAE (L1), Huber Loss (Smooth L1 - Robust ต่อ Outliers)
* **Object Detection & Metric Losses:** IoU, GIoU, DIoU, CIoU Loss, Triplet Loss

### 5. Optimizers, Gradient Descent & Learning Rates
* **Gradient Descent Math:** $\theta_{t+1} = \theta_t - \alpha \nabla_{\theta} J(\theta)$
* **Batch Variants:** Batch GD, Stochastic GD (SGD), Mini-batch GD
* **Evolution of Optimizers:** SGD $\rightarrow$ Momentum $\rightarrow$ RMSprop $\rightarrow$ Adam $\rightarrow$ **AdamW** (Decoupled Weight Decay)
* **Learning Rate Schedulers:** StepLR, ReduceLROnPlateau, Cosine Annealing with Warmup

### 6. Training Monitoring & Troubleshooting
* **Training Dynamics:** Epochs, Iterations/Steps, Batch Size
* **Loss Curve Diagnostics:** วิธีวิเคราะห์การแกว่ง การดิ่ง หรือการพุ่งขึ้นของ Train vs Validation Loss Curves
* **Vanishing & Exploding Gradients:** สาเหตุและวิธีแก้ (ReLU/GELU, ResNet Skip Connections, Gradient Clipping, BatchNorm)
* **Troubleshooting Matrix:** คู่มือแก้ปัญหา Loss = `NaN`/`Inf`, Loss Plateau, และ Loss Oscillation

---

## 🛠️ วิธีการรันสคริปต์ตัวอย่าง

เปิด Terminal ใน VS Code และ activate environment:

```bash
conda activate dip_env
```

รันสคริปต์ตัวอย่างตามต้องการ:

```bash
# 1. รันตัวอย่าง Pipeline และการลบ Outliers
python ml_model_training/01_pipeline_and_outliers.py

# 2. รันตัวอย่าง Cross-Validation และแก้ปัญหา Overfitting
python ml_model_training/02_overfitting_cross_val.py

# 3. รันตัวอย่างการคำนวณและพล็อตกราฟ Evaluation Metrics
python ml_model_training/03_evaluation_metrics_demo.py
```
