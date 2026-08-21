# 🏋️ Template โปรเจกต์ที่ 4: การตรวจจับท่าทางและข้อต่อร่างกาย (Pose Estimation & Fitness Tracker with YOLO-Pose)

ยินดีต้อนรับสู่ Template สำหรับทำโปรเจกต์ **Pose Estimation & Keypoints Tracking (การตรวจจับท่าทางและข้อต่อร่างกาย)** เหมาะสำหรับงานวิเคราะห์ชีวกลศาสตร์ (Biomechanics), ติดตามการออกกำลังกาย, กายภาพบำบัด และตรวจจับพฤติกรรมเสี่ยง

---

## 💡 ไอเดียหัวข้อโปรเจกต์ที่นักศึกษาสามารถนำไปทำได้
* 🏋️ **ระบบนับจำนวนครั้งการออกกำลังกายอัตโนมัติ (Smart AI Fitness Rep Counter):** นับจำนวนครั้ง Squat, Push-up, Bicep Curl พร้อมเช็คว่าฟอร์มถูกต้องหรือไม่
* 🪑 **ระบบแจ้งเตือนท่านั่งทำงานผิดสุขลักษณะ (Ergonomics Sitting Posture Alert):** แจ้งเตือนเมื่อศีรษะก้มต่ำเกินไป (Text Neck) หรือหลังค่อม (Slouching)
* 👴 **ระบบตรวจจับการหกล้มของผู้สูงอายุ (Elderly Fall Detection):** แจ้งเตือนฉุกเฉินเมื่อแนวกระดูกสันหลังเอียงเป็นแนวนอนและลงไปกองกับพื้นอย่างรวดเร็ว
* 🥋 **ระบบวิเคราะห์ทักษะกีฬา (Sports Biomechanics Analysis):** วิเคราะห์มุมสวิงไม้กอล์ฟ, ท่าชู้ตบาสเกตบอล

---

## 🦴 ผังจุดข้อต่อ 17 Keypoints มาตรฐาน (COCO 17-Keypoints Topology)

YOLO-Pose และ MediaPipe จะสกัดพิกัด $(x, y)$ ของจุดข้อต่อ 17 จุดทั่วร่างกาย:

```
                  (0) จมูก
              (1) ╭───┴───╮ (2) ตา
            (3) ╭─╯       ╰─╮ (4) หู
                │           │
       (5) ไหล่ซ้าย ═══════ (6) ไหล่ขวา
            │                   │
       (7) ศอกซ้าย         (8) ศอกขวา
            │                   │
       (9) ข้อมือซ้าย     (10) ข้อมือขวา
            │                   │
      (11) สะโพกซ้าย ═════ (12) สะโพกขวา
            │                   │
      (13) เข่าซ้าย       (14) เข่าขวา
            │                   │
      (15) ข้อเท้าซ้าย   (16) ข้อเท้าขวา
```

---

## 📐 คณิตศาสตร์การคำนวณมุมข้อต่อ (Joint Angle Calculation)

เมื่อต้องการวัดมุมข้อต่อ เช่น **มุมข้อเข่า ($\angle \text{Hip-Knee-Ankle}$)** กำหนดจุด $A(x_a, y_a)$, $B(x_b, y_b)$, $C(x_c, y_c)$:

```
           A (สะโพก: Hip)
            \
             \   θ (มุมข้อเข่า)
              B (ข้อเข่า: Knee) ──────── C (ข้อเท้า: Ankle)
```

$$\vec{BA} = (x_a - x_b, y_a - y_b), \quad \vec{BC} = (x_c - x_b, y_c - y_b)$$
$$\theta = \arccos \left( \frac{\vec{BA} \cdot \vec{BC}}{\|\vec{BA}\| \|\vec{BC}\|} \right) \times \frac{180^\circ}{\pi}$$

* **ตรรกะนับรอบ Squat:**
  * **ท่าย่อลง (Down Stage):** มุมข้อเข่า $\theta < 90^\circ$
  * **ท่ายืนขึ้น (Up Stage):** มุมข้อเข่า $\theta > 160^\circ$ $\rightarrow$ **นับบวก 1 ครั้ง (Count + 1)**

---

## 🚀 การรันสคริปต์ตัวอย่าง

### 1. รันระบบนับจำนวนรอบ Squat ด้วยกล้อง Webcam:
```bash
python ml_model_training/project_templates/04_pose_estimation_and_keypoints/squat_fitness_counter.py
```

### 2. รันสกัด Keypoints และวาดโครงกระดูก (Skeleton):
```bash
python ml_model_training/project_templates/04_pose_estimation_and_keypoints/yolo_pose_infer.py
```
