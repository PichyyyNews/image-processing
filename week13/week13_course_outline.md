# โครงสร้างเนื้อหารายสัปดาห์ - สัปดาห์ที่ 13
## โมเดลการหาจุดข้อต่อร่างกายมนุษย์ด้วย MediaPipe (Pose & Landmark Detection)

> **วิชา:** การประมวลผลภาพดิจิทัล (Digital Image Processing)  
> **รหัสวิชา:** 31909-2007  
> **เวลาเรียน:** 5 ชั่วโมง (บรรยาย 2 ชั่วโมง, ปฏิบัติ 3 ชั่วโมง)

---

## 1. วัตถุประสงค์การเรียนรู้ประจำสัปดาห์ (Learning Objectives)
1. **เข้าใจหลักการตรวจจับ Landmark (CLO 1):** อธิบายความแตกต่างระหว่าง Bounding Box กับ Keypoint/Landmark Estimation ของสถิติท่าทางร่างกายมนุษย์
2. **การใช้งานไลบรารี MediaPipe (CLO 2):** เขียนสคริปต์ Python ใช้ MediaPipe Hands, Pose, และ Face Mesh สกัดพิกัด 2D/3D Landmarks ($x, y, z$)
3. **การคำนวณเวกเตอร์มุมองศา (CLO 3):** นำพิกัด Landmark มาคำนวณระยะทางยูคลิด (Euclidean Distance) และมุมองศาข้อต่อ ($\theta$) ใน OpenCV
4. **สร้างแอปพลิเคชันตอบสนองไร้สัมผัส (CLO 3):** พัฒนาระบบ Virtual Canvas (วาดภาพด้วยนิ้ว) หรือ Fitness Counter (ระบบนับจำนวนครั้งออกกำลังกาย)

---

## 2. แผนการเรียนรู้ประจำสัปดาห์

```mermaid
flowchart LR
    A[Webcam / Video Feed] --> B[MediaPipe Landmark Pipeline]
    B --> C[Extract 33 Pose Keypoints<br>or 21 Hand Keypoints]
    C --> D[Vector Geometry Math<br>Angle & Distance Calculation]
    D --> E[OpenCV Interactive Visualiser]
```

* **บรรยาย (2 ชั่วโมง):**
  * สถาปัตยกรรม BlazeFace & Palm Detection Network ใน MediaPipe
  * การส่งกลับพิกัด Normalize ($x, y, z \in [0.0, 1.0]$)
  * คณิตศาสตร์เวกเตอร์: $\cos\theta = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$ สำหรับคำนวณมุมข้อศอก/เข่า
* **ปฏิบัติการ (3 ชั่วโมง) - LAB 13:**
  * ติดตั้ง `mediapipe` package
  * เขียนสคริปต์ `mediapipe_demo.py` สแกนจุดข้อต่อมือและแกนร่างกาย
  * สร้างโปรแกรม Air Canvas (ใช้นิ้วชี้วาดภาพบนอากาศ)

---

## 3. ฟังก์ชันและคำสั่งสำคัญประจำสัปดาห์

| ไลบรารี / โมดูล | ฟังก์ชัน / คำสั่ง | วัตถุประสงค์ |
|---|---|---|
| **`mediapipe.solutions.pose`** | `Pose(min_detection_confidence=0.5)` | สร้างตัวตรวจจับข้อต่อร่างกาย (33 จุด) |
| **`mediapipe.solutions.hands`** | `Hands(max_num_hands=2)` | สร้างตัวตรวจจับจุดนิ้วมือ (21 จุด) |
| **`mediapipe.solutions.drawing_utils`** | `draw_landmarks()` | วาดเส้นเชื่อมข้อต่อบน OpenCV Frame |
