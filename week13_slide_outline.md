# โครงร่างสไลด์นำเสนอ - สัปดาห์ที่ 13
## โมเดลการหาจุดข้อต่อร่างกายมนุษย์ด้วย MediaPipe (Pose & Landmark Detection)

---

### Slide 1: หน้าปก (Title Slide)
* **หัวข้อ:** Human Pose & Hand Landmark Tracking with MediaPipe & Vector Geometry
* **วิชา:** การประมวลผลภาพดิจิทัล (Digital Image Processing)
* **สัปดาห์ที่ 13:** ตรวจจับข้อต่อและสร้างระบบโต้ตอบไร้สัมผัส (Touchless Interaction)

---

### Slide 2: ทำไมต้อง MediaPipe? (Lightweight Cross-Platform Solutions)
* พัฒนาโดย Google สแกนพิกัด Keypoint ระดับมิลลิวินาทีบน CPU
* **โมดูลหลัก:**
  * **MediaPipe Hands:** 21 จุดต่อหนึ่งข้างมือ
  * **MediaPipe Pose:** 33 จุดทั่วร่างกาย (ข้อศอก, เข่า, สะโพก, ข้อเท้า)
  * **MediaPipe Face Mesh:** 468 จุดบนใบหน้า

---

### Slide 3: พิกัด Keypoint Normalized
* ค่า $x, y$ ถูก Normalize ตามความกว้างและความสูงของภาพ ($0.0 - 1.0$)
* ค่า $z$ บอกความลึก (Depth) สัมพัทธ์เทียบกับตำแหน่งสะโพก/ข้อมือ
* การแปลงพิกัดกลับสู่พิกเซลจริง:
  $$X_{\text{pixel}} = x_{\text{norm}} \times W, \quad Y_{\text{pixel}} = y_{\text{norm}} \times H$$

---

### Slide 4: คณิตศาสตร์คำนวณมุมข้อต่อ ($\theta$)
* คำนวณมุมระหว่าง 3 จุดพิกัด $A(x_a, y_a), B(x_b, y_b), C(x_c, y_c)$ โดยให้ $B$ เป็นจุดหมุน:
  $$\theta = \arctan2(y_c - y_b, x_c - x_b) - \arctan2(y_a - y_b, x_a - x_b)$$
  $$\theta = |\theta \times \frac{180}{\pi}|$$

---

### Slide 5: ตัวอย่างการประยุกต์ใช้งานจริง
1. **Fitness Counter:** นับจำนวนครั้งการยกดัมเบลล์ (Curl) เมื่อมุมข้อศอกเปลี่ยนจาก $> 160^\circ$ เป็น $< 30^\circ$
2. **Drowsiness Detector:** วัดระยะทางระหว่างเปลือกตาบน-ล่าง หากแคบติดต่อกัน 3 วินาที ให้ส่งเสียงเตือน
3. **Air Canvas:** จรดปลายนิ้วชี้เพื่อวาดภาพบนหน้าจอ OpenCV

---

### Slide 6: สรุปปฏิบัติการ LAB 13
* เขียนสคริปต์สแกนพิกัดจุดข้อมือและนิ้วชี้
* คำนวณระยะทางระหว่างปลายนิ้วชี้และนิ้วโป้ง (Pinch Gesture)
* หากแตะกัน ให้ถือเป็นคำสั่ง "คลิก" หรือ "เริ่มวาดเส้น" บนกระจกหน้าจอ
