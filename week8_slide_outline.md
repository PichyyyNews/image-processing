# โครงร่างสไลด์นำเสนอ - สัปดาห์ที่ 8
## สรุปทบทวนและเตรียมพร้อมสอบกลางภาค (Midterm Exam Guide & Overview)

---

### Slide 1: หน้าปก (Title Slide)
* **หัวข้อ:** สรุปบทเรียนครึ่งเทอมแรก และข้อแนะนำการสอบกลางภาค
* **วิชา:** การประมวลผลภาพดิจิทัล (Digital Image Processing)
* **ผู้สอน:** สาขาวิชาเทคโนโลยีสารสนเทศ

---

### Slide 2: สรุปภาพรวมครึ่งแรกของภาคเรียน (Weeks 1–7 Roadmap)
* **รากฐานการประมวลผลภาพ (Spatial Domain & Pixel Manipulation)**
  * Week 1: Environment setup, Conda, OpenCV I/O, BGR Representation
  * Week 2: Array Slicing, Resizing, Rotation, Affine Transformation
  * Week 3: Contrast Enhancement, CLAHE, Spatial Filtering & Denoising
  * Week 4: Edge Detection (Canny/Sobel), Morphological Operations
  * Week 5: Contour Extraction, Bounding Box, Moments & Auto-Cropper
  * Week 6: Frequency Domain (DFT/FFT), Image Inpainting
  * Week 7: Classical Feature Matching (ORB, SIFT, Homography)

---

### Slide 3: ประเด็นเน้นย้ำทฤษฎี (Key Theoretical Concepts)
* **ข้อแตกต่างระหว่าง Spatial Domain และ Frequency Domain:**
  * Spatial: ทำงานกับค่าพิกเซล $f(x, y)$ โดยตรง (Convolution, Mask Filter)
  * Frequency: แปลงด้วย 2D DFT $F(u, v)$ เพื่อกรองความถี่ต่ำ/สูง (Low-pass/High-pass)
* **สูตรการปรับปรุงภาพ:**
  * Gamma Correction: $s = c \cdot r^\gamma$
  * Histogram Equalization & CLAHE
  * Structural Element ใน Morphology (Erosion vs Dilation)

---

### Slide 4: สูตรคำนวณและเมทริกซ์ที่ต้องจำ (Essential Math Formulas)
* **2D Rotation Matrix:**
  $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$
* **Canny Edge Detection Steps:**
  1. Gaussian Smoothing $\rightarrow$ 2. Intensity Gradient Calculation $\rightarrow$ 3. Non-Maximum Suppression $\rightarrow$ 4. Hysteresis Thresholding
* **Contour Area & Center of Mass (Moments):**
  $$\bar{x} = \frac{M_{10}}{M_{00}}, \quad \bar{y} = \frac{M_{01}}{M_{00}}$$

---

### Slide 5: โครงสร้างและแนวทางข้อสอบปฏิบัติการ (Practical Lab Exam Guide)
* **โจทย์สมมติ:** ระบบสแกนเอกสารและคัดแยกป้ายสินค้าจากภาพถ่ายที่สว่างไม่สม่ำเสมอและมีจุดรบกวน
* **ขั้นตอนการแก้ปัญหาที่ต้องเขียนโปรแกรม:**
  1. โหลดภาพและทำ Normalization / Grayscale
  2. กำจัด Noise ด้วย Median Blur หรือ Bilateral Filter
  3. ปรับฮิสโตแกรมด้วย CLAHE
  4. สกัดขอบด้วย Canny และกระชับพื้นที่ด้วย Morphology Closing
  5. ค้นหา Contour และตีกรอบ Bounding Box
  6. Crop ROI และ Save ออกเป็นไฟล์ภาพสะอาด

---

### Slide 6: ข้อควรระวังและเทคนิคการดีบักใน VS Code (Troubleshooting Tips)
* ตรวจสอบชนิดข้อมูล Array (`img.dtype` ต้องเป็น `uint8` สำหรับ OpenCV Display)
* เช็กสัดส่วนภาพและพิกัด ROI ($y_1:y_2, x_1:x_2$)
* ตรวจสอบว่าพิกัด Bounding Box ไม่ติดลบหรือเกินขนาดภาพ (`np.clip`)
* ใช้ Breakpoint ใน VS Code ส่องดูค่า Matrix ก่อนรันผ่านลูป
