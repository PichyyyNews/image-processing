# คู่มือทบทวนและเตรียมสอบกลางภาค (Midterm Examination Review Guide)
## วิชา: การประมวลผลภาพดิจิทัล (Digital Image Processing) — 31909-2007

---

## 1. สรุปสาระสำคัญประจำบทเรียน 1–7 (Core Summary)

### บทที่ 1: พื้นฐานภาพและการจัดสภาพแวดล้อม (Basic Setup & Image I/O)
* **โครงสร้างภาพดิจิทัล:** ภาพสีแบบ BGR (Blue, Green, Red) เป็น Tensor ขนาด $H \times W \times 3$ แบบชนิดข้อมูล `np.uint8` (ค่า $0-255$)
* **คำสั่งสำคัญ:**
  ```python
  import cv2
  img = cv2.imread('image.jpg') # อ่านภาพในรูปแบบ BGR
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # แปลงเป็นภาพเทา
  cv2.imwrite('output.png', gray) # บันทึกภาพ
  ```

### บทที่ 2: การดำเนินการเชิงเรขาคณิต (Geometric Transformation)
* **Resizing:** การปรับขนาดภาพโดยใช้ Interpolation (`cv2.INTER_AREA` ย่อภาพ, `cv2.INTER_CUBIC` ขยายภาพ)
* **Rotation & Affine:** 
  ```python
  M = cv2.getRotationMatrix2D(center, angle, scale)
  rotated = cv2.warpAffine(img, M, (w, h))
  ```

### บทที่ 3: การปรับปรุงคุณภาพและการกรองสัญญาณรบกวน (Enhancement & Filtering)
* **การปรับความสว่างและความคมชัด:** CLAHE (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`) ช่วยปรับสมดุลแสงเฉพาะพื้นที่
* **ตัวกรองลด Noise:**
  * **Gaussian Blur (`cv2.GaussianBlur`):** เหมาะสำหรับ Gaussian Noise
  * **Median Blur (`cv2.medianBlur`):** เหมาะที่สุดสำหรับ Salt & Pepper Noise (จุดขาวดำจุดมด)
  * **Bilateral Filter (`cv2.bilateralFilter`):** ลด Noise โดยคงความคมของขอบภาพเอาไว้

### บทที่ 4: การสกัดเส้นขอบและสัณฐานวิทยา (Edge Detection & Morphology)
* **Canny Edge Detector:**
  $$g(x,y) = \sqrt{G_x^2 + G_y^2}$$
  ใช้ Threshold สองระดับ (Low & High Threshold) ในการสกัดขอบ
* **Morphological Operations:**
  * **Erosion (กร่อน):** ลดขอบวัตถุ ขจัดNoise เล็กๆ
  * **Dilation (ขยาย):** เติมเต็มรอยรั่วหรือเชื่อมวัตถุใกล้เคียง
  * **Opening:** Erosion ตามด้วย Dilation (ลบNoise รอบวัตถุ)
  * **Closing:** Dilation ตามด้วย Erosion (ปิดรอยร้าวในวัตถุ)

### บทที่ 5: การตรวจจับโครงร่างและสกัดวัตถุ (Contour Detection & Auto-Cropper)
* **Contour Extraction:**
  ```python
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
      x, y, w, h = cv2.boundingRect(cnt)
      roi = img[y:y+h, x:x+w]
  ```

### บทที่ 6: โดเมนความถี่และการซ่อมแซมภาพ (Frequency Domain & Inpainting)
* **2D Fourier Transform:** แปลงภาพ $f(x,y)$ สู่โดเมนความถี่ $F(u,v)$ ด้วย `cv2.dft()`
* **Image Inpainting:** ซ่อมแซมรอยขีดข่วนหรือลบลายน้ำด้วย `cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)`

### บทที่ 7: การจับคู่จุดเด่นเชิงคลาสสิก (Classical Feature Matching)
* **ORB & SIFT Feature Matcher:**
  ```python
  orb = cv2.ORB_create()
  kp1, des1 = orb.detectAndCompute(img1, None)
  kp2, des2 = orb.detectAndCompute(img2, None)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1, des2)
  ```

---

## 2. ตัวอย่างแนวข้อสอบและวิธีคิด (Practice Exam & Solution)

### โจทย์ทฤษฎี (Theoretical Question)
**โจทย์:** หากภาพมีสัญญาณรบกวนชนิด Salt & Pepper (จุดเม็ดทรายขาวดำ) กระจายทั่วภาพ ควรเลือกใช้ Filter ใดระหว่าง Gaussian Blur และ Median Blur? เพราะเหตุใด?
**แนวตอบ:** ควรเลือกใช้ **Median Blur** เนื่องจาก Median Blur นำค่ามัธยฐานในหน้าต่าง Filter มาแทนที่พิกเซลเป้าหมาย ซึ่งค่ามัธยฐานจะคัดเอาค่าสุดโต่ง (0 หรือ 255 ของ Salt & Pepper Noise) ออกได้อย่างสมบูรณ์ ในขณะที่ Gaussian Blur เป็นการถัวเฉลี่ยถ่วงน้ำหนัก จะทำให้จุด Noise กระจายกลายเป็นรอยเบลอเปรอะเปื้อนกว้างขึ้น

---

## 3. โจทย์ปฏิบัติการจำลอง (Practice Lab Challenge)

เขียนสคริปต์ Python ใน VS Code เพื่อรับภาพ `test_sample.jpg` จากนั้น:
1. แปลงภาพเป็น Grayscale
2. ใช้ Median Blur ขนาด kernel 5x5
3. ใช้ Thresholding แบบ Otsu เพื่อแยกวัตถุจากพื้นหลัง
4. หา Contour ของวัตถุที่มีพื้นที่กว้างกว่า 1,000 พิกเซล
5. ตีกรอบสีแดงบนภาพต้นฉบับและเซฟผลลัพธ์เป็น `result.jpg`

```python
import cv2
import numpy as np

# 1. Load image
img = cv2.imread('test_sample.jpg')
if img is None:
    raise FileNotFoundError("Image not found!")

# 2. Grayscale & Median Blur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 5)

# 3. Otsu Thresholding
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 4. Find Contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. Filter by Area & Draw Box
output = img.copy()
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save result
cv2.imwrite('result.jpg', output)
print("Processing finished successfully!")
```
