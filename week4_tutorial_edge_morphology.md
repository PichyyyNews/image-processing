# Week 4: การสกัดเส้นขอบและการดัดแปลงสัณฐานวิทยา (Edge Detection & Morphological Operations)

สัปดาห์นี้เราจะเรียนรู้วิธีการค้นหา **เส้นขอบ (Edge)** ของวัตถุในภาพดิจิทัล ซึ่งเป็นขั้นตอนสำคัญในการแยกวัตถุออกจากพื้นหลัง และการใช้ **Morphological Operations** เพื่อปรับปรุงคุณภาพของเส้นขอบให้สะอาดสมบูรณ์ พร้อมสำหรับส่งต่อไปยังขั้นตอน Contour Detection ในสัปดาห์ถัดไป

---

## 1. แนวคิดพื้นฐาน: Image Gradient (อนุพันธ์ของภาพ)

เส้นขอบ (Edge) คือบริเวณในภาพที่ค่าความสว่างของพิกเซลเปลี่ยนแปลงอย่างรุนแรง ในทางคณิตศาสตร์ เราใช้ **อนุพันธ์ (Derivative)** เพื่อวัดอัตราการเปลี่ยนแปลงนี้:

*   **Gradient แกน X** ($G_x$): วัดการเปลี่ยนแปลงในแนวนอน → ตรวจจับเส้นขอบแนวตั้ง
*   **Gradient แกน Y** ($G_y$): วัดการเปลี่ยนแปลงในแนวตั้ง → ตรวจจับเส้นขอบแนวนอน
*   **ขนาดรวม (Magnitude):** $|G| = \sqrt{G_x^2 + G_y^2}$ — ยิ่งค่ามากยิ่งเป็นขอบชัด
*   **ทิศทาง (Direction):** $\theta = \arctan(G_y / G_x)$ — บอกทิศทางของเส้นขอบ

> [!TIP]
> บริเวณที่ Gradient Magnitude สูง = มีเส้นขอบ, บริเวณที่ต่ำ = เป็นพื้นที่สีสม่ำเสมอ (พื้นหลังหรือพื้นผิวเรียบ)

---

## 2. Sobel Edge Detection — อนุพันธ์อันดับหนึ่ง

**Sobel Filter** ใช้ Kernel ขนาด 3×3 ในการประมาณค่าอนุพันธ์อันดับหนึ่งของภาพ โดยแยกคำนวณเป็น 2 ทิศทาง:

**Sobel Kernel แกน X** (ตรวจจับขอบแนวตั้ง):
```
| -1 |  0 | +1 |
| -2 |  0 | +2 |
| -1 |  0 | +1 |
```

**Sobel Kernel แกน Y** (ตรวจจับขอบแนวนอน):
```
| -1 | -2 | -1 |
|  0 |  0 |  0 |
| +1 | +2 | +1 |
```

**ตัวอย่างโค้ด:**
```python
import cv2
import numpy as np

# อ่านภาพและแปลงเป็น Grayscale
img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# คำนวณ Sobel Gradient แยกแกน X และ Y
# ใช้ cv2.CV_64F เพื่อรองรับค่าลบของอนุพันธ์
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# คำนวณขนาดรวม (Magnitude) ของ Gradient
magnitude = cv2.magnitude(sobel_x, sobel_y)

# แปลงกลับเป็น uint8 เพื่อแสดงผล
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)
magnitude_abs = cv2.convertScaleAbs(magnitude)

cv2.imshow('Original', gray)
cv2.imshow('Sobel X (Vertical Edges)', sobel_x_abs)
cv2.imshow('Sobel Y (Horizontal Edges)', sobel_y_abs)
cv2.imshow('Sobel Magnitude (All Edges)', magnitude_abs)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

> [!IMPORTANT]
> **ทำไมต้องใช้ `cv2.CV_64F`?** เพราะผลอนุพันธ์มีทั้งค่าบวก (เปลี่ยนจากดำ→ขาว) และค่าลบ (เปลี่ยนจากขาว→ดำ) หากใช้ `cv2.CV_8U` (uint8) ค่าลบจะถูกตัดเป็น 0 ทำให้สูญเสียเส้นขอบฝั่งที่ค่าลดลง ดังนั้นควรคำนวณในรูปแบบ Float ก่อน แล้วค่อยแปลงเป็น Absolute Value ด้วย `cv2.convertScaleAbs()` ทีหลัง

**อธิบายพารามิเตอร์ `cv2.Sobel()`:**

| พารามิเตอร์ | ความหมาย | ตัวอย่างค่า |
|---|---|---|
| `src` | ภาพต้นฉบับ (Grayscale) | `gray` |
| `ddepth` | ชนิดข้อมูลผลลัพธ์ | `cv2.CV_64F` |
| `dx` | อันดับอนุพันธ์แกน X | `1` (ทำอนุพันธ์), `0` (ไม่ทำ) |
| `dy` | อันดับอนุพันธ์แกน Y | `0` (ไม่ทำ), `1` (ทำอนุพันธ์) |
| `ksize` | ขนาด Kernel (ต้องเป็นเลขคี่) | `3`, `5`, `7` |

---

## 3. Laplacian Edge Detection — อนุพันธ์อันดับสอง

**Laplacian** คำนวณอนุพันธ์อันดับสอง (Second-order Derivative) ของภาพ ตรวจจับเส้นขอบได้ทุกทิศทางในครั้งเดียว

สมการ: $\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$

**Laplacian Kernel ขนาด 3×3:**
```
|  0 |  1 |  0 |
|  1 | -4 |  1 |
|  0 |  1 |  0 |
```

**ตัวอย่างโค้ด:**
```python
import cv2

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# สำคัญ! ต้อง Blur ก่อนเพื่อลด Noise (Laplacian ไวต่อ Noise มาก)
blurred = cv2.GaussianBlur(gray, (3,3), 0)

# คำนวณ Laplacian
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

cv2.imshow('Original', gray)
cv2.imshow('Laplacian Edges', laplacian_abs)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

> [!WARNING]
> **Laplacian ไวต่อ Noise มาก!** หากไม่ทำ Gaussian Blur ก่อน Laplacian จะตรวจเจอ Noise เป็นเส้นขอบเต็มไปหมด ทำให้ผลลัพธ์ไม่สามารถใช้งานได้ ดังนั้น **ต้อง Blur ก่อนเสมอ**

**จุดเด่นและจุดด้อย:**

| จุดเด่น | จุดด้อย |
|---|---|
| ตรวจจับขอบได้ทุกทิศทาง | ไวต่อ Noise มาก |
| คำนวณง่าย ใช้ Kernel เดียว | เส้นขอบมักหนาและไม่คม |
| เหมาะเป็น Focus Measure (วัดความชัดของภาพ) | ไม่ให้ข้อมูลทิศทางของขอบ |

---

## 4. Canny Edge Detection — มาตรฐานทองคำ ⭐

**Canny Edge Detection** เป็นอัลกอริทึมหาเส้นขอบที่ได้รับความนิยมมากที่สุด เพราะให้ผลลัพธ์เส้นขอบที่บาง คมชัด และต่อเนื่อง

### 4.1 ขั้นตอนการทำงานของ Canny (5 Steps)

1.  **Gaussian Smoothing** — ลด Noise ด้วย Gaussian Blur (ทำในตัว)
2.  **Gradient Calculation** — คำนวณ Gradient Magnitude และ Direction ด้วย Sobel
3.  **Non-Maximum Suppression (NMS)** — ทำให้เส้นขอบบางเหลือ 1 พิกเซล โดยเก็บเฉพาะจุดที่มี Magnitude สูงสุดในทิศทางของ Gradient
4.  **Double Thresholding** — แบ่งพิกเซลเป็น 3 กลุ่ม:
    *   **Strong Edge:** Magnitude > Threshold สูง → เป็นขอบแน่นอน ✅
    *   **Weak Edge:** Threshold ต่ำ < Magnitude < Threshold สูง → อาจเป็นขอบ ❓
    *   **ไม่ใช่ Edge:** Magnitude < Threshold ต่ำ → ไม่ใช่ขอบ ❌
5.  **Edge Tracking by Hysteresis** — Weak Edge จะถูกเก็บไว้ก็ต่อเมื่อเชื่อมต่อกับ Strong Edge เท่านั้น (ทำให้เส้นไม่ขาดตอน)

### 4.2 ตัวอย่างโค้ด Canny พื้นฐาน

```python
import cv2

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny Edge Detection
# threshold1 = Threshold ต่ำ, threshold2 = Threshold สูง
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

cv2.imshow('Original', gray)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 4.3 ตัวอย่างโค้ด Canny พร้อมปรับ Threshold ด้วย Trackbar (Interactive)

```python
import cv2

def nothing(x):
    pass

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Canny Edge Detection')
cv2.createTrackbar('Threshold 1', 'Canny Edge Detection', 50, 500, nothing)
cv2.createTrackbar('Threshold 2', 'Canny Edge Detection', 150, 500, nothing)

while True:
    t1 = cv2.getTrackbarPos('Threshold 1', 'Canny Edge Detection')
    t2 = cv2.getTrackbarPos('Threshold 2', 'Canny Edge Detection')
    
    edges = cv2.Canny(gray, t1, t2)
    
    cv2.imshow('Canny Edge Detection', edges)
    
    if cv2.waitKey(1) & 0xFF == 27:  # กด ESC เพื่อออก
        break

cv2.destroyAllWindows()
```

> [!TIP]
> **เคล็ดลับการตั้ง Threshold:**
> *   อัตราส่วนที่แนะนำ: `threshold1:threshold2 = 1:2` หรือ `1:3`
> *   เส้นขอบน้อยเกินไป → ลด Threshold ทั้งสองลง
> *   เส้นขอบเยอะเกินไป (มี Noise) → เพิ่ม Threshold ทั้งสองขึ้น
> *   ลองใช้ Trackbar Interactive ในโค้ดด้านบนเพื่อหาค่าที่เหมาะสมกับภาพของตนเอง

### 4.4 ตัวอย่างการคำนวณ Auto Threshold ด้วยค่า Median

```python
import cv2
import numpy as np

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# คำนวณค่า Median ของความสว่างภาพ
median_val = np.median(gray)

# ตั้ง Threshold อัตโนมัติจาก Median
sigma = 0.33
lower = int(max(0, (1.0 - sigma) * median_val))
upper = int(min(255, (1.0 + sigma) * median_val))

edges = cv2.Canny(gray, lower, upper)

print(f"Median: {median_val}, Lower: {lower}, Upper: {upper}")

cv2.imshow('Auto Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 5. ตารางเปรียบเทียบ Sobel vs Laplacian vs Canny

| เกณฑ์เปรียบเทียบ | Sobel | Laplacian | Canny |
|---|---|---|---|
| ประเภทอนุพันธ์ | อันดับหนึ่ง (1st) | อันดับสอง (2nd) | อันดับหนึ่ง + กลั่นกรอง |
| ทิศทางการตรวจจับ | แยก X, Y | ทุกทิศทาง | ทุกทิศทาง |
| ความหนาของเส้นขอบ | ค่อนข้างหนา | หนาปานกลาง | บาง 1 พิกเซล (NMS) |
| ความไวต่อ Noise | ปานกลาง | สูงมาก | ต่ำ (มี Blur ในตัว) |
| ต้อง Blur ก่อน? | แนะนำ | จำเป็น | ไม่จำเป็น (ทำในตัว) |
| ให้ข้อมูลทิศทาง? | ✅ ได้ | ❌ ไม่ได้ | ✅ ได้ (ใช้ภายใน) |
| ความนิยมในงานจริง | ปานกลาง | ต่ำ | สูงมาก ⭐ |

> [!NOTE]
> **สรุป:** ในงานจริง Canny เป็นตัวเลือกแรกเสมอ ใช้ Sobel เมื่อต้องการข้อมูลทิศทาง Gradient ส่วน Laplacian มักใช้เป็นตัววัดความคมชัดของภาพ (Focus Measure)

---

## 6. Structuring Element — แม่พิมพ์สำหรับ Morphological Operations

ก่อนจะเข้าสู่ Morphological Operations เราต้องรู้จัก **Structuring Element (SE)** ก่อน ซึ่งเป็น Kernel รูปร่างเล็ก ๆ ที่ใช้เป็น "แม่พิมพ์" กวาดทั่วภาพ

### 6.1 รูปร่างของ Structuring Element

**ตัวอย่างโค้ด:**
```python
import cv2
import numpy as np

# 1. สี่เหลี่ยม (MORPH_RECT) — เหมาะกับวัตถุขอบตรง
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
print("RECT:\n", kernel_rect)
# [[1 1 1 1 1]
#  [1 1 1 1 1]
#  [1 1 1 1 1]
#  [1 1 1 1 1]
#  [1 1 1 1 1]]

# 2. วงรี (MORPH_ELLIPSE) — เหมาะกับวัตถุขอบโค้ง (นิยมมากที่สุด)
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
print("ELLIPSE:\n", kernel_ellipse)
# [[0 0 1 0 0]
#  [1 1 1 1 1]
#  [1 1 1 1 1]
#  [1 1 1 1 1]
#  [0 0 1 0 0]]

# 3. กากบาท (MORPH_CROSS) — เหมาะกับงานรักษาโครงสร้างแนวเส้น
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
print("CROSS:\n", kernel_cross)
# [[0 0 1 0 0]
#  [0 0 1 0 0]
#  [1 1 1 1 1]
#  [0 0 1 0 0]
#  [0 0 1 0 0]]
```

### 6.2 การเลือกขนาด Structuring Element

| ขนาด SE | ผลลัพธ์ | เหมาะกับ |
|---|---|---|
| เล็ก (3×3) | ผลกระทบน้อย ละเอียดสูง | Noise เล็ก, รายละเอียดสำคัญ |
| กลาง (5×5) | สมดุลพอดี | งานทั่วไป |
| ใหญ่ (7×7 ขึ้นไป) | ผลกระทบมาก หยาบ | Noise ใหญ่, รอยแตกกว้าง |

> [!TIP]
> เริ่มต้นลองที่ขนาด **3×3 หรือ 5×5** ก่อน ถ้าผลยังไม่เพียงพอค่อย ๆ เพิ่มขนาดหรือเพิ่ม `iterations`

---

## 7. Erosion (การกัดกร่อน)

**Erosion** ทำให้บริเวณสีขาว (Foreground) ใน Binary Image **หดเล็กลง** ขอบถูกกัดเข้าด้านใน

**กฎ:** พิกเซลผลลัพธ์ = ขาว (1) **ก็ต่อเมื่อ** พิกเซลทุกตัวภายใต้ Kernel เป็นขาว (1) ทั้งหมด

**ตัวอย่างโค้ด:**
```python
import cv2
import numpy as np

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold สร้าง Binary Image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# สร้าง Structuring Element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# Erosion — กัดกร่อนพิกเซลขาว
eroded_1 = cv2.erode(binary, kernel, iterations=1)  # กัด 1 รอบ
eroded_2 = cv2.erode(binary, kernel, iterations=2)  # กัด 2 รอบ
eroded_3 = cv2.erode(binary, kernel, iterations=3)  # กัด 3 รอบ

cv2.imshow('Original Binary', binary)
cv2.imshow('Eroded 1x', eroded_1)
cv2.imshow('Eroded 2x', eroded_2)
cv2.imshow('Eroded 3x', eroded_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**ผลลัพธ์ที่เห็น:**
*   จุดขาวเล็ก ๆ ที่เล็กกว่า Kernel จะหายไป
*   ขอบวัตถุหดตัวเข้าด้านใน
*   ยิ่ง `iterations` มาก วัตถุยิ่งบางลง

---

## 8. Dilation (การขยาย)

**Dilation** ทำให้บริเวณสีขาว (Foreground) ใน Binary Image **ขยายใหญ่ขึ้น** ขอบขยายออกด้านนอก

**กฎ:** พิกเซลผลลัพธ์ = ขาว (1) **ถ้ามีพิกเซลใดก็ตาม** ภายใต้ Kernel เป็นขาว (1) อย่างน้อย 1 ตัว

**ตัวอย่างโค้ด:**
```python
import cv2
import numpy as np

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold สร้าง Binary Image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# สร้าง Structuring Element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# Dilation — ขยายพิกเซลขาว
dilated_1 = cv2.dilate(binary, kernel, iterations=1)  # ขยาย 1 รอบ
dilated_2 = cv2.dilate(binary, kernel, iterations=2)  # ขยาย 2 รอบ
dilated_3 = cv2.dilate(binary, kernel, iterations=3)  # ขยาย 3 รอบ

cv2.imshow('Original Binary', binary)
cv2.imshow('Dilated 1x', dilated_1)
cv2.imshow('Dilated 2x', dilated_2)
cv2.imshow('Dilated 3x', dilated_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**ผลลัพธ์ที่เห็น:**
*   รูโหว่ดำเล็ก ๆ ในวัตถุขาวจะถูกอุด
*   เส้นขอบที่ขาดตอนจะเชื่อมต่อกัน
*   ยิ่ง `iterations` มาก วัตถุยิ่งหนาขึ้น

---

## 9. Opening & Closing — การดำเนินการสัณฐานวิทยาผสม

### 9.1 Opening (Erosion → Dilation) — กำจัดจุดขาวเล็ก ๆ

**Opening = Erosion ก่อน แล้วตามด้วย Dilation**

*   Erosion กัดจุดขาวเล็ก ๆ ให้หายไป (แต่วัตถุหลักก็บางลง)
*   Dilation คืนขนาดวัตถุหลักกลับมา (แต่จุดที่หายไปแล้วไม่กลับมา)
*   **ใช้เมื่อ:** ต้องการลบจุด Noise ขาวเล็ก ๆ บนพื้นหลังดำ

```python
import cv2
import numpy as np

img = cv2.imread('noisy_binary.jpg', 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Opening = cv2.erode() + cv2.dilate() (แต่ใช้ morphologyEx สะดวกกว่า)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

cv2.imshow('Original (with white noise)', binary)
cv2.imshow('After Opening (noise removed)', opened)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 9.2 Closing (Dilation → Erosion) — อุดรูโหว่ดำเล็ก ๆ

**Closing = Dilation ก่อน แล้วตามด้วย Erosion**

*   Dilation ขยายวัตถุขาวเพื่ออุดรูดำ (แต่วัตถุหลักก็บวมขึ้น)
*   Erosion ลดขนาดวัตถุหลักกลับมา (แต่รูที่ถูกอุดแล้วยังคงอุดอยู่)
*   **ใช้เมื่อ:** ต้องการอุดรูหรือเชื่อมรอยแตกในวัตถุ Foreground ขาว

```python
import cv2
import numpy as np

img = cv2.imread('binary_with_holes.jpg', 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Closing = cv2.dilate() + cv2.erode()
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original (with dark holes)', binary)
cv2.imshow('After Closing (holes filled)', closed)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 9.3 สรุป Opening vs Closing

| เปรียบเทียบ | Opening | Closing |
|---|---|---|
| ลำดับ | Erosion → Dilation | Dilation → Erosion |
| กำจัด | จุดขาวเล็ก ๆ (White Noise) | รูดำเล็ก ๆ (Dark Holes) |
| การจำ | "เปิดประตูปัดฝุ่นออก" | "ปิดรอยรั่วอุดรูให้เรียบ" |

---

## 10. Morphological Gradient — สกัดเส้นขอบจากการหักลบ

**Morphological Gradient = Dilation – Erosion**

ผลลัพธ์คือ "แถบขอบ" ระหว่างขอบด้านนอก (Dilation) กับขอบด้านใน (Erosion) ของวัตถุ

```python
import cv2
import numpy as np

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# Morphological Gradient
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('Original Binary', binary)
cv2.imshow('Morphological Gradient (Edge)', gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**จุดเด่นของ Morphological Gradient:**
*   เส้นขอบปิดสนิท (Closed Contour) เสมอ
*   ไม่ต้องใช้อนุพันธ์ ทำงานกับ Binary Image โดยตรง
*   เหมาะเป็นอินพุตสำหรับ `cv2.findContours()` ในสัปดาห์ที่ 5

---

## 11. Top Hat & Black Hat — สกัดรายละเอียดที่ซ่อนอยู่

### 11.1 Top Hat (White Top Hat)
**Top Hat = ภาพต้นฉบับ – Opening**

ดึงจุดสว่างเล็ก ๆ บนพื้นหลังมืดออกมา (สิ่งที่ Opening ลบทิ้ง)

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
```

### 11.2 Black Hat
**Black Hat = Closing – ภาพต้นฉบับ**

ดึงจุดมืดเล็ก ๆ บนพื้นหลังสว่างออกมา (สิ่งที่ Closing เติมเข้าไป)

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
```

> [!NOTE]
> Top Hat และ Black Hat มีประโยชน์ในงานที่ต้องตรวจจับรายละเอียดเล็ก ๆ ที่อาจถูกกลืนไปกับพื้นหลัง เช่น ตัวอักษรบนพื้นผิวที่ไม่สม่ำเสมอ หรือรอยตำหนิเล็ก ๆ บนชิ้นงานอุตสาหกรรม

---

## 12. ตัวอย่างโค้ดรวม: Edge Detection + Morphology Pipeline

โค้ดด้านล่างนี้แสดง Pipeline ที่สมบูรณ์ตั้งแต่ภาพต้นฉบับจนถึง Binary Edge Mask สุดท้าย:

```python
import cv2
import numpy as np

# ===== ขั้นตอนที่ 1: อ่านภาพและแปลง Grayscale =====
img = cv2.imread('pcb_board.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===== ขั้นตอนที่ 2: ลด Noise ด้วย Gaussian Blur =====
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# ===== ขั้นตอนที่ 3: สกัดเส้นขอบด้วย Canny =====
edges = cv2.Canny(blurred, 50, 150)

# ===== ขั้นตอนที่ 4: Dilation — ทำเส้นขอบหนาขึ้น =====
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilated = cv2.dilate(edges, kernel_dilate, iterations=1)

# ===== ขั้นตอนที่ 5: Closing — อุดช่องว่างในเส้นขอบ =====
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)

# ===== ขั้นตอนที่ 6: Opening — กำจัดจุดเล็ก ๆ ที่ไม่ใช่เส้นขอบ =====
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
final_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

# ===== แสดงผลลัพธ์ทุกขั้นตอน =====
cv2.imshow('1. Original', img)
cv2.imshow('2. Grayscale', gray)
cv2.imshow('3. Blurred', blurred)
cv2.imshow('4. Canny Edges', edges)
cv2.imshow('5. Dilated', dilated)
cv2.imshow('6. Closed', closed)
cv2.imshow('7. Final Mask (Opening)', final_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===== บันทึกผลลัพธ์ =====
cv2.imwrite('edge_mask_result.jpg', final_mask)
print("Pipeline completed! Binary Edge Mask saved as 'edge_mask_result.jpg'")
```

---

## 13. ตัวอย่างการเปรียบเทียบ Morphological Operations ทั้งหมดในภาพเดียว

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# ทำ Morphological Operations ทุกแบบ
eroded    = cv2.erode(binary, kernel, iterations=1)
dilated   = cv2.dilate(binary, kernel, iterations=1)
opened    = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient  = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
tophat    = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
blackhat  = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

# แสดงผลเปรียบเทียบ
titles = ['Original', 'Erosion', 'Dilation', 'Opening',
          'Closing', 'Gradient', 'Top Hat', 'Black Hat']
images = [binary, eroded, dilated, opened,
          closed, gradient, tophat, blackhat]

plt.figure(figsize=(16, 8))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=12)
    plt.axis('off')
plt.tight_layout()
plt.suptitle('Morphological Operations Comparison', fontsize=14, y=1.02)
plt.show()
```

---

## 💻 ปฏิบัติการ (3 ชั่วโมง) — LAB 4: Custom Object Edge Extraction

### ภารกิจ

ให้นักศึกษาเขียนสคริปต์ Python ตรวจจับเส้นขอบของวัตถุ (เช่น ชิ้นส่วนอิเล็กทรอนิกส์, แผงวงจร PCB, หรือวัตถุที่ครูผู้สอนกำหนดให้) จากภาพถ่าย โดยใช้ Canny Edge Detection ร่วมกับ Morphological Operations เพื่อสร้าง **Binary Edge Mask** ที่สะอาดและสมบูรณ์

### ขั้นตอนแนะนำ

1.  **อ่านภาพและแปลงเป็น Grayscale**
    ```python
    img = cv2.imread('target_object.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ```

2.  **ลด Noise ด้วย Gaussian Blur หรือ Bilateral Filter**
    ```python
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    # หรือใช้ Bilateral Filter เพื่อรักษาขอบ:
    # blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    ```

3.  **ทดลองใช้ Sobel, Laplacian และ Canny เปรียบเทียบผลลัพธ์**
    ```python
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))

    laplacian = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F))

    canny = cv2.Canny(blurred, 50, 150)
    ```

4.  **เลือก Canny เป็นตัวหลัก ปรับ Threshold ให้เหมาะกับภาพ**
    *   ใช้ Trackbar Interactive หรือ Auto Threshold (Median Method) ช่วยหาค่าที่เหมาะสม

5.  **ปรับปรุงเส้นขอบด้วย Morphological Operations**
    ```python
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    # Dilation — ทำเส้นหนาขึ้น เชื่อมเส้นที่ขาด
    dilated = cv2.dilate(canny, kernel, iterations=1)

    # Closing — อุดช่องว่างในเส้นขอบ
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Opening — กำจัดจุดเล็ก ๆ ที่ไม่ใช่เส้นขอบ
    final_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    ```

6.  **บันทึกภาพผลลัพธ์ลงไฟล์**
    ```python
    cv2.imwrite('output/edge_mask.jpg', final_mask)
    ```

7.  **ใช้ Breakpoint ใน VS Code สังเกต:**
    *   ค่า Gradient ที่ได้จาก Sobel (`sobel_x.dtype`, `sobel_x.min()`, `sobel_x.max()`)
    *   ผลลัพธ์ของ Canny ก่อนและหลัง Morphological Operations
    *   ขนาดและรูปร่างของ Structuring Element (`kernel.shape`, `print(kernel)`)

### เกณฑ์การประเมิน

| เกณฑ์ | คะแนน |
|---|---|
| สคริปต์รันได้สำเร็จไม่มี Error | 20% |
| เลือก Canny Threshold ได้เหมาะสม (เส้นขอบครบ ไม่มี Noise เกิน) | 25% |
| ใช้ Morphological Operations ปรับปรุงเส้นขอบ (ต่อเนื่อง ไม่ขาดตอน) | 25% |
| Binary Edge Mask สุดท้ายสะอาดและมีโครงร่างวัตถุที่ชัดเจน | 20% |
| การจัดระเบียบโค้ดและ Comment อธิบาย | 10% |

> [!IMPORTANT]
> **เป้าหมายสำคัญ:** Binary Edge Mask ที่ได้จากแล็บนี้จะถูกนำไปใช้ต่อในสัปดาห์ที่ 5 (Contour Detection) ดังนั้น Mask ยิ่งสะอาดเท่าไหร่ Contour Detection จะยิ่งแม่นยำเท่านั้น!
