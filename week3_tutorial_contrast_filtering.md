# Week 3: การจัดการแสง สี และการทำความสะอาดข้อมูลภาพ (Contrast & Filtering)

สัปดาห์นี้เราจะเรียนรู้เกี่ยวกับการจัดการแสง สี และการทำความสะอาดข้อมูลภาพ ซึ่งเป็นขั้นตอนสำคัญใน Image Preprocessing เพื่อเตรียมข้อมูลภาพให้พร้อมสำหรับการนำไปใช้งานต่อในโมเดล AI 

## 1. การปรับปรุงแสงและความคมชัด (Brightness & Contrast)

การปรับแสงและคอนทราสต์ช่วยให้ภาพที่มืดหรือสว่างเกินไปกลับมามีรายละเอียดที่ชัดเจนขึ้น
โดยสมการพื้นฐานที่ใช้ในการปรับคือ: `g(x,y) = \alpha * f(x,y) + \beta`
*   `\alpha` (Alpha): ควบคุมคอนทราสต์ (Contrast)
*   `\beta` (Beta): ควบคุมความสว่าง (Brightness)

**ตัวอย่างโค้ด:**
```python
import cv2
import numpy as np

# อ่านภาพ
img = cv2.imread('input.jpg')

# ปรับ Alpha (Contrast) = 1.5 และ Beta (Brightness) = 50
alpha = 1.5 
beta = 50 
adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

cv2.imshow('Original', img)
cv2.imshow('Adjusted', adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 2. กราฟฮิสโตแกรม (Histogram)

ฮิสโตแกรมคือการพล็อตกราฟแจกแจงความถี่ของค่าความสว่างพิกเซลในภาพ (ตั้งแต่ 0 ถึง 255) ช่วยให้เราเข้าใจภาพรวมของการกระจายตัวของแสงในภาพ

**ตัวอย่างโค้ดพล็อตกราฟแยกแชนเนลสี (RGB):**
```python
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('input.jpg')
color = ('b','g','r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0,256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])

plt.title('Color Histogram')
plt.show()
```

## 3. การปรับสมดุลฮิสโตแกรมด้วย CLAHE

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** เป็นเทคนิคที่ช่วยปรับคอนทราสต์ของภาพให้ดีขึ้นโดยแบ่งภาพออกเป็นส่วนๆ (Tiles) แล้วทำ Histogram Equalization ในแต่ละส่วน ช่วยป้องกันไม่ให้เกิดสัญญาณรบกวนมากเกินไปในพื้นที่ที่มีคอนทราสต์ต่ำ เหมาะสำหรับภาพถ่ายมุมมืด

**ตัวอย่างโค้ด:**
```python
import cv2

# CLAHE นิยมทำกับภาพระดับสีเทา (Grayscale)
img = cv2.imread('dark_image.jpg', 0)

# สร้าง object CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv2.imshow('Original', img)
cv2.imshow('CLAHE', cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 4. ปัญหา Noise และทฤษฎีการกรองภาพ (Spatial Filtering)

ภาพถ่ายมักมี **Noise (สัญญาณรบกวน)** ปะปนมาด้วย เช่น:
1.  **Gaussian Noise**: สัญญาณรบกวนแบบสุ่มที่เกิดจากเซนเซอร์กล้อง
2.  **Salt & Pepper Noise**: จุดขาว-ดำที่กระจายตามภาพ มักเกิดจากข้อผิดพลาดในการรับส่งข้อมูล

เราสามารถกำจัด Noise ได้ด้วยการกรองภาพ (Filtering):

*   **Gaussian Blur**: เบลอภาพโดยให้ความสำคัญกับพิกเซลตรงกลางมากที่สุด เหมาะสำหรับลด Gaussian Noise
*   **Median Blur**: ใช้ค่ามัธยฐานของพิกเซลรอบข้าง เหมาะสำหรับลด Salt & Pepper Noise ได้ดีเยี่ยม
*   **Bilateral Filter**: ช่วยลด Noise พร้อมทั้งรักษารายละเอียดของเส้นขอบวัตถุ (Edge-preserving)

**ตัวอย่างโค้ดตัวกรองกำจัด Noise:**
```python
import cv2

img = cv2.imread('noisy_image.jpg')

# 1. Gaussian Blur
gaussian = cv2.GaussianBlur(img, (5,5), 0)

# 2. Median Blur (เหมาะกับ Salt & Pepper Noise)
median = cv2.medianBlur(img, 5)

# 3. Bilateral Filter
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

cv2.imshow('Original', img)
cv2.imshow('Gaussian Blur', gaussian)
cv2.imshow('Median Blur', median)
cv2.imshow('Bilateral Filter', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 💻 ปฏิบัติการ (3 ชั่วโมง) - LAB 3: Enhancement & Filtering Pipeline

**ภารกิจ:** 
ให้นักศึกษาเขียนสคริปต์ประมวลผลล้าง Noise ของภาพถ่ายวัตถุที่ถ่ายในสภาวะแสงน้อยให้สะอาดและชัดเจน (Preprocessing Pipeline) เพื่อเตรียมพร้อมสำหรับนำไปให้ระบบอ่านรหัส QR code หรือจดจำใบหน้าในอนาคต

**ขั้นตอนแนะนำ:**
1.  อ่านภาพและแปลงเป็น Grayscale 
2.  ใช้ **CLAHE** ปรับสมดุลความสว่างให้เห็นรายละเอียดวัตถุชัดขึ้น
3.  เลือกใช้ **Filter** ที่เหมาะสม (เช่น Bilateral Filter หรือ Median Blur) เพื่อลบจุดเม็ดทรายหรือ Noise โดยไม่ทำให้ขอบวัตถุเบลอจนเกินไป
4.  เขียนสคริปต์บันทึกภาพผลลัพธ์ (Cleaned Image) ลงในโฟลเดอร์ที่กำหนด
5.  ฝึกฝนการเปิด-ปิดจุด Breakpoint ใน VS Code เพื่อดูค่าความสว่างของตัวแปร Array ในแต่ละขั้นตอน
