# คู่มือการเขียนโค้ด (Code Guide) - สัปดาห์ที่ 6

**Lab 6: Photo Inpainting and Denoising Tool**

ในสัปดาห์นี้เราจะมาฝึกเขียนสคริปต์ 3 ส่วนหลัก ได้แก่:
1. การแปลงภาพเข้าสู่โดเมนความถี่ (Fourier Transform)
2. การลบลายน้ำหรือซ่อมแซมภาพ (Image Inpainting)
3. การคำนวณตัวชี้วัดคุณภาพภาพ (MSE และ PSNR)

---

## 1. การแปลงภาพด้วย 2D Discrete Fourier Transform (DFT)

โค้ดส่วนนี้แสดงการแปลงภาพขาวดำ (Grayscale) ให้อยู่ในโดเมนความถี่ และการหาผลลัพธ์ของ Magnitude Spectrum

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. โหลดภาพต้นฉบับในโหมด Grayscale
img = cv2.imread('sample_image.jpg', cv2.IMREAD_GRAYSCALE)

# 2. แปลงชนิดข้อมูลภาพเป็น float32 เพื่อเตรียมเข้าสมการคณิตศาสตร์
img_float32 = np.float32(img)

# 3. แปลงภาพด้วย DFT (ผลลัพธ์ที่ได้จะมี 2 Channel: Real และ Imaginary)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)

# 4. ทำการ Shift ความถี่ต่ำสุด (แกนหลัก) ให้มาอยู่กึ่งกลางภาพ
dft_shift = np.fft.fftshift(dft)

# 5. คำนวณหาขนาด (Magnitude) เพื่อนำมาแสดงผลภาพ
# สูตรคือ 20 * log(magnitude) เพื่อให้ค่าสเกลอยู่ในช่วงที่ตาคนมองเห็นความแตกต่างชัดขึ้น
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# พล็อตภาพเปรียบเทียบ
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```

---

## 2. การซ่อมแซมภาพ (Image Inpainting)

สมมติว่าเรามีรูปภาพที่มีรอยแตกหรือมีลายน้ำทับอยู่ เราจำเป็นต้องเตรียม **Mask** ซึ่งเป็นภาพขาวดำขนาดเท่าภาพต้นฉบับ โดยบริเวณที่เป็นสีขาว (255) ใน Mask คือบริเวณที่เราสั่งให้โปรแกรม "ซ่อมแซม"

```python
import cv2
import numpy as np

# 1. โหลดรูปภาพที่มีความเสียหายหรือมีลายน้ำทับ
damaged_img = cv2.imread('watermarked_image.jpg')

# 2. สร้าง Mask หรือโหลด Mask ขาวดำ (สีขาวคือจุดที่เสีย สีดำคือจุดปกติ)
# (ในทางปฏิบัติ เราอาจใช้ Thresholding ช่วยดึงลายน้ำออกมาเป็น Mask ได้)
mask = cv2.imread('watermark_mask.png', cv2.IMREAD_GRAYSCALE)

# 3. ซ่อมแซมภาพด้วย cv2.inpaint()
# พารามิเตอร์: ภาพที่มีรอยตำหนิ, Mask ภาพขาวดำ, รัศมีการซ่อมแซม, อัลกอริทึมที่ใช้
inpaint_radius = 3

# อัลกอริทึมที่ 1: Navier-Stokes (อิงพลศาสตร์ของไหล)
res_ns = cv2.inpaint(damaged_img, mask, inpaint_radius, cv2.INPAINT_NS)

# อัลกอริทึมที่ 2: Telea (ถมจากขอบเข้าในสุด)
res_telea = cv2.inpaint(damaged_img, mask, inpaint_radius, cv2.INPAINT_TELEA)

# แสดงผลลัพธ์
cv2.imshow('Damaged Image', damaged_img)
cv2.imshow('Mask', mask)
cv2.imshow('Inpainted - NS', res_ns)
cv2.imshow('Inpainted - Telea', res_telea)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 3. การคำนวณคุณภาพภาพ (MSE และ PSNR)

ฟังก์ชันการคำนวณเปรียบเทียบรูปภาพสองรูป (เช่น ภาพก่อนถูกบีบอัด vs ภาพหลังถูกบีบอัด)

```python
import cv2
import numpy as np
import math

def calculate_psnr(img1, img2):
    # 1. คำนวณหา MSE (Mean Squared Error)
    # ต้องมั่นใจว่าขนาดและจำนวนช่องสี (Channels) ตรงกัน
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    
    # ถ้าภาพเหมือนกันทุกพิกเซล MSE = 0 หมายความว่า PSNR เข้าใกล้ Infinity (ไม่มี Noise เลย)
    if mse == 0:
        return float('inf')
        
    # 2. กำหนดค่าสูงสุดของพิกเซล (สำหรับภาพ 8-bit จะเท่ากับ 255)
    max_pixel = 255.0
    
    # 3. คำนวณ PSNR
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# การใช้งาน:
# สมมติเรามีภาพต้นฉบับ และภาพที่จำลองการโดนบีบอัด (JPEG)
original = cv2.imread('hq_image.jpg')
compressed = cv2.imread('low_quality_jpeg.jpg')

value_psnr = calculate_psnr(original, compressed)
print(f"PSNR Value: {value_psnr:.2f} dB")
# หากค่า > 30 dB ถือว่าภาพยังคงคุณภาพสูงเมื่อมองด้วยตาเปล่า
```
