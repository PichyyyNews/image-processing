# คู่มือการเขียนโค้ด (Code Guide) - สัปดาห์ที่ 7

**Lab 7: Target Object Matching Script**

ในสัปดาห์นี้เราจะมาฝึกเขียนสคริปต์ 3 ส่วนหลัก ได้แก่:
1. การตรวจหาจุดสำคัญและดึงตัวบรรยายลักษณะภาพด้วย ORB
2. การจับคู่ด้วย Brute-Force Matcher ร่วมกับการใช้ Ratio Test
3. การคำนวณ Homography และวาดเส้นขอบเขตสีเขียวล้อมรอบวัตถุในกล้องวิดีโอสดหรือภาพปลายทาง

---

## 1. การสกัดจุดสำคัญและ Descriptor ด้วย ORB

โค้ดส่วนนี้แสดงวิธีการโหลดภาพต้นฉบับ ปรับเป็นเกรย์สเกล และประมวลผลดึงจุดเด่น (Keypoints) รวมถึงเวกเตอร์คำอธิบาย (Descriptors) ออกมาแสดงผลภาพ

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. โหลดรูปภาพที่ต้องการค้นหา (Grayscale)
img = cv2.imread('logo.jpg', cv2.IMREAD_GRAYSCALE)

# 2. สร้างออบเจกต์ ORB Detector
# เราสามารถจำกัดจำนวนจุดสูงสุดได้ เช่น nfeatures=1000
orb = cv2.ORB_create(nfeatures=1000)

# 3. ค้นหาจุดสำคัญ (Keypoints) และคำนวณ Descriptors
# keypoints: ลิสต์ข้อมูลตำแหน่งและองศาของจุดสนใจ
# descriptors: NumPy Array ขนาด (จำนวนจุด, 32) เก็บเวกเตอร์ไบนารี
keypoints, descriptors = orb.detectAndCompute(img, None)

print(f"จำนวนจุดสำคัญที่ตรวจพบ: {len(keypoints)}")
print(f"ขนาดโครงสร้างเมทริกซ์ Descriptor: {descriptors.shape}")

# 4. วาดจุดสำคัญลงบนภาพต้นฉบับเพื่อสบตาตรวจสอบ
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS จะช่วยวาดวงกลมแสดงขนาดและทิศทางของแต่ละจุดด้วย
img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), 
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# แสดงรูปภาพจุดเด่น
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoints')
plt.axis('off')
plt.show()
```

---

## 2. การจับคู่และการกรอง Good Matches ด้วย Ratio Test

โค้ดนี้จำลองการหาคู่ฟีเจอร์เด่นระหว่างสองภาพโดยใช้ Brute-Force Matcher (BFMatcher) วัดระยะทางแบบ Hamming พร้อมกรอง Outliers ด้วยเกณฑ์ Ratio Test

```python
import cv2

# โหลดภาพอ้างอิงและภาพเป้าหมาย
img_ref = cv2.imread('logo.jpg', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('shelf.jpg', cv2.IMREAD_GRAYSCALE)

# สกัดฟีเจอร์ด้วย ORB
orb = cv2.ORB_create(nfeatures=1500)
kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
kp_scene, des_scene = orb.detectAndCompute(img_scene, None)

# 1. สร้าง BFMatcher โดยกำหนดให้ใช้วัดระยะห่างแบบ Hamming Distance
# crossCheck=False เพื่อยอมให้เราทำ KNN Match และเลือก Ratio Test เองได้
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# 2. ทำการจับคู่แบบ K-Nearest Neighbors (k=2)
matches = bf.knnMatch(des_ref, des_scene, k=2)

# 3. ใช้ Ratio Test ของ Lowe กรองเฉพาะจุดที่มีความมั่นใจสูง
good_matches = []
for m, n in matches:
    # m คือจุดสบอันดับ 1, n คือจุดสบอันดับ 2
    # ตรวจสอบว่าระยะห่างของอันดับ 1 ต้องน้อยกว่าอันดับ 2 อย่างชัดเจน
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"จำนวนคู่จับคู่ผ่านเกณฑ์ที่ดี: {len(good_matches)}")

# 4. วาดเส้นเชื่อมคู่ฟีเจอร์เด่นเพื่อตรวจสอบความถูกต้องเชิงพิกัด
matched_img = cv2.drawMatches(img_ref, kp_ref, img_scene, kp_scene, good_matches, None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Feature Matches', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 3. การทำ Homography ค้นหาพิกัดและล้อมกรอบวัตถุ

โค้ดขั้นสูงในการหาแมทริกซ์แปลงระนาบเพื่อคำนวณหามุมพิกัดของวัตถุเป้าหมาย และวาดกรอบสี่เหลี่ยมสีเขียวล้อมรอบบนภาพที่มีการหมุนบิดเอี้ยวตัว

```python
import cv2
import numpy as np

# (ต่อจากขั้นตอนการกรอง good_matches ด้านบน)
# ตรวจสอบว่าคู่จับคู่เด่นผ่านเกณฑ์ขั้นต่ำ 4 คู่เพื่อคำนวณ Homography (ปกติแนะนำอย่างน้อย 10 คู่เพื่อความมั่นใจ)
if len(good_matches) >= 10:
    # 1. ดึงพิกัด (x, y) ของจุดสำคัญจากภาพอ้างอิงและภาพเป้าหมายเฉพาะส่วนที่จับคู่ผ่านเกณฑ์
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 2. หาเมทริกซ์ Homography (H) โดยใช้ RANSAC ในการกรองจุดจับคู่ที่หลอกตาออก
    # พารามิเตอร์: พิกัดต้นทาง, พิกัดปลายทาง, อัลกอริทึม, เกณฑ์ระยะห่างยอมรับได้ (RansacReprojThreshold)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 3. กำหนดขอบมุมทั้ง 4 ของภาพอ้างอิงต้นฉบับ
    h_ref, w_ref = img_ref.shape
    ref_corners = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)

    # 4. แปลงพิกัดขอบมุมอ้างอิงไปยังพิกัดปลายทางในภาพจริงผ่านสมการ Homography
    scene_corners = cv2.perspectiveTransform(ref_corners, H)

    # 5. วาดกรอบสีเขียวล้อมรอบพิกัดปลายทางที่ทำนายได้ลงบนภาพจริงสีสีสัน
    img_scene_color = cv2.imread('shelf.jpg')
    # วาดรูปหลายเหลี่ยมเชื่อมกัน 4 มุม โดยให้เส้นบรรจบกันครบวง (isClosed=True)
    cv2.polylines(img_scene_color, [np.int32(scene_corners)], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.imshow('Object Detected', img_scene_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"มีคู่ฟีเจอร์เด่นไม่เพียงพอ - ตรวจพบคู่เด่นเพียง {len(good_matches)} คู่ (ต้องการอย่างน้อย 10 คู่)")
```
