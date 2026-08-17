# คู่มือสรุปเตรียมสอบปลายภาค (Final Examination Prep Guide)
## วิชา: การประมวลผลภาพดิจิทัล (Digital Image Processing) — 31909-2007

---

## 1. แนวข้อสอบภาคทฤษฎี (Sample Theoretical Questions)

### ข้อที่ 1: การเปรียบเทียบ IoU
**โจทย์:** Bounding Box เฉเฉลยมีขนาด $100 \times 100$ พิกเซล Bounding Box ที่โมเดลทายทับซ้อนกันเป็นพื้นที่ $60 \times 60$ พิกเซล และพื้นที่รวม Union เท่ากับ $12,000$ พิกเซล จงคำนวณค่า IoU และสรุปว่ากรอบนี้ผ่านเกณฑ์ IoU Threshold $0.50$ หรือไม่?
**คำนวณ:**
$$\text{Area of Overlap} = 60 \times 60 = 3,600$$
$$\text{IoU} = \frac{3,600}{12,000} = 0.30$$
**สรุป:** ค่า $\text{IoU} = 0.30$ ซึ่งน้อยกว่า Threshold $0.50$ ดังนั้น กรอบนี้ **ไม่ผ่านเกณฑ์ (False Positive)**

---

## 2. แม่แบบโค้ดข้อสอบปฏิบัติการปลายภาค (`final_lab_challenge.py`)

```python
import cv2
import numpy as np

def run_final_lab_challenge(video_path, model_onnx_path):
    """
    สคริปต์โซลูชันข้อสอบปฏิบัติการปลายภาค
    1. โหลดโมเดล ONNX ผ่าน OpenCV DNN
    2. อ่านวิดีโอ ประมวลผลทำนายวัตถุ
    3. นับจำนวนวัตถุและแสดงผลบน GUI
    """
    net = cv2.dnn.readNetFromONNX(model_onnx_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (224, 224), swapRB=True)
        net.setInput(blob)
        output = net.forward()

        # Display result
        cv2.putText(frame, "Final Exam Pipeline Running", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Final Exam", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
```
