# คู่มือการจัดทำโครงงาน Mini-Project (Mini-Project Development Guide)
## วิชา: การประมวลผลภาพดิจิทัล (Digital Image Processing) — 31909-2007

---

## 1. 템플릿ไฟล์ `README.md` สำหรับโครงงาน

นักศึกษาทุกคนต้องจัดทำไฟล์ `README.md` ประจำโครงงานตนเองโดยใช้โครงสร้างดังนี้:

```markdown
# 📷 [ชื่อโครงงานของคุณ]

## 📝 คำอธิบายโครงงาน (Project Description)
[อธิบายว่าโปรเจกต์นี้ทำอะไร แก้ไขปัญหาอะไร]

## 🛠️ เครื่องมือและแพ็คเกจที่ใช้ (Technologies Used)
- Python 3.10
- OpenCV 4.6.0
- Ultralytics YOLOv8 / MediaPipe / PyTorch

## ⚙️ วิธีการติดตั้งและสั่งรัน (Installation & Execution)
1. Activate Environment:
   ```bash
   conda activate dip_env
   ```
2. Run Main Program:
   ```bash
   python main.py
   ```
```

---

## 2. ตัวอย่างแม่แบบโค้ดโครงงานแบบประยุกต์ (`main_template.py`)

```python
import cv2
import time

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not detected.")
        return

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Preprocessing Step
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Compute FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 3. Draw Overlay
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Mini-Project Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
