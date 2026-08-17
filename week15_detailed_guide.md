# บทที่ 15: การทดสอบปลายภาคเรียนและการสรุปภาพรวมวิชา (Final Examination & Course Synthesis)
> **หลักสูตร:** การประมวลผลภาพดิจิทัล (Digital Image Processing)  
> **เครื่องมือ:** VS Code, Conda, Python, OpenCV, PyTorch, ONNX, YOLO, MediaPipe

---

## ภาพรวมของบทเรียน

สัปดาห์ที่ 15 เป็นสัปดาห์สรุปประมวลองค์ความรู้ทั้งหมดของรายวิชาการประมวลผลภาพดิจิทัล (Digital Image Processing) ซึ่งครอบคลุมตั้งแต่เทคนิคพื้นฐานในโดเมนเชิงพื้นที่ (Spatial Domain) และโดเมนเชิงความถี่ (Frequency Domain) ไปจนถึงการประยุกต์ใช้แบบจำลองโครงข่ายประสาทเทียมระดับสูง (Deep Learning Models) 

---

## บทที่ 1: ตารางสรุปเปรียบเทียบเทคโนโลยีตลอดหลักสูตร

| หมวดหมู่ | เทคโนโลยี / อัลกอริทึม | จุดเด่น | ข้อจำกัด | ฟังก์ชัน OpenCV / Python หลัก |
|---|---|---|---|---|
| **Spatial Filtering** | Gaussian, Median, Bilateral | ประมวลผลเร็ว ลบ Noise ได้ดี | ไม่สามารถเข้าใจความหมายวัตถุ | `cv2.GaussianBlur()`, `cv2.medianBlur()` |
| **Edge & Morphology** | Canny, Sobel, Opening/Closing | สกัดเส้นขอบชัดเจน ใช้ทรัพยากรน้อย | ไวต่อการเปลี่ยนแปลงของแสง | `cv2.Canny()`, `cv2.morphologyEx()` |
| **Contour Extraction** | `findContours`, Bounding Box | แยกวัตถุออกจากพื้นหลัง ตัด ROI ได้ | วัตถุต้องมี Contrast สูง | `cv2.findContours()`, `cv2.boundingRect()` |
| **Feature Matching** | ORB, SIFT, Homography | ทนทานต่อการหมุน/ย่อขยายขนาด | ทำงานช้าถ้าภาพมีรายละเอียดซับซ้อน | `cv2.ORB_create()`, `cv2.BFMatcher()` |
| **Deep Learning CNN** | PyTorch CNN, MobileNetV3 | เรียนรู้ฟีเจอร์ซับซ้อนอัตโนมัติ | ต้องใช้ข้อมูลเทรนจำนวนมาก | `torch.nn.Conv2d`, `torchvision.models` |
| **ONNX Deployment** | ONNX Export & OpenCV DNN | ขนาดเล็ก รันได้โดยไม่ต้องมี PyTorch | ต้องปรับ Shape Input ให้ตรงกัน | `cv2.dnn.readNetFromONNX()` |
| **Object Detection** | YOLOv8 / YOLOv11 | ตรวจจับหลายวัตถุเรียลไทม์ (FPS สูง) | ต้องการ GPU หากเทรนชุดข้อมูลใหญ่ | `from ultralytics import YOLO` |
| **Landmark Tracking** | MediaPipe Hands / Pose | สกัด 2D/3D Keypoints มนุษย์บน CPU | ประสิทธิภาพลดลงถ้าแสงสว่างไม่พอ | `mp.solutions.pose`, `mp.solutions.hands` |

---

## บทที่ 2: สรุปหลักจริยธรรมและความเป็นส่วนตัว (AI Ethics & PDPA)

ในการนำโปรเจกต์ Computer Vision ไปใช้งานจริง ต้องคำนึงถึง 3 หลักการสำคัญ:
1. **Data Anonymization:** การเปิดเผยภาพสาธารณะต้องทำ Anonymization เบลอใบหน้า (`cv2.GaussianBlur` บน ROI) และป้ายทะเบียนรถ
2. **Bias Mitigation:** ชุดข้อมูลที่นำมาฝึกฝนโมเดลต้องมีความหลากหลายของเพศ สีผิว และสภาพแสง เพื่อป้องกันความลำเอียงของ AI
3. **Transparency:** ระบบระบุชีวมิติ (Biometric Authentication) ต้องมีการแจ้งเตือนและขอความยินยอมจากผู้ใช้งานก่อนเสมอ
