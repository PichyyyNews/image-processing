# คำแนะนำซอร์สโค้ด - สัปดาห์ที่ 12 (Code Guide)
## การทำงานกับ Custom YOLO Model Training & Validation

---

## 1. คำอธิบายไฟล์โค้ดประจำสัปดาห์

| ชื่อไฟล์ | วัตถุประสงค์หลัก | คำสั่งสำหรับรัน |
|---|---|---|
| **[`train_custom_yolo.py`](train_custom_yolo.py)** | สคริปต์ Python สั่งเทรน YOLOv8 บน Custom Dataset และประเมินผล mAP | `python train_custom_yolo.py` |

---

## 2. โครงสร้างโค้ดหลักใน `train_custom_yolo.py`

```python
from ultralytics import YOLO

def train_and_eval():
    model = YOLO("yolov8n.pt")
    
    # สั่งเทรนโมเดล
    results = model.train(
        data="custom_dataset/data.yaml",
        epochs=10,
        imgsz=640,
        name="custom_yolo_model"
    )
    
    # วัดผล
    metrics = model.val()
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)

if __name__ == "__main__":
    train_and_eval()
```
