"""
สคริปต์ฝึกสอนโมเดล Custom YOLO Object Detection ด้วย Python API
"""

import sys
import io
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
from ultralytics import YOLO

def train_yolo_detection():
    print("=" * 65)
    print(" 🚀 STARTING CUSTOM YOLO OBJECT DETECTION TRAINING")
    print("=" * 65)

    # 1. ตรวจสอบไฟล์ data.yaml
    data_yaml = "sample_yolo_data/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"⚠️ ไม่พบไฟล์ '{data_yaml}' กำลังสร้างชุดข้อมูลจำลองอัตโนมัติ...")
        from create_sample_dataset import generate_sample_yolo_dataset
        generate_sample_yolo_dataset("sample_yolo_data")

    # 2. โหลด Pretrained YOLO Model (yolo11n.pt หรือ yolov8n.pt)
    model_name = "yolo11n.pt"
    print(f"\n📦 Loading pretrained backbone: {model_name}")
    model = YOLO(model_name)

    # 3. กำหนดการเทรน
    print("\n⚙️ Training Settings:")
    print(f"   - Dataset Config : {data_yaml}")
    print(f"   - Image Size     : 640")
    print(f"   - Epochs         : 5 (สำหรับทดสอบ Demo)")
    print(f"   - Batch Size     : 8")
    print(f"   - Optimizer      : auto (SGD / AdamW)")

    # 4. สั่งเทรนโมเดล
    results = model.train(
        data=data_yaml,
        epochs=5,
        imgsz=640,
        batch=8,
        project="runs_detect",
        name="custom_yolo_exp",
        exist_ok=True,
        verbose=True
    )

    print("\n" + "=" * 65)
    print(" ✅ TRAINING COMPLETED!")
    print(f" 💾 Best Model Weight: runs_detect/custom_yolo_exp/weights/best.pt")
    print("=" * 65)

if __name__ == '__main__':
    train_yolo_detection()
