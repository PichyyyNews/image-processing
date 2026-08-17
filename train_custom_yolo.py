"""
สคริปต์สาธิตการสั่งเทรนโมเดล YOLOv8 บนชุดข้อมูล Custom Dataset และวัดผลประสิทธิภาพ
วิชา: การประมวลผลภาพดิจิทัล (Digital Image Processing) - สัปดาห์ที่ 12
"""

import os
import sys

def train_custom_yolo(data_yaml_path="data.yaml", epochs=5):
    """
    สั่งเทรนโมเดล YOLOv8 Nano บนชุดข้อมูลที่กำหนดใน data.yaml
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics package is not installed. Run: pip install ultralytics")
        return

    # สร้างไฟล์ data.yaml ตัวอย่างหากยังไม่มีในระบบ
    if not os.path.exists(data_yaml_path):
        print(f"Creating sample '{data_yaml_path}' config file...")
        sample_yaml = """path: ./dataset
train: train/images
val: val/images

nc: 2
names: ['item_type_a', 'item_type_b']
"""
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            f.write(sample_yaml)

    print("Initializing YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")

    print(f"Starting training pipeline with config: {data_yaml_path}...")
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=8,
            project="runs/custom_train",
            name="exp_custom_yolo",
            exist_ok=True
        )
        print("[SUCCESS] Training pipeline completed!")
    except Exception as e:
        print(f"[NOTE] Training simulation finished or halted: {e}")
        print("In a complete dataset setup, best.pt will be saved to runs/custom_train/exp_custom_yolo/weights/best.pt")

if __name__ == "__main__":
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "data.yaml"
    train_custom_yolo(yaml_file, epochs=1)
