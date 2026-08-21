"""
สคริปต์ฝึกสอนโมเดล YOLO Instance Segmentation
"""

import sys
import io
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ultralytics import YOLO

def train_yolo_segmentation():
    print("=" * 65)
    print(" ✂️ STARTING YOLO INSTANCE SEGMENTATION TRAINING")
    print("=" * 65)

    # 1. โหลดโมเดล YOLO-Seg Pretrained (yolo11n-seg.pt หรือ yolov8n-seg.pt)
    model_name = "yolo11n-seg.pt"
    print(f"📦 Loading base model: {model_name}")
    model = YOLO(model_name)

    # 2. กำหนด Path data.yaml (หรือใช้ coco8-seg สำหรับทดสอบ)
    dataset_yaml = "coco8-seg.yaml"

    print("\n⚙️ Training Settings:")
    print(f"   - Dataset Config : {dataset_yaml}")
    print(f"   - Image Size     : 640")
    print(f"   - Epochs         : 3 (สำหรับ Demo)")
    print(f"   - Batch Size     : 8")

    # 3. สั่งเทรนโมเดล
    results = model.train(
        data=dataset_yaml,
        epochs=3,
        imgsz=640,
        batch=8,
        project="runs_segment",
        name="custom_seg_exp",
        exist_ok=True,
        verbose=True
    )

    print("\n" + "=" * 65)
    print(" ✅ SEGMENTATION TRAINING COMPLETED!")
    print(f" 💾 Model Weights Saved: runs_segment/custom_seg_exp/weights/best.pt")
    print("=" * 65)

if __name__ == '__main__':
    train_yolo_segmentation()
