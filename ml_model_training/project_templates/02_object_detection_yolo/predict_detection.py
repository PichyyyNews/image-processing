"""
สคริปต์ทำนายผล YOLO Object Detection บนรูปภาพ หรือเปิดกล้อง Real-time Webcam
"""

import sys
import io
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import numpy as np
from ultralytics import YOLO

def detect_objects_and_overlay(source_img_path=None, model_path="yolo11n.pt", conf_thresh=0.25):
    print("=" * 65)
    print(" 🎯 RUNNING YOLO OBJECT DETECTION INFERENCE")
    print("=" * 65)

    # 1. โหลดโมเดล
    print(f"📦 Loading weights from: {model_path}")
    model = YOLO(model_path)

    # 2. เตรียมภาพ
    if source_img_path is None or not cv2.haveImageReader(source_img_path):
        print("🖼️ Creating sample testing image...")
        img = np.ones((480, 640, 3), dtype=np.uint8) * 230
        cv2.circle(img, (200, 240), 60, (0, 0, 220), -1)
        cv2.rectangle(img, (380, 180), (520, 320), (220, 100, 0), -1)
    else:
        img = cv2.imread(source_img_path)

    # 3. รันการทำนายผล (Inference)
    results = model.predict(source=img, conf=conf_thresh, verbose=False)
    result = results[0]

    # 4. วนลูปอ่านพิกัด Bounding Box
    annotated_img = img.copy()
    boxes = result.boxes
    print(f"\n🚀 Detected {len(boxes)} object(s) in image:")

    for idx, box in enumerate(boxes):
        # พิกัดกล่อง [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        cls_name = result.names[cls_id]

        print(f"   [{idx+1}] Class: '{cls_name}' (ID: {cls_id}) | Conf: {conf*100:.2f}% | Box: [{x1}, {y1}, {x2}, {y2}]")

        # วาด Bounding Box ด้วย OpenCV
        color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)

        # วาดป้ายข้อความ Label Badge
        label = f"{cls_name} {conf*100:.1f}%"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated_img, (x1, y1 - 25), (x1 + w, y1), color, -1)
        cv2.putText(annotated_img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 5. บันทึกผลลัพธ์
    out_file = "yolo_detection_result.jpg"
    cv2.imwrite(out_file, annotated_img)
    print(f"\n💾 Saved detection output to: '{out_file}'")

if __name__ == '__main__':
    detect_objects_and_overlay()
