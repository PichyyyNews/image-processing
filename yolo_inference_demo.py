"""
สคริปต์สาธิตการใช้งาน YOLOv8 Inference ร่วมกับการวาด Overlay ใน OpenCV
วิชา: การประมวลผลภาพดิจิทัล (Digital Image Processing) - สัปดาห์ที่ 11
"""

import sys
import cv2
import numpy as np

def run_yolo_demo(source_path=None):
    """
    โหลดโมเดล Ultralytics YOLOv8 และประมวลผลสแกนวัตถุ
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please run: pip install ultralytics")
        return

    print("Loading YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")

    # หากไม่ได้ระบุภาพ ให้สร้างภาพจำลองที่มีรูปทรงเพื่อทดสอบ
    if source_path is None or not sys.argv[1:]:
        print("No input image specified. Creating a test synthetic image...")
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # วาดรูปจำลอง
        cv2.rectangle(img, (100, 100), (250, 300), (50, 50, 200), -1)
        cv2.circle(img, (400, 250), 80, (200, 50, 50), -1)
    else:
        img = cv2.imread(source_path)
        if img is None:
            print(f"[ERROR] Could not read image: {source_path}")
            return

    # รัน Inference
    results = model(img, conf=0.25)[0]

    output_img = img.copy()
    detected_count = len(results.boxes)
    print(f"[INFO] Total Objects Detected: {detected_count}")

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        label = f"{class_name}: {conf:.2f}"
        print(f" - Found {class_name} at [{x1}, {y1}, {x2}, {y2}] with conf {conf:.2f}")

        # วาด Bounding Box
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_img, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # บันทึกภาพเอาต์พุต
    cv2.imwrite("yolo_demo_output.jpg", output_img)
    print("[SUCCESS] Detection result saved to 'yolo_demo_output.jpg'")

if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_yolo_demo(img_arg)
