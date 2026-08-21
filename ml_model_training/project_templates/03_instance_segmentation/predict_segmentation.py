"""
สคริปต์ทำนายผล Instance Segmentation พร้อมคำนวณพื้นที่ Polygon Mask และวาด Overlay สีโปร่งแสง
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

def segment_and_measure(source_img_path=None, model_path="yolo11n-seg.pt"):
    print("=" * 65)
    print(" ✂️ RUNNING INSTANCE SEGMENTATION INFERENCE & AREA MEASUREMENT")
    print("=" * 65)

    # 1. โหลดโมเดล
    model = YOLO(model_path)

    # 2. จัดเตรียมรูปภาพ
    if source_img_path is None or not cv2.haveImageReader(source_img_path):
        print("🖼️ Generating sample image with geometric shapes...")
        img = np.ones((500, 500, 3), dtype=np.uint8) * 240
        # วาดวัตถุรูปทรงอิสระ
        pts = np.array([[100, 150], [200, 100], [300, 180], [250, 300], [120, 280]], np.int32)
        cv2.fillPoly(img, [pts], (50, 180, 50))
    else:
        img = cv2.imread(source_img_path)

    # 3. รัน Inference
    results = model.predict(source=img, verbose=False)
    result = results[0]

    annotated_img = img.copy()
    overlay = img.copy()

    # 4. ตรวจสอบว่าพบ Masks หรือไม่
    if result.masks is not None:
        masks = result.masks.xy # ดึงพิกัด Polygon ในหน่วยพิกเซลจริง
        boxes = result.boxes

        print(f"\n🔍 Detected {len(masks)} segmented object(s):")
        for i, (polygon, box) in enumerate(zip(masks, boxes)):
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            conf = box.conf[0].item()

            # คำนวณพื้นที่รูปหลายเหลี่ยม (Polygon Area in Pixels)
            polygon_int = np.array(polygon, dtype=np.int32)
            area_pixels = cv2.contourArea(polygon_int)

            print(f"   [{i+1}] Object: '{cls_name}' | Conf: {conf*100:.1f}% | Mask Area: {area_pixels:,.1f} px²")

            # วาด Mask สีโปร่งแสง (Alpha Blending)
            color = (0, 200, 0)
            cv2.fillPoly(overlay, [polygon_int], color)
            cv2.polylines(annotated_img, [polygon_int], True, (0, 100, 0), 2)

        # รวมภาพ Overlay โปร่งแสง 40%
        cv2.addWeighted(overlay, 0.4, annotated_img, 0.6, 0, annotated_img)

    # 5. บันทึกผลลัพธ์
    out_path = "segmentation_result.jpg"
    cv2.imwrite(out_path, annotated_img)
    print(f"\n💾 Segmentation visualization saved to: '{out_path}'")

if __name__ == '__main__':
    segment_and_measure()
