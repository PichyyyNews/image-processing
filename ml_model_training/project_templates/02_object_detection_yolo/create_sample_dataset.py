"""
สคริปต์ช่วยสร้างชุดข้อมูลจำลอง (Synthetic YOLO Dataset)
สำหรับให้นักศึกษาทดสอบรันการเทรน YOLO Object Detection ได้ทันที
"""

import os
import cv2
import numpy as np

def generate_sample_yolo_dataset(base_dir="sample_yolo_data"):
    print("=" * 65)
    print(" 🛠️ GENERATING SAMPLE YOLO OBJECT DETECTION DATASET")
    print("=" * 65)

    # 1. สร้างโครงสร้างไดเรกทอรี
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'labels', split), exist_ok=True)

    # 2. สร้างไฟล์ data.yaml
    yaml_content = f"""path: {os.path.abspath(base_dir)}
train: images/train
val: images/val

nc: 2
names:
  0: red_circle
  1: blue_box
"""
    yaml_path = os.path.join(base_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"📄 Created config file: '{yaml_path}'")

    # 3. วาดภาพจำลองและสร้างไฟล์ Label .txt
    def create_images(split, count):
        for i in range(count):
            img = np.ones((640, 640, 3), dtype=np.uint8) * 240
            labels = []

            # วัตถุที่ 1: วงกลมสีแดง (Class 0)
            cx1, cy1, r1 = np.random.randint(100, 300), np.random.randint(100, 300), np.random.randint(40, 70)
            cv2.circle(img, (cx1, cy1), r1, (0, 0, 220), -1)
            # พิกัด Normalized [class, x_center, y_center, w, h]
            labels.append(f"0 {cx1/640:.6f} {cy1/640:.6f} {(r1*2)/640:.6f} {(r1*2)/640:.6f}")

            # วัตถุที่ 2: สี่เหลี่ยมสีน้ำเงิน (Class 1)
            x1, y1, w, h = np.random.randint(350, 500), np.random.randint(350, 500), np.random.randint(80, 120), np.random.randint(80, 120)
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (220, 100, 0), -1)
            cx2, cy2 = x1 + w/2, y1 + h/2
            labels.append(f"1 {cx2/640:.6f} {cy2/640:.6f} {w/640:.6f} {h/640:.6f}")

            # บันทึกภาพ
            img_filename = f"img_{split}_{i:03d}.jpg"
            img_path = os.path.join(base_dir, 'images', split, img_filename)
            cv2.imwrite(img_path, img)

            # บันทึก Label
            txt_filename = f"img_{split}_{i:03d}.txt"
            txt_path = os.path.join(base_dir, 'labels', split, txt_filename)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(labels))

    create_images('train', 20)
    create_images('val', 6)
    print(f"✅ Generated 20 Train images and 6 Val images in '{base_dir}/'")

if __name__ == '__main__':
    generate_sample_yolo_dataset()
