"""
สคริปต์รัน Inference ไฟล์โมเดล ONNX ผ่านเอนจิน OpenCV DNN Module
วิชา: การประมวลผลภาพดิจิทัล (Digital Image Processing) - สัปดาห์ที่ 10
"""

import sys
import cv2
import numpy as np

def run_inference(onnx_path, image_path=None):
    """
    โหลดไฟล์ .onnx ด้วย cv2.dnn.readNetFromONNX และประมวลผลทำนายผลภาพ
    """
    print(f"Loading ONNX model from: {onnx_path}")
    net = cv2.dnn.readNetFromONNX(onnx_path)
    if net.empty():
        print("Error: Could not load ONNX model.")
        return

    # หากไม่ได้ระบุรูปภาพ ให้สร้างภาพจำลอง (Synthetic Image) สำหรับทดสอบ
    if image_path is None or not sys.argv[1:]:
        print("No image provided. Creating a dummy test image (224x224)...")
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

    # เตรียม Blob สำหรับส่งเข้า DNN (Scale, Resize, Normalization, Swap BGR->RGB)
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0 / 255.0,
        size=(224, 224),
        mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    output = net.forward()

    # คำนวณ Softmax Probabilities
    exp_out = np.exp(output - np.max(output))
    probabilities = exp_out / np.sum(exp_out, axis=1, keepdims=True)

    predicted_class = np.argmax(probabilities)
    confidence = probabilities[0][predicted_class]

    print(f"[RESULT] Predicted Class ID: {predicted_class}")
    print(f"[RESULT] Confidence Score: {confidence * 100:.2f}%")

if __name__ == "__main__":
    onnx_file = "mobilenet_v3_custom.onnx"
    img_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_inference(onnx_file, img_file)
