"""
สคริปต์ทำ Fine-Tuning ด้วย MobileNetV3 และส่งออกเป็นไฟล์โมเดลมาตรฐาน ONNX
วิชา: การประมวลผลภาพดิจิทัล (Digital Image Processing) - สัปดาห์ที่ 10
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms

def build_transfer_model(num_classes=2):
    """
    โหลดโมเดล MobileNetV3 Small Pre-trained จาก ImageNet
    ทำการแช่แข็ง Weights ในส่วน Feature Extractor และดัดแปลง Classifier Layer ตัวสุดท้าย
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # แช่แข็ง Weights เพื่อไม่ต้องคำนวณ Gradient ใหม่
    for param in model.parameters():
        param.requires_grad = False

    # ดัดแปลง Linear Layer ตัวสุดท้ายของ Classifier
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model

def export_to_onnx(model, save_path="mobilenet_v3_custom.onnx"):
    """
    ส่งออกโมเดล PyTorch เป็นไฟล์ฟอร์แมต ONNX (พร้อมระบบ Safe Fallback)
    """
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        print(f"[SUCCESS] ONNX model successfully saved to: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"[WARNING] ONNX export encountered an issue: {e}")
        # Fallback to TorchScript saving if ONNX exporter package is missing
        ts_path = save_path.replace(".onnx", ".pt")
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(ts_path)
        print(f"[FALLBACK SUCCESS] Saved TorchScript model to: {os.path.abspath(ts_path)}")

if __name__ == "__main__":
    print("Building MobileNetV3 Transfer Learning Model...")
    model = build_transfer_model(num_classes=2)
    print(model)

    print("\nExporting model to ONNX format...")
    export_to_onnx(model, "mobilenet_v3_custom.onnx")
