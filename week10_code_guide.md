# คำแนะนำซอร์สโค้ด - สัปดาห์ที่ 10 (Code Guide)
## การทำงานกับ Transfer Learning & OpenCV ONNX Inference

---

## 1. คำอธิบายไฟล์โค้ดประจำสัปดาห์

| ชื่อไฟล์ | วัตถุประสงค์หลัก | คำสั่งสำหรับรัน |
|---|---|---|
| **[`train_transfer_onnx.py`](train_transfer_onnx.py)** | สคริปต์ PyTorch โหลด MobileNetV3, แช่แข็ง weights, Fine-tune FC Layer และ export เป็น `.onnx` | `python train_transfer_onnx.py` |
| **[`infer_onnx.py`](infer_onnx.py)** | สคริปต์ OpenCV DNN โหลดไฟล์ `.onnx` มาทดสอบประมวลผลทำนายภาพ/วิดีโอ | `python infer_onnx.py` |

---

## 2. โครงสร้างโค้ดหลักใน `train_transfer_onnx.py`

```python
import torch
import torch.nn as nn
from torchvision import models, transforms

def get_mobilenet_model(num_classes=2):
    # Load pre-trained MobileNetV3 Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Freeze Feature Extractor
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace Final Classifier Head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
```

---

## 3. โครงสร้างโค้ดหลักใน `infer_onnx.py`

```python
import cv2
import numpy as np

def run_onnx_inference(onnx_path, image_path, labels):
    net = cv2.dnn.readNetFromONNX(onnx_path)
    img = cv2.imread(image_path)
    
    # Preprocess image into blob
    blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (224, 224), (0.485*255, 0.456*255, 0.406*255), swapRB=True)
    net.setInput(blob)
    output = net.forward()
    
    # Softmax probabilities
    exp_out = np.exp(output - np.max(output))
    probs = exp_out / np.sum(exp_out)
    class_id = np.argmax(probs)
    
    print(f"Prediction: {labels[class_id]} ({probs[0][class_id]*100:.1f}%)")
```
