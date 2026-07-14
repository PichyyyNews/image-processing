# คู่มือการเขียนโค้ด (Code Guide) - สัปดาห์ที่ 9

**Lab 9: Build & Train CNN in VS Code**

ในสัปดาห์นี้เราจะฝึกเขียนโค้ดเชิงลึก 4 ส่วนหลัก ได้แก่:
1. การนิยามโครงสร้างคลาสแบบจำลอง CNN (`SimpleCNN`) โดยใช้ PyTorch
2. การโหลดจัดการฐานข้อมูลรูปภาพ MNIST ผ่านคลาส DataLoader
3. การเขียนลูปฝึกสอนและประเมินผลตัวแบบ (Train Loop & Validation Loop)
4. การใช้ OpenCV ในการประมวลผลเตรียมรูปภาพตัวเลขที่วาดเองจากภายนอก ส่งป้อนทำนายผลผ่านตัวแบบที่ถูกฝึกสอนเสร็จแล้ว (Inference pipeline)

---

## 1. การสร้างคลาสโครงสร้างตัวแบบ CNN ด้วย PyTorch

เราจะสืบทอดคุณสมบัติมาจาก `torch.nn.Module` เพื่อสร้างโมเดลโครงข่ายประสาทที่มี Convolution 2 ชั้น และตามด้วย Linear Layers สำหรับแปลงขนาดมิติ Output

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Layer 1: รับภาพขาวดำ 1 Channel (28x28) -> ส่งออก 16 Channels (ตัวกรอง 16 ตัว ขนาด 3x3)
        # padding=1 เพื่อรักษามิติขนาดความกว้างยาวให้คงที่เท่าเดิม 28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        
        # Layer 2: รับ 16 Channels -> ส่งออก 32 Channels (ตัวกรอง ขนาด 3x3)
        # padding=1 รักษามิติกว้างยาวเท่าเดิมก่อนทำ Pooling
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Layer สำหรับลดมิติเชิงพื้นที่: Max Pooling ขนาด 2x2 สไลด์ทีละ 2 พิกเซล
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # หมายเหตุการเปลี่ยนแปลงขนาดขนาดภาพหลังจากประมวลผล:
        # ภาพเริ่มที่: 28x28 พิกเซล
        # หลัง conv1: 28x28 พิกเซล
        # หลัง pool ครั้งที่ 1: 14x14 พิกเซล
        # หลัง conv2: 14x14 พิกเซล
        # หลัง pool ครั้งที่ 2: 7x7 พิกเซล
        # ขนาด Matrix สุดท้ายก่อนแปลงเป็นเวกเตอร์ตรง: 32 channels * 7 * 7 พิกเซล = 1568 พารามิเตอร์
        
        # Fully Connected layers (Linear Layer)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # Output 10 คลาสสิกแทนเลข 0-9

    def forward(self, x):
        # ลำดับการประมวลผลส่งต่อข้อมูลไปข้างหน้า
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> MaxPool
        
        # แผ่ขนาดรูปภาพ 2 มิติ (32x7x7) ให้กลายเป็นเวกเตอร์ 1 มิติ (ขนาด 1568)
        # x.size(0) คือขนาด Batch ของข้อมูลที่ถูกส่งเข้ามา
        x = x.view(x.size(0), -1)
        
        # ส่งต่อเวกเตอร์เข้า Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ทดสอบโครงสร้างโมเดลโดยใช้ Dummy Tensor
if __name__ == "__main__":
    model = SimpleCNN()
    dummy_input = torch.randn(1, 1, 28, 28)  # Batch=1, Channel=1, W=28, H=28
    output = model(dummy_input)
    print("โครงสร้างโมเดล:")
    print(model)
    print(f"\nมิติผลลัพธ์การสุ่มรันพยากรณ์: {output.shape}")  # ควรเป็น torch.Size([1, 10])
```

---

## 2. การจัดการสตรีมโหลดข้อมูล Dataset และ DataLoader

เราจะแปลงข้อมูลรูปภาพพิกเซลทั่วไปให้อยู่ในรูปเวกเตอร์ Tensor และทำการ Normalize ข้อมูลลดขนาดลงให้อยู่ในช่วง $[0, 1]$ หรือ $[-1, 1]$

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. นิยามขั้นตอนการจัดเตรียมรูปภาพ (Preprocessing Pipeline)
# transforms.ToTensor(): แปลงภาพเป็น Tensor และแปลงช่วงพิกเซลจาก [0, 255] เป็น [0.0, 1.0]
# transforms.Normalize((0.5,), (0.5,)): ปรับค่าพิกเซลให้อยู่ในช่วง [-1.0, 1.0] 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. โหลดชุดข้อมูล MNIST (ดาวน์โหลดฟรีอัตโนมัติหากยังไม่มี)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3. จัดทำสตรีม DataLoader เพื่อดึงข้อมูลประมวลผลเป็นกลุ่มย่อย (Batch)
# batch_size=64: โหลดประมวลผลทีละ 64 รูปพร้อมๆ กัน
# shuffle=True: สลับลอจิกรูปภาพแบบสุ่มในแต่ละรอบ Epoch
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(f"จำนวน Batch ทั้งหมดในชุดฝึกสอน: {len(train_loader)}")
print(f"จำนวนรูปภาพในชุดทดสอบรวม: {len(test_dataset)} รูป")
```

---

## 3. การเขียนคำสั่งลูปฝึกสอนโมเดล (Train Loop)

กระบวนการวนลูปเพื่อคำนวณหาค่าเกรเดียนต์และทำการป้อนปรับเปลี่ยนค่าน้ำหนักโมเดลในแต่ละ Epoch

```python
import torch.optim as optim

# ตรวจสอบการรองรับการประมวลผลแบบ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ระบบเลือกฮาร์ดแวร์ประมวลผล: {device}")

model = SimpleCNN().to(device)

# กำหนด Loss Function (Cross Entropy สำหรับจำแนก 10 คลาส)
criterion = nn.CrossEntropyLoss()

# กำหนด Optimizer (เราใช้ Adam พร้อมระบุอัตราความไวในการเรียนรู้ lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# เริ่มคำสั่งวนลูปฝึกสอนโมเดล
epochs = 3
for epoch in range(epochs):
    model.train()  # ปรับสถานะโมเดลให้อยู่ในโหมดเทรน
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # โอนย้ายข้อมูลภาพและเฉลยไปคำนวณบนอุปกรณ์หลัก (CPU หรือ GPU)
        images, labels = images.to(device), labels.to(device)
        
        # 1. เคลียร์ค่าเกรเดียนต์ที่คงเหลือจากการประมวลผล Batch ก่อนหน้า
        optimizer.zero_grad()
        
        # 2. ป้อนภาพเข้าสู่ตัวแบบทำนายคำตอบ (Forward Pass)
        outputs = model(images)
        
        # 3. คำนวณความต่างความสูญเสียเมื่อเทียบกับเฉลยจริง
        loss = criterion(outputs, labels)
        
        # 4. คำนวณอนุพันธ์ทิศทางเกรเดียนต์ย้อนกลับ (Backward Pass)
        loss.backward()
        
        # 5. สั่ง Optimizer อัปเดตค่าน้ำหนักน้ำหนักโมเดล
        optimizer.step()
        
        running_loss += loss.item()
        
        # ดีบักความคืบหน้าทุกๆ 200 batch
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {running_loss/200:.4f}")
            running_loss = 0.0

# สั่งเซฟน้ำหนักโมเดลเก็บลงเครื่อง
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("บันทึกไฟล์น้ำหนักโมเดลเป็น 'mnist_cnn.pth' เรียบร้อย!")
```

---

## 4. สคริปต์ทำนายรูปภาพตัวเลขเขียนลายมือจริง (Inference Pipeline)

โค้ดนี้สาธิตวิธีการนำไฟล์น้ำหนักโมเดลที่เซฟไว้ มาโหลดใช้งานทำนายรูปภาพไฟล์ข้างนอกของจริงที่นักศึกษาสร้างขึ้นมาเอง เช่น โหลดภาพ `digit.png` ผ่าน OpenCV ปรับเตรียมมิติ และแปลงให้เป็น Tensor ป้อนให้ PyTorch วิเคราะห์หาผลเฉลยพยากรณ์คลาสตัวเลข

```python
import cv2
import torch
import numpy as np

# 1. โหลดแบบจำลองและตั้งค่าให้อยู่ในโหมดประเมิน (Evaluation Mode)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
model.to(device)
model.eval()

# 2. โหลดรูปภาพตัวเลขภายนอกในโหมดขาวดำ (Grayscale)
# หากรูปที่วาดมาเป็นกระดาษขาวดินสอดำ ให้เขียนคำสั่งอินเวอร์สกลับสี
image_path = 'digit.png'
img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_raw is None:
    print(f"ข้อผิดพลาด: ไม่พบรูปภาพที่ตำแหน่ง {image_path}")
else:
    # 3. ปรับกระบวนการ Preprocessing ให้ตรงตามสเปก MNIST
    # ก. ปรับขนาดรูปภาพภาพให้ลดเหลือ 28x28 พิกเซล
    img_resized = cv2.resize(img_raw, (28, 28))
    
    # ข. MNIST เป็นตัวเลขเขียนสีขาวบนฉากหลังดำสนิท 
    # หากผู้เรียนเตรียมรูปมาเป็นเลขสีดำบนกระดาษขาว ให้สั่ง Invert สี:
    # img_final = cv2.bitwise_not(img_resized)
    img_final = img_resized # กรณีรูปถูกเตรียมมาเป็นตัวเลขขาวบนพื้นหลังดำอยู่แล้ว
    
    # ค. นำสเกลภาพแปลงเป็นเวกเตอร์คณิตศาสตร์และปรับค่าช่วงพิกเซลเป็น [-1, 1]
    img_norm = (img_final.astype(np.float32) / 255.0 - 0.5) / 0.5
    
    # ง. จัดมิติ Tensor ให้กลายเป็น (Batch, Channel, Height, Width)
    # เพิ่มมิติ Batch size เป็น 1 และ Channel size เป็น 1
    tensor_img = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)
    tensor_img = tensor_img.to(device)
    
    # 4. ส่งข้อมูลพิกเซลเข้าทำนายผลใน PyTorch
    with torch.no_grad():
        outputs = model(tensor_img)
        # ดึงดัชนีคลาสความน่าจะเป็นที่มีคะแนนทำนายสูงสุด
        predicted_prob, predicted_class = torch.max(outputs, 1)
        
    print("=" * 45)
    print(f"ผลลัพธ์การจำแนกภาพตัวเลขจากโมเดล:")
    print(f"  --> ทำนายเป็นตัวเลข: {predicted_class.item()}")
    print(f"  --> ค่าคะแนนความมั่นใจ (Raw score): {predicted_prob.item():.4f}")
    print("=" * 45)
    
    # แสดงภาพก่อนและหลังจัดเรียงพิกเซล
    cv2.imshow("Input Image to Model", img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```
