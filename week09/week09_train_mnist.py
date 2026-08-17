# train_mnist.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. นิยามสถาปัตยกรรมโมเดล CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # รับภาพ Grayscale 1 Channel -> ผลิต 16 Channels ฟิลเตอร์ขนาด 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # รับ 16 Channels -> ผลิต 32 Channels ฟิลเตอร์ขนาด 3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # ตัวลดขนาดข้อมูลภาพ Max Pooling ขนาด 2x2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layer (ขนาดอินพุตคำนวณจาก: 32 channels * 7 * 7 พิกเซล = 1568)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # ทำนายเลข 0-9 ทั้งหมด 10 คลาส

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # ปรับรูปร่างเป็นเวกเตอร์ตรง 1 มิติ
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    # 2. ตั้งค่าอุปกรณ์ประมวลผล (เลือกใช้ CUDA หากเครื่องสนับสนุน GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"กำลังเริ่มฝึกสอนโมเดลโดยใช้: {device}")

    # 3. กำหนด Pipeline การทำภาพ Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # ปรับสเกลให้อยู่ในช่วง [-1, 1]
    ])

    # 4. ดาวน์โหลดคลังรูปภาพ MNIST
    print("กำลังโหลดชุดข้อมูล MNIST...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 5. จัดการแปลงข้อมูลเป็นกลุ่มย่อย Batch Size = 64
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 6. เรียกใช้งานโครงสร้างโมเดลและส่งไปยังตัวแปรประมวลผลหลัก
    model = SimpleCNN().to(device)

    # 7. นิยามสูตรคำนวณ Loss และ Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 8. เริ่มลูปฝึกสอนจริง (Train Loop)
    epochs = 3
    print("เริ่มขั้นตอนการฝึกสอนโมเดล (Training Loop)...")
    for epoch in range(epochs):
        model.train()  # ตั้งโหมดเทรน
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # เคลียร์เกรเดียนต์
            optimizer.zero_grad()
            
            # ทำนายผล
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # คำนวณความเบี่ยงเบนย้อนกลับและอัปเดตน้ำหนัก
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (batch_idx + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {running_loss/200:.4f}")
                running_loss = 0.0

        # 9. ทดสอบความแม่นยำหลังจบแต่ละ Epoch (Evaluation Phase)
        model.eval()  # ตั้งโหมดประเมินผลลัพธ์
        correct = 0
        total = 0
        with torch.no_grad():  # ไม่ต้องการจำการคำนวณเกรเดียนต์เพื่อเซฟความเร็วและแรม
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"==> ประสิทธิภาพเมื่อจบ Epoch ที่ {epoch+1}: ความแม่นยำบน Test Set = {accuracy:.2f}%")

    # 10. บันทึกผลน้ำหนัก
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("การฝึกฝนเสร็จสมบูรณ์! น้ำหนักตัวแบบถูกบันทึกในไฟล์ 'mnist_cnn.pth'")

if __name__ == "__main__":
    train()
