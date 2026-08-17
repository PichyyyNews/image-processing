# infer_mnist.py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. โครงสร้างโมเดล (ต้องเขียนให้ตรงกับตอนเทรน)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. สร้างโครงสร้างและโหลดไฟล์น้ำหนัก
    model = SimpleCNN()
    try:
        model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
        print("โหลดไฟล์น้ำหนัก 'mnist_cnn.pth' สำเร็จ!")
    except FileNotFoundError:
        print("ข้อผิดพลาด: ไม่พบไฟล์น้ำหนัก 'mnist_cnn.pth' กรุณารัน train_mnist.py ก่อน!")
        return

    model.to(device)
    model.eval()  # ปรับเปลี่ยนเข้าโหมดพยากรณ์/วิเคราะห์

    # 3. โหลดและจัดการภาพด้วย OpenCV
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"ข้อผิดพลาด: ไม่พบไฟล์รูปภาพปลายทางที่กำหนด {image_path}")
        return

    # ก. ปรับขนาดภาพให้กลายเป็น 28x28 พิกเซลพอดีตัว
    img_resized = cv2.resize(img_gray, (28, 28))

    # ข. ประเมินปรับข้อมูลพิกเซลให้อยู่ในช่วง [-1, 1] ตามตอนที่โมเดลเรียนรู้
    img_norm = (img_resized.astype(np.float32) / 255.0 - 0.5) / 0.5

    # ค. จัดรูปแบบ Tensor ให้พร้อมส่งต่อ: เพิ่มมิติ Batch และ Channel (1, 1, 28, 28)
    tensor_img = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)
    tensor_img = tensor_img.to(device)

    # 4. ส่งรูปภาพเข้าไปรับพยากรณ์จากโมเดล PyTorch
    with torch.no_grad():
        outputs = model(tensor_img)
        # คำนวณหาค่าความน่าจะเป็นรายคลาสด้วย Softmax
        probabilities = F.softmax(outputs, dim=1)
        # ดึงดัชนีคลาสตัวเลขที่ได้คะแนนสูงสุด
        predicted_val = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_val].item() * 100

    print("\n" + "="*50)
    print(f"วิเคราะห์รูปภาพ: {image_path}")
    print(f"==> คำพยากรณ์จากโมเดล: ตัวเลข {predicted_val}")
    print(f"==> ค่าความมั่นใจ: {confidence:.2f}%")
    print("="*50 + "\n")

    # แสดงภาพพิกเซลขยายใหญ่ที่ถูกส่งเข้าวิเคราะห์
    cv2.imshow(f"Preprocessed Image (28x28) - Predicted: {predicted_val}", cv2.resize(img_resized, (300, 300), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict('digit.png')
