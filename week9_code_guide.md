# คู่มือการเขียนโค้ด (Code Guide) - สัปดาห์ที่ 9
**หัวข้อ:** พื้นฐาน Machine Learning, Data Pipeline, Decision Tree, Perceptron, และ Simple CNN (PyTorch)

ในคู่มือโค้ดสัปดาห์นี้จะถูกแบ่งออกเป็น 4 ส่วนหลักเพื่อให้นักศึกษาเข้าใจพื้นฐานทีละส่วน:
1. การจัดการข้อมูล (Data Pipeline: Get, Preprocess, Split) และการสร้างโมเดลจำแนกด้วย **Decision Tree**
2. การเขียนฟังก์ชันจำลองการทำงานและผลการคำนวณของ **Perceptron** ด้วยมือในภาษา Python
3. โครงสร้างและการเทรนแบบจำลอง **CNN** บน MNIST ด้วย PyTorch
4. การรันทำนายภาพผลลัพธ์ลายมือภายนอกด้วย OpenCV (Inference Pipeline)

---

## 1. Data Pipeline และการรันโมเดล Decision Tree

โค้ดนี้สาธิตขั้นตอนของ Data Pipeline โดยการจำลองการสกัดคุณลักษณะ (Features) ของรูปทรงเรขาคณิต (วงกลม, สามเหลี่ยม, สี่เหลี่ยม) ป้อนเข้าการ Preprocess การแบ่งชุดข้อมูล (Train/Test Split) และส่งต่อไปประมวลผลบนแบบจำลอง Decision Tree ของ Scikit-learn

```python
# pipeline_decision_tree.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

# =====================================================================
# ขั้นตอนที่ 1: Get Data (รวบรวมข้อมูลจำลองคุณลักษณะของรูปทรงเรขาคณิต)
# คุณลักษณะ (Features): [Circularity (ความกลม), Vertices (จำนวนยอดมุมที่นับได้)]
# ป้ายเฉลย (Labels): 0 = วงกลม, 1 = สามเหลี่ยม, 2 = สี่เหลี่ยม
# =====================================================================
# สร้างข้อมูลตัวอย่าง (Dataset) 15 ข้อมูล
features = np.array([
    [0.98, 0], [0.95, 0], [0.97, 1], [0.99, 0], [0.96, 0],  # วงกลม (กลมสูง มุมน้อย/ไม่มีมุม)
    [0.55, 3], [0.58, 3], [0.60, 3], [0.52, 3], [0.57, 3],  # สามเหลี่ยม (กลมปานกลาง ยอดมุม 3)
    [0.72, 4], [0.75, 4], [0.70, 4], [0.78, 4], [0.74, 4]   # สี่เหลี่ยม (กลมค่อนข้างสูง ยอดมุม 4)
])
labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

print("--- 1. Get Data ---")
print(f"ขนาดข้อมูล Features: {features.shape} | Labels: {labels.shape}")

# =====================================================================
# ขั้นตอนที่ 2: Data Splitting (แบ่งกลุ่มข้อมูลเป็น Train Set และ Test Set)
# =====================================================================
# แบ่งข้อมูลเป็น Train 70% และ Test 30% โดยสุ่มสลับ (random_state ช่วยควบคุมค่าสุ่มให้คงที่)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

print("\n--- 2. Data Splitting ---")
print(f"ชุดฝึกสอน (Train Set): {X_train.shape} ข้อมูล")
print(f"ชุดทดสอบ (Test Set):  {X_test.shape} ข้อมูล")

# =====================================================================
# ขั้นตอนที่ 3: Data Preprocessing (ทำ Normalization สเกลข้อมูลให้สมดุล)
# =====================================================================
# ใช้ StandardScaler ปรับค่าเฉลี่ยเป็น 0 และความแปรปรวนเป็น 1 (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # ใช้ fit ของเทรนห้ามฟิตข้อมูลเทส เพื่อกันข้อมูลรั่วไหล

print("\n--- 3. Data Preprocessing (Scaled Train Features) ---")
print(X_train_scaled[:3]) # แสดงตัวอย่าง 3 ข้อมูลแรกที่ทำ Preprocessing แล้ว

# =====================================================================
# ขั้นตอนที่ 4: Model Training (เทรนโมเดล Decision Tree)
# =====================================================================
# ประกาศตัวแบบและสั่งเรียนรู้จากข้อมูลฝึกสอน
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train_scaled, y_train)

print("\n--- 4. Model Training Completed ---")

# =====================================================================
# ขั้นตอนที่ 5: Model Evaluation (ประเมินและทำนายผลลัพธ์)
# =====================================================================
# ทดสอบความถูกต้องบนข้อมูลทดสอบ (Test Set)
accuracy = clf.score(X_test_scaled, y_test)
print(f"\n--- 5. Evaluation ---")
print(f"ความถูกต้องจำแนก (Test Accuracy): {accuracy * 100:.2f}%")

# แสดงโครงสร้างเงื่อนไขจำแนกผลของต้นไม้ตัดสินใจ (Decision Tree Structure)
tree_rules = export_text(clf, feature_names=["Circularity", "Vertices"])
print("\nโครงสร้างต้นไม้เงื่อนไขการแยกกลุ่ม:")
print(tree_rules)
```

---

## 2. โค้ดคำนวณการทำงานของ Perceptron (Neural Network) ด้วยมือ

สคริปต์นี้เขียนเลียนแบบการทำโจทย์ Perceptron เพื่อตรวจเฉลยการคำนวณสมการ $z = \sum w_i x_i + b$ และเปรียบเทียบผลฟังก์ชันกระตุ้น (Activation Functions) ทั้ง 3 รูปแบบ

```python
# perceptron_calculation.py
import numpy as np

def step_function(z):
    return 1 if z >= 0 else 0

def relu_function(z):
    return max(0.0, z)

def sigmoid_function(z):
    return 1.0 / (1.0 + np.exp(-z))

# 1. กำหนดอินพุต ค่าน้ำหนัก และไบแอส ตามแบบฝึกหัด
x = np.array([0.5, 0.8])
w = np.array([0.4, -0.6])
b = 0.1

print("=" * 50)
print("  การคำนวณ Perceptron ในรูปแบบภาษา Python")
print("=" * 50)
print(f"อินพุต (Inputs: x)       : {x}")
print(f"ค่าน้ำหนัก (Weights: w)   : {w}")
print(f"ไบแอส (Bias: b)          : {b}")

# 2. คำนวณหาผลรวมเชิงเส้น (Linear Combination: z)
z = np.dot(w, x) + b
print("-" * 50)
print(f"ผลรวมเชิงเส้น (z) = w1*x1 + w2*x2 + b")
print(f"               = ({w[0]} * {x[0]}) + ({w[1]} * {x[1]}) + {b}")
print(f"               = {z:.4f}")
print("-" * 50)

# 3. ส่งข้อมูลผ่านฟังก์ชันกระตุ้นรูปแบบต่างๆ
print(f"ผลลัพธ์ผ่าน Activation Functions:")
print(f"  -> A. Step Function  : {step_function(z)}")
print(f"  -> B. ReLU Function  : {relu_function(z):.4f}")
print(f"  -> C. Sigmoid        : {sigmoid_function(z):.4f}")
print("=" * 50)
```

---

## 3. การสร้างและฝึกสอนโมเดล Simple CNN ด้วย PyTorch

เราจะสืบทอดโครงสร้างจาก `torch.nn.Module` และทำการโหลดจัดเตรียมชุดข้อมูลตัวเลขเขียนด้วยลายมือ MNIST

```python
# train_mnist.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. นิยามโครงข่ายประสาทเทียม SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # รับภาพ Grayscale 1 Channel -> แปลงสกัดส่งต่อ 16 Channels ด้วยฟิลเตอร์ 3x3
        # padding=1 ช่วยรักษาความกว้างและสูงของภาพให้เป็น 28x28 พิกเซลเท่าเดิม
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        
        # รับ 16 Channels -> สกัดได้ 32 Channels ตัวกรองขนาด 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # เลเยอร์ย่อยขนาดพื้นที่ Max Pooling (ขนาด 2x2, stride 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # คำนวณมิติมวลภาพก่อนต่อเข้า Linear:
        # ภาพเริ่มต้น: 28x28
        # หลัง Conv1 & Pooling ครั้งที่ 1: 14x14 พิกเซล
        # หลัง Conv2 & Pooling ครั้งที่ 2: 7x7 พิกเซล
        # มิติก่อนเข้า Linear layer = 32 channels * 7 * 7 พิกเซล = 1568 ตัวแปร
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # Output 10 คลาสสิกแทนเลข 0-9

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> MaxPool
        
        # ปรับรูปโครงสร้างมิติเมทริกซ์ 2 มิติให้เป็นเวกเตอร์แถวตรง 1 มิติ (Flatten)
        x = x.view(x.size(0), -1)
        
        # ส่งเข้าชั้นจำแนกคำพยากรณ์
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    # ตรวจสอบตัวเลือกการประมวลผล GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ระบบเลือกประมวลผลผ่าน: {device}")
    
    # 2. นิยามการทำ Preprocessing ของภาพใน PyTorch
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # ปรับระดับความเข้มช่วงสีเป็น [-1, 1]
    ])
    
    # 3. โหลดชุดข้อมูล MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 4. แบ่งข้อมูลส่งต่อเป็นกลุ่มด้วย DataLoader (Batch Size = 64)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    model = SimpleCNN().to(device)
    
    # 5. กำหนดสูตรความสูญเสียและ Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. ลูปการฝึกสอนโมเดล (Train Loop)
    epochs = 3
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()       # เคลียร์ทิศทางความชันเดิม
            outputs = model(images)      # พยากรณ์คำตอบ (Forward Pass)
            loss = criterion(outputs, labels) # หาค่าความต่างสูญเสีย (Loss)
            loss.backward()             # หาค่าเกรเดียนต์ย้อนกลับ (Backward Pass)
            optimizer.step()            # ปรับปรุงค่าน้ำหนัก (Update Weights)
            
            running_loss += loss.item()
            if (batch_idx + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {running_loss/200:.4f}")
                running_loss = 0.0
                
        # ขั้นประเมินผลลัพธ์เมื่อจบรอบการเรียนรู้
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"==> Accuracy on Test Set: {accuracy:.2f}%")
        
    # เซฟเก็บไฟล์น้ำหนักโมเดล
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("บันทึกไฟล์น้ำหนักเป็น 'mnist_cnn.pth' สำเร็จ!")

if __name__ == "__main__":
    train()
```

---

## 4. โค้ดทำนายผลลัพธ์ภาพตัวเลขวาดมือภายนอก (Inference Pipeline)

สคริปต์นี้นำเข้าไฟล์น้ำหนัก `.pth` และเขียนกระบวนการ Preprocessing รูปภาพภายนอกที่นักศึกษาวาดขึ้นเองผ่าน OpenCV (ขนาด $28 \times 28$, ทำการสเกล, ปรับขนาดมิติของ Tensor) ส่งให้โมเดลประมวลผลคำตอบออกมา

```python
# infer_mnist.py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ต้องออกแบบโครงสร้างให้ตรงเป๊ะกับสถาปัตยกรรมโมเดลที่ใช้ฝึกสอน
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
    
    # 1. โหลดโมเดลและไฟล์น้ำหนัก
    model = SimpleCNN()
    try:
        model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
        print("โหลดค่าน้ำหนักเรียบร้อยแล้ว!")
    except FileNotFoundError:
        print("ข้อผิดพลาด: ไม่พบไฟล์ 'mnist_cnn.pth' กรุณารันสคริปต์ฝึนสอนก่อน!")
        return
        
    model.to(device)
    model.eval() # ปรับโมเดลให้อยู่ในโหมดประเมิน/ทำนาย (ปิด Dropout/BatchNorm)
    
    # 2. โหลดรูปภาพขาวดำด้วย OpenCV
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"ข้อผิดพลาด: ไม่พบรูปภาพที่ตำแหน่ง {image_path}")
        return
        
    # 3. ดำเนินการ Data Preprocessing (ต้องทำเหมือนกับชุดข้อมูล MNIST)
    # ก. ปรับขนาดภาพภาพให้เหมาะสม (28x28 พิกเซล)
    img_resized = cv2.resize(img_gray, (28, 28))
    
    # ข. MNIST เป็นตัวเลขเขียนสีขาวบนฉากหลังดำ
    # หมายเหตุ: หากรูปวาดเขียนมาเป็น ตัวเลขดำบนกระดาษขาว ให้เปิดใช้คำสั่งด้านล่างนี้เพื่อสลับสี:
    # img_final = cv2.bitwise_not(img_resized)
    img_final = img_resized 
    
    # ค. ปรับระดับข้อมูลและช่วงพิกเซลสีให้เปรียบเทียบอยู่ในช่วง [-1, 1]
    img_norm = (img_final.astype(np.float32) / 255.0 - 0.5) / 0.5
    
    # ง. จัดเรียงมิติข้อมูลภาพให้ตรงรูปแบบ Tensor (Batch, Channel, Height, Width) -> (1, 1, 28, 28)
    tensor_img = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)
    tensor_img = tensor_img.to(device)
    
    # 4. ประมวลผลทำนายคำตอบภาพ
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = F.softmax(outputs, dim=1)
        predicted_val = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_val].item() * 100
        
    print("\n" + "="*50)
    print(f"รูปภาพภายนอกที่ป้อนเข้ามา: {image_path}")
    print(f"==> ผลลัพธ์ตัวเลขที่โมเดลทำนายได้: {predicted_val}")
    print(f"==> ระดับเปอร์เซ็นต์ความมั่นใจ: {confidence:.2f}%")
    print("="*50 + "\n")
    
    # ขยายมิติเปิดหน้าต่าง GUI โชว์ภาพที่โมเดลใช้มองจริง
    cv2.imshow(f"Image for Model - Predict: {predicted_val}", cv2.resize(img_final, (300, 300), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict('digit.png')
```
