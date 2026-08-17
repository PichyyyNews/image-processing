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
