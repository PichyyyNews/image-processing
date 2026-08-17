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
