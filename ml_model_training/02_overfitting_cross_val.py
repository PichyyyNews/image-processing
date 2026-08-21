"""
สคริปต์ตัวอย่าง: การตรวจจับ Overfitting/Underfitting, Cross-Validation และ Regularization
วิชา: การประมวลผลภาพดิจิทัลและปัญญาประดิษฐ์ (DIP & ML Fundamentals)
"""

import sys
import io

# Reconfigure stdout for UTF-8 on Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score

def demonstrate_overfitting_underfitting():
    """
    สร้างโจทย์เปรียบเทียบระหว่าง:
    1. Underfitting Model (Decision Tree depth=1)
    2. Balanced Fit Model (Decision Tree depth=4)
    3. Overfitting Model (Decision Tree depth=20 หรือ unconstrained)
    """
    print("=" * 65)
    print(" 📊 OVERFITTING vs UNDERFITTING DEMONSTRATION")
    print("=" * 65)

    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_clusters_per_class=2,
        flip_y=0.1, # เพิ่ม Noise สภาพแวดล้อม
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Underfitting Model (Max Depth = 1)": DecisionTreeClassifier(max_depth=1, random_state=42),
        "Optimal Fit Model   (Max Depth = 4)": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Overfitting Model  (Max Depth = 20)": DecisionTreeClassifier(max_depth=20, random_state=42)
    }

    print(f"{'Model Configuration':38s} | {'Train Acc':10s} | {'Test Acc':10s} | Status")
    print("-" * 75)

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        diff = train_acc - test_acc
        status = "Good Fit"
        if diff > 0.15:
            status = "Overfitting ⚠️"
        elif train_acc < 0.70:
            status = "Underfitting ⚠️"

        print(f"{name:38s} | {train_acc*100:8.2f}% | {test_acc*100:8.2f}% | {status}")

def demonstrate_stratified_kfold():
    """
    สาธิตการใช้งาน Stratified K-Fold Cross Validation กับชุดข้อมูล Imbalanced
    """
    print("\n" + "=" * 65)
    print(" 🔄 STRATIFIED K-FOLD CROSS-VALIDATION DEMO")
    print("=" * 65)

    # สร้างชุดข้อมูล Imbalanced (Ratio 90:10)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        weights=[0.90, 0.10],
        random_state=42
    )

    clf = LogisticRegression(random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')

    print(f"5-Fold Stratified CV Accuracy Scores: {np.round(cv_scores * 100, 2)}")
    print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% (± {cv_scores.std() * 100:.2f}%)")

def demonstrate_regularization():
    """
    สาธิตการใช้ L2 Regularization (Ridge) ช่วยลดการ Overfitting
    """
    print("\n" + "=" * 65)
    print(" 🛡️ REGULARIZATION (L2 RIDGE) DEMO")
    print("=" * 65)

    X, y = make_classification(n_samples=300, n_features=50, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # RidgeClassifier พร้อมปรับค่า alpha (L2 Regularization Strength)
    alphas = [0.01, 1.0, 100.0]
    for alpha in alphas:
        model = RidgeClassifier(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Alpha (L2 Penalty) = {alpha:6.2f} -> Train Acc: {train_acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%")

if __name__ == '__main__':
    demonstrate_overfitting_underfitting()
    demonstrate_stratified_kfold()
    demonstrate_regularization()
