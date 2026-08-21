"""
สคริปต์ตัวอย่าง: การคำนวณและการวาดกราฟตัวชี้วัดประสิทธิภาพโมเดล (Evaluation Metrics)
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
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_curve, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def evaluate_classification_metrics():
    """
    คำนวณและแสดงผล Confusion Matrix, Accuracy, Precision, Recall, F1-Score, และ ROC-AUC
    """
    print("=" * 65)
    print(" 📈 CLASSIFICATION EVALUATION METRICS DEMO")
    print("=" * 65)

    # 1. Generate Binary Classification Dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        weights=[0.7, 0.3], # Class 0 = 70%, Class 1 = 30%
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 2. Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Predict Classes & Probabilities
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1]

    # 4. Confusion Matrix Computation
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n1. Confusion Matrix Components:")
    print(f"   - True Positives  (TP): {tp:3d} (ทายว่าเป็น 1 และเป็น 1 จริง)")
    print(f"   - True Negatives  (TN): {tn:3d} (ทายว่าเป็น 0 และเป็น 0 จริง)")
    print(f"   - False Positives (FP): {fp:3d} (ทายว่าเป็น 1 แต่เป็น 0 จริง - Type I Error)")
    print(f"   - False Negatives (FN): {fn:3d} (ทายว่าเป็น 0 แต่เป็น 1 จริง - Type II Error)")

    # 5. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)

    print(f"\n2. Computed Performance Metrics:")
    print(f"   - Accuracy  : {acc*100:.2f}%")
    print(f"   - Precision : {prec*100:.2f}%")
    print(f"   - Recall    : {rec*100:.2f}%")
    print(f"   - F1-Score  : {f1*100:.2f}%")
    print(f"   - ROC-AUC   : {auc:.4f}")

    print("\n3. Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    # 6. Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix Visualization")
    plt.tight_layout()
    plt.savefig("ml_model_training/confusion_matrix.png")
    plt.close()
    print(" Saved Confusion Matrix plot to 'ml_model_training/confusion_matrix.png'")

    # 7. Save ROC Curve Plot
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (Recall / TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("ml_model_training/roc_curve.png")
    plt.close()
    print(" Saved ROC Curve plot to 'ml_model_training/roc_curve.png'")

def evaluate_regression_metrics():
    """
    คำนวณและแสดงผล MSE, RMSE, MAE, และ R2 Score สำหรับโจทย์ Regression
    """
    print("\n" + "=" * 65)
    print(" 📉 REGRESSION EVALUATION METRICS DEMO")
    print("=" * 65)

    X, y = make_regression(n_samples=500, n_features=10, noise=10.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Regression Performance Summary:")
    print(f" - Mean Squared Error (MSE)  : {mse:.2f}")
    print(f" - Root Mean Sq. Error (RMSE): {rmse:.2f}")
    print(f" - Mean Absolute Error (MAE) : {mae:.2f}")
    print(f" - R-squared Score (R2)      : {r2:.4f} ({r2*100:.2f}%)")

if __name__ == '__main__':
    evaluate_classification_metrics()
    evaluate_regression_metrics()
