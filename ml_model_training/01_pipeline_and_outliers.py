"""
สคริปต์ตัวอย่าง: ML Training Pipeline และการตรวจจับ/จัดการ Outliers
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

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def generate_synthetic_data_with_outliers(n_samples=200, seed=42):
    """สร้างข้อมูลจำลองพร้อมจำลองการแทรก Outliers"""
    np.random.seed(seed)
    X, y = make_regression(n_samples=n_samples, n_features=2, noise=15.0, random_state=seed)

    # แปลงเป็น DataFrame
    df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
    df['Target'] = y

    # แทรก Outliers 5 จุด (จุดที่มีค่าสูงผิดปกติ)
    outlier_indices = [10, 35, 70, 110, 150]
    df.loc[outlier_indices, 'Feature_1'] += 15.0
    df.loc[outlier_indices, 'Target'] += 400.0

    return df

def remove_outliers_iqr(df, columns):
    """
    ฟังก์ชันคัดกรองลบ Outliers โดยใช้วิธี IQR (Interquartile Range)
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    removed_rows = initial_rows - len(df_clean)
    print(f"[IQR Cleanup] Removed {removed_rows} outlier rows from {initial_rows} total rows.")
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3.0):
    """
    ฟังก์ชันคัดกรองลบ Outliers โดยใช้วิธี Z-Score (|Z| > threshold)
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    for col in columns:
        mean_val = df_clean[col].mean()
        std_val = df_clean[col].std()
        z_scores = (df_clean[col] - mean_val) / std_val
        df_clean = df_clean[np.abs(z_scores) <= threshold]

    removed_rows = initial_rows - len(df_clean)
    print(f"[Z-Score Cleanup] Removed {removed_rows} outlier rows from {initial_rows} total rows.")
    return df_clean

def run_pipeline_demo():
    print("=" * 65)
    print(" 🚀 ML TRAINING PIPELINE & OUTLIER HANDLING DEMO")
    print("=" * 65)

    # 1. Generate Dataset
    df = generate_synthetic_data_with_outliers()
    print(f"\n1. Generated Dataset Shape: {df.shape}")

    # 2. Clean Outliers
    df_cleaned = remove_outliers_iqr(df, columns=['Feature_1', 'Feature_2', 'Target'])

    # 3. Separate Features and Target
    X = df_cleaned[['Feature_1', 'Feature_2']]
    y = df_cleaned['Target']

    # 4. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n2. Train Set Size: {X_train.shape[0]}, Test Set Size: {X_test.shape[0]}")

    # 5. Build Scikit-Learn Pipeline (Scaler + Regressor)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # 6. Train Pipeline
    pipeline.fit(X_train, y_train)
    print("\n3. Pipeline Trained Successfully!")

    # 7. Evaluate Model
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n4. Evaluation Metrics on Test Set:")
    print(f"   - Mean Squared Error (MSE) : {mse:.2f}")
    print(f"   - R-squared (R2 Score)    : {r2:.4f} ({r2*100:.2f}%)")

    # 8. Save Trained Model Pipeline
    os.makedirs('ml_model_training/saved_models', exist_ok=True)
    model_filename = 'ml_model_training/saved_models/pipeline_model.joblib'
    joblib.dump(pipeline, model_filename)
    print(f"\n5. Model Pipeline Saved to: '{model_filename}'")

if __name__ == '__main__':
    run_pipeline_demo()
