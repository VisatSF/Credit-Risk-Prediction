"""
🤖 train.py — Huấn luyện mô hình rủi ro tín dụng
===================================================
Pipeline:
    1. Tải dữ liệu đã tiền xử lý từ data/processed/
    2. Huấn luyện Logistic Regression
    3. Huấn luyện CatBoost
    4. Huấn luyện Random Forest
    5. Lưu tất cả mô hình → models/

Sử dụng:
    python src/train.py

Yêu cầu:
    Chạy preprocessing.py trước
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

# ============================================================
# 📁 Đường dẫn
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def load_processed_data() -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, list[str]]:
    """
    📥 Tải dữ liệu đã tiền xử lý
    """
    print("=" * 70)
    print("📥 TẢI DỮ LIỆU ĐÃ TIỀN XỬ LÝ")
    print("=" * 70)

    required_files = [
        "X_train.pkl", "X_test.pkl",
        "y_train.pkl", "y_test.pkl",
        "feature_cols.pkl",
    ]

    for f in required_files:
        path = os.path.join(PROCESSED_DIR, f)
        if not os.path.exists(path):
            print(f"   ❌ Không tìm thấy: {path}")
            print(f"   💡 Hãy chạy preprocessing.py trước!")
            sys.exit(1)

    X_train = joblib.load(os.path.join(PROCESSED_DIR, "X_train.pkl"))
    X_test = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(PROCESSED_DIR, "y_train.pkl"))
    y_test = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    feature_cols = joblib.load(os.path.join(PROCESSED_DIR, "feature_cols.pkl"))

    print(f"   ✅ X_train: {X_train.shape}")
    print(f"   ✅ X_test:  {X_test.shape}")
    print(f"   ✅ Features: {len(feature_cols)} cột")
    print()

    return X_train, X_test, y_train, y_test, feature_cols


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    📈 Huấn luyện Logistic Regression

    - class_weight='balanced': xử lý mất cân bằng lớp
    - max_iter=1000: đảm bảo hội tụ
    """
    print("=" * 70)
    print("📈 HUẤN LUYỆN: Logistic Regression")
    print("=" * 70)

    start = time.time()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"   ✅ Hoàn tất trong {elapsed:.2f}s")
    print(f"   🔹 class_weight: balanced")
    print(f"   🔹 max_iter: 1000")
    print(f"   🔹 Số iterations thực tế: {model.n_iter_[0]}")
    print()

    return model


def train_catboost(
    X_train: np.ndarray,
    y_train: pd.Series,
) -> CatBoostClassifier:
    """
    🐱 Huấn luyện CatBoost

    - iterations=300, learning_rate=0.01
    - Loss: Logloss, Metric: AUC
    """
    print("=" * 70)
    print("🐱 HUẤN LUYỆN: CatBoost")
    print("=" * 70)

    start = time.time()

    model = CatBoostClassifier(
        iterations=300,
        loss_function="Logloss",
        learning_rate=0.01,
        custom_metric=["AUC"],
        verbose=0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"   ✅ Hoàn tất trong {elapsed:.2f}s")
    print(f"   🔹 iterations: 300")
    print(f"   🔹 learning_rate: 0.01")
    print(f"   🔹 loss_function: Logloss")
    print()

    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    🌲 Huấn luyện Random Forest

    - n_estimators=300
    - n_jobs=-1: sử dụng tất cả CPU cores
    """
    print("=" * 70)
    print("🌲 HUẤN LUYỆN: Random Forest")
    print("=" * 70)

    start = time.time()

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"   ✅ Hoàn tất trong {elapsed:.2f}s")
    print(f"   🔹 n_estimators: 300")
    print(f"   🔹 n_jobs: -1 (all cores)")
    print()

    return model


def save_models(
    lr_model: LogisticRegression,
    cb_model: CatBoostClassifier,
    rf_model: RandomForestClassifier,
) -> None:
    """
    💾 Lưu tất cả mô hình
    """
    print("=" * 70)
    print("💾 LƯU MÔ HÌNH")
    print("=" * 70)

    models = {
        "logistic_regression.pkl": lr_model,
        "catboost.pkl": cb_model,
        "random_forest.pkl": rf_model,
    }

    for filename, model in models.items():
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, path)
        size_kb = os.path.getsize(path) / 1024
        print(f"   ✅ {filename} ({size_kb:.0f} KB)")

    print(f"\n   📁 Thư mục: {MODELS_DIR}")
    print()


# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    print("\n" + "🔷" * 35)
    print("   🤖 TRAINING PIPELINE — Credit Risk")
    print("🔷" * 35 + "\n")

    total_start = time.time()

    # 1. Tải dữ liệu
    X_train, X_test, y_train, y_test, feature_cols = load_processed_data()

    # 2. Huấn luyện các mô hình
    lr_model = train_logistic_regression(X_train, y_train)
    cb_model = train_catboost(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # 3. Lưu mô hình
    save_models(lr_model, cb_model, rf_model)

    total_elapsed = time.time() - total_start

    print("=" * 70)
    print(f"🎉 TRAINING HOÀN TẤT! (Tổng thời gian: {total_elapsed:.2f}s)")
    print("=" * 70)
    print(f"   📊 Đã huấn luyện 3 mô hình:")
    print(f"      1. 📈 Logistic Regression")
    print(f"      2. 🐱 CatBoost")
    print(f"      3. 🌲 Random Forest")
    print(f"   → Tiếp theo: python src/evaluate.py")
    print()


if __name__ == "__main__":
    main()
