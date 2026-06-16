"""
📦 preprocessing.py — Tiền xử lý dữ liệu rủi ro tín dụng
==========================================================
Pipeline:
    1. Đọc dữ liệu thô từ data/raw/
    2. Xử lý missing values
    3. Mã hóa biến phân loại (OrdinalEncoder)
    4. Chia train/test (80/20, stratified)
    5. Chuẩn hóa đặc trưng (RobustScaler)
    6. Lưu dữ liệu đã xử lý → data/processed/
    7. Lưu encoder & scaler → models/

Sử dụng:
    python src/preprocessing.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.model_selection import train_test_split

# ============================================================
# 📁 Đường dẫn
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "credit_risk_dataset.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    📥 Bước 1: Đọc dữ liệu thô
    """
    print("=" * 70)
    print("📥 BƯỚC 1: ĐỌC DỮ LIỆU THÔ")
    print("=" * 70)

    data = pd.read_csv(path)
    print(f"   ✅ Đã đọc: {path}")
    print(f"   📊 Kích thước: {data.shape[0]:,} dòng × {data.shape[1]} cột")
    print(f"   📋 Các cột: {list(data.columns)}")
    print()
    return data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    🔧 Bước 2: Xử lý giá trị thiếu (missing values)

    - person_emp_length: tạo cờ missing → điền median
    - loan_int_rate: tạo cờ missing → điền median theo loan_grade
    """
    print("=" * 70)
    print("🔧 BƯỚC 2: XỬ LÝ GIÁ TRỊ THIẾU")
    print("=" * 70)

    # Thống kê missing trước khi xử lý
    missing_before = data.isnull().sum()
    missing_cols = missing_before[missing_before > 0]
    if len(missing_cols) > 0:
        print("   ⚠️  Missing values trước khi xử lý:")
        for col, count in missing_cols.items():
            pct = count / len(data) * 100
            print(f"      • {col}: {count:,} ({pct:.1f}%)")
    else:
        print("   ✅ Không có missing values")
        return data

    # --- person_emp_length ---
    data["person_emp_length_missing"] = data["person_emp_length"].isna().astype(int)
    emp_median = data["person_emp_length"].median()
    data["person_emp_length"] = data["person_emp_length"].fillna(emp_median)
    print(f"\n   🔹 person_emp_length:")
    print(f"      → Tạo cờ 'person_emp_length_missing'")
    print(f"      → Điền median = {emp_median}")

    # --- loan_int_rate ---
    data["int_rate_missing"] = data["loan_int_rate"].isna().astype(int)
    data["loan_int_rate"] = (
        data.groupby("loan_grade")["loan_int_rate"]
        .transform(lambda x: x.fillna(x.median()))
    )
    print(f"   🔹 loan_int_rate:")
    print(f"      → Tạo cờ 'int_rate_missing'")
    print(f"      → Điền median theo từng loan_grade")

    # Xác nhận
    missing_after = data.isnull().sum().sum()
    print(f"\n   ✅ Missing values sau xử lý: {missing_after}")
    print()
    return data


def encode_categoricals(data: pd.DataFrame) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """
    🏷️ Bước 3: Mã hóa biến phân loại (Ordinal Encoding)

    Thứ tự mã hóa phản ánh mức độ rủi ro tăng dần.
    """
    print("=" * 70)
    print("🏷️ BƯỚC 3: MÃ HÓA BIẾN PHÂN LOẠI")
    print("=" * 70)

    cols_to_encode = [
        "loan_grade",
        "loan_intent",
        "cb_person_default_on_file",
        "person_home_ownership",
    ]

    encoder = OrdinalEncoder(categories=[
        ["G", "F", "E", "D", "C", "B", "A"],                                          # loan_grade: G(rủi ro cao)=0 → A(an toàn)=6
        ["VENTURE", "EDUCATION", "PERSONAL", "HOMEIMPROVEMENT", "MEDICAL", "DEBTCONSOLIDATION"],  # loan_intent
        ["N", "Y"],                                                                     # cb_person_default_on_file
        ["OWN", "MORTGAGE", "OTHER", "RENT"],                                           # person_home_ownership
    ])

    data[cols_to_encode] = encoder.fit_transform(data[cols_to_encode])

    print(f"   ✅ Đã mã hóa {len(cols_to_encode)} cột:")
    for col in cols_to_encode:
        unique_vals = sorted(data[col].unique())
        print(f"      • {col}: {unique_vals}")
    print()

    return data, encoder


def split_and_scale(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, RobustScaler, list[str]]:
    """
    ✂️ Bước 4: Chia dữ liệu & Chuẩn hóa

    - Train/Test split: 80/20, stratified theo loan_status
    - Chuẩn hóa: RobustScaler (bền vững với outliers)
    """
    print("=" * 70)
    print("✂️  BƯỚC 4: CHIA DỮ LIỆU & CHUẨN HÓA")
    print("=" * 70)

    feature_cols = [c for c in data.columns if c != "loan_status"]
    X = data[feature_cols]
    y = data["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   📊 Features: {len(feature_cols)} cột")
    print(f"   📋 Danh sách: {feature_cols}")
    print(f"   ✂️  Train: {X_train_scaled.shape[0]:,} mẫu")
    print(f"   ✂️  Test:  {X_test_scaled.shape[0]:,} mẫu")
    print(f"   ⚖️  Tỷ lệ default (train): {y_train.mean():.2%}")
    print(f"   ⚖️  Tỷ lệ default (test):  {y_test.mean():.2%}")
    print()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def save_artifacts(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    encoder: OrdinalEncoder,
    scaler: RobustScaler,
    feature_cols: list[str],
) -> None:
    """
    💾 Bước 5: Lưu dữ liệu đã xử lý & các transformer
    """
    print("=" * 70)
    print("💾 BƯỚC 5: LƯU ARTIFACTS")
    print("=" * 70)

    # Lưu dữ liệu đã xử lý
    joblib.dump(X_train, os.path.join(PROCESSED_DIR, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(PROCESSED_DIR, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(PROCESSED_DIR, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(PROCESSED_DIR, "y_test.pkl"))
    joblib.dump(feature_cols, os.path.join(PROCESSED_DIR, "feature_cols.pkl"))
    print(f"   ✅ Dữ liệu → {PROCESSED_DIR}")

    # Lưu encoder & scaler
    joblib.dump(encoder, os.path.join(MODELS_DIR, "encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"   ✅ Encoder & Scaler → {MODELS_DIR}")
    print()


# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    print("\n" + "🔶" * 35)
    print("   📦 PREPROCESSING PIPELINE — Credit Risk")
    print("🔶" * 35 + "\n")

    # 1. Đọc dữ liệu
    data = load_raw_data()

    # 2. Xử lý missing values
    data = handle_missing_values(data)

    # 3. Mã hóa biến phân loại
    data, encoder = encode_categoricals(data)

    # 4. Chia dữ liệu & chuẩn hóa
    X_train, X_test, y_train, y_test, scaler, feature_cols = split_and_scale(data)

    # 5. Lưu artifacts
    save_artifacts(X_train, X_test, y_train, y_test, encoder, scaler, feature_cols)

    print("=" * 70)
    print("🎉 PREPROCESSING HOÀN TẤT!")
    print("=" * 70)
    print(f"   → Tiếp theo: python src/train.py")
    print()


if __name__ == "__main__":
    main()
