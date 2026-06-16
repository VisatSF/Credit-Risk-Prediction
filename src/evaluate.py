"""
📊 evaluate.py — Đánh giá mô hình rủi ro tín dụng
====================================================
Pipeline:
    1. Tải mô hình & dữ liệu test
    2. Tạo Classification Report cho từng mô hình
    3. Vẽ Confusion Matrix so sánh
    4. Vẽ ROC Curve so sánh
    5. Phân tích Feature Importance
    6. Báo cáo tổng hợp hiệu suất & Business Insights
    7. Lưu tất cả biểu đồ → outputs/figures/

Sử dụng:
    python src/evaluate.py

Yêu cầu:
    Chạy preprocessing.py và train.py trước
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# 📁 Đường dẫn
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="Set2")


def load_models_and_data():
    """
    📥 Tải mô hình đã huấn luyện & dữ liệu test
    """
    print("=" * 70)
    print("📥 TẢI MÔ HÌNH & DỮ LIỆU TEST")
    print("=" * 70)

    # Kiểm tra file tồn tại
    required = {
        "data": ["X_test.pkl", "y_test.pkl", "feature_cols.pkl"],
        "models": ["logistic_regression.pkl", "catboost.pkl", "random_forest.pkl"],
    }

    for category, files in required.items():
        base = PROCESSED_DIR if category == "data" else MODELS_DIR
        for f in files:
            path = os.path.join(base, f)
            if not os.path.exists(path):
                print(f"   ❌ Không tìm thấy: {path}")
                print(f"   💡 Hãy chạy preprocessing.py và train.py trước!")
                sys.exit(1)

    X_test = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_test = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    feature_cols = joblib.load(os.path.join(PROCESSED_DIR, "feature_cols.pkl"))

    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    cb_model = joblib.load(os.path.join(MODELS_DIR, "catboost.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

    print(f"   ✅ X_test: {X_test.shape}")
    print(f"   ✅ Features: {len(feature_cols)} cột")
    print(f"   ✅ Mô hình: Logistic Regression, CatBoost, Random Forest")
    print()

    return X_test, y_test, feature_cols, lr_model, cb_model, rf_model


def print_classification_reports(X_test, y_test, lr_model, cb_model, rf_model):
    """
    📋 In Classification Report cho từng mô hình
    """
    print("=" * 70)
    print("📋 CLASSIFICATION REPORTS")
    print("=" * 70)

    models = {
        "📈 Logistic Regression": lr_model,
        "🐱 CatBoost": cb_model,
        "🌲 Random Forest": rf_model,
    }

    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        print(f"\n{'─' * 50}")
        print(f"   {name}")
        print(f"{'─' * 50}")
        print(classification_report(y_test, y_pred, target_names=["Non-default", "Default"]))

    return predictions


def plot_confusion_matrices(y_test, rf_pred, cb_pred):
    """
    🔲 Vẽ Confusion Matrix so sánh Random Forest vs CatBoost
    """
    print("=" * 70)
    print("🔲 CONFUSION MATRIX")
    print("=" * 70)

    rf_cm = confusion_matrix(y_test, rf_pred)
    cb_cm = confusion_matrix(y_test, cb_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay(confusion_matrix=rf_cm).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title("🌲 Random Forest", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    ConfusionMatrixDisplay(confusion_matrix=cb_cm).plot(
        ax=axes[1], cmap="Blues", colorbar=False
    )
    axes[1].set_title("🐱 CatBoost", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.suptitle("Confusion Matrix Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Đã lưu: {path}")
    plt.close(fig)
    print()


def plot_roc_curves(X_test, y_test, rf_model, cb_model):
    """
    📈 Vẽ ROC Curve so sánh
    """
    print("=" * 70)
    print("📈 ROC CURVE")
    print("=" * 70)

    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    cb_proba = cb_model.predict_proba(X_test)[:, 1]

    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
    cb_fpr, cb_tpr, _ = roc_curve(y_test, cb_proba)

    rf_auc = roc_auc_score(y_test, rf_proba)
    cb_auc = roc_auc_score(y_test, cb_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rf_fpr, rf_tpr, linewidth=2, label=f"🌲 Random Forest (AUC = {rf_auc:.4f})")
    ax.plot(cb_fpr, cb_tpr, linewidth=2, label=f"🐱 CatBoost (AUC = {cb_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Random Baseline")
    ax.set_title("ROC Curve Comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Đã lưu: {path}")
    plt.close(fig)
    print()


def plot_feature_importance(X_test, feature_cols, rf_model, cb_model):
    """
    🏆 Vẽ biểu đồ Feature Importance so sánh
    """
    print("=" * 70)
    print("🏆 FEATURE IMPORTANCE")
    print("=" * 70)

    rf_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    cb_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": cb_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Random Forest
    top_rf = rf_importance.head(10)
    axes[0].barh(top_rf["feature"][::-1], top_rf["importance"][::-1], color="#4C72B0")
    axes[0].set_title("🌲 Random Forest", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Importance")

    # CatBoost
    top_cb = cb_importance.head(10)
    axes[1].barh(top_cb["feature"][::-1], top_cb["importance"][::-1], color="#DD8452")
    axes[1].set_title("🐱 CatBoost", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Importance")

    plt.suptitle("Feature Importance Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Đã lưu: {path}")

    # In top 5
    print(f"\n   🌲 Random Forest — Top 5:")
    for i, row in rf_importance.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")

    print(f"\n   🐱 CatBoost — Top 5:")
    for i, row in cb_importance.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")

    plt.close(fig)
    print()

    return cb_importance


def generate_final_report(X_test, y_test, rf_model, cb_model, cb_importance):
    """
    📊 Báo cáo tổng hợp hiệu suất & Business Insights
    """
    print("=" * 70)
    print("📊 BÁO CÁO TỔNG HỢP HIỆU SUẤT")
    print("=" * 70)

    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    cb_probs = cb_model.predict_proba(X_test)[:, 1]
    rf_preds = rf_model.predict(X_test)
    cb_preds = cb_model.predict(X_test)

    rf_auc = roc_auc_score(y_test, rf_probs)
    cb_auc = roc_auc_score(y_test, cb_probs)

    # Bảng so sánh
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FINAL MODEL PERFORMANCE REPORT")
    report_lines.append("=" * 70)

    for name, probs, preds, emoji in [
        ("Random Forest", rf_probs, rf_preds, "🌲"),
        ("CatBoost", cb_probs, cb_preds, "🐱"),
    ]:
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        block = f"""
{emoji} {name}
{'─' * 70}
   ROC-AUC   : {auc:.4f}
   Accuracy  : {acc:.4f}
   Precision : {prec:.4f}
   Recall    : {rec:.4f}
   F1-Score  : {f1:.4f}"""
        print(block)
        report_lines.append(block)

    # Top Risk Factors
    top_factors_block = f"""
{'=' * 70}
🏆 TOP YẾU TỐ RỦI RO (CATBOOST)
{'=' * 70}"""
    print(top_factors_block)
    report_lines.append(top_factors_block)

    for i, feature in enumerate(cb_importance.head(5)["feature"].tolist(), 1):
        line = f"   {i}. {feature}"
        print(line)
        report_lines.append(line)

    # Business Insights
    insights_block = f"""
{'=' * 70}
💡 BUSINESS INSIGHTS
{'=' * 70}
   • Higher loan-to-income ratio increases default risk.
   • Lower income borrowers are more likely to default.
   • Poor loan grades are strongly associated with default.
   • Home ownership status contributes to risk prediction.
   • Loan purpose contains meaningful risk information."""
    print(insights_block)
    report_lines.append(insights_block)

    # Mô hình tốt nhất
    selected_model = "CatBoost" if cb_auc >= rf_auc else "Random Forest"
    selected_emoji = "🐱" if cb_auc >= rf_auc else "🌲"

    selection_block = f"""
{'=' * 70}
🏅 MÔ HÌNH ĐƯỢC CHỌN: {selected_emoji} {selected_model}
{'=' * 70}
   🌲 RF  ROC-AUC : {rf_auc:.4f}
   🐱 CB  ROC-AUC : {cb_auc:.4f}
{'=' * 70}"""
    print(selection_block)
    report_lines.append(selection_block)

    # Lưu báo cáo text
    report_path = os.path.join(REPORTS_DIR, "model_performance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n   ✅ Báo cáo đã lưu: {report_path}")
    print()


# ============================================================
# 🚀 MAIN
# ============================================================
def main():
    print("\n" + "🔶" * 35)
    print("   📊 EVALUATION PIPELINE — Credit Risk")
    print("🔶" * 35 + "\n")

    # 1. Tải dữ liệu & mô hình
    X_test, y_test, feature_cols, lr_model, cb_model, rf_model = load_models_and_data()

    # 2. Classification Reports
    predictions = print_classification_reports(X_test, y_test, lr_model, cb_model, rf_model)

    # Lấy predictions cho RF và CB
    rf_pred = predictions["🌲 Random Forest"]
    cb_pred = predictions["🐱 CatBoost"]

    # 3. Confusion Matrices
    plot_confusion_matrices(y_test, rf_pred, cb_pred)

    # 4. ROC Curves
    plot_roc_curves(X_test, y_test, rf_model, cb_model)

    # 5. Feature Importance
    cb_importance = plot_feature_importance(X_test, feature_cols, rf_model, cb_model)

    # 6. Final Report
    generate_final_report(X_test, y_test, rf_model, cb_model, cb_importance)

    print("=" * 70)
    print("🎉 EVALUATION HOÀN TẤT!")
    print("=" * 70)
    print(f"   📁 Biểu đồ:  {FIGURES_DIR}")
    print(f"   📁 Báo cáo:  {REPORTS_DIR}")
    print()


if __name__ == "__main__":
    main()
