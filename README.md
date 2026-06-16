# 📊 Phân Tích Rủi Ro Tín Dụng (Credit Risk Analysis)

> 🎯 **Mục tiêu:** Khám phá và phân tích bộ dữ liệu rủi ro tín dụng nhằm phát hiện các tín hiệu ban đầu của **rủi ro vỡ nợ** (default risk), từ đó xây dựng mô hình dự đoán khả năng vỡ nợ của khách hàng.

---

## 📁 Cấu Trúc Dự Án

```
Credit Risk/
├── data/
│   ├── raw/                        # Dữ liệu thô
│   │   └── credit_risk_dataset.csv # Bộ dữ liệu chính (32,581 bản ghi)
│   └── processed/                  # Dữ liệu đã xử lý
├── notebooks/
│   └── predict-credit-risk.ipynb        # Notebook EDA (Tiếng Việt)
├── models/                         # Các mô hình đã huấn luyện
├── outputs/                        # Kết quả xuất ra
├── src/                            # Mã nguồn
├── requirements.txt                # Các thư viện cần thiết
└── README.md                       # Tài liệu hướng dẫn
```

---

## 📝 Mô Tả Bộ Dữ Liệu

Bộ dữ liệu `credit_risk_dataset.csv` chứa **32,581 bản ghi** với **12 đặc trưng** mô tả thông tin cá nhân và khoản vay:

+----+-----------------------------+--------------+--------------------------------------------------------------------------+
| #  | Tên Biến                    | Kiểu Dữ Liệu | Mô Tả                                                                    |
+----+-----------------------------+--------------+--------------------------------------------------------------------------+
| 1  | person_age                  | int64        | Tuổi của người vay                                                       |
| 2  | person_income               | int64        | Thu nhập hàng năm                                                        |
| 3  | person_home_ownership       | object       | Tình trạng sở hữu nhà (RENT, OWN, MORTGAGE, OTHER)                       |
| 4  | person_emp_length           | float64      | Thời gian làm việc (năm)                                                 |
| 5  | loan_intent                 | object       | Mục đích vay (PERSONAL, EDUCATION, MEDICAL, VENTURE,                     |
|    |                             |              | DEBTCONSOLIDATION, HOMEIMPROVEMENT)                                      |
| 6  | loan_grade                  | object       | Xếp hạng khoản vay (A–G)                                                 |
| 7  | loan_amnt                   | int64        | Số tiền vay                                                              |
| 8  | loan_int_rate               | float64      | Lãi suất khoản vay                                                       |
| 9  | loan_status                 | int64        | Biến mục tiêu — Trạng thái vay (0 = Trả được, 1 = Vỡ nợ)                 |
| 10 | loan_percent_income         | float64      | Tỷ lệ khoản vay trên thu nhập                                            |
| 11 | cb_person_default_on_file   | object       | Có tiền sử vỡ nợ hay không (Y/N)                                         |
| 12 | cb_person_cred_hist_length  | int64        | Thời gian lịch sử tín dụng (năm)                                         |
+----+-----------------------------+--------------+--------------------------------------------------------------------------+

> ⚠️ **Lưu ý:** Có giá trị thiếu ở cột `person_emp_length` (895 giá trị) và `loan_int_rate` (3,116 giá trị).

---

## 🔬 Phương Pháp Phân Tích

### 1. Phân Tích Khám Phá Dữ Liệu (EDA)

Sử dụng phân tích **đơn biến** và **đa biến** kết hợp trực quan hóa dữ liệu để đánh giá tác động của từng yếu tố lên khả năng vỡ nợ:

- **Phân bố biến mục tiêu:** ~78.18% trả được nợ vs ~21.82% vỡ nợ (mất cân bằng lớp)
- **Phân tích theo nhóm:** Cross-tabulation và groupby để so sánh tỷ lệ default
- **Trực quan hóa:** Boxplot, countplot, barplot bằng Matplotlib & Seaborn

### 2. Tiền Xử Lý Dữ Liệu

- **Mã hóa biến phân loại:** Sử dụng `OrdinalEncoder` cho các biến `loan_grade`, `loan_intent`, `cb_person_default_on_file`, `person_home_ownership`
- **Xử lý giá trị thiếu:**
  - `person_emp_length`: Điền bằng giá trị median
  - `loan_int_rate`: Điền bằng median theo nhóm `loan_grade`
  - Tạo cờ nhị phân (`_missing`) để đánh dấu các giá trị bị thiếu
- **Chuẩn hóa dữ liệu:** `RobustScaler` (ít bị ảnh hưởng bởi outliers)
- **Chia tập dữ liệu:** 80/20 train/test với `stratify` để giữ cân bằng lớp

### 3. Xây Dựng Mô Hình

Ba mô hình được huấn luyện và so sánh:

+--------------------------+----------+---------------------+------------------+--------------------+
| Mô Hình                  | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
+--------------------------+----------+---------------------+------------------+--------------------+
| **Logistic Regression**  | 79%      | 0.51                | 0.78             | 0.62               |
| **CatBoost**             | 93%      | 0.99                | 0.69             | 0.81               |
| **Random Forest**        | 93%      | 0.98                | 0.71             | 0.82               |
+--------------------------+----------+---------------------+------------------+--------------------+

### 4. Phân Tích Tầm Quan Trọng Đặc Trưng (Feature Importance)

So sánh feature importance giữa **Random Forest** và **CatBoost** để xác nhận các insight từ EDA.

---

## 📋 Tổng Kết Các Phát Hiện Chính
+----+----------------------------------+--------------------------------------------------------------+------------------+
| #  | Biến                             | Phát hiện                                                    | Mức độ ảnh hưởng |
+----+----------------------------------+--------------------------------------------------------------+------------------+
| 1  | 💰 `person_income`              | Thu nhập thấp → rủi ro vỡ nợ cao hơn                         | ⭐⭐⭐          |
| 2  | 📉 `loan_percent_income`        | Tỷ lệ vay/thu nhập càng cao → rủi ro càng lớn                | ⭐⭐⭐⭐       |
| 3  | 🏷️ `loan_grade`                 | Grade thấp (D–G) → tín hiệu rủi ro rất mạnh                  | ⭐⭐⭐⭐⭐     |
| 4  | 📜 `cb_person_default_on_file`  | Có tiền sử vỡ nợ → xác suất default cao gấp đôi              | ⭐⭐⭐⭐       |
| 5  | 🎯 `loan_intent`                | Vay gộp nợ, y tế, cải tạo nhà → rủi ro cao hơn               | ⭐⭐⭐          |
| 6  | 📅 `cb_person_cred_hist_length` | Tác động đơn biến chưa rõ ràng                               | ⭐               |
| 7  | 🏠 `person_home_ownership`      | Sở hữu nhà → an toàn hơn thuê nhà                            | ⭐⭐⭐          |
+----+----------------------------------+--------------------------------------------------------------+------------------+

> 🔑 **Kết luận tổng quát:** Các biến `loan_grade`, `loan_percent_income` và `cb_person_default_on_file` là **ba yếu tố dự báo mạnh nhất** cho rủi ro vỡ nợ, nên được ưu tiên trong quá trình xây dựng mô hình.

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### Yêu Cầu Hệ Thống
- Python >= 3.9

### Cài Đặt

```bash
# Clone dự án
git clone <repository-url>
cd "Credit Risk"

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# hoặc
venv\Scripts\activate           # Windows

# Cài đặt các thư viện
pip install -r requirements.txt
```

### Chạy Notebook

```bash
jupyter notebook notebooks/predict-credit-risk.ipynb
```

---

## 🛠️ Công Nghệ Sử Dụng

+----------------------+------------------------------------------------------------------+
| Thư Viện             | Mục Đích                                                         |
+----------------------+------------------------------------------------------------------+
| **pandas**           | Xử lý và phân tích dữ liệu                                       |
| **numpy**            | Tính toán số học                                                 |
| **matplotlib**       | Trực quan hóa dữ liệu cơ bản                                     |
| **seaborn**          | Trực quan hóa thống kê nâng cao                                  |
| **scikit-learn**     | Tiền xử lý dữ liệu và xây dựng mô hình Machine Learning          |
|                      | (Logistic Regression, Random Forest)                             |
| **catboost**         | Mô hình Gradient Boosting (CatBoost)                             |
+----------------------+------------------------------------------------------------------+

---

## 📄 Giấy Phép

Dự án này được phát triển cho mục đích nghiên cứu và học tập.
