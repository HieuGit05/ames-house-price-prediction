# Ames House Price Prediction

Dự án Machine Learning dự đoán giá bán nhà trên bộ dữ liệu **Ames Housing**. Dự án xây dựng một quy trình hồi quy có thể tái lập, bao gồm khám phá dữ liệu, xử lý dữ liệu thiếu, feature engineering, cross-validation, tối ưu siêu tham số và triển khai mô hình bằng Streamlit.

Dự án được thực hiện nhằm củng cố kiến thức về Machine Learning và xây dựng một sản phẩm cá nhân có thể trình bày trong CV hoặc portfolio.

---

## Kết quả mô hình

Các mô hình được đánh giá bằng **5-fold cross-validation** trên tập huấn luyện. Mô hình cuối cùng được lựa chọn dựa trên CV RMSE, không dựa trên kết quả của tập test.

### Kết quả cross-validation

| Model             |           CV RMSE |            CV MAE |      CV R² |
| ----------------- | ----------------: | ----------------: | ---------: |
| Lasso Regression  | **19,629.33 USD** | **13,516.27 USD** | **0.9351** |
| Ridge Regression  |     19,734.29 USD |     13,559.35 USD |     0.9343 |
| Linear Regression |     23,042.10 USD |     15,054.95 USD |     0.9106 |

Lasso Regression đạt CV RMSE thấp nhất và được lựa chọn làm mô hình cuối cùng.

### Kết quả trên tập test

Tập test được giữ riêng và chỉ sử dụng một lần sau khi hoàn tất quá trình lựa chọn mô hình.

| Chỉ số    |       Kết quả |
| --------- | ------------: |
| Test MAE  | 14,105.60 USD |
| Test RMSE | 21,111.96 USD |
| Test R²   |        0.9419 |

Kết quả cho thấy mô hình giải thích được khoảng **94.19% sự biến thiên của giá nhà** trên tập test.

---

## Chức năng chính

* Phân tích dữ liệu khám phá bằng biểu đồ và thống kê mô tả.
* Kiểm tra và xử lý các giá trị bị thiếu.
* Xử lý các biến phân loại dạng nominal và ordinal.
* Tạo thêm các đặc trưng liên quan đến tuổi nhà, diện tích và tiện ích.
* Xử lý ngoại lệ trên tập huấn luyện.
* Xây dựng preprocessing pipeline bằng scikit-learn.
* So sánh Linear Regression, Ridge Regression và Lasso Regression.
* Tối ưu siêu tham số bằng GridSearchCV.
* Đánh giá mô hình bằng MAE, RMSE và R² trên đơn vị USD.
* Lưu toàn bộ pipeline bằng Joblib.
* Xây dựng giao diện dự đoán giá nhà bằng Streamlit.

---

## Bộ dữ liệu

Dự án sử dụng bộ dữ liệu **Ames Housing** từ cuộc thi House Prices trên Kaggle.

Thông tin chính:

* Số lượng quan sát: `1,460`
* Số biến đầu vào ban đầu: `80`
* Biến mục tiêu: `SalePrice`
* Loại bài toán: Regression
* Đơn vị giá: USD

Bộ dữ liệu mô tả nhiều đặc điểm của căn nhà như:

* Chất lượng tổng thể.
* Diện tích sử dụng.
* Diện tích tầng hầm.
* Số phòng tắm.
* Số chỗ để xe.
* Năm xây dựng.
* Khu vực sinh sống.
* Chất lượng vật liệu và nội thất.

---

## Feature engineering

Dự án tạo thêm một số đặc trưng nhằm tổng hợp thông tin từ các biến ban đầu.

| Feature        | Ý nghĩa                                               |
| -------------- | ----------------------------------------------------- |
| `HouseAge`     | Tuổi của căn nhà tại thời điểm bán                    |
| `RemodAge`     | Số năm kể từ lần sửa chữa gần nhất                    |
| `TotalSF`      | Tổng diện tích tầng hầm, tầng một và tầng hai         |
| `TotalBath`    | Tổng số phòng tắm, có tính trọng số cho phòng tắm nhỏ |
| `TotalPorchSF` | Tổng diện tích các khu vực hiên nhà                   |
| `HasGarage`    | Căn nhà có garage hay không                           |
| `HasBsmt`      | Căn nhà có tầng hầm hay không                         |
| `HasFireplace` | Căn nhà có lò sưởi hay không                          |

Ví dụ:

```python
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

df["TotalSF"] = (
    df["TotalBsmtSF"]
    + df["1stFlrSF"]
    + df["2ndFlrSF"]
)

df["TotalBath"] = (
    df["FullBath"]
    + 0.5 * df["HalfBath"]
    + df["BsmtFullBath"]
    + 0.5 * df["BsmtHalfBath"]
)
```

---

## Quy trình xây dựng mô hình

Quy trình huấn luyện được tổ chức theo thứ tự:

1. Đọc và kiểm tra dữ liệu.
2. Phân tích dữ liệu khám phá.
3. Chia dữ liệu thành tập train và test.
4. Loại ngoại lệ chỉ trên tập train.
5. Tạo các đặc trưng mới.
6. Phân chia numerical features và categorical features.
7. Xây dựng preprocessing pipeline.
8. Đánh giá các baseline model bằng cross-validation.
9. Tối ưu Ridge và Lasso bằng GridSearchCV.
10. Chọn mô hình dựa trên CV RMSE.
11. Huấn luyện mô hình được chọn trên toàn bộ tập train.
12. Đánh giá một lần trên tập test.
13. Lưu pipeline và metadata của mô hình.
14. Sử dụng model trong ứng dụng Streamlit.

Quy trình này giúp hạn chế data leakage và đảm bảo kết quả đánh giá khách quan hơn.

---

## Tiền xử lý dữ liệu

### Numerical features

Các biến số được xử lý bằng:

* `SimpleImputer(strategy="median")`
* `RobustScaler()`

`RobustScaler` được sử dụng vì ít bị ảnh hưởng bởi các giá trị ngoại lệ hơn StandardScaler.

### Categorical features

Các biến phân loại được xử lý bằng:

* `SimpleImputer(strategy="most_frequent")`
* `OneHotEncoder(handle_unknown="ignore")`

Thiết lập `handle_unknown="ignore"` giúp pipeline vẫn dự đoán được khi dữ liệu mới xuất hiện category chưa từng có trong tập huấn luyện.

### Target transformation

Biến mục tiêu `SalePrice` có phân phối lệch phải, vì vậy dự án sử dụng phép biến đổi logarithm:

```python
y_log = np.log1p(y)
```

Khi dự đoán, kết quả được chuyển lại về đơn vị USD:

```python
y_pred = np.expm1(y_pred_log)
```

---

## Các mô hình được sử dụng

### Linear Regression

Được sử dụng làm baseline để so sánh với các mô hình có regularization.

### Ridge Regression

Ridge sử dụng L2 regularization để giảm ảnh hưởng của đa cộng tuyến và hạn chế hệ số mô hình quá lớn.

### Lasso Regression

Lasso sử dụng L1 regularization, có khả năng đưa một số hệ số về 0 và thực hiện lựa chọn đặc trưng gián tiếp.

Sau quá trình cross-validation và tuning, Lasso Regression đạt CV RMSE tốt nhất và được chọn làm mô hình cuối cùng.

---

## Cấu trúc dự án

```text
ames-house-price-prediction/
├── app/
│   └── app.py
│
├── data/
│   └── train.csv
│
├── models/
│   ├── house_price_model.pkl
│   └── metrics.json
│
├── notebooks/
│   └── house_price_analysis.ipynb
│
├── reports/
│
├── src/
│   └── train.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Mô tả thư mục

* `app/`: chứa ứng dụng Streamlit.
* `data/`: chứa dữ liệu sử dụng trong dự án.
* `models/`: chứa model đã huấn luyện và kết quả đánh giá.
* `notebooks/`: chứa notebook phân tích và xây dựng mô hình.
* `reports/`: có thể chứa báo cáo, slide hoặc hình ảnh kết quả.
* `src/`: chứa script huấn luyện có thể chạy lại.
* `requirements.txt`: danh sách thư viện cần cài đặt.

---

## Cài đặt dự án

### 1. Clone repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ames-house-price-prediction
```

### 2. Tạo môi trường ảo

```bash
python -m venv .venv
```

### 3. Kích hoạt môi trường

Windows:

```bash
.venv\Scripts\activate
```

Linux hoặc macOS:

```bash
source .venv/bin/activate
```

### 4. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## Huấn luyện lại mô hình

Để chạy toàn bộ quá trình preprocessing, cross-validation, tuning và lưu model:

```bash
python src/train.py
```

Sau khi chạy thành công, model và metrics sẽ được lưu trong thư mục:

```text
models/
├── house_price_model.pkl
└── metrics.json
```

---

## Chạy ứng dụng Streamlit

```bash
streamlit run app/app.py
```

Sau đó mở địa chỉ được Streamlit hiển thị trong terminal, thông thường là:

```text
http://localhost:8501
```

Ứng dụng cho phép người dùng nhập các thông tin cơ bản của căn nhà và nhận kết quả dự đoán giá bán.

---

## Công nghệ sử dụng

* Python
* Pandas
* NumPy
* Matplotlib
* scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

---

## Hạn chế của dự án

* Bộ dữ liệu chỉ phản ánh thị trường nhà ở tại Ames, Iowa.
* Mô hình không thể áp dụng trực tiếp cho thị trường bất động sản Việt Nam.
* Một số biến ordinal được xử lý như categorical features để đơn giản hóa pipeline.
* Các mô hình tuyến tính có thể chưa biểu diễn tốt những quan hệ phi tuyến phức tạp.
* Ứng dụng Streamlit chỉ sử dụng một nhóm đặc trưng quan trọng để người dùng dễ nhập dữ liệu.
* Chưa có automated testing.
* Chưa có CI/CD.
* Chưa triển khai theo dõi chất lượng model sau deployment.

---

## Hướng phát triển

Trong tương lai, dự án có thể được cải thiện theo các hướng:

* So sánh với Random Forest, Gradient Boosting, XGBoost và LightGBM.
* Sử dụng SHAP để giải thích ảnh hưởng của từng đặc trưng.
* Thêm biểu đồ feature importance.
* Xây dựng unit test cho feature engineering và inference.
* Thêm GitHub Actions để kiểm tra code tự động.
* Tạo Dockerfile để chuẩn hóa môi trường chạy.
* Deploy ứng dụng trên Streamlit Community Cloud.
* Xây dựng API dự đoán bằng FastAPI.
* Theo dõi model performance và data drift.

---

## Tác giả

**Đặng Trung Hiếu**

Sinh viên năm 4, định hướng AI/Machine Learning.

* Lĩnh vực quan tâm: Machine Learning, Data Science và Natural Language Processing.
* Mục tiêu: tìm kiếm cơ hội thực tập AI/Machine Learning và phát triển các dự án có khả năng ứng dụng thực tế.

---

## License

Dự án được phát hành theo giấy phép MIT License.
