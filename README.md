# 🏠 Dự đoán giá nhà Ames Housing

Giá nhà phụ thuộc vào nhiều yếu tố như diện tích, chất lượng xây dựng, vị trí, số phòng, năm xây dựng và các tiện ích đi kèm.

Dự án này xây dựng một hệ thống Machine Learning nhằm dự đoán giá bán nhà trên bộ dữ liệu **Ames Housing**. Hệ thống thực hiện các bước từ phân tích dữ liệu, xử lý dữ liệu, xây dựng mô hình đến triển khai ứng dụng Web bằng Streamlit.

## 📌 Mục tiêu dự án

- Phân tích các yếu tố ảnh hưởng đến giá nhà:
  - Chất lượng tổng thể
  - Diện tích sử dụng
  - Diện tích tầng hầm
  - Số phòng tắm
  - Số chỗ để xe
  - Năm xây dựng
  - Khu vực sinh sống

- Xây dựng mô hình hồi quy để dự đoán giá bán nhà.

- So sánh hiệu quả giữa các mô hình Machine Learning.

- Đánh giá mô hình bằng các chỉ số:
  - MAE
  - RMSE
  - R²

- Xây dựng ứng dụng Web bằng Streamlit để người dùng nhập thông tin căn nhà và nhận giá dự đoán.

## 📂 Dataset

- Nguồn dữ liệu: Ames Housing Dataset
- File dữ liệu: `data/train.csv`
- Số lượng bản ghi: khoảng `1,460`
- Số lượng đặc trưng ban đầu: khoảng `80`
- Biến mục tiêu: `SalePrice`
- Loại bài toán: Regression
- Đơn vị giá: USD

## 🔎 Một số đặc trưng quan trọng

- `OverallQual` – Chất lượng tổng thể của căn nhà
- `GrLivArea` – Diện tích sử dụng trên mặt đất
- `TotalBsmtSF` – Tổng diện tích tầng hầm
- `GarageCars` – Số ô tô garage có thể chứa
- `GarageArea` – Diện tích garage
- `FullBath` – Số phòng tắm đầy đủ
- `BedroomAbvGr` – Số phòng ngủ
- `YearBuilt` – Năm xây dựng
- `YearRemodAdd` – Năm sửa chữa gần nhất
- `Neighborhood` – Khu vực sinh sống
- `1stFlrSF` – Diện tích tầng một
- `2ndFlrSF` – Diện tích tầng hai
- `Fireplaces` – Số lượng lò sưởi
- `SalePrice` – Giá bán của căn nhà

## 🛠️ Feature Engineering

Dự án tạo thêm một số đặc trưng mới:

- `HouseAge` – Tuổi của căn nhà tại thời điểm bán
- `RemodAge` – Số năm kể từ lần sửa chữa gần nhất
- `TotalSF` – Tổng diện tích tầng hầm, tầng một và tầng hai
- `TotalBath` – Tổng số phòng tắm
- `TotalPorchSF` – Tổng diện tích hiên nhà
- `HasGarage` – Căn nhà có garage hay không
- `HasBsmt` – Căn nhà có tầng hầm hay không
- `HasFireplace` – Căn nhà có lò sưởi hay không

## 🔄 Pipeline thực hiện

### 1. Khám phá dữ liệu

- Kiểm tra kích thước dữ liệu
- Kiểm tra kiểu dữ liệu
- Phân tích giá trị thiếu
- Phân tích phân phối của `SalePrice`
- Kiểm tra tương quan giữa các đặc trưng
- Phát hiện một số giá trị ngoại lệ

### 2. Tiền xử lý dữ liệu

- Xử lý giá trị thiếu
- Điền median cho dữ liệu số
- Điền giá trị xuất hiện nhiều nhất cho dữ liệu phân loại
- Chuẩn hóa dữ liệu số bằng `RobustScaler`
- Mã hóa biến phân loại bằng `OneHotEncoder`
- Đưa toàn bộ quá trình tiền xử lý vào `Pipeline`

### 3. Chia dữ liệu

- Chia dữ liệu theo tỷ lệ:
  - Train: 80%
  - Test: 20%
- Sử dụng `random_state=42`
- Chỉ xử lý ngoại lệ trên tập train
- Giữ nguyên tập test để đánh giá cuối cùng

### 4. Huấn luyện mô hình

Các mô hình được sử dụng:

- Linear Regression
- Ridge Regression
- Lasso Regression

Mô hình được đánh giá bằng 5-fold cross-validation.

### 5. Tuning mô hình

- Sử dụng `GridSearchCV`
- Tuning tham số `alpha` cho Ridge và Lasso
- Chọn mô hình dựa trên CV RMSE
- Không lựa chọn mô hình dựa trực tiếp trên tập test

### 6. Đánh giá mô hình

Các chỉ số đánh giá:

- MAE – Sai số tuyệt đối trung bình
- RMSE – Căn bậc hai của sai số bình phương trung bình
- R² – Mức độ giải thích biến thiên của giá nhà

### 7. Triển khai

- Lưu model bằng Joblib
- Tích hợp model vào ứng dụng Streamlit
- Cho phép người dùng nhập thông tin căn nhà
- Hiển thị giá nhà dự đoán theo đơn vị USD

## 🤖 Các mô hình sử dụng

- Linear Regression
- Ridge Regression
- Lasso Regression

Mô hình có kết quả cross-validation tốt nhất là **Lasso Regression**.

## 📊 Kết quả mô hình

### Kết quả cross-validation

| Model | CV RMSE | CV MAE | CV R² |
|---|---:|---:|---:|
| Lasso Regression | **19,629.33 USD** | **13,516.27 USD** | **0.9351** |
| Ridge Regression | 19,734.29 USD | 13,559.35 USD | 0.9343 |
| Linear Regression | 23,042.10 USD | 15,054.95 USD | 0.9106 |

### Kết quả trên tập test

| Chỉ số | Kết quả |
|---|---:|
| MAE | 14,105.60 USD |
| RMSE | 21,111.96 USD |
| R² | 0.9419 |

Kết quả cho thấy mô hình có khả năng dự đoán khá tốt trên tập dữ liệu Ames Housing.

## ⚙️ Cài đặt và chạy dự án

### Yêu cầu

- Python 3.8 trở lên
- Git
- pip

### Clone repository


git clone https://github.com/HieuGit05/ames-house-price-prediction.git
cd ames-house-price-prediction
Tạo môi trường ảo
Windows
python -m venv .venv
.venv\Scripts\activate
macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
Cài đặt thư viện
pip install -r requirements.txt
📓 Chạy notebook

Mở Jupyter Notebook:

jupyter notebook

Sau đó mở file:

notebooks/houseprice_rewritten.ipynb

Chọn:

Cell → Run All

để chạy toàn bộ quy trình phân tích và huấn luyện mô hình.

🌐 Chạy ứng dụng Web

Chạy Streamlit:

streamlit run app/app.py

Truy cập địa chỉ:

http://localhost:8501

Người dùng nhập các thông tin của căn nhà và hệ thống sẽ trả về giá dự đoán.

📁 Cấu trúc thư mục
ames-house-price-prediction/
├── app/
│   └── app.py
├── data/
│   └── train.csv
├── models/
│   ├── house_price_model.pkl
│   └── metrics.json
├── notebooks/
│   └── houseprice_rewritten.ipynb
├── .gitignore
├── README.md
└── requirements.txt

Trong đó:

app/: chứa source code ứng dụng Streamlit
data/: chứa dữ liệu gốc
models/: chứa model đã huấn luyện và file metrics
notebooks/: chứa notebook phân tích và xây dựng mô hình
.gitignore: khai báo các file không đưa lên GitHub
requirements.txt: danh sách thư viện cần cài đặt
README.md: mô tả dự án
🧰 Công nghệ sử dụng
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Joblib
Streamlit
Jupyter Notebook
⚠️ Hạn chế
Bộ dữ liệu chỉ phản ánh thị trường nhà ở tại Ames, Iowa.
Mô hình chưa thể áp dụng trực tiếp cho thị trường bất động sản Việt Nam.
Dự án mới thử nghiệm các mô hình hồi quy tuyến tính.
Ứng dụng Streamlit còn đơn giản.
Chưa có automated testing.
Chưa có CI/CD.
Chưa triển khai model trên server thực tế.
🚀 Hướng phát triển
Thử nghiệm Random Forest
Thử nghiệm Gradient Boosting
Thử nghiệm XGBoost hoặc LightGBM
Sử dụng SHAP để giải thích kết quả dự đoán
Thêm biểu đồ feature importance
Viết unit test cho phần xử lý dữ liệu
Deploy ứng dụng lên Streamlit Community Cloud
Xây dựng API bằng FastAPI
👨‍💻 Tác giả
Đặng Trung Hiếu
