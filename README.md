# Wine Quality - Machine Learning Final Project

Project này được nâng cấp theo yêu cầu trong `ML.pdf` của môn Nhập môn Học máy.

## 1. Các nội dung đã bổ sung

- Mô tả và EDA cho Red Wine / White Wine.
- Tiền xử lý: kiểm tra missing value, mã hóa 3 lớp, StandardScaler.
- Chia dữ liệu **70% train / 15% validation / 15% test**.
- Baseline: Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree.
- **LSTM**: 3 cấu hình.
- **CNN 1D**: 3 cấu hình.
- Early Stopping + ReduceLROnPlateau.
- Accuracy, Precision, Recall, F1-score.
- Confusion Matrix.
- Training/validation loss và accuracy curves.
- Phân tích overfitting/underfitting.
- Lưu model `.keras`, history và kết quả CSV khi chạy thực nghiệm.
- Streamlit dashboard để trình diễn.

## 2. Cài đặt

Khuyến nghị Python 3.11 hoặc 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu PowerShell chặn activate, có thể chạy trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Chạy Streamlit

```powershell
streamlit run app.py
```

## 4. Chạy toàn bộ thực nghiệm

Red Wine:

```powershell
python train_models.py --wine red
```

White Wine:

```powershell
python train_models.py --wine white
```

Kết quả được lưu trong:

```text
outputs/
  red/
  white/
```

Mỗi thư mục gồm:

- `baseline_results.csv`
- `neural_results.csv`
- `all_results.csv`
- `models/*.keras`
- `figures/*_loss.png`
- `figures/*_accuracy.png`
- `results/*_history.csv`

## 5. Cách biểu diễn cho LSTM/CNN

Wine Quality là dữ liệu dạng bảng, không phải chuỗi tự nhiên. Theo yêu cầu môn học, 11 thuộc tính được biểu diễn thành sequence dài 11, mỗi bước có 1 giá trị đặc trưng:

```text
[feature_1, feature_2, ..., feature_11]
        -> reshape -> (11, 1)
        -> LSTM / CNN 1D
```

Trong báo cáo cần nêu rõ đây là cách chuyển đổi để thực nghiệm; thứ tự thuộc tính không mang ý nghĩa chuỗi tự nhiên nên đây là một hạn chế của cách áp dụng.

## 6. Quy trình thực nghiệm

```text
Raw data
   -> Clean/check
   -> Quality -> Low/Medium/High
   -> Train/Validation/Test
   -> StandardScaler (fit train only)
   -> Baseline / LSTM / CNN
   -> Evaluation
   -> Accuracy / Precision / Recall / F1
   -> Confusion Matrix
   -> Training curves
   -> Overfitting / Underfitting analysis
```

## 7. Lưu ý cho báo cáo

Không lấy `test` để điều chỉnh hyperparameter. Dùng validation để chọn cấu hình; test chỉ dùng đánh giá cuối cùng. Báo cáo cần so sánh model, tốc độ huấn luyện, ảnh hưởng hyperparameter và các lỗi dự đoán.
