# Tiền xử lý ClinVar Conflicting

Dùng Python 3.13.3 để tái lập môi trường đã kiểm chứng. Giải nén và chạy các lệnh từ thư mục `clinvar`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe preprocess.py --input source/clinvar_conflicting.csv --output data_rerun
```

`source/clinvar_conflicting.csv` là CSV gốc. `data/` chứa kết quả đã chạy sẵn. Tham số `--output` cho phép ghi kết quả chạy lại sang thư mục khác.

| Tệp trong `data/` | Nội dung |
|---|---|
| `X_{train,test}.csv` | Tám đặc trưng đã làm sạch, trước điền thiếu, mã hóa và chuẩn hóa |
| `y_{train,test}.csv` | `CLASS` theo `source_row`. Giá trị 0 là nhất quán, 1 là mâu thuẫn |
| `ids_{train,test}.csv` | Khóa biến thể và bốn trường cấu thành khóa |
| `scaled_{train,test}.npz` | Ma trận thưa CSR có chuẩn hóa cột số cho Logistic Regression và K-means |
| `unscaled_{train,test}.npz` | Ma trận thưa CSR giữ thang đo cột số cho Random Forest |
| `source_rows_{train,test}.npy` | Mã truy vết theo đúng thứ tự hàng của cả hai ma trận |
| `preprocessor_{scaled,unscaled}.joblib` | Pipeline đã fit trên train. Dùng với các phiên bản trong `requirements.txt` |
| `feature_names_{scaled,unscaled}.json` | Tên cột theo vị trí trong ma trận |
| `raw_column_profile.csv`, `excluded_rows.csv`, `report.json` | Hồ sơ cột gốc, các dòng bị loại và kết quả kiểm tra |

`source_row` bắt đầu từ 1 và đếm bản ghi dữ liệu trong CSV sau tiêu đề. Khi đọc `X_*.csv` trực tiếp, dùng `index_col="source_row"` để mã truy vết nằm ngoài tập đặc trưng. Hàm `load_split` thực hiện bước này và kiểm tra đồng bộ X, y và IDs.

## Nạp dữ liệu cho từng mô hình

```python
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from preprocess import load_matrix, load_split

data = "data"
_, y_train, ids_train = load_split(data, "train")
_, y_test, ids_test = load_split(data, "test")

X_train_scaled, rows_train = load_matrix(data, "train", scaled=True)
X_test_scaled, rows_test = load_matrix(data, "test", scaled=True)
assert (rows_train == y_train.index.to_numpy()).all()
assert (rows_test == y_test.index.to_numpy()).all()

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_predictions = lr.predict(X_test_scaled)

X_train_rf, _ = load_matrix(data, "train", scaled=False)
X_test_rf, _ = load_matrix(data, "test", scaled=False)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_rf, y_train)
rf_predictions = rf.predict(X_test_rf)

# k=3 là ví dụ sử dụng. Chọn k bằng dữ liệu train.
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X_train_scaled)
clusters_test = kmeans.predict(X_test_scaled)
```

K-means chỉ nhận ma trận X. Sau khi fit, có thể ghép cụm với `CLASS` qua `source_row` để mô tả thành phần từng cụm. Mã cụm do thuật toán gán.

Để biến đổi dữ liệu cùng cấu trúc tám cột đã làm sạch, nạp pipeline bằng `joblib.load` và gọi `.transform(X)`. Hàm `clean` trong mã nguồn mô tả các quy tắc xử lý đầu vào gốc.

## Cross-validation theo biến thể

Với Logistic Regression và Random Forest, dùng X trước biến đổi và bọc bộ tiền xử lý chưa fit cùng mô hình. Mỗi fold tính lại trung vị, danh mục và hệ số chuẩn hóa trên phần huấn luyện của fold.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from preprocess import load_split, make_preprocessor

X, y, ids = load_split("data", "train")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
models = [
    (True, LogisticRegression(max_iter=1000, random_state=42)),
    (False, RandomForestClassifier(random_state=42)),
]
for scaled, model in models:
    pipe = Pipeline([("prep", make_preprocessor(scaled)), ("model", model)])
    scores = cross_val_score(
        pipe, X, y, groups=ids["variant_id"], cv=cv,
        scoring="average_precision", error_score="raise",
    )
    print(type(model).__name__, scores)
```

Nếu đánh giá K-means theo fold, dùng `GroupKFold` với `variant_id` và `Pipeline([("prep", make_preprocessor()), ("model", KMeans(...))])`. Gọi `fit` chỉ với X của phần huấn luyện mỗi fold. Dành tập test cho lần đánh giá sau khi chọn cấu hình.

Xem [báo cáo quyết định và xác minh](REPORT.md) để biết kết quả khảo sát và kiểm tra.
