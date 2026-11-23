# Huấn luyện ClinVar Random Forest

Từ thư mục gốc repo, cài thư viện và chạy toàn bộ quy trình:

```bash
python -m pip install -r src/requirements.txt
python src/train_model.py
```

Script dùng `clinvar/source/clinvar_conflicting.csv`, chia train/test phân tầng 80/20 với seed 42, dò siêu tham số trên 5 fold của train, đánh giá test một lần và tạo tài liệu trong `document/`. Các phép điền thiếu và one-hot nằm trong pipeline để được fit riêng bên trong mỗi fold. Không dùng các trường định danh/truy vết làm đầu vào.
