# Huấn luyện ClinVar Random Forest

Từ thư mục gốc repo, cài thư viện và chạy toàn bộ quy trình:

```bash
python -m pip install -r src/requirements.txt
python src/train_model.py
```

Script dùng `clinvar/source/clinvar_conflicting.csv`, chia train/test phân tầng 80/20 với seed 42, dò 12 cấu hình bằng 5-fold CV trên train, đánh giá test một lần và tạo tài liệu trong `document/`. Các phép điền thiếu và one-hot nằm trong pipeline để được fit riêng bên trong mỗi fold. Không dùng các trường định danh/truy vết làm đầu vào.

Kết quả hiện tại là hold-out ngẫu nhiên theo dòng. Nó không đảm bảo gene hoặc vị trí chỉ xuất hiện ở một phía, nên không đại diện cho đánh giá trên gene/biến thể hoàn toàn mới. AP CV tốt nhất là điểm chọn cấu hình sau khi thử nhiều cấu hình và có thể lạc quan; AP test là kết quả cuối trong phép chia này. Ngưỡng 0,5 chỉ được dùng để minh họa Precision/Recall/F1, không phải ngưỡng tối ưu theo nhu cầu ứng dụng. Impurity-based feature importance là mô tả nội bộ mô hình, không phải bằng chứng nhân quả.

Sau khi chạy, đọc `document/GIAI_THICH_CHO_NGUOI_HOC.txt` để học phương pháp và `document/BAO_CAO_XAY_DUNG_MO_HINH.docx` để xem kết quả, giới hạn và biểu đồ. Để tái lập, cần cung cấp đúng CSV nguồn và các phiên bản thư viện trong `src/requirements.txt`.
