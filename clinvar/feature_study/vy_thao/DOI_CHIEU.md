# Đối chiếu xử lý ClinVar với Vy và Thảo

**Kết luận.** Bộ 24 đầu vào đã xử lý được các góp ý chính về vùng transcript và độ phủ của chú giải protein. Thay đổi có bằng chứng rõ nhất cho vòng train tiếp theo là biểu diễn riêng số 0 của mỗi nguồn tần suất alen. Với Logistic Regression, kết hợp ba cờ này và biến đổi log tăng average precision (AP) cho lớp mâu thuẫn ở cả 5 fold theo biến thể và cả 5 fold theo gene.

## Kiểm tra góp ý trên train

| Nội dung nguồn | Kết quả đối chiếu và quyết định |
|---|---|
| Vy tr. 11–12, Thảo §3.4a về INTRON | Bộ hiện tại dùng bốn trạng thái EXON/INTRON. Train có 44.988 dòng chỉ exon, 7.035 chỉ intron, 31 có cả hai và 96 thiếu cả hai. Trong 96 dòng cuối, 6 dòng vẫn có đủ cả bốn trường coding. Giữ tên vùng `unresolved` khi thiếu cả hai. |
| Thảo §3.4b–c về thiếu theo vùng | Train có 25.892 missense. Tỷ lệ thiếu SIFT, PolyPhen, BLOSUM62 trong nhóm này lần lượt là 23,63%, 23,80%, 21,12%. Ở 26.258 dòng còn lại, các tỷ lệ tương ứng là 99,72%, 99,71%, 99,66%. Giữ các giá trị thực ở những dòng non-missense có chú giải. Hai bảng độ phủ bên dưới cho phép kiểm tra lại theo đúng tập train. |
| Vy tr. 11–12 về BLOSUM62 | Trong 5.469 ô trống trên missense của train, nhánh nghiên cứu hiện tại khôi phục được 5.442 điểm 0 từ cặp axit amin. 27 ô còn lại thiếu căn cứ để gán 0 bằng quy tắc này. Giữ cờ hiện diện gốc trong bộ đã khóa. |
| Vy tr. 11 về CADD thiếu | Train có 857 dòng thiếu CADD_RAW, với 11,90% mang CLASS=1, so với 25,43% ở nhóm có điểm. Thử thêm cờ vào bộ đã khóa cho ΔAP +0,0001 với Logistic Regression và −0,0019 với Random Forest. Chênh lệch tỷ lệ lớp tự nó chưa đủ để chọn cờ này. |
| Vy tr. 12 và Thảo §3.3c về AF | Tỷ lệ bằng 0 trên train lần lượt là 54,63% (ESP), 36,69% (ExAC), 58,02% (TGP). Cờ ghi đúng sự kiện `AF == 0`. Dữ liệu nguồn đã gộp trường hợp thiếu vào số 0. Tên cột thực trong CSV là `AF_EXAC`, còn danh sách nhóm trong notebook của Thảo ở cell 28 ghi `AF_EXA`. |

[PDF của Vy](https://drive.google.com/file/d/13Q5P43aJcsYW8344dQCX7fhGwaAVz0j7/view) · [Notebook của Vy](https://drive.google.com/file/d/1nZopPo8BR4W1DIb6a7THnxnYoIoA1RDS/view) · [PDF của Thảo](https://drive.google.com/file/d/1Ib2sleQpkijvG7tqSxXAqgF9oe_G-EH0/view) · [Notebook của Thảo](https://drive.google.com/file/d/15dA729CxRRbKMIAS7pqTurA1Q0lPlqxt/view)

[Độ phủ protein](protein_coverage.csv) · [Độ phủ bốn trường coding theo vùng](coding_coverage.csv) · [Thống kê AF](af_coverage.csv)

## Thử nghiệm trên cùng 52.150 dòng train

Mốc so sánh là bộ đã khóa `combined_without_mc`. Giữ cùng mô hình, seed, cách mã hóa Consequence và mã dòng mỗi fold. Trong nhánh log, thay mỗi AF bằng `log10(1 + 1.000.000 × AF)` để số 0 hữu hạn và các AF dương nhỏ dễ phân biệt. Hệ số một triệu là lựa chọn cho lần thử này. Cờ số 0 tạo riêng cho từng nguồn. Trung vị và chuẩn hóa học trong mỗi fold. Các số dưới đây là AP trung bình trên 5 fold.

| Biểu diễn | Logistic Regression | Random Forest |
|---|---:|---:|
| Bộ 24 đầu vào đã khóa | 0,3421 | 0,4847 |
| AF gốc + ba cờ bằng 0 | 0,4149 | 0,4860 |
| Log AF | 0,3213 | 0,4847 |
| Log AF + ba cờ bằng 0 | **0,4207** | 0,4859 |
| Bộ đã khóa + cờ có CADD | 0,3423 | 0,4828 |

Khi chia theo gene, thử log AF + cờ 0 trên cùng năm fold của lần kiểm tra gene trước. AP của Logistic Regression tăng từ 0,3275 lên 0,4145 và của Random Forest tăng từ 0,4523 lên 0,4594. Cả hai tăng ở 5/5 fold. Riêng trên missense, ΔAP của Random Forest là −0,0029 khi chia theo biến thể và +0,0004 khi chia theo gene. Vì vậy, ưu tiên cấu hình mới cho Logistic Regression, tiếp tục dùng Random Forest hiện tại làm mốc nếu mục tiêu đặt nặng nhóm missense.

Các phép so sánh này dùng lại train để nghiên cứu feature sau khi kết quả test cũ đã được biết. Các chỉ số của cấu hình mới hiện được đo bằng cross-validation. Trước khi công bố chỉ số cuối, cần khóa cấu hình, học lại ngưỡng từ dự đoán ngoài fold và đánh giá trên dữ liệu độc lập phù hợp với mục tiêu triển khai.

[Kết quả theo fold biến thể](cv_folds.csv) · [Tổng hợp theo nhóm](cv_summary.csv) · [Kết quả theo fold gene](gene_folds.csv)

Tái lập từ thư mục `clinvar` bằng `python evaluate_af_and_cadd.py` và `python evaluate_af_and_cadd.py --gene`. Đã đối chiếu ma trận train tạo lại từ CSV và hai preprocessor hiện hành với ma trận đã xuất. Cả hai khớp từng phần tử, kích thước 52.150 × 83.
