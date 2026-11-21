# Đánh giá đặc trưng bổ sung cho ClinVar

Phần [đối chiếu bổ sung với Vy và Thảo](vy_thao/DOI_CHIEU.md) kiểm tra cách biểu diễn AF, cờ có CADD và thiếu có cấu trúc trên train sau khi bộ dưới đây đã khóa.

Bộ đã khóa tăng average precision (AP) cho `CLASS=1` ở cả hai mô hình trên test 13.038 biến thể. Logistic Regression tăng từ **0,3367 lên 0,3508** và Random Forest tăng từ **0,4652 lên 0,4919**. Trên missense, AP của Random Forest tăng từ **0,4564 lên 0,4879**. Khi validation chỉ gồm gene chưa gặp, thay đổi AP toàn tập còn **+0,0007** với Logistic Regression và **−0,0040** với Random Forest. Mức cải thiện trên test hiện tại do đó áp dụng cho cách chia theo biến thể của bộ dữ liệu này. [Chỉ số test](holdout_threshold_metrics.csv) · [Kiểm tra theo gene](gene_fold_metrics.csv)

## Đối chiếu Week 1

| Nguồn | Vị trí và đề xuất liên quan |
|---|---|
| Vy | `Seminar_ML_Tuan1_BaoCao.pdf` tr. 5, 10–12 và `01_EDA_ClinVar.ipynb` cell 12–14. Giữ INTRON dù thiếu nhiều, gộp với EXON, giữ SIFT/PolyPhen/BLOSUM62, kiểm tra điểm BLOSUM62 bằng 0 và thử tần suất SYMBOL. [Tài liệu của Vy](https://drive.google.com/drive/folders/18-5Rzq4cUFUtx5-BSPq_ccMuLKHsdjNo) |
| Sơn | `main.pdf` §5.3–5.4, §6 và Bảng 10 tr. 25–28. Tách vị trí, mã hóa chú giải kèm trạng thái thiếu, khôi phục BLOSUM62 bằng 0 khi có căn cứ, chia theo gene. Đề xuất bỏ Protein_position vì nghi trùng CDS_position và giữ một thang CADD. [Tài liệu của Sơn](https://drive.google.com/drive/folders/1T5GGkzTAMNXZzxXAzhSQZfCwRPwkpaAt) |
| Thảo | `Week1_Report.pdf` tr. 9–11 và `week1_clinvar.ipynb` cell 28–30, 51–52. Đo thiếu có cấu trúc theo vùng và missense. Notebook bỏ `Feature` và `CLNDISDB` trong 46 cột hiện có. [Tài liệu của Thảo](https://drive.google.com/drive/folders/1n7EP1ezxe2TwxNkc_zDJB-BJiK3sJMz6) |
| Trọng Toàn | `Seminar_ML_Tuan1.docx`, phần đặc trưng Ensembl VEP và thống kê missing. Nêu điểm in-silico và EXON/INTRON là tín hiệu nên xem xét, mô tả việc tách điểm nếu chuỗi có điểm trong ngoặc. CSV này chỉ có nhãn phân loại SIFT/PolyPhen. Tệp `missing_value.txt` tải từ thư mục có 0 byte. [Báo cáo của Toàn](https://docs.google.com/document/d/1PHGeuKr1Oq-S9HaplBi-qZQEdbGFDmQz/edit) |
| Hồng Dương | `Seminar_ML_Tuan1.docx` §2, §3, §4.2 và `missing_value.txt`. Giữ cờ EXON/INTRON và chỉ tách cú pháp hợp lệ, thử trung vị kèm cờ cho các điểm, cân nhắc gộp mức hiếm của SYMBOL, bỏ motif khỏi baseline. [Báo cáo của Dương](https://docs.google.com/document/d/1pTrZxV6XAj4tqvbz8KBz00JKRjOq8bkx/edit) |

Các tài liệu xác nhận nhiều đề xuất trùng nhau, nhất là giữ thông tin EXON/INTRON và xử lý độ phủ của chú giải protein. Chưa tìm thấy danh sách quyết định chung đã được cả nhóm duyệt. Bảng 46 cột ghi các đề xuất theo từng nguồn, bao gồm các điểm bất đồng về `BIOTYPE`, `CLNVI` và cách dùng `SYMBOL`. [Bảng quyết định 46 cột](quyet_dinh_46_cot.md)

## Khảo sát trên 52.150 dòng train

| Chú giải | Có dữ liệu | Trong 25.892 biến thể missense |
|---|---:|---:|
| LoFtool | 48.758 (93,5%) | 24.368 (94,1%) |
| SIFT | 19.847 (38,1%) | 19.773 (76,4%) |
| PolyPhen | 19.806 (38,0%) | 19.729 (76,2%) |
| BLOSUM62 | 20.511 (39,3%) | 20.423 (78,9%) |
| INTRON | 7.066 (13,5%) | 0 |

- Spearman giữa ba AF là 0,786–0,835 trên 52.150 cặp mỗi ô. Hai thang CADD có ρ = 0,999985 trên 51.293 cặp. Ba AF được giữ cố định cho phép so sánh các nhóm ứng viên. ORIGIN, STRAND và SSR là mã được lưu dưới dạng số. [Heatmap Spearman](spearman.png) · [Số cặp từng ô](spearman_pair_counts.csv)
- Heatmap độ phủ dùng từng hệ quả bên trong `Consequence`. Biến thể có nhiều hệ quả xuất hiện ở nhiều hàng. Bảng nguồn có đủ 46 trường, tách theo hệ quả và CLASS. [Heatmap độ phủ](coverage.png) · [Độ phủ từng cột](coverage_by_consequence_and_class.csv) · [Mức phân loại theo nhóm](categorical_levels.csv)
- Đối chiếu cặp axit amin đơn trên missense với ma trận NCBI khôi phục được 5.442 ô BLOSUM62 bằng 0 trên train. Cờ `has_BLOSUM62` đo việc có giá trị trong CSV gốc. `DISTANCE` chỉ có 12–19 dòng mỗi fold và mỗi cột motif có tối đa một quan sát trong một fold. [Kiểm tra cú pháp](chan_doan_du_lieu.md) · [Độ phủ cột thưa theo fold](sparse_coverage_by_fold.csv) · [Ma trận BLOSUM62](https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62)

## So sánh trên cùng 5 fold

Dùng `StratifiedGroupKFold` với seed 42 và khóa CHROM/POS/REF/ALT. Toàn bộ trung vị, chuẩn hóa và từ điển mã hóa được fit trong phần huấn luyện của từng fold. Thử cả hai baseline ở đầu vòng so sánh, sau đó giữ cố định one-hot cho mọi cấu hình ứng viên. Logistic Regression dùng L2 mặc định và số đầu vào được chuẩn hóa. Random Forest dùng 80 cây, độ sâu tối đa 16, `min_samples_leaf=5` và 6 feature ứng viên mỗi nút. Hai mô hình dùng cấu hình cố định qua các phép so sánh. [Mã dòng và fold](fold_assignments.csv) · [Kết quả từng fold, kể cả recall và số mẫu dương](cv_fold_metrics.csv)

Mỗi ô ghi AP trung bình. Số trong ngoặc là chênh lệch với baseline one-hot.

| Cấu hình | Logistic Regression toàn tập | Random Forest toàn tập |
|---|---:|---:|
| Baseline one-hot | 0,3292 | 0,4602 |
| Baseline multi-hot | 0,3298 (+0,0006) | 0,4635 (+0,0033) |
| LoFtool, cờ | 0,3331 (+0,0039) | 0,4623 (+0,0021) |
| LoFtool, cờ và điểm | 0,3326 (+0,0034) | 0,4820 (+0,0217) |
| SIFT/PolyPhen/BLOSUM62, cờ | 0,3308 (+0,0016) | 0,4611 (+0,0009) |
| SIFT/PolyPhen/BLOSUM62, cờ và giá trị | 0,3268 (−0,0024) | 0,4603 (+0,0000) |
| SIFT/PolyPhen/BLOSUM62, cờ và giá trị, khôi phục BLOSUM62 | 0,3269 (−0,0023) | 0,4614 (+0,0012) |
| Cả bốn chú giải, cờ | 0,3347 (+0,0055) | 0,4634 (+0,0032) |
| Cả bốn chú giải, cờ và giá trị | 0,3327 (+0,0035) | 0,4772 (+0,0170) |
| Độ dài alen | 0,3293 (+0,0001) | 0,4577 (−0,0026) |
| Vùng và vị trí transcript | 0,3377 (+0,0085) | 0,4693 (+0,0091) |
| Thay CADD_PHRED bằng CADD_RAW | 0,3306 (+0,0014) | 0,4603 (+0,0001) |
| MC | 0,3312 (+0,0020) | 0,4573 (−0,0030) |
| Kết hợp, giữ MC | 0,3428 (+0,0136) | 0,4755 (+0,0153) |
| **Kết hợp đã khóa, bỏ MC** | **0,3421 (+0,0129)** | **0,4847 (+0,0245)** |

Trên missense, thêm điểm LoFtool sau cờ tăng AP Random Forest **0,0213** ở 5/5 fold. Thêm ba cờ protein vào baseline tăng AP Logistic Regression **0,0113** ở 5/5 fold. Thêm giá trị SIFT/PolyPhen/BLOSUM62 sau ba cờ giảm AP missense của Logistic Regression **0,0021** và Random Forest **0,0031**. Khi khôi phục BLOSUM62, chênh lệch so với chỉ dùng cờ còn **−0,0017** và **−0,0003**. [Bảng tổng hợp theo nhóm](cv_summary.csv) · [So sánh cờ, giá trị và bỏ từng nhóm](cv_ablation.csv)

Chênh lệch AP của bộ đã khóa với baseline one-hot trên toàn fold, cùng recall tại ngưỡng 0,5.

| Fold | CLASS=1 | ΔAP Logistic Regression | ΔAP Random Forest | Recall Logistic Regression | Recall Random Forest |
|---|---:|---:|---:|---:|---:|
| 1 | 2.623 | +0,0120 | +0,0113 | 0,15% | 2,59% |
| 2 | 2.593 | +0,0084 | +0,0220 | 0,31% | 2,28% |
| 3 | 2.591 | +0,0195 | +0,0379 | 0,04% | 1,81% |
| 4 | 2.641 | +0,0112 | +0,0270 | 0,19% | 3,48% |
| 5 | 2.699 | +0,0137 | +0,0241 | 0,19% | 1,44% |

### Kiểm tra theo gene

Trong 5 fold theo biến thể, tần suất SYMBOL tăng AP toàn tập **0,0101** với Logistic Regression và **0,0353** với Random Forest. Tần suất được học trong từng fold. Trong 5 fold giữ gene cùng nhóm, tần suất gene chưa gặp bằng 0 và ΔAP còn **−0,0002** và **−0,0094**. Tập validation của từng fold có 418–480 gene và 1.963–3.162 mẫu dương. Với bộ đã khóa, ΔAP missense khi chia theo gene là **+0,0135** với Logistic Regression và **+0,0001** với Random Forest. Nhánh SYMBOL nằm ngoài bộ xuất. [Số gene và mẫu dương từng fold](gene_fold_coverage.csv) · [Chỉ số từng fold theo gene](gene_fold_metrics.csv)

## Bộ đặc trưng đã khóa

Áp dụng quy tắc chọn chung cho hai mô hình. Ít nhất một mô hình phải tăng AP toàn tập từ 0,002, tăng ở ít nhất 4/5 fold và có mức tăng lớn hơn sai số chuẩn mô tả của chênh lệch fold. Cả hai mô hình phải có ΔAP từ −0,002 trở lên. Trong số các cấu hình đạt điều kiện, lấy mức tăng AP trung bình qua hai mô hình cao nhất. Ngưỡng vào nhánh kết hợp có thêm điều kiện missense tăng từ 0,005 với ít nhất 50 mẫu dương mỗi fold. Đây là tiêu chí lựa chọn trên train. [Cấu hình và ngưỡng đã khóa](locked.json)

| Quyết định | Lý do |
|---|---|
| Giữ ba AF, CLNVC, Consequence, IMPACT và BIOTYPE | Giữ các biến đối chiếu của baseline. Consequence dùng one-hot theo tổ hợp. |
| Thay CADD_PHRED bằng CADD_RAW | Cấu hình kết hợp dùng một thang duy nhất. Thử độc lập cho lợi ích nhỏ trên toàn tập và ΔAP missense +0,0083 với Logistic Regression. |
| Giữ cờ và điểm LoFtool | Random Forest hưởng lợi từ giá trị điểm, nhất là trên missense. |
| Giữ cờ SIFT, PolyPhen và BLOSUM62 | Cờ cải thiện AP missense của Logistic Regression khi thử riêng. Đóng góp riêng trong cấu hình kết hợp nhỏ và phụ thuộc mô hình. Ba giá trị điểm chưa cho lợi ích ổn định sau cờ. |
| Giữ nhóm transcript | Cả hai mô hình tăng AP khi thử riêng. Dùng chỉ số, tổng số và tỷ lệ EXON/INTRON, đầu và độ dài khoảng CDS/protein, cùng vùng suy ra từ EXON/INTRON. |
| Loại MC và các biến cấu trúc alen đã thử | MC có giao hệ quả với Consequence ở 51.060/51.472 dòng có cả hai và làm giảm AP Random Forest ở cả 5 fold trong phép bỏ từng nhóm. Nhóm độ dài alen chưa có lợi ích nhất quán. |
| Loại khóa, nhãn, hồ sơ lâm sàng, SYMBOL và trường kỹ thuật rất thưa | Khóa dùng để nhóm và truy vết. Hồ sơ lâm sàng có nguy cơ phụ thuộc quá trình tạo nhãn. Tần suất SYMBOL giảm hiệu quả khi kiểm tra gene mới. |

Bản xuất có **24 cột đầu vào đã chuẩn bị**, tạo **83 cột sau mã hóa** cho cả hai mô hình. Danh sách cột và thứ tự đi kèm CSV, ma trận CSR, mã dòng, nhãn, bộ tiền xử lý và mô hình đã fit. [Đầu vào](pipeline/input_columns.json) · [Tên cột sau mã hóa](pipeline/feature_names.json)

## Đánh giá test sau khi khóa

Một vòng đánh giá trên 13.038 biến thể, gồm 3.287 mẫu dương. Trong đó có 6.516 missense và 1.700 mẫu dương. Precision, recall và F1 bên dưới dùng ngưỡng tối đa hóa F1 trên dự đoán ngoài fold của train, khóa riêng cho mỗi cấu hình và mô hình trước khi đọc nhãn test. AP không phụ thuộc ngưỡng. [Kết quả ở ngưỡng đã học](holdout_threshold_metrics.csv) · [Kết quả ở ngưỡng 0,5](holdout_metrics.csv)

| Nhóm | Mô hình | AP baseline → đã khóa | Precision baseline → đã khóa | Recall baseline → đã khóa | F1 baseline → đã khóa |
|---|---|---:|---:|---:|---:|
| Toàn test | Logistic Regression | 0,3367 → 0,3508 | 30,40% → 30,22% | 92,64% → 95,10% | 0,4578 → 0,4586 |
| Toàn test | Random Forest | 0,4652 → 0,4919 | 43,13% → 44,85% | 66,99% → 66,81% | 0,5248 → 0,5367 |
| Missense | Logistic Regression | 0,2478 → 0,2763 | 27,67% → 27,80% | 92,53% → 95,82% | 0,4261 → 0,4310 |
| Missense | Random Forest | 0,4564 → 0,4879 | 44,71% → 46,05% | 58,94% → 59,94% | 0,5085 → 0,5208 |
