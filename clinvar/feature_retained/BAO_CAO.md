# Tiền xử lý và đánh giá đặc trưng ClinVar

Mục tiêu là dự đoán CLASS=1, tức diễn giải mâu thuẫn giữa các đánh giá ClinVar. Average precision (AP) đo khả năng xếp các biến thể lớp 1 lên đầu theo xác suất dự đoán.

| Phạm vi | Biến thể | CLASS=1 |
| --- | --- | --- |
| Toàn bộ | 65.188 | 16.434 |
| Train | 52.150 | 13.147 |
| Test lịch sử | 13.038 | 3.287 |

Số dòng đếm theo source_row duy nhất. Cột CLASS=1 đếm nhãn bằng 1 trong từng tập. [Nhãn train](y_train.csv) · [Nhãn test](y_test.csv)

### Hai cấu hình đã chọn

| Mô hình | Thay đổi so với mốc 24 đầu vào | Đầu vào → sau mã hóa | AP mốc → cuối |
| --- | --- | --- | --- |
| Logistic Regression | Log AF, ba cờ AF bằng 0, multi-hot hệ quả có nhóm hiếm | 27 → 56 | 0,3421 → 0,4208 |
| Random Forest | Giữ AF gốc, multi-hot hệ quả có nhóm hiếm | 24 → 53 | 0,4847 → 0,4957 |

AP cho CLASS=1 là trung bình không trọng số trên cùng 5 fold theo biến thể từ 52.150 dòng train. Bộ mốc locked_24 có 24 đầu vào và 83 cột sau mã hóa. [Tổng hợp CV](cv_summary.csv) · [Cấu hình đã xuất](selected.json)

Trong Logistic Regression, lợi ích chính đến từ log AF kết hợp ba cờ bằng 0. Khi thêm multi-hot có nhóm hiếm sau bước AF, AP đổi +0,00013 và số cột sau mã hóa giảm từ 86 xuống 56. Random Forest giữ AF gốc theo kết quả CV. [Kết quả từng cấu hình](cv_summary.csv)

<!-- pagebreak -->

## Danh sách đầu vào cuối

| Nhóm | Tên cột đầu vào | Mô hình |
| --- | --- | --- |
| Điểm chú giải | CADD_RAW, LoFtool | Cả hai |
| Vị trí exon và intron | EXON_index, EXON_total, EXON_fraction, INTRON_index, INTRON_total, INTRON_fraction, transcript_region | Cả hai |
| Vị trí CDS và protein | CDS_position_start, CDS_position_span, Protein_position_start, Protein_position_span | Cả hai |
| Loại biến thể và hệ quả | CLNVC, IMPACT, BIOTYPE, Consequence | Cả hai |
| Có chú giải trong CSV | has_LoFtool, has_SIFT, has_PolyPhen, has_BLOSUM62 | Cả hai |
| AF biến đổi | log_AF_ESP, log_AF_EXAC, log_AF_TGP, zero_AF_ESP, zero_AF_EXAC, zero_AF_TGP | Logistic Regression |
| AF gốc | AF_ESP, AF_EXAC, AF_TGP | Random Forest |

Danh sách có 21 cột chung. Sáu dẫn xuất AF của Logistic Regression cho 27 đầu vào, mã hóa thành 56 cột. Ba AF gốc của Random Forest cho 24 đầu vào, mã hóa thành 53 cột. Số cột là độ dài schema đã fit trên toàn tập train. [Đầu vào Logistic Regression](LogisticRegression/input_columns.json) · [Đầu vào Random Forest](RandomForest/input_columns.json) · [56 cột sau mã hóa](LogisticRegression/feature_names.json) · [53 cột sau mã hóa](RandomForest/feature_names.json)

Giữ tỷ lệ và nhóm vùng của EXON/INTRON vì hai trường bổ sung nhau. Giữ điểm và cờ LoFtool. Các giá trị SIFT, PolyPhen, BLOSUM62 và tọa độ cDNA được bảo toàn trong bảng sạch. Đầu vào cuối của ba chú giải protein dùng cờ có giá trị trong CSV gốc.

<!-- pagebreak -->

## Tương quan cờ thiếu và tương quan điểm

![Tương quan cờ thiếu trên train](hinh_1_co_thieu.png)
**Hình 1. Các cột nào thường thiếu cùng nhau?** Tương quan Pearson giữa cờ thiếu 0/1 trên 52.150 dòng train. EXON/INTRON = −0,990, SIFT/PolyPhen = 0,991 và hai cờ thiếu CADD = 1,000. Bốn nhóm vùng có 44.988 chỉ exon, 7.035 chỉ intron, 31 có cả hai và 96 chưa xác định. Hai trường bổ sung nhau nên giữ cả hai. Hình của Sơn dùng 16 cột trên toàn bộ 65.188 dòng. [Ma trận cờ thiếu train](missing_indicator_corr_train.csv) · [Hình 12 của Sơn, mục 5.2](https://drive.google.com/drive/folders/1T5GGkzTAMNXZzxXAzhSQZfCwRPwkpaAt)

![Spearman và số cặp trên train](hinh_2_spearman.png)
**Hình 2. Hai thang điểm nào có thứ hạng gần trùng?** Mỗi ô ghi Spearman của giá trị điểm và số cặp cùng quan sát. CADD_PHRED/CADD_RAW có ρ ≈ 0,999985 trên 51.293 cặp từ train. Các ô hiển thị hệ số làm tròn. Ô thiếu ở một trong hai trường nằm ngoài phép tính. [Giá trị Spearman](../feature_study/spearman.csv) · [Số cặp](../feature_study/spearman_pair_counts.csv)

<!-- pagebreak -->

## Chú giải protein và cách dùng CADD

![Độ phủ chú giải protein trên train](hinh_3_do_phu.png)
**Hình 3. Chú giải protein có mặt ở nhóm nào?** Tử số là số dòng có chú giải gốc, mẫu số là số dòng của nhóm train. Missense được xác định khi Consequence có từ missense_variant. Điểm BLOSUM62 khôi phục nằm ngoài tử số. [Số đếm độ phủ](audit_train.csv) · [Độ phủ theo hệ quả](../feature_study/coverage.png)

Trên train, khôi phục 5.442 điểm BLOSUM62 bằng 0 trong 5.469 ô thiếu missense khi cặp axit amin hợp lệ có điểm ma trận bằng 0. Còn 27 ô thiếu. Bảng sạch giữ điểm âm, chú giải ngoài missense, PolyPhen unknown và mức tin cậy SIFT. [Kiểm tra protein](audit_train.csv)

### CADD theo từng giai đoạn

| Giai đoạn | Thang điểm và xử lý thiếu |
| --- | --- |
| Bộ cơ sở lịch sử, 8 đầu vào | CADD_PHRED. Trung vị 14,06 học trên các giá trị PHRED hợp lệ trong toàn bộ 52.150 dòng train ở bản xuất lịch sử. |
| Bảng sạch trước nội suy | Giữ CADD_PHRED và CADD_RAW trước điền thiếu. |
| Bộ mốc mở rộng và hai cấu hình cuối | CADD_RAW. Bảo toàn cả điểm RAW âm hữu hạn. |
| Cross-validation | Học trung vị cột số từ phần huấn luyện của từng fold. Preprocessor xuất học trên toàn bộ train. |

Trung vị lịch sử 14,06 dùng 51.293 giá trị PHRED hợp lệ của train. Hình 1 đo sự đồng thời thiếu của hai thang. Hình 2 đo tương quan thứ hạng của các giá trị cùng có điểm. [Thống kê bộ cơ sở](../data/report.json) · [Schema mốc](../feature_study/pipeline/input_columns.json) · [Quy tắc điền thiếu](../preprocess.py)

<!-- pagebreak -->

## Tọa độ thiếu và độ phủ của cDNA

| Giá trị gốc | start | end | span | status |
| --- | --- | --- | --- | --- |
| 73 | 73 | 73 | 1 | single |
| 73-75 | 73 | 75 | 3 | range |
| 1476-? | 1476 | NaN | NaN | partial |
| ?-73 | NaN | 73 | NaN | partial |
| ? | NaN | NaN | NaN | unknown |
| Ô trống | NaN | NaN | NaN | missing |
| 75-73 | NaN | NaN | NaN | invalid |

Ví dụ tính bằng cùng quy tắc cho CDS, protein và cDNA. Tọa độ đơn có span = 1. Khoảng đầy đủ có span = end − start + 1. NaN trong bộ nhớ được ghi thành ô trống trong CSV. [Hàm tách tọa độ](../retained_features.py)

**1476-? qua ba bước.** Bảng sạch lưu start=1476, end=NaN, span=NaN và status=partial. Khi chuẩn bị CDS/protein, cả hai bộ cuối đặt start của dòng partial về NaN. span vẫn là NaN, còn end nằm ngoài schema cuối. Khi biến đổi, start và span thiếu nhận trung vị của cột tương ứng từ phần fit. [Hàm chuẩn bị X](../study_retained.py) · [Bộ điền thiếu](../preprocess.py)

cDNA được giữ trong bảng sạch và thử trong nhánh tọa độ với start, end, span, status và cờ có cDNA khi thiếu CDS. Cả hai bộ cuối dùng start và span của CDS/protein.

### Đếm dòng có cDNA và thiếu CDS trên train

| Nhóm hệ quả | Tổng dòng trong nhóm | Có cDNA và thiếu CDS |
| --- | --- | --- |
| Toàn bộ train | 52.150 | 860 |
| 5′ UTR | 519 | 518 |
| 3′ UTR | 345 | 342 |

Đếm source_row duy nhất trong 52.150 dòng train. Tử số lọc cDNA_position có giá trị và CDS_position thiếu theo quy ước ký hiệu thiếu. Mẫu số 519 và 345 là số dòng train chứa hệ quả 5′ UTR và 3′ UTR tương ứng. Hai nhóm giao 0 dòng trên train và 0 dòng trong tử số. Có 0 dòng thỏa điều kiện bên ngoài hai nhóm. [Bảng sạch train](clean_train.csv) · [Số đếm kiểm tra](audit_train.csv)

<!-- pagebreak -->

## Bằng chứng chọn và loại thử nghiệm

So từng mô hình với chính mô hình đó dùng locked_24. Quy tắc yêu cầu đủ 5 fold hợp lệ ở cả bốn tổ hợp biến thể/gene × toàn tập/missense. ΔAP toàn tập theo biến thể phải đạt ít nhất +0,002 và tăng ở ít nhất 4/5 fold. Không tổ hợp nào được có ΔAP dưới −0,002. Các điều kiện được tính trên số chưa làm tròn. Trong các cấu hình đạt, chọn ΔAP toàn tập theo biến thể cao nhất. [Quy tắc chọn](../study_retained.py)

| Mô hình và thử nghiệm | AP mốc → thử | ΔAP | Fold tăng | Căn cứ |
| --- | --- | --- | --- | --- |
| Logistic Regression · Tọa độ bổ sung | 0,3421 → 0,3429 | +0,00077 | 4/5 | Dưới ngưỡng +0,002 |
| Logistic Regression · CHROM one-hot | 0,3421 → 0,3523 | +0,01015 | 5/5 | ΔAP toàn tập theo gene −0,01070 |
| Random Forest · Hệ quả nhóm hiếm | 0,4847 → 0,4957 | +0,01106 | 5/5 | Đủ 5 fold ở cả bốn tổ hợp, đạt mọi ngưỡng |

AP và ΔAP trong bảng đo trên toàn tập, chia theo biến thể. ΔAP là trung bình chênh lệch AP thử trừ mốc theo từng fold. Cột fold tăng đếm fold có ΔAP > 0. [Tổng hợp thử nghiệm](cv_summary.csv) · [Kết quả ghép cặp](cv_folds.csv)

Nhánh tọa độ tăng ở 4/5 fold cho Logistic Regression, nhưng mức tăng trung bình +0,00077 thấp hơn +0,002. CHROM đạt mức tăng và số fold theo biến thể, nhưng mức giảm −0,01070 trên toàn tập theo gene vượt giới hạn. Random Forest với hệ quả nhóm hiếm đạt +0,01106 trên toàn tập theo biến thể, +0,00954 trên missense theo biến thể, +0,01027 trên toàn tập theo gene và +0,00368 trên missense theo gene. [ΔAP theo bốn tổ hợp](cv_summary.csv)

| Nhóm CV trên train | Logistic Regression | Random Forest |
| --- | --- | --- |
| Missense theo biến thể | 0,2803 → 0,4052 | 0,4804 → 0,4899 |
| Toàn tập theo gene | 0,3275 → 0,4147 | 0,4523 → 0,4626 |
| Missense theo gene | 0,2680 → 0,3948 | 0,4467 → 0,4503 |

AP mốc → cuối, trung bình không trọng số của 5 fold trong từng nhóm. Mỗi dòng thuộc đúng một fold trong mỗi cách chia. [Phân bổ fold](fold_assignments.csv) · [AP theo nhóm](cv_summary.csv)

<!-- pagebreak -->

## Phụ lục thử nghiệm Logistic Regression

| Thay đổi | AP mốc → thử | ΔAP | Fold tăng | ΔAP thấp nhất | Quyết định |
| --- | --- | --- | --- | --- | --- |
| AF gốc + cờ 0 | 0,3421 → 0,4149 | +0,07274 | 5/5 | +0,07274 | Đạt, xếp sau |
| Log AF | 0,3421 → 0,3213 | −0,02081 | 0/5 | −0,02081 | Chưa đủ phép đo |
| Log AF + cờ 0 | 0,3421 → 0,4207 | +0,07856 | 5/5 | +0,07856 | Đạt, xếp sau |
| Cờ có CADD_RAW | 0,3421 → 0,3423 | +0,00014 | 3/5 | −0,00005 | Chưa đủ phép đo |
| Giá trị protein | 0,3421 → 0,3419 | −0,00026 | 2/5 | −0,00129 | Δ dưới ngưỡng, dưới 4 fold tăng |
| Tọa độ bổ sung | 0,3421 → 0,3429 | +0,00077 | 4/5 | +0,00048 | Δ dưới ngưỡng |
| Hệ quả nhóm hiếm | 0,3421 → 0,3421 | −0,00005 | 2/5 | −0,00015 | Δ dưới ngưỡng, dưới 4 fold tăng |
| IMPACT thứ bậc | 0,3421 → 0,3423 | +0,00016 | 3/5 | −0,00029 | Δ dưới ngưỡng, dưới 4 fold tăng |
| CHROM one-hot | 0,3421 → 0,3523 | +0,01015 | 5/5 | −0,01070 | giảm quá giới hạn |
| Log AF + cờ 0 + hệ quả nhóm hiếm | 0,3421 → 0,4208 | +0,07869 | 5/5 | +0,07869 | Đã chọn |

AP và số fold tăng tính trên toàn tập theo biến thể. ΔAP thấp nhất lấy trong các tổ hợp có kết quả. Các dòng “Chưa đủ phép đo” thiếu ít nhất một tổ hợp 5 fold và không được xếp hạng. Các điều kiện bị vi phạm cho biết lý do không đạt. “Đạt, xếp sau” là cấu hình qua ngưỡng nhưng có AP toàn tập theo biến thể thấp hơn bộ cuối. Tra từng fold bằng config, model, scheme, subgroup và fold. [Tổng hợp](cv_summary.csv) · [Chỉ số theo fold](cv_folds.csv)

### Giải nghĩa nhánh thử

**Tọa độ bổ sung.** Thêm CDS_position_end, CDS_position_status, Protein_position_end, Protein_position_status, cDNA_position_start, cDNA_position_end, cDNA_position_span, cDNA_position_status, has_cDNA_without_CDS. Giữ mút start đã biết của khoảng partial trong nhánh này.

**Giá trị protein.** Thêm SIFT_call, SIFT_low_confidence, PolyPhen, BLOSUM62_score, BLOSUM62_source vào bộ mốc đã có cờ hiện diện.

**Các nhánh khác.** Log AF thay ba AF gốc. Cờ AF và cờ có CADD_RAW thêm các chỉ báo tương ứng. IMPACT thứ bậc thay one-hot bằng mã 0–3. CHROM thêm one-hot. Hệ quả nhóm hiếm thay one-hot cả chuỗi bằng multi-hot theo từng từ. [Định nghĩa các nhánh](../study_retained.py)

<!-- pagebreak -->

## Phụ lục thử nghiệm Random Forest

| Thay đổi | AP mốc → thử | ΔAP | Fold tăng | ΔAP thấp nhất | Quyết định |
| --- | --- | --- | --- | --- | --- |
| AF gốc + cờ 0 | 0,4847 → 0,4860 | +0,00129 | 4/5 | −0,00293 | Δ dưới ngưỡng, giảm quá giới hạn |
| Log AF | 0,4847 → 0,4847 | −0,00001 | 2/5 | −0,00012 | Chưa đủ phép đo |
| Log AF + cờ 0 | 0,4847 → 0,4859 | +0,00123 | 4/5 | −0,00288 | Δ dưới ngưỡng, giảm quá giới hạn |
| Cờ có CADD_RAW | 0,4847 → 0,4828 | −0,00189 | 2/5 | −0,00189 | Chưa đủ phép đo |
| Giá trị protein | 0,4847 → 0,4801 | −0,00453 | 0/5 | −0,00511 | Δ dưới ngưỡng, dưới 4 fold tăng, giảm quá giới hạn |
| Tọa độ bổ sung | 0,4847 → 0,4741 | −0,01058 | 0/5 | −0,01127 | Δ dưới ngưỡng, dưới 4 fold tăng, giảm quá giới hạn |
| Hệ quả nhóm hiếm | 0,4847 → 0,4957 | +0,01106 | 5/5 | +0,00368 | Đã chọn |
| IMPACT thứ bậc | 0,4847 → 0,4814 | −0,00332 | 0/5 | −0,00332 | Δ dưới ngưỡng, dưới 4 fold tăng, giảm quá giới hạn |
| CHROM one-hot | 0,4847 → 0,4836 | −0,00105 | 3/5 | −0,01212 | Δ dưới ngưỡng, dưới 4 fold tăng, giảm quá giới hạn |
| Log AF + cờ 0 + hệ quả nhóm hiếm | 0,4847 → 0,4946 | +0,00990 | 5/5 | +0,00650 | Đạt, xếp sau |

AP và số fold tăng tính trên toàn tập theo biến thể. ΔAP thấp nhất lấy trong các tổ hợp có kết quả. Các dòng “Chưa đủ phép đo” thiếu ít nhất một tổ hợp 5 fold và không được xếp hạng. Các điều kiện bị vi phạm cho biết lý do không đạt. “Đạt, xếp sau” là cấu hình qua ngưỡng nhưng có AP toàn tập theo biến thể thấp hơn bộ cuối. Tra từng fold bằng config, model, scheme, subgroup và fold. [Tổng hợp](cv_summary.csv) · [Chỉ số theo fold](cv_folds.csv)

Hệ quả có dưới 50 dòng fit hoặc chỉ gặp lúc validation bật nhóm hiếm. Hệ quả trống có cờ riêng. Trung vị, chuẩn hóa và từ vựng được fit lại ở từng fold. Các mô tả cột thử ở phụ lục Logistic Regression áp dụng cho cả hai mô hình. [Bộ mã hóa hệ quả](../consequence_encoder.py)

<!-- pagebreak -->

## Phụ lục kiểm chứng và test lịch sử

| Mô hình | AP cơ sở 8 trường | AP bộ 24 đã khóa |
| --- | --- | --- |
| Logistic Regression | 0,3367 | 0,3508 |
| Random Forest | 0,4652 | 0,4919 |

AP CLASS=1 đo một lần trên test lịch sử 13.038 dòng, gồm 3.287 mẫu dương. Bộ cơ sở one-hot dùng 8 trường gốc. Bộ mở rộng lịch sử dùng 24 đầu vào, 83 cột sau mã hóa. Test của lần cập nhật mới phục vụ kiểm tra phép biến đổi. [Chỉ số test lịch sử](../feature_study/holdout_threshold_metrics.csv)

### Nghiệm thu bảng sạch

Trên 52.150 dòng train, đếm theo hai trường phân số hợp lệ cho 44.988 chỉ exon, 7.035 chỉ intron, 31 có cả hai và 96 chưa xác định. Nhóm chưa xác định gồm dòng thiếu cả hai trường hợp lệ. [Số đếm vùng và lỗi cú pháp](audit_train.csv)

Cả 22 kiểm thử đều đạt trong lần bàn giao. Bốn ma trận khớp từng phần tử khi đọc lại CSV và nạp lại preprocessor. Mã dòng và nhãn căn đúng thứ tự, khóa biến thể giữa train và test tách biệt, tên cột duy nhất và dữ liệu ma trận hữu hạn. [Kiểm tra đầu ra](../study_retained.py) · [Từ điển cột](column_dictionary.csv)

ORIGIN được lưu để kiểm tra, với các bit quan sát khớp header GRCh37 tham chiếu. Phiên bản VCF gốc và thời điểm ORIGIN có sẵn khi dự đoán cần xác minh. CLNDN được giữ trong bảng sạch với trạng thái chỉ placeholder, có cả placeholder và tên bệnh, hoặc có tên bệnh. SYMBOL dùng để chia fold theo gene. [Đối chiếu mã ORIGIN](origin_train.csv) · [Phụ lục quyết định đủ 46 cột](QUYET_DINH_46_COT.md)

Tái lập dữ liệu và CV bằng `python study_retained.py`. Tái sinh báo cáo từ các kết quả đã lưu bằng `python retained_publication.py`. Kiểm thử bằng `python -m unittest -q`. [Hướng dẫn workspace](../README.md)

<!-- pagebreak -->

## Phụ lục đối chiếu 46 trường gốc

| Trường gốc | Vai trò | Trường gốc | Vai trò |
| --- | --- | --- | --- |
| CHROM | Ứng viên | Feature_type | Truy vết / kiểm tra |
| POS | Truy vết / kiểm tra | Feature | Truy vết / kiểm tra |
| REF | Truy vết / kiểm tra | BIOTYPE | Ứng viên |
| ALT | Truy vết / kiểm tra | EXON | Giữ theo đề xuất |
| AF_ESP | Giữ theo đề xuất | INTRON | Giữ theo đề xuất |
| AF_EXAC | Giữ theo đề xuất | cDNA_position | Ứng viên |
| AF_TGP | Giữ theo đề xuất | CDS_position | Ứng viên |
| CLNDISDB | Truy vết / kiểm tra | Protein_position | Giữ theo đề xuất |
| CLNDISDBINCL | Truy vết / kiểm tra | Amino_acids | Giữ theo đề xuất |
| CLNDN | Truy vết / kiểm tra | Codons | Truy vết / kiểm tra |
| CLNDNINCL | Truy vết / kiểm tra | DISTANCE | Truy vết / kiểm tra |
| CLNHGVS | Truy vết / kiểm tra | STRAND | Truy vết / kiểm tra |
| CLNSIGINCL | Truy vết / kiểm tra | BAM_EDIT | Truy vết / kiểm tra |
| CLNVC | Giữ theo đề xuất | SIFT | Giữ theo đề xuất |
| CLNVI | Truy vết / kiểm tra | PolyPhen | Giữ theo đề xuất |
| MC | Truy vết / kiểm tra | MOTIF_NAME | Truy vết / kiểm tra |
| ORIGIN | Ứng viên | MOTIF_POS | Truy vết / kiểm tra |
| SSR | Truy vết / kiểm tra | HIGH_INF_POS | Truy vết / kiểm tra |
| CLASS | Truy vết / kiểm tra | MOTIF_SCORE_CHANGE | Truy vết / kiểm tra |
| Allele | Truy vết / kiểm tra | LoFtool | Giữ theo đề xuất |
| Consequence | Giữ theo đề xuất | CADD_PHRED | Giữ theo đề xuất |
| IMPACT | Giữ theo đề xuất | CADD_RAW | Ứng viên |
| SYMBOL | Truy vết / kiểm tra | BLOSUM62 | Giữ theo đề xuất |

Các vai trò phản ánh quyết định lưu trong bảng sạch và phạm vi thử. Danh sách ở trang 2 ghi chính xác những cột đi vào mô hình cuối. [Nguồn đề xuất, cách xử lý và trạng thái của từng trường](QUYET_DINH_46_COT.md)
