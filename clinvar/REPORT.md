# Báo cáo tiền xử lý ClinVar Conflicting

CSV từ GitHub của Kevin Arvai trùng từng byte với CSV trong ZIP tải từ Kaggle. Bản gốc được lưu trong `source/` với SHA-256 `53061e207d437af397028760be5bc566d216f6932e21db5a10322b843c41689a`. [Kaggle](https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting) · [CSV của tác giả](https://github.com/arvkevi/clinvar-kaggle/blob/master/clinvar_conflicting.csv)

## Khảo sát và kết quả chia tập

| Tập | Số dòng | Số cột | CLASS=0 | CLASS=1 | % CLASS=1 |
|---|---:|---:|---:|---:|---:|
| CSV gốc | 65.188 | 46 | 48.754 | 16.434 | 25,2102% |
| X đã làm sạch | 65.188 | 8 | 48.754 | 16.434 | 25,2102% |
| Train, trước biến đổi | 52.150 | 8 | 39.003 | 13.147 | 25,2100% |
| Test, trước biến đổi | 13.038 | 8 | 9.751 | 3.287 | 25,2109% |
| Train, sau biến đổi | 52.150 | 64 | 39.003 | 13.147 | 25,2100% |
| Test, sau biến đổi | 13.038 | 64 | 9.751 | 3.287 | 25,2109% |

Sau chuẩn hóa khóa `CHROM/POS/REF/ALT`, có 65.188 khóa riêng biệt, 0 nhóm lặp và 0 nhóm mang đồng thời hai nhãn. Có 0 dòng trùng toàn bộ, 0 dòng lỗi khóa hoặc nhãn và 0 dòng bị loại. `data/raw_column_profile.csv` ghi kiểu suy luận từ dữ liệu, số ô thiếu và số giá trị phân biệt của từng cột. Trong tám đặc trưng, chỉ `CADD_PHRED` (1.092 ô) và `BIOTYPE` (16 ô) có giá trị thiếu.

## Quyết định cho 46 cột

| Cột gốc | Xử lý và lý do |
|---|---|
| `CLASS` | Lưu riêng làm nhãn. Tác giả gán lớp 1 dựa trên sự hiện diện của `CLNSIGCONF`. `CLNSIG`, `CLNSIGCONF`, `CLNREVSTAT` đã bị tác giả bỏ khỏi CSV. |
| `CHROM`, `POS`, `REF`, `ALT` | Dùng để lập khóa nhóm và truy vết. Vị trí và trình tự cụ thể có thể khiến mô hình ghi nhớ danh tính biến thể. |
| `AF_ESP`, `AF_EXAC`, `AF_TGP` | Giữ dạng số. Tần số alen trong các nguồn quần thể ESP, ExAC và 1000 Genomes cung cấp thông tin về độ hiếm. |
| `CADD_PHRED` | Giữ dạng số. Điểm theo thang Phred biểu thị thứ hạng tác động do CADD dự đoán. |
| `CLNVC` | Giữ và one-hot. Đây là loại biến thể như thay thế một nucleotide, mất đoạn hoặc chèn đoạn. |
| `Consequence`, `IMPACT`, `BIOTYPE` | Giữ và one-hot. Chúng mô tả hệ quả dự đoán của VEP, mức độ tác động và loại transcript. Một tổ hợp hệ quả phân cách bằng `&` được mã hóa như một mức để giữ nguyên cách ghi của nguồn. |
| `CLNSIGINCL`, `CLNDISDB`, `CLNDISDBINCL`, `CLNDN`, `CLNDNINCL`, `ORIGIN`, `SSR` | Loại. Chúng mô tả ý nghĩa lâm sàng của alen thành phần, tên/mã bệnh, nguồn gốc alen trong hồ sơ và cờ nghi ngờ từ nguồn. `CLNSIGINCL` chứa thông tin phân loại gần với mục tiêu. Các trường còn lại có thể phản ánh cách thu thập và tổng hợp hồ sơ ClinVar. |
| `CLNHGVS`, `CLNVI`, `SYMBOL`, `Feature`, `Allele` | Loại. Biểu diễn HGVS, mã ngoài, ký hiệu gene, mã transcript/đặc trưng và chuỗi alen có thể giúp ghi nhớ địa điểm, gene, transcript hoặc danh tính bản ghi. |
| `MC` | Loại. Hệ quả phân tử theo Sequence Ontology trùng chủ đề với `Consequence` và `IMPACT`. |
| `CADD_RAW` | Loại. Giữ một thang điểm CADD để tránh biểu diễn hai lần cùng nguồn tín hiệu. |
| `LoFtool`, `BLOSUM62`, `SIFT`, `PolyPhen` | Loại ở bộ ban đầu. Các điểm độ dung nạp mất chức năng, thay thế axit amin và dự đoán ảnh hưởng protein khác nhau về phạm vi áp dụng và độ phủ. |
| `Feature_type`, `STRAND`, `BAM_EDIT`, `EXON`, `INTRON`, `cDNA_position`, `CDS_position`, `Protein_position`, `Amino_acids`, `Codons`, `DISTANCE`, `MOTIF_NAME`, `MOTIF_POS`, `HIGH_INF_POS`, `MOTIF_SCORE_CHANGE` | Loại ở bộ ban đầu. Đây là chi tiết về loại/chiều/hiệu chỉnh transcript, vị trí và thay đổi phân tử, hoặc chú giải motif. Một số trường có độ phủ rất thấp và nhiều trường cần quy tắc riêng cho phân số, khoảng hoặc cặp giá trị. |

Danh sách cho phép trong mã chỉ đưa tám cột đã chọn vào X. Việc loại trường lâm sàng dựa trên quy trình tạo nhãn và khả năng các trường mang thông tin phát sinh trong quá trình nộp hồ sơ. Phạm vi đánh giá là biến thể thuộc bộ dữ liệu đã qua lọc của tác giả. [Quy trình tạo CSV](https://github.com/arvkevi/clinvar-kaggle/blob/master/process_clinvar.py) · [Định dạng VEP](https://useast.ensembl.org/info/docs/tools/vep/vep_formats.html) · [Phương pháp CADD](https://cadd.bihealth.org/info)

## Làm sạch, chia tập và fit

- Cắt khoảng trắng. Quy về giá trị thiếu với ô rỗng và các ký hiệu như `.`, `-`, `NA`, `NULL`, `?`. Số không đọc được, vô hạn, âm, hoặc tần số vượt 1 được chuyển thành thiếu. Giữ các tần số 0 vì tác giả đã điền 0 cho tần số thiếu và CSV không còn phân biệt được hai trường hợp.
- Chuẩn hóa chữ hoa ở nhiễm sắc thể và alen, bỏ tiền tố `chr`, đổi `M` thành `MT`, bỏ số 0 đầu vị trí. Kiểm tra khóa trên các nhiễm sắc thể người chính và chuỗi alen kiểu CSV này. Dòng trùng toàn bộ giữ lần xuất hiện đầu tiên. Dòng bị loại có `source_row` và cờ lý do. Các dòng hợp lệ cùng khóa vẫn ở chung một tập, kể cả khi nhãn khác nhau. Quy tắc nhóm dựa trên bốn trường, chưa căn trái indel hoặc nhận dạng tương đương giữa các bản lắp ráp.
- Chia 80/20 theo nhãn với `random_state=42` khi mỗi khóa chỉ có một dòng. Nếu có khóa lặp, dùng `StratifiedGroupKFold` 5 fold và lấy fold test có tổng sai lệch tỷ lệ dòng và tỷ lệ lớp 1 nhỏ nhất so với mục tiêu, trong số các fold có đủ hai lớp ở cả hai tập.
- `Pipeline` và `ColumnTransformer` fit trên train. Các cột số điền bằng trung vị train. Với bộ này, trung vị của `CADD_PHRED` là 14,06. Nếu cả cột số thiếu trong một fold, bộ điền giữ cột và dùng 0. Biến phân loại dùng mức `__MISSING__` rồi one-hot. Mức mới được biểu diễn bằng 0 ở tất cả cột one-hot của biến tương ứng.
- Logistic Regression và K-means dùng `StandardScaler` cho bốn cột số sau khi điền thiếu. Random Forest giữ thang số gốc. Cả hai phiên bản có cùng 64 tên cột. Với K-means, các khối one-hot và các cột số chuẩn hóa cùng tham gia khoảng cách Euclid. Các trường VEP liên quan có thể ảnh hưởng đồng thời lên khoảng cách.

## Xác minh

- Đã chạy từ CSV gốc hai lần độc lập trong cùng môi trường. Toàn bộ 19 tệp đầu ra có SHA-256 trùng giữa hai lần chạy.
- Đã đọc lại X, y, IDs và ma trận. `source_row` khớp toàn bộ theo thứ tự và 0 khóa xuất hiện ở cả train lẫn test. Cả hai ma trận train/test ở mỗi phiên bản dùng cùng pipeline và danh sách tên cột. Ma trận chứa số hữu hạn, với `CLASS` và các mã truy vết nằm ngoài tập đặc trưng.
- Đã nạp lại cả hai pipeline, biến đổi X và đối chiếu chính xác với ma trận lưu sẵn. Đã thử mức phân loại chưa thấy và thu được đúng cấu trúc.
- Đã chạy Logistic Regression và Random Forest qua 5 fold theo nhóm và thử dự đoán trên ma trận test. Đã fit K-means chỉ với X train và thử gán cụm trên X test. Đây là kiểm tra tương thích đầu vào. Chất lượng mô hình và lựa chọn số cụm chưa được đánh giá.
- Đã thử nhánh khóa lặp với một nhóm mang hai nhãn, một bản sao chính xác và một dòng lỗi khóa/nhãn. Kiểm tra giữ nhóm cùng tập và ghi lý do loại dòng đều đạt.

Các kiểm tra trên áp dụng cho CSV và các phiên bản thư viện đi kèm. Khả năng chuyển sang bản ClinVar khác, dự đoán trên biến thể mới theo thời gian và mức ảnh hưởng của các nguồn chú giải bên ngoài chưa được xác minh.
