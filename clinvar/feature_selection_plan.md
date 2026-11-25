# Kế hoạch bổ sung đặc trưng cho ClinVar Conflicting

Mục tiêu là kiểm tra liệu các nhóm chú giải đang bị bỏ qua có giúp nhận diện biến thể mang nhãn diễn giải mâu thuẫn (`CLASS=1`) hay không. Đơn vị dự đoán là biến thể. Các điểm dự đoán tác động protein là đầu vào ứng viên, còn nhãn đánh giá là sự mâu thuẫn giữa các diễn giải.

## 1. Chốt mốc so sánh và tổng hợp ý kiến nhóm

Pipeline hiện chọn 8 trường từ 46 cột gốc của 65.188 bản ghi. Hai phiên bản 64 và 42 cột sau mã hóa khác nhau ở cách biểu diễn `Consequence`. Cả hai dùng chung 52.150 dòng train và 13.038 dòng test. Báo cáo hiện có xác nhận khả năng chạy mô hình, chưa đo mức cải thiện do bổ sung đặc trưng.

Trước khi sửa pipeline, lập **một bảng quyết định cho đủ 46 cột**. Đọc báo cáo, notebook, kết quả missing và các đề xuất của cả năm thành viên trong thư mục Week 1. Với từng cột, ghi tên tệp và vị trí đề xuất, lý do chuyên môn, độ phủ trên train, cách mã hóa hiện tại và quyết định dự kiến. Đánh dấu các cột nhóm từng yêu cầu giữ dù thiếu nhiều và các điểm còn bất đồng. Nếu tài liệu không nêu một cột, ghi rõ điều đó. Danh sách ưu tiên dưới đây dựa trên workspace, cần đối chiếu với bảng này.

## 2. Khảo sát tín hiệu trên train trước khi chọn cột

- Dùng mã dòng train hiện có để lấy lại các trường từ CSV gốc. Thống kê số quan sát hợp lệ theo cột, theo `Consequence` và theo `CLASS`. Phân biệt thiếu do trường không áp dụng với thiếu chú giải khi ngữ cảnh cho phép. Giữ trạng thái chưa xác định khi thiếu căn cứ.
- Vẽ heatmap Spearman giữa các trường số, trước khi điền thiếu. Ghi số cặp có dữ liệu tại từng ô. Kiểm tra riêng ba nguồn AF và cặp `CADD_RAW`–`CADD_PHRED`. Tạo thêm heatmap tỷ lệ có chú giải theo nhóm hệ quả để thấy SIFT, PolyPhen và BLOSUM62 bao phủ những biến thể nào. Chú thích ô quá ít quan sát. Dùng ablation để quyết định giá trị bổ sung khi các cột tương quan.
- Với biến phân loại, lập bảng chéo theo nhóm và đếm số dòng mỗi mức. Rà soát thời điểm có thể biết từng trường lúc dự đoán. Loại `CLASS` và `CLNSIGINCL` khỏi đầu vào. Các trường hồ sơ lâm sàng như `CLNDN`, `CLNDISDB` cần có căn cứ về nguồn gốc và thời điểm trước khi xét dùng. Giữ khóa biến thể để đối chiếu dòng.

## 3. Thứ tự thử nghiệm

Số đếm dưới đây tính từ hồ sơ toàn bộ CSV hiện có. Khi triển khai, tính lại độ phủ trên train và trên từng nhóm biến thể.

| Ưu tiên | Tín hiệu cần thử | Cách xử lý và câu hỏi kiểm chứng |
|---|---|---|
| 1 | `LoFtool` có mặt ở 60.975 dòng. `SIFT`, `PolyPhen`, `BLOSUM62` lần lượt có mặt ở 24.836, 24.796 và 25.593 dòng | Giữ LoFtool và BLOSUM62 dạng số, bảo toàn điểm BLOSUM62 âm. Điền trung vị học trong fold và thêm cờ có điểm. SIFT và PolyPhen có bốn mức quan sát mỗi cột, mã hóa phân loại với mức thiếu riêng. So sánh ba cấu hình gồm baseline, baseline cộng cờ có chú giải, và baseline cộng cờ cùng giá trị. Đánh giá riêng nhóm missense. |
| 2 | Cấu trúc alen và vị trí trên transcript | Từ REF/ALT suy ra độ dài và chênh lệch độ dài cho alen đơn, quy định riêng cách xử lý ALT nhiều alen. Khảo sát cú pháp EXON/INTRON trước khi tách thứ tự và tổng số. Với CDS_position và Protein_position, kiểm tra giá trị đơn, khoảng, phần chưa biết. Chỉ tính vị trí tương đối khi có mẫu số phù hợp. Kiểm tra quy tắc bằng ví dụ thật và đếm lỗi phân tích. |
| 3 | Tín hiệu có thể trùng hoặc rất thưa | Thử thay CADD_PHRED bằng CADD_RAW trong cùng cấu hình. Đối chiếu MC với Consequence trước khi thêm. DISTANCE chỉ có 108 dòng có dữ liệu và mỗi trường motif chỉ có 2 dòng. Ghi nhận trong bảng quyết định và báo cáo độ phủ theo fold trước khi diễn giải tác động. |

Nếu nhóm đề xuất dùng `SYMBOL`, thử thành một nhánh riêng và đánh giá bổ sung bằng cách giữ toàn bộ biến thể cùng gene trong một fold. Ghi cả số gene và số mẫu dương mỗi fold. Điều này giúp phát hiện lợi ích chủ yếu xuất hiện khi gene đã gặp trong train.

## 4. Đánh giá và điều kiện chốt

1. Trên train, đo Logistic Regression và Random Forest với cả hai baseline hiện có. Dùng cùng năm fold phân tầng theo nhãn, giữ các bản ghi cùng khóa biến thể trong cùng fold. Cố định cách mã hóa Consequence khi so sánh từng nhóm đặc trưng. Fit toàn bộ bước học từ dữ liệu trong phần huấn luyện của từng fold.
2. Chạy từng nhóm ứng viên độc lập trên cùng baseline, sau đó ghép các nhóm có triển vọng và bỏ lần lượt từng nhóm để đo đóng góp riêng. Giữ nguyên cấu hình mô hình và ngân sách tinh chỉnh giữa các lần so sánh.
3. Dùng **average precision (AP) cho CLASS=1** làm chỉ số chính vì lớp này chiếm khoảng 25,2%. Báo AP từng fold, chênh lệch ghép cặp với baseline và số mẫu dương. Báo thêm precision, recall và F1 với ngưỡng xác định trên train. Tách kết quả cho missense, các hệ quả còn lại, nhóm có và thiếu chú giải protein.
4. Ưu tiên giữ nhóm có lợi ích lặp lại qua nhiều fold hoặc có lợi ích rõ trong nhóm biến thể đã xác định trước và đủ mẫu. Nếu chênh lệch nhỏ so với dao động giữa fold, ghi “chưa đủ bằng chứng”. Khóa tập feature, cấu hình và ngưỡng rồi mới đánh giá trên test một lần.

**Bàn giao khi triển khai** gồm bảng quyết định 46 cột có nguồn, hai heatmap có chú thích độ phủ, bảng so sánh mô hình theo nhóm đặc trưng và danh sách feature cuối có lý do giữ hoặc loại. Phiên bản pipeline mở rộng cần được xuất riêng, kiểm tra cùng mã dòng, nhãn và thứ tự cột giữa các phép so sánh.

**Tài liệu để đối chiếu** · [Báo cáo hiện trạng](baseline_preprocessing_report.md) · [Hồ sơ 46 cột](data/raw_column_profile.csv) · [Mã tiền xử lý](preprocess.py) · [Tài liệu Week 1 của nhóm](https://drive.google.com/drive/folders/1gsyXN82c_NebRZRxWzKR-4JVbULry05i)
