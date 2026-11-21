# Kiểm tra cú pháp và chú giải trên train

- MC và Consequence cùng có dữ liệu ở 51,472 dòng. Tập hệ quả trùng hoàn toàn ở 41,766 dòng và giao nhau ở 51,060 dòng sau khi quy `nonsense` về `stop_gained`.
- ALT có dấu phẩy ở 0 dòng. Các dòng này bật cờ ALT nhiều alen và bỏ độ dài ALT lẫn chênh lệch. Độ dài REF vẫn xác định.
- BLOSUM62 có 11,299 điểm âm hợp lệ.

- Khôi phục được 5,442 điểm BLOSUM62 bằng 0 trên missense có cặp axit amin đơn khớp điểm 0 theo ma trận NCBI. Heatmap và cờ BLOSUM62 giữ trạng thái ghi nhận gốc của CSV. [Ma trận NCBI](https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62)

| Trường | Có dữ liệu | Phân tích được | Giá trị lỗi/không rõ | Ví dụ đơn hoặc phân số | Ví dụ khoảng | Ví dụ lỗi |
|---|---:|---:|---:|---|---|---|
| EXON | 45,019 | 45,019 | 0 | 1/1, 4/4, 1/7 | — | — |
| INTRON | 7,066 | 7,066 | 0 | 6/27, 8/17, 3/20 | — | — |
| CDS_position | 44,165 | 44,130 | 35 | 522, 509, 100 | 874-875, 764-765, 2403-2405 | ?-73, 1476-?, 1155-? |
| Protein_position | 44,165 | 44,130 | 35 | 174, 170, 34 | 801-802, 181-188, 244-252 | ?-25, 492-?, 385-? |

Tỷ lệ EXON/INTRON dùng mẫu số tổng ghi trong chính trường đó. CDS/protein chỉ cho phép tọa độ dương đơn hoặc khoảng tăng, và không tính vị trí tương đối.

Cờ hiện diện đo việc trường có giá trị trong CSV. Với SIFT/PolyPhen/BLOSUM62, các hệ quả không ghi nhận thay thế axit amin là ngữ cảnh dự kiến ít áp dụng. Trường hợp thiếu trên missense và các ngữ cảnh chưa rõ vẫn được ghi là thiếu, chưa xác định nguyên nhân. Giá trị PolyPhen `unknown` là một mức được ghi nhận.

Các điểm VEP, CADD và tần số quần thể được dùng theo snapshot của bộ dữ liệu này. Việc chúng khả dụng ở thời điểm diễn giải ban đầu chưa được kiểm chứng. Hồ sơ bệnh và ý nghĩa lâm sàng thành phần vẫn ở ngoài đầu vào.
