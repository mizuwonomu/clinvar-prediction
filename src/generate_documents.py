"""Sinh tài liệu tiếng Việt từ kết quả thật trong document/results.json."""
import json
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "document"


def pct(value):
    return f"{value:.4f}"


def main():
    r = json.loads((DOC / "results.json").read_text(encoding="utf-8"))
    tr, te = r["train_metrics"], r["test_metrics"]
    params = r["best_params"]
    (DOC / "AGENT_HANDOFF.txt").write_text(f"""BÀN GIAO MÔ HÌNH CLINVAR CONFLICTING

Bối cảnh: repo phân tích dữ liệu ClinVar Conflicting. Mã huấn luyện mới ở src/train_model.py; mã tạo tài liệu ở src/generate_documents.py. Dữ liệu nguồn: {r['source']}. Kết quả thực nghiệm: document/results.json; mô hình: document/random_forest_model.joblib; hình ở document/figures/.

Thiết kế dữ liệu: dùng train_test_split với stratify=y, test_size=0.20, random_state=42; train {r['split']['train_rows']:,} dòng ({r['split']['train_positive_rate']:.2%} dương), test {r['split']['test_rows']:,} dòng ({r['split']['test_positive_rate']:.2%} dương). Tập test chỉ được dùng sau khi chọn cấu hình. Dùng StratifiedKFold 5 fold trên train để tìm cấu hình theo Average Precision. Pipeline fit imputer và one-hot encoder trong mỗi fold.

Đặc trưng sử dụng: {', '.join(r['features'])}. Đã bỏ các cột không nằm trong danh sách này, bao gồm khóa/định danh (CHROM, POS, REF, ALT), trường truy vết và annotation dễ tạo số chiều thưa như CLNHGVS, CLNVI, MOTIF_* và các cột khác. CLASS là nhãn, không phải đầu vào.

Siêu tham số tốt nhất: {json.dumps(params, ensure_ascii=False)}. AP CV={r['cv']['best_average_precision']:.4f}.

Điểm test tại ngưỡng xác suất 0.5: Accuracy={te['accuracy']:.4f}; Precision={te['precision']:.4f}; Recall={te['recall']:.4f}; F1={te['f1']:.4f}; AP={te['average_precision']:.4f}; ROC-AUC={te['roc_auc']:.4f}. Train AP={tr['average_precision']:.4f}, train F1={tr['f1']:.4f}. Chênh AP train-test={tr['average_precision']-te['average_precision']:+.4f}. Khoảng cách này và learning curve hữu ích để xem dấu hiệu overfit nhưng không chứng minh khả năng tổng quát hóa ngoài mẫu này.

Hướng tiếp theo: kiểm tra hiệu năng theo gene/vị trí để đánh giá biến thể mới; cân nhắc tối ưu ngưỡng theo mục tiêu lâm sàng bằng validation hoặc OOF, không dùng test; so sánh với baseline và mô hình khác trong cùng phép chia; xác nhận trên cohort độc lập trước khi dùng trong thực tế.
""", encoding="utf-8")

    (DOC / "GIAI_THICH_CHO_NGUOI_HOC.txt").write_text(f"""GIẢI THÍCH MÔ HÌNH CLINVAR CHO NGƯỜI MỚI

Bài toán là đoán cột CLASS: 1 nghĩa là biến thể được ghi nhận có mâu thuẫn, 0 nghĩa là không có ghi nhận mâu thuẫn theo bộ dữ liệu. Mô hình học từ một số chú giải như tần suất alen, kiểu biến thể, hệ quả, mức ảnh hưởng và loại sinh học. Các cột chỉ để nhận diện hoặc truy vết như vị trí, chuỗi HGVS hay motif không được đưa vào mô hình này.

Vì sao chia Train và Test?
Hãy tưởng tượng Train là bộ câu hỏi ôn tập. Mô hình được phép học từ đó. Test là đề thi được cất riêng đến cuối. Nếu chấm trên chính câu đã học, điểm số có thể cao nhưng không cho biết mô hình xử lý dữ liệu mới tốt ra sao. Ở đây 80% dữ liệu dùng để học, 20% giữ riêng để đánh giá công bằng.

Stratified sampling (lấy mẫu phân tầng) có nghĩa là giữ tỷ lệ lớp 0 và lớp 1 gần giống nhau trong cả hai phần. ClinVar có ít mẫu dương hơn mẫu âm. Nếu chia ngẫu nhiên mà không phân tầng, một phần có thể tình cờ thiếu mẫu dương và điểm đánh giá sẽ dao động hoặc gây hiểu nhầm. Phép chia này dùng random_state=42 để có thể tái tạo lại.

Cross-validation (kiểm tra chéo) là chia phần Train thành 5 lượt: mỗi lượt giữ một phần làm bài kiểm tra tạm, còn bốn phần để học. Cấu hình tốt được chọn dựa trên điểm AP trung bình của 5 lượt. Tập Test 20% không tham gia chọn cấu hình. Cách này giúp lựa chọn bớt phụ thuộc vào một lần chia may rủi.

Overfitting, hay “học vẹt”, xảy ra khi mô hình nhớ quá kỹ các ví dụ đã thấy: lỗi trên Train rất thấp nhưng lỗi trên dữ liệu mới cao hơn. Underfitting, hay “học không đến nơi đến chốn”, xảy ra khi mô hình quá đơn giản nên dự đoán kém cả Train lẫn Test. Learning curve cho thấy điểm theo kích thước dữ liệu trên Train và validation; khoảng cách lớn kéo dài có thể là dấu hiệu học vẹt, còn cả hai đường thấp có thể là dấu hiệu mô hình chưa học đủ.

Random Forest hoạt động thế nào?
1. Ngẫu nhiên hóa: mỗi cây nhận một mẫu ngẫu nhiên có hoàn lại từ dữ liệu Train; một dòng có thể được lấy nhiều lần, dòng khác không được lấy ở cây đó. Khi chia nhánh, cây cũng xem một tập con đặc trưng ngẫu nhiên. Vì vậy các cây không hoàn toàn giống nhau.
2. Bagging: đây là cách huấn luyện nhiều cây trên các mẫu bootstrap (lấy ngẫu nhiên có hoàn lại). Cây sâu có thể học những quy tắc chi tiết, còn các giới hạn độ sâu và số mẫu tối thiểu ở lá giúp hạn chế quy tắc quá riêng lẻ.
3. Tổng hợp (Aggregation): mỗi cây đưa ra xác suất thuộc lớp 1. Rừng lấy trung bình các xác suất để có dự đoán chung. Nhiều cây bỏ phiếu thường ổn định hơn một cây đơn lẻ.

Kết quả thật của lần chạy này
Test AP={te['average_precision']:.4f}; Accuracy={te['accuracy']:.4f}; Precision={te['precision']:.4f}; Recall={te['recall']:.4f}; F1={te['f1']:.4f}. Trên Train, AP={tr['average_precision']:.4f}, F1={tr['f1']:.4f}. AP (Average Precision) tóm tắt chất lượng xếp hạng mẫu dương, hữu ích khi hai lớp không cân bằng. Precision trả lời trong các dự đoán dương có bao nhiêu đúng; Recall trả lời trong các ca dương thật mô hình tìm được bao nhiêu; F1 cân bằng hai chỉ số đó. Accuracy có thể trông tốt chỉ vì lớp phổ biến.

Điểm Train và Test gần nhau là dấu hiệu đáng mừng, nhưng không phải bằng chứng chắc chắn rằng mô hình sẽ hoạt động ở bệnh viện hoặc trên quần thể khác. Cần đánh giá trên dữ liệu độc lập và xem hậu quả của bỏ sót/đánh dấu nhầm trước khi áp dụng.
""", encoding="utf-8")

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(.7); sec.bottom_margin = Inches(.7)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"; styles["Normal"].font.size = Pt(10)
    styles["Title"].font.name = "Aptos Display"; styles["Title"].font.size = Pt(24)
    styles["Title"].font.color.rgb = RGBColor(31, 78, 121)
    title = doc.add_paragraph("Báo cáo xây dựng mô hình ClinVar", style="Title")
    doc.add_paragraph("Phân loại biến thể mâu thuẫn bằng Random Forest Classifier")
    doc.add_paragraph(f"Tóm tắt: mô hình đạt AP {te['average_precision']:.4f} trên tập test giữ riêng gồm {r['split']['test_rows']:,} dòng. Đây là điểm đo trên lần chia phân tầng đã định; cần dữ liệu ngoài để xác nhận khả năng áp dụng.")

    doc.add_heading("Phần 1  Tổng quan phương pháp đánh giá", level=1)
    doc.add_paragraph(f"Dữ liệu có {r['split']['train_rows']+r['split']['test_rows']:,} dòng. Dùng hold-out 80/20, stratify theo CLASS, random_state=42. Tỷ lệ lớp dương train là {r['split']['train_positive_rate']:.2%}, test là {r['split']['test_positive_rate']:.2%}. Stratification giữ tỷ lệ lớp gần nhau để test đại diện hơn cho mất cân bằng vốn có.")
    doc.add_paragraph("Chỉ phần train được dùng trong StratifiedKFold 5 fold để chọn siêu tham số theo Average Precision. Pipeline điền thiếu và one-hot encoding được fit trong mỗi fold. Tập test chỉ dùng để báo cáo cuối, tránh rò rỉ thông tin khi chọn mô hình.")
    doc.add_heading("Phần 2  Cấu trúc mô hình Random Forest", level=1)
    doc.add_paragraph("Mô hình tạo nhiều cây từ các mẫu bootstrap và các tập con đặc trưng, sau đó trung bình xác suất dự đoán. Tìm kiếm ngẫu nhiên đánh giá 12 cấu hình trên 5 fold. Các tham số tốt nhất:")
    for key, value in params.items():
        doc.add_paragraph(f"{key}: {value}", style="List Bullet")
    doc.add_paragraph("Đầu vào gồm: " + ", ".join(r["features"]) + ". Các cột định danh, truy vết, annotation thưa và CLASS bị loại khỏi X.")
    doc.add_heading("Phần 3  Biểu đồ và đánh giá", level=1)
    doc.add_paragraph("Điểm ở ngưỡng 0.5. AP CV trên train = %.4f. AP train = %.4f; AP test = %.4f (chênh train-test = %+.4f). F1 train = %.4f; F1 test = %.4f. Khoảng cách nhỏ là dấu hiệu hữu ích khi xem overfitting, nhưng chỉ một hold-out không chứng minh tổng quát hóa; cần đọc learning curve và đánh giá ngoài." % (r['cv']['best_average_precision'], tr['average_precision'], te['average_precision'], tr['average_precision']-te['average_precision'], tr['f1'], te['f1']))
    table = doc.add_table(rows=1, cols=3); table.style = "Light Shading Accent 1"
    for cell, val in zip(table.rows[0].cells, ["Chỉ số", "Train", "Test"]): cell.text = val
    for key, label in [("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),("f1","F1"),("average_precision","Average Precision"),("roc_auc","ROC-AUC")]:
        cells=table.add_row().cells; cells[0].text=label; cells[1].text=pct(tr[key]); cells[2].text=pct(te[key])
    for name, caption in [("learning_curve.png","Learning curve: AP Train và validation theo số mẫu."), ("roc_curve.png","ROC curve trên hold-out test."), ("precision_recall_curve.png","Precision-Recall trên hold-out test."), ("feature_importance.png","Độ quan trọng impurity-based của 20 đặc trưng đứng đầu."), ("confusion_matrix.png","Ma trận nhầm lẫn tại ngưỡng 0.5." )]:
        doc.add_paragraph(caption)
        doc.add_picture(str(DOC/"figures"/name), width=Inches(5.8))
    doc.add_heading("Phần 4  Kết luận và ma trận nhầm lẫn", level=1)
    tn, fp = r["confusion_matrix"][0]; fn, tp = r["confusion_matrix"][1]
    doc.add_paragraph(f"Ở ngưỡng 0.5, mô hình nhận đúng {tp:,} mẫu dương và {tn:,} mẫu âm; bỏ sót {fn:,} mẫu dương, đồng thời báo dương nhầm {fp:,} mẫu âm. Chọn ngưỡng khác sẽ đổi trade-off Precision/Recall và cần thực hiện trên validation/OOF, không tối ưu trực tiếp trên test.")
    doc.add_paragraph("Kết quả cho thấy hiệu năng đo được trong phép chia này. Bước tiếp theo nên kiểm tra nhóm gene/vị trí chưa gặp và xác nhận trên cohort độc lập trước khi cân nhắc sử dụng thực tế.")
    doc.save(DOC/"BAO_CAO_XAY_DUNG_MO_HINH.docx")


if __name__ == "__main__":
    main()
