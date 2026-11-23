"""Sinh báo cáo và tài liệu giảng dạy từ document/results.json."""
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "document"


def f(value):
    return f"{value:.4f}"


def main():
    r = json.loads((DOC / "results.json").read_text(encoding="utf-8"))
    tr, te, split = r["train_metrics"], r["test_metrics"], r["split"]
    params = r["best_params"]
    tn, fp = r["confusion_matrix"][0]
    fn, tp = r["confusion_matrix"][1]
    prevalence = r.get("test_prevalence_baseline_ap", split["test_positive_rate"])

    (DOC / "AGENT_HANDOFF.txt").write_text(f"""BÀN GIAO RANDOM FOREST CLINVAR

Mã nguồn: src/train_model.py; tài liệu được tạo bởi src/generate_documents.py.
Dữ liệu: {r['source']}. Kết quả: document/results.json; mô hình fit cuối: document/random_forest_model.joblib.

Thiết kế: {split['method']}, test_size={split['test_size']}, random_state={split['random_state']}. Train {split['train_rows']:,} dòng, tỷ lệ CLASS=1 là {split['train_positive_rate']:.2%}; test {split['test_rows']:,} dòng, tỷ lệ CLASS=1 là {split['test_positive_rate']:.2%}. Đây là phép chia ngẫu nhiên theo dòng, không phải đánh giá trên gene hoặc vị trí hoàn toàn mới.

Đặc trưng: {', '.join(r['features'])}. CLASS là nhãn. Các trường định danh và truy vết bị loại. Tiền xử lý nằm trong Pipeline; imputer và encoder được fit bên trong từng fold.

Tìm kiếm: 12 cấu hình, StratifiedKFold 5 fold trên train, Average Precision làm tiêu chí. Cấu hình: {json.dumps(params, ensure_ascii=False)}. AP tốt nhất CV={r['cv']['best_average_precision']:.4f}; đây là điểm chọn cấu hình, có thể lạc quan do chọn cấu hình cao nhất. Không xem đây là ước lượng độc lập cuối cùng.

Hold-out test tại ngưỡng quy ước 0.5: Accuracy={te['accuracy']:.4f}; Precision={te['precision']:.4f}; Recall={te['recall']:.4f}; F1={te['f1']:.4f}; AP={te['average_precision']:.4f}; ROC-AUC={te['roc_auc']:.4f}. Tỷ lệ lớp dương test={prevalence:.4f}, là mốc AP của bộ phân loại xếp hạng ngẫu nhiên. Ma trận nhầm lẫn (TN, FP; FN, TP)=[[{tn}, {fp}], [{fn}, {tp}]].

Diễn giải: test chỉ được dùng sau khi chọn cấu hình. Ngưỡng 0.5 là điểm vận hành để minh họa, không được tối ưu theo nhu cầu lâm sàng. Feature importance là impurity-based, không phải tác động nhân quả hay bằng chứng sinh học. Cần đánh giá tách theo gene/vị trí, hiệu chỉnh xác suất và xác nhận trên cohort độc lập trước khi cân nhắc ứng dụng.
""", encoding="utf-8")

    (DOC / "GIAI_THICH_CHO_NGUOI_HOC.txt").write_text(f"""TÀI LIỆU HỌC MÔ HÌNH RANDOM FOREST CHO CLINVAR

Mục tiêu của bài học

Ta xây dựng một bộ phân loại để dự đoán CLASS trong bộ dữ liệu ClinVar Conflicting. CLASS=1 là bản ghi được gán nhãn mâu thuẫn theo quy tắc của bộ dữ liệu; CLASS=0 là bản ghi không được gán nhãn mâu thuẫn. Đây là nhãn của snapshot dữ liệu, không phải chẩn đoán bệnh nhân và không đồng nghĩa với pathogenicity.

Mô hình dùng tám trường: AF_ESP, AF_EXAC, AF_TGP, CADD_PHRED, CLNVC, Consequence, IMPACT và BIOTYPE. CLASS không được đưa vào X. Các trường vị trí/định danh như CHROM, POS, REF, ALT cùng thông tin truy vết bị loại để giảm nguy cơ mô hình ghi nhớ bản ghi.

1. Vì sao cần chia dữ liệu?

Train là phần mô hình được phép học. Test được giữ lại để đo hiệu năng sau khi quy trình chọn mô hình đã hoàn tất. Trong lần chạy này, train có {split['train_rows']:,} dòng và test có {split['test_rows']:,} dòng. Stratify giữ tỷ lệ CLASS=1 gần nhau: {split['train_positive_rate']:.2%} ở train và {split['test_positive_rate']:.2%} ở test. Seed 42 giúp tái lập phép chia.

Giới hạn cần nhớ: phép chia này ngẫu nhiên theo dòng. Nó đo khả năng phân loại các dòng cùng snapshot khi cả train và test có thể chứa mẫu từ cùng gene hoặc vùng sinh học. Nó không trả lời trực tiếp câu hỏi mô hình sẽ ra sao trên gene, vị trí hoặc cohort hoàn toàn mới. Vì vậy không gọi đây là đánh giá tổng quát hóa ngoài miền.

2. Tiền xử lý không được học trước trên toàn bộ dữ liệu

Pipeline điền giá trị số thiếu bằng trung vị và thêm cờ thiếu; các biến phân loại được điền bằng mức phổ biến nhất rồi one-hot encode. Pipeline nằm bên trong quy trình cross-validation nên mỗi fold chỉ học trung vị và danh mục từ phần fit của fold. Điều này ngăn thông tin phân phối validation/test lọt vào bước tiền xử lý.

Lưu ý với AF: tài liệu nguồn cho biết một số giá trị AF thiếu đã được biểu diễn thành 0 trước khi CSV được tạo. Pipeline không thể khôi phục thiếu hụt đã bị mã hóa như vậy; 0 được xem như một giá trị hợp lệ và không bật cờ thiếu.

3. Random Forest hoạt động ra sao?

Mỗi cây được huấn luyện từ mẫu bootstrap, tức lấy mẫu có hoàn lại từ tập fit. Tại các nút, cây chỉ xét một tập con đặc trưng. Cây sâu có thể học quy tắc chi tiết; giới hạn max_depth, min_samples_leaf và min_samples_split kiểm soát độ phức tạp. Random Forest tổng hợp xác suất từ nhiều cây. Phép tổng hợp thường giảm độ biến động so với một cây đơn, nhưng không tự loại bỏ thiên lệch dữ liệu hay leakage.

4. Chọn cấu hình và đọc AP

Twelve cấu hình được so sánh bằng StratifiedKFold 5 fold trên train; tiêu chí là Average Precision (AP). AP hữu ích khi lớp dương ít hơn lớp âm vì tập trung vào thứ tự xếp hạng lớp dương. Tỷ lệ CLASS=1 trong test là {prevalence:.4f}; đó là mốc AP của xếp hạng ngẫu nhiên trong trường hợp này. AP CV tốt nhất là {r['cv']['best_average_precision']:.4f}. Vì đây là giá trị cao nhất sau khi thử 12 cấu hình, nó có thể lạc quan và không nên trình bày như đánh giá độc lập.

5. Kết quả của hold-out test

Tại ngưỡng xác suất quy ước 0.5, AP={f(te['average_precision'])}, ROC-AUC={f(te['roc_auc'])}, Precision={f(te['precision'])}, Recall={f(te['recall'])}, F1={f(te['f1'])}, Accuracy={f(te['accuracy'])}. Ma trận nhầm lẫn theo thứ tự TN, FP / FN, TP là [[{tn}, {fp}], [{fn}, {tp}]]. Mô hình tìm đúng {tp:,} mẫu dương và bỏ sót {fn:,}; đồng thời báo dương nhầm {fp:,} mẫu âm.

AP và ROC-AUC đánh giá xếp hạng trên nhiều ngưỡng. Precision, Recall và F1 phụ thuộc ngưỡng. Ngưỡng 0.5 ở đây chỉ là điểm vận hành để minh họa; không có căn cứ để gọi nó là ngưỡng tối ưu. Nếu mục tiêu ưu tiên giảm bỏ sót hay giảm báo động giả, cần xác định chi phí sai lầm, chọn ngưỡng bằng validation/OOF trên train, rồi mới đánh giá một lần trên test.

Accuracy không nên đọc riêng khi dữ liệu lệch lớp. F1 không thể hiện chi phí lâm sàng. Điểm train AP={f(tr['average_precision'])} cao hơn test AP={f(te['average_precision'])}. Điểm train được đo trên chính các dòng dùng fit mô hình cuối nên thường lạc quan; chênh lệch chỉ là dấu hiệu chẩn đoán, không phải ước lượng không thiên lệch của generalization gap.

6. Đọc biểu đồ và feature importance

Learning curve cho thấy điểm train và validation khi tăng số mẫu. Khoảng cách kéo dài gợi ý variance/overfit; cả hai điểm thấp có thể gợi ý mô hình chưa đủ khả năng. Đường cong không chứng minh hiệu năng trên gene mới.

ROC curve mô tả trade-off giữa tỷ lệ dương tính giả và tỷ lệ dương tính thật. Precision–Recall curve hữu ích hơn khi quan tâm lớp dương hiếm. Confusion matrix là kết quả tại ngưỡng 0.5. Feature importance trong báo cáo là impurity-based importance: nó phân bổ mức giảm impurity trong các lần chia cây. Chỉ số này có thể thiên lệch về biến có nhiều cách chia hoặc nhiều mức, chịu ảnh hưởng của các biến tương quan và không phải tác động nhân quả. Hãy coi nó là mô tả của mô hình, không phải kết luận sinh học.

7. Kết luận đúng mức và bước tiếp theo

Kết quả cho thấy mô hình có khả năng phân biệt nhãn trong hold-out ngẫu nhiên của snapshot này. Nó chưa chứng minh khả năng áp dụng lâm sàng, hiệu quả trên biến thể tương lai, hay khả năng tổng quát sang gene/cohort khác. Bước đánh giá tiếp theo nên giữ toàn bộ gene (hoặc nhóm vị trí/biến thể tương đương) ở một phía của phép chia, báo khoảng bất định qua nhiều fold, kiểm tra calibration và so sánh với baseline trong cùng phép chia. Sau đó cần xác nhận trên cohort độc lập.
""", encoding="utf-8")

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(.65)
    sec.bottom_margin = Inches(.65)
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.5)
    normal.paragraph_format.space_after = Pt(5)
    title_style = doc.styles["Title"]
    title_style.font.name = "Aptos Display"
    title_style.font.size = Pt(24)
    title_style.font.color.rgb = RGBColor(0, 0, 0)

    doc.add_paragraph("Báo cáo mô hình Random Forest cho ClinVar", style="Title")
    doc.add_paragraph("Phân loại nhãn mâu thuẫn trong bộ ClinVar Conflicting")
    p = doc.add_paragraph()
    p.add_run("Kết luận chính. ").bold = True
    p.add_run(f"Trên hold-out ngẫu nhiên theo dòng, mô hình đạt AP {te['average_precision']:.4f} (tỷ lệ lớp dương {prevalence:.4f}). Đây là kết quả trong cùng snapshot dữ liệu; phép đánh giá chưa tách gene hoặc vị trí giữa train và test.")

    doc.add_heading("1. Câu hỏi và phạm vi", level=1)
    doc.add_paragraph("Mục tiêu là dự đoán CLASS theo định nghĩa có sẵn trong bộ dữ liệu, không phải dự đoán pathogenicity hay đưa ra chẩn đoán. Đầu vào gồm tám chú giải: " + ", ".join(r["features"]) + ". Các khóa biến thể và trường truy vết không được dùng làm đặc trưng.")
    doc.add_paragraph("Phạm vi kết quả bị giới hạn bởi snapshot nguồn và phép chia ngẫu nhiên theo dòng. Mẫu ở cùng gene có thể xuất hiện ở cả train và test; do đó không diễn giải điểm test như khả năng dự đoán trên gene mới.")
    doc.add_paragraph("Các tần suất AF trong CSV nguồn có thể đã được tác giả mã hóa thiếu thành 0; nếu vậy, mô hình không thể phân biệt AF thật bằng 0 với AF bị thiếu chỉ từ các cột hiện tại. Cờ thiếu do imputer tạo ra chỉ phản ánh ô còn thiếu khi đọc CSV, không khôi phục thông tin đã mất trước đó.")

    doc.add_heading("2. Quy trình huấn luyện", level=1)
    doc.add_paragraph(f"Dữ liệu có {split['train_rows'] + split['test_rows']:,} dòng: {split['train_rows']:,} train và {split['test_rows']:,} test (80/20), phân tầng theo CLASS với random_state={split['random_state']}. Tỷ lệ dương lần lượt là {split['train_positive_rate']:.2%} và {split['test_positive_rate']:.2%}.")
    doc.add_paragraph("Điền thiếu và one-hot encoding nằm trong Pipeline. RandomizedSearchCV so sánh 12 cấu hình bằng StratifiedKFold 5 fold trên train, tối ưu AP. Tập test không tham gia chọn siêu tham số. Điểm AP CV tốt nhất là %.4f; vì lấy giá trị cao nhất trong các cấu hình đã thử, đây là điểm phục vụ lựa chọn và có thể lạc quan." % r["cv"]["best_average_precision"])
    doc.add_paragraph("Cấu hình chọn được: " + "; ".join(f"{k.removeprefix('model__')}={v}" for k, v in params.items()) + ".")

    doc.add_heading("3. Kết quả trên test", level=1)
    doc.add_paragraph("Bảng phân biệt chỉ số xếp hạng và chỉ số tại ngưỡng quy ước 0,5. Ngưỡng này không được tối ưu theo test và chưa được chọn theo một mục tiêu lâm sàng.")
    table = doc.add_table(rows=1, cols=3)
    table.style = "Light Shading Accent 1"
    for c, value in zip(table.rows[0].cells, ["Chỉ số", "Train apparent", "Test"]):
        c.text = value
    labels = [("average_precision", "Average Precision"), ("roc_auc", "ROC-AUC"),
              ("precision", "Precision @ 0,5"), ("recall", "Recall @ 0,5"),
              ("f1", "F1 @ 0,5"), ("accuracy", "Accuracy @ 0,5")]
    for key, label in labels:
        cells = table.add_row().cells
        cells[0].text, cells[1].text, cells[2].text = label, f(tr[key]), f(te[key])
    doc.add_paragraph(f"Tỷ lệ CLASS=1 trên test là {prevalence:.4f}, làm mốc AP cho xếp hạng ngẫu nhiên. Tại ngưỡng 0,5, mô hình có TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}. Do đó precision và recall thể hiện một trade-off cụ thể, không phải chất lượng độc lập với ngưỡng.")
    doc.add_paragraph(f"AP train apparent={tr['average_precision']:.4f}, AP test={te['average_precision']:.4f}. Điểm train được đo trên chính các dòng dùng fit mô hình cuối nên thường lạc quan; chênh lệch chỉ là dấu hiệu chẩn đoán, không phải ước lượng không thiên lệch của generalization gap.")

    doc.add_heading("4. Biểu đồ", level=1)
    for name, caption in [
        ("learning_curve.png", "Hình 1. Learning curve tính trên train; dải thể hiện độ lệch chuẩn giữa các fold."),
        ("precision_recall_curve.png", "Hình 2. Precision–Recall trên hold-out test; phù hợp để xem hiệu năng lớp dương."),
        ("roc_curve.png", "Hình 3. ROC trên hold-out test; tổng hợp trade-off qua các ngưỡng."),
        ("confusion_matrix.png", "Hình 4. Ma trận nhầm lẫn tại ngưỡng 0,5."),
        ("feature_importance.png", "Hình 5. Impurity-based importance. Đây là mô tả nội bộ mô hình, không phải mức đóng góp nhân quả.")]:
        p = doc.add_paragraph(caption)
        p.paragraph_format.keep_with_next = True
        doc.add_picture(str(DOC / "figures" / name), width=Inches(5.8))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("5. Giới hạn và bước đánh giá kế tiếp", level=1)
    for item in [
        "Đánh giá theo dòng có thể đặt mẫu cùng gene hoặc vùng sinh học ở cả train và test; cần thêm phép chia nhóm theo gene/vị trí.",
        "Nguồn AF có thể đã điền 0 cho giá trị thiếu; điều này gộp hai trạng thái và làm hạn chế cách diễn giải ảnh hưởng của AF.",
        "Ngưỡng 0,5 chỉ là quy ước. Chọn ngưỡng bằng validation hoặc OOF theo mục tiêu sai lầm, rồi khóa trước khi mở test.",
        "Feature importance impurity-based có thể thiên lệch và không cho biết quan hệ nhân quả; cần permutation importance hoặc ablation trên fold nếu muốn so sánh đóng góp.",
        "Xác nhận trên cohort độc lập và kiểm tra calibration trước mọi suy luận về khả năng dùng ngoài bộ dữ liệu này."]:
        doc.add_paragraph(item, style="List Bullet")
    doc.save(DOC / "BAO_CAO_XAY_DUNG_MO_HINH.docx")


if __name__ == "__main__":
    main()
