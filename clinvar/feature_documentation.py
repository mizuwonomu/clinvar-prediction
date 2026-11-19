"""Bảng quyết định, thống kê nghiệm thu và báo cáo tiếng Việt cho bộ giữ feature."""

import pandas as pd

from consequence_encoder import _terms
from feature_study import OUTPUT, POSITIONS
from preprocess import AF, require
from retained_features import KEEP_RAW


REQUIRED = {*AF, "EXON", "INTRON", "SIFT", "PolyPhen", "BLOSUM62", "Amino_acids",
            "Protein_position", "CLNVC", "IMPACT", "Consequence", "CADD_PHRED", "LoFtool"}
TRIAL = {"CHROM", "ORIGIN", "CDS_position", "cDNA_position", "CADD_RAW", "BIOTYPE"}
DRIVE_MISSING = (
    "INTRON", "PolyPhen", "SIFT", "BLOSUM62", "CLNVI", "BAM_EDIT", "Amino_acids", "Codons",
    "CDS_position", "Protein_position", "EXON", "cDNA_position", "LoFtool", "CADD_PHRED", "CADD_RAW", "MC",
)
RULES = {
    "CHROM": ("Chuỗi chuẩn hóa trong khóa", "Khóa bắt buộc", "Mô hình thử one-hot riêng vì Vy đề nghị giữ."),
    "POS": ("Số vị trí dạng chuỗi trong khóa", "Khóa bắt buộc", "Dùng để ghép khóa, tránh ghi nhớ tọa độ gốc."),
    "REF": ("Chuỗi REF, ref_length", "Khóa bắt buộc", "Độ dài alen trong thử nghiệm trước chưa tăng AP ổn định."),
    "ALT": ("Chuỗi ALT, alt_length, alt_minus_ref, multiple_alt", "Độ dài ALT và chênh lệch để trống nếu nhiều alen",
            "Độ dài alen trong thử nghiệm trước chưa tăng AP ổn định."),
    **{col: ("Giá trị nguồn, AF hợp lệ, log và cờ bằng 0", "Trống, ngoài miền và sai kiểu có trạng thái riêng",
             "Vy muốn log AF. Tách ảnh hưởng của log và cờ 0 theo kết quả CV.") for col in AF},
    **{col: ("Giá trị nguồn, chỉ số, tổng số, tỷ lệ và vùng transcript", "Giữ trạng thái missing và invalid riêng",
             "Giữ tín hiệu vùng đã có trong bộ 24 cột, kể cả khi trường bổ sung thiếu.")
       for col in ("EXON", "INTRON")},
    "SIFT": ("Nhãn nguồn, cờ hiện diện, kết luận và cờ tin cậy thấp", "Thiếu chú giải giữ trống trước khi fit",
             "Ước lượng giá trị bổ sung của kết luận và mức tin cậy sau cờ."),
    "PolyPhen": ("Nhãn nguồn, cờ hiện diện và mức unknown", "unknown là mức quan sát, ô thiếu giữ trống",
                 "Kiểm tra giá trị phân loại kèm nhóm protein."),
    "BLOSUM62": ("Điểm nguồn, cờ hiện diện, điểm sau kiểm tra và nguồn điểm",
                 "Khôi phục 0 khi ô trống trên missense và cặp khớp điểm 0, giữ dấu âm",
                 "Vy đề xuất khôi phục điểm 0. Bộ 24 dùng cờ, nhánh giá trị được đo lại."),
    "Amino_acids": ("Cặp nguồn và cờ hiện diện", "Thiếu giữ trống",
                    "Dùng kiểm tra điểm 0, tránh lặp tín hiệu với điểm BLOSUM62 trong X."),
    **{col: ("Tọa độ nguồn, đầu, cuối, độ dài khoảng và trạng thái",
             "Giữ một mút nếu mút còn lại chưa rõ. Chỉ tính độ dài khi biết đủ hai mút",
             "Vy đề xuất bỏ CDS/cDNA vì trùng. Bộ 24 đã dùng CDS và protein. cDNA bổ sung độ phủ khi CDS thiếu.")
       for col in (*POSITIONS, "cDNA_position")},
    "CLNVC": ("Nhãn nguồn và giá trị sạch", "Thiếu để trống trước one-hot", "Giữ loại biến thể trong mốc."),
    "IMPACT": ("Nhãn nguồn và mã 0–3 từ MODIFIER đến HIGH", "Thiếu hoặc mức chưa biết giữ trống",
               "So thứ bậc được Vy đề xuất với one-hot hiện hành."),
    "Consequence": ("Chuỗi nguồn và danh sách tách bằng &", "Danh sách rỗng nếu không có chú giải",
                    "So multi-hot có gộp hiếm theo fold với one-hot hiện hành."),
    "ORIGIN": ("Mã nguồn, các bit giải mã và trạng thái", "0 là unknown, mã kết hợp tách thành các bit",
               "Hoãn thử vì snapshot tổng hợp không chứng minh ORIGIN có trước thời điểm dự đoán."),
    "CADD_PHRED": ("Điểm nguồn và điểm số", "Thiếu giữ trống", "Vy ưu tiên PHRED, bộ 24 đang dùng RAW. Hai thang gần trùng."),
    "CADD_RAW": ("Điểm nguồn, điểm số và cờ hiện diện", "Thiếu giữ trống, bảo toàn điểm âm",
                 "Dùng một thang trong X. Cờ có CADD chưa cho lợi ích ổn định."),
    "LoFtool": ("Điểm nguồn, điểm số và cờ hiện diện", "Thiếu giữ trống trước khi fit",
                "Giữ điểm và cờ theo bộ 24, giá trị đã cải thiện AP RF trong lần trước."),
    "SYMBOL": ("Tên gene nguồn và tên sạch để nhóm", "Biến thể thiếu gene nhóm theo khóa riêng khi chia fold",
               "Tần suất gene thử trước giảm lợi ích trên gene mới."),
    "BIOTYPE": ("Nhãn nguồn và giá trị sạch", "Thiếu giữ trống trước one-hot",
                "Vy và Sơn đề nghị bỏ. Giữ biến của bộ 24 để so sánh đồng nhất."),
    "CLNDN": ("Chuỗi nguồn, cờ chỉ placeholder và trạng thái", "Tách mục theo |, giữ riêng named, mixed, placeholder_only, missing",
               "Chỉ dùng kiểm tra. Snapshot chưa chứng minh thông tin có trước dự đoán."),
    "CLASS": ("Nhãn trong y_train.csv và y_test.csv", "Nhãn chỉ nhận 0 hoặc 1", "Tách khỏi bảng sạch và X."),
}


def decision_table(columns, out, selected=None):
    fields = {}
    text = (OUTPUT / "quyet_dinh_46_cot.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("| `"):
            col, internal, reason, _, coding, _, vy, son, other = (
                part.strip() for part in line.strip("|").split("|")
            )
            fields[col.strip("`")] = (internal, reason, coding, vy, son, other)
    require(set(fields) == set(columns) and len(fields) == 46, "Không khớp bảng nguồn 46 cột.")
    head = [
        "# Quyết định giữ và sử dụng 46 trường ClinVar", "",
        "Nhóm **Giữ theo đề xuất** gồm các trường kế hoạch yêu cầu bảo toàn trong bảng sạch, kể cả khi chỉ dùng "
        "để tạo dẫn xuất. **Ứng viên** gồm các trường có cách biểu diễn cần đối chiếu hoặc đang chờ nguồn. "
        "**Truy vết hoặc kiểm tra** gồm khóa, nhãn và các trường nằm ngoài X. "
        "Nguồn từng đề xuất phản ánh tài liệu cá nhân Week 1.", "",
        "| Cột gốc | Vai trò | Nguồn đề xuất | Bảng sạch / truy vết | Xử lý thiếu | Trạng thái trong X | Lý do |",
        "|---|---|---|---|---|---|---|",
    ]
    for col in columns:
        internal, original_reason, coding, vy, son, other = fields[col]
        role = "Giữ theo đề xuất" if col in REQUIRED else "Ứng viên" if col in TRIAL else "Truy vết hoặc kiểm tra"
        retain, missing, reason = RULES.get(
            col, ("Chỉ tra CSV nguồn", "Theo CSV nguồn, không nội suy", original_reason),
        )
        state = f"Mốc: {coding}"
        if selected:
            usage = []
            for label, schema in selected.items():
                aliases = ({"transcript_region"} if col in {"EXON", "INTRON"} else
                           {"has_cDNA_without_CDS"} if col in {"cDNA_position", "CDS_position"} else set())
                active = [name for name in schema if name in aliases | {col}
                          or name.startswith((f"{col}_", f"has_{col}", f"zero_{col}", f"log_{col}"))]
                short = {"LogisticRegression": "LR", "RandomForest": "RF"}[label]
                usage.append(f"{short}: {', '.join(active) or 'ngoài X'}")
            state = " · ".join(usage)
        source = " · ".join((internal, vy, son, other))
        safe = [value.replace("|", "\\|") for value in (f"`{col}`", role, source, retain, missing, state, reason)]
        head.append("| " + " | ".join(safe) + " |")
    head.extend(["", "[Nguồn chi tiết của các thành viên](../feature_study/KET_QUA.md) · "
                 "[Báo cáo Vy](https://drive.google.com/file/d/13Q5P43aJcsYW8344dQCX7fhGwaAVz0j7/view) · "
                 "[Notebook Vy](https://drive.google.com/file/d/1nZopPo8BR4W1DIb6a7THnxnYoIoA1RDS/view)"])
    (out / "QUYET_DINH_46_COT.md").write_text("\n".join(head) + "\n", encoding="utf-8")


def dictionary(clean, out):
    rows = [{"column": "source_row", "type": "int64", "source": "CSV nguồn",
             "meaning": "Số dòng dữ liệu bắt đầu từ 1 sau tiêu đề. Khóa căn hàng, nằm ngoài X."}]
    for col in clean:
        if col.endswith("_raw") and col[:-4] in KEEP_RAW:
            source = col[:-4]
            meaning = "Nội dung ô nguồn, gồm mã hoặc ký hiệu thiếu nếu có."
        elif col in RULES:
            source = col
            meaning = RULES[col][0] + ". " + RULES[col][1] + "."
        elif col.startswith(("zero_", "log_")):
            source = col.split("_", 1)[1]
            meaning = ("1 nếu giá trị hợp lệ đúng bằng 0, 0 nếu không. Cờ không xác định thiếu của nguồn."
                       if col.startswith("zero_") else "log10(1 + 10^6 × AF). Giá trị 0 giữ bằng 0, AF lỗi/thiếu để trống.")
        elif col.endswith("_status"):
            source = col.removesuffix("_status")
            meaning = ("Trạng thái trước nội suy. Các nhóm xuất hiện được thống kê trong audit_train.csv. "
                       "Trạng thái tọa độ phân biệt missing, unknown, single, range, partial và invalid.")
        elif col.endswith(("_start", "_end", "_span", "_index", "_total", "_fraction")):
            source, part = col.rsplit("_", 1)
            meaning = {
                "start": "Mút đầu hoặc tọa độ đơn. Ô trống khi mút đầu chưa biết.",
                "end": "Mút cuối hoặc tọa độ đơn. Ô trống khi mút cuối chưa biết.",
                "span": "Độ dài bao gồm hai mút. Ô trống khi thiếu mút hoặc cú pháp sai.",
                "index": "Thứ tự từ 1 đến tổng số. Ô trống nếu không có phân số hợp lệ.",
                "total": "Tổng số trên transcript. Ô trống nếu không có phân số hợp lệ.",
                "fraction": "Thứ tự chia tổng số. Ô trống nếu không có phân số hợp lệ.",
            }[part]
        elif col.startswith("has_") and col != "has_cDNA_without_CDS":
            source = col[4:]
            meaning = "1 có chú giải nguồn hợp lệ, 0 không có. Cờ BLOSUM62 không tính điểm khôi phục."
        else:
            source, meaning = {
                "ref_length": ("REF", "Số ký tự alen tham chiếu."),
                "alt_length": ("ALT", "Số ký tự ALT đơn. Để trống nếu có nhiều alen."),
                "alt_minus_ref": ("REF, ALT", "Độ dài ALT đơn trừ REF. Để trống nếu có nhiều alen."),
                "multiple_alt": ("ALT", "1 nếu ALT gồm nhiều alen, 0 nếu ALT đơn."),
                "transcript_region": ("EXON, INTRON", "exon, intron, exon_and_intron hoặc unresolved."),
                "has_cDNA_without_CDS": ("cDNA_position, CDS_position", "1 khi nguồn có cDNA và thiếu CDS, 0 cho các trường hợp còn lại."),
                "SIFT_call": ("SIFT", "Kết luận sau khi tách hậu tố low_confidence. Thiếu giữ trống."),
                "SIFT_low_confidence": ("SIFT", "1 có hậu tố low_confidence, 0 có kết luận không mang hậu tố. Thiếu giữ trống."),
                "BLOSUM62_score": ("BLOSUM62, Amino_acids, Consequence", "Điểm quan sát, hoặc 0 đã xác minh. Ô trống khi chưa có điểm."),
                "BLOSUM62_source": ("BLOSUM62, Amino_acids, Consequence", "observed, recovered hoặc missing."),
                "Consequence_terms": ("Consequence", "Các hệ quả duy nhất đã sắp xếp, phân tách bằng &. Ô trống khi không có từ."),
                "IMPACT_rank": ("IMPACT", "MODIFIER=0, LOW=1, MODERATE=2, HIGH=3. Thiếu/mức lạ giữ trống."),
                "ORIGIN_flags": ("ORIGIN", "Các bit nguồn, cách nhau bằng |. Dùng kiểm tra dữ liệu."),
                "CLNDN_unknown_only": ("CLNDN", "1 khi mọi mục đều là not_specified/not_provided, 0 cho các trạng thái còn lại."),
            }[col]
        rows.append(dict(column=col, type=str(clean[col].dtype), source=source, meaning=meaning))
    pd.DataFrame(rows).to_csv(out / "column_dictionary.csv", index=False, encoding="utf-8-sig")


def audit(raw, clean, out):
    keys = [col for col in clean if col.endswith("_status")] + ["transcript_region", "BLOSUM62_source"]
    rows = [(col, str(state), int(n)) for col in keys for state, n in clean[col].value_counts(dropna=False).items()]
    syntax_invalid = {col: int(clean[f"{col}_status"].eq("invalid").sum())
                      for col in ("EXON", "INTRON", *POSITIONS, "cDNA_position")}
    rows.extend((f"{col}_status", "invalid", 0) for col, n in syntax_invalid.items() if n == 0)
    missense = raw.Consequence.map(lambda value: "missense_variant" in _terms(value))
    missing = int((missense & clean.BLOSUM62.isna()).sum())
    recovered = int(clean.BLOSUM62_source.eq("recovered").sum())
    cdna_extra = int(clean.has_cDNA_without_CDS.sum())
    for subgroup, mask in [("missense", missense), ("non_missense", ~missense)]:
        for col in ("SIFT", "PolyPhen", "BLOSUM62", "Amino_acids"):
            rows.append((col, f"observed_{subgroup}", int(clean.loc[mask, f"has_{col}"].sum())))
    rows.extend([("BLOSUM62", "missing_missense", missing), ("BLOSUM62", "recovered_missense", recovered),
                 ("BLOSUM62", "observed_negative", int(clean.BLOSUM62.lt(0).sum())),
                 ("cDNA_position", "observed_without_CDS", cdna_extra)])
    utr = {}
    for term in ("5_prime_UTR_variant", "3_prime_UTR_variant"):
        mask = raw.Consequence.map(lambda value: term in _terms(value))
        stats = (int(mask.sum()), int((mask & raw.cDNA_position.notna()).sum()),
                 int((mask & clean.has_cDNA_without_CDS.eq(1)).sum()))
        utr[term] = stats
        rows.extend((term, q, n) for q, n in zip(("n", "with_cDNA", "with_cDNA_without_CDS"), stats))
    pd.DataFrame(rows, columns=["column_or_group", "state", "n"]).to_csv(out / "audit_train.csv", index=False)
    origin = clean.groupby(["ORIGIN_raw", "ORIGIN_flags", "ORIGIN_status"], dropna=False).size()
    origin.rename("n").to_csv(out / "origin_train.csv")
    missing_corr = raw[list(DRIVE_MISSING)].isna().astype("int8").corr()
    missing_corr.to_csv(out / "missing_indicator_corr_train.csv")
    region = clean.transcript_region.value_counts().to_dict()
    return {"region": region, "missing": missing, "recovered": recovered, "cdna": cdna_extra,
            "utr": utr, "missing_corr": missing_corr, "invalid": syntax_invalid}
