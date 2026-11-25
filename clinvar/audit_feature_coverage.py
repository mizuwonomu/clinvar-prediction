"""Khảo sát dữ liệu train trước khi chạy so sánh mô hình."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consequence_encoder import _terms
from feature_study import COUNTS, MOTIFS, POSITIONS, PROTEIN, SCORES, number, position_features, recover_blosum
from preprocess import AF, require


DECISIONS = (
    (["CHROM", "POS"], "baseline_preprocessing_report.md:40", "Khóa nhóm và truy vết biến thể.", "Khóa", "Loại khỏi X"),
    (["REF", "ALT"], "baseline_preprocessing_report.md:40; kế hoạch §3.2", "Chuỗi gốc định danh. Độ dài có thể biểu thị cấu trúc alen.",
     "Khóa", "Thử độ dài, chênh lệch và cờ ALT nhiều alen"),
    (AF, "baseline_preprocessing_report.md:41", "Độ hiếm từ ba nguồn. Số 0 bao gồm giá trị thiếu đã được nguồn điền.",
     "Số, trung vị; chuẩn hóa cho LR", "Giữ baseline"),
    (["CADD_PHRED"], "baseline_preprocessing_report.md:42", "Thứ hạng tác động theo CADD.",
     "Số, trung vị; chuẩn hóa cho LR", "Giữ baseline; thử thay bằng CADD_RAW"),
    (["CLNVC"], "baseline_preprocessing_report.md:43", "Loại biến thể trong bộ nguồn.",
     "One-hot", "Giữ baseline; thời điểm ngoài bộ nguồn chưa xác nhận"),
    (["Consequence"], "baseline_preprocessing_report.md:44", "Hệ quả VEP trên transcript.",
     "One-hot theo tổ hợp hoặc multi-hot theo hệ quả", "Cố định one-hot cho các nhóm ứng viên"),
    (["IMPACT", "BIOTYPE"], "baseline_preprocessing_report.md:44", "Mức độ tác động VEP và loại transcript.",
     "One-hot", "Giữ baseline"),
    (["CLNSIGINCL"], "baseline_preprocessing_report.md:45; kế hoạch §2", "Chứa diễn giải lâm sàng gần mục tiêu.",
     "Loại", "Loại khỏi X"),
    (["CLNDISDB", "CLNDISDBINCL", "CLNDN", "CLNDNINCL"], "baseline_preprocessing_report.md:45; kế hoạch §2",
     "Mã/tên bệnh và quan hệ hồ sơ. Chưa xác nhận khả dụng trước thời điểm dự đoán.",
     "Loại", "Loại khỏi X"),
    (["ORIGIN", "SSR"], "baseline_preprocessing_report.md:45", "Nguồn gốc và cờ nghi ngờ phản ánh cách tổng hợp hồ sơ.",
     "Loại", "Loại khỏi X"),
    (["CLNHGVS", "CLNVI", "Feature", "Allele"], "baseline_preprocessing_report.md:46",
     "Định danh hoặc chuỗi có thể giúp ghi nhớ biến thể hay transcript.", "Loại", "Loại khỏi X"),
    (["SYMBOL"], "baseline_preprocessing_report.md:46; điều kiện ở kế hoạch §3", "Tần suất xuất hiện có thể phụ thuộc gene đã gặp.",
     "Loại", "Thử riêng tần suất gene học trong fold và kiểm tra cả fold theo gene"),
    (["MC"], "baseline_preprocessing_report.md:47; kế hoạch §3.3", "Chú giải SO có thể trùng một phần hệ quả VEP.",
     "Loại", "Thử multi-hot từng hệ quả SO"),
    (["CADD_RAW"], "baseline_preprocessing_report.md:48; kế hoạch §3.3", "Hai thang của cùng nguồn CADD.",
     "Loại", "Thử thay CADD_PHRED; đọc cả điểm âm"),
    (["LoFtool"], "baseline_preprocessing_report.md:49; kế hoạch §3.1", "Độ dung nạp mất chức năng theo gene.",
     "Loại", "Thử cờ hiện diện rồi thêm điểm số, điền trung vị trong fold"),
    (["BLOSUM62"], "baseline_preprocessing_report.md:49; kế hoạch §3.1", "Điểm thay thế axit amin, có giá trị âm.",
     "Loại", "Thử cờ hiện diện rồi thêm điểm số, giữ điểm âm"),
    (["SIFT", "PolyPhen"], "baseline_preprocessing_report.md:49; kế hoạch §3.1", "Mức dự đoán tác động protein, độ phủ phụ thuộc hệ quả.",
     "Loại", "Thử cờ hiện diện rồi thêm one-hot, có mức thiếu riêng"),
    (["EXON", "INTRON"], "baseline_preprocessing_report.md:50; kế hoạch §3.2", "Thứ tự và tổng số exon/intron trên transcript.",
     "Loại", "Thử chỉ số, tổng số và tỷ lệ chỉ số/tổng"),
    (list(POSITIONS), "baseline_preprocessing_report.md:50; kế hoạch §3.2", "Tọa độ CDS/protein đơn hoặc khoảng. CSV thiếu mẫu số độ dài.",
     "Loại", "Thử đầu khoảng và độ dài khoảng"),
    (["Feature_type", "STRAND", "BAM_EDIT"], "baseline_preprocessing_report.md:50", "Loại, chiều hoặc trạng thái hiệu chỉnh transcript.",
     "Loại", "Ngoài các nhóm thử đã định"),
    (["cDNA_position", "Amino_acids", "Codons"], "baseline_preprocessing_report.md:50", "Cần quy tắc riêng theo cú pháp và ngữ cảnh.",
     "Loại", "Ngoài các nhóm thử đã định"),
    (["DISTANCE", *MOTIFS], "baseline_preprocessing_report.md:50; kế hoạch §3.3", "Chú giải khoảng cách hoặc motif rất thưa.",
     "Loại", "Đo độ phủ từng fold, loại khỏi X"),
    (["CLASS"], "baseline_preprocessing_report.md:39; kế hoạch §2", "Nhãn mục tiêu từ sự hiện diện của CLNSIGCONF.",
     "Lưu nhãn", "Loại khỏi X"),
)
NUMERIC_AUDIT = [*AF, "CADD_PHRED", "CADD_RAW", "LoFtool", "BLOSUM62",
                 "DISTANCE", "STRAND", "MOTIF_POS", "MOTIF_SCORE_CHANGE", "SSR", "POS", "ORIGIN"]
COVERAGE_COLUMNS = [*SCORES, "MC", "EXON", "INTRON", "cDNA_position", *POSITIONS,
                    "Amino_acids", "Codons", "BAM_EDIT", "DISTANCE", *MOTIFS, "CADD_PHRED", "CADD_RAW"]
VY_PROPOSALS = (
    (["CLNHGVS", "CLNVI"], "tr. 8, 12 · Loại mã định danh"),
    (["SSR", "DISTANCE", "CLNSIGINCL", "CLNDNINCL", "CLNDISDBINCL", *MOTIFS],
     "tr. 10, 12 · Loại trường gần như trống"),
    (["Feature", "CADD_RAW", "MC", "Allele", "CLNDISDB", "CDS_position", "cDNA_position", "Codons"],
     "tr. 12 · Loại vì trùng. Tr. 11 có danh sách giữ chung cho vị trí và thay đổi phân tử"),
    (["POS", "STRAND", "Feature_type", "BIOTYPE", "BAM_EDIT"], "tr. 12 · Loại vì ít giá trị"),
    (["CHROM", "CLNVC", "ORIGIN", "IMPACT"], "tr. 12 · Giữ và mã hóa"),
    (AF, "tr. 12 · Biến đổi log10"),
    (["REF", "ALT"], "tr. 12 · Tính chênh lệch độ dài ALT − REF"),
    (["Consequence"], "tr. 12 · Tách hệ quả, gộp mức hiếm"),
    (["SYMBOL"], "tr. 12 · Mã hóa bằng tần suất gene"),
    (["CLNDN"], "tr. 12 · Tạo cờ bệnh chưa rõ"),
    (["EXON", "INTRON"], "tr. 10–12 · Gộp EXON và INTRON thành cột vùng"),
    (["BLOSUM62"], "tr. 5, 11–12 và notebook cell 12–14 · Giữ, kiểm tra điểm 0 ở missense"),
    (["SIFT", "PolyPhen"], "tr. 11–12 · Giữ, mã hóa trạng thái không áp dụng khi có căn cứ"),
    (["CADD_PHRED", "LoFtool"], "tr. 11–12 · Điền trung vị"),
    (["Protein_position", "Amino_acids"], "tr. 11 · Giữ nhóm vị trí và thay đổi phân tử"),
)
SON_PROPOSALS = (
    (["CHROM"], 25, "Giữ dạng chuỗi"),
    (["POS", "Feature_type", "BIOTYPE"], 25, "Loại"),
    (["Feature"], 25, "Tách thông tin rồi cân nhắc loại vì trùng SYMBOL"),
    (["STRAND"], 25, "Đọc như mã phân loại"),
    (["EXON", "INTRON", "CDS_position"], 25, "Tách cú pháp và đánh dấu thiếu có cấu trúc"),
    (["cDNA_position"], 25, "Tách để tạo vùng rồi loại"),
    (["Protein_position"], 25, "Đề xuất loại vì trùng CDS, mức trùng còn cần kiểm"),
    (["SYMBOL"], 25, "Mã hóa và chia theo gene"),
    (["REF", "ALT"], 26, "Tách độ dài alen"),
    (["Allele"], 26, "Đề xuất loại vì trùng ALT, giữ ý nghĩa dấu - khi khảo sát"),
    (["CLNVC"], 26, "Giữ và mã hóa"),
    (["Amino_acids", "Codons"], 26, "Tách thành biến phụ và đánh dấu thiếu có cấu trúc"),
    (["MC"], 26, "Tách hệ quả, kiểm tra trùng rồi cân nhắc loại"),
    (["Consequence"], 26, "Tách và mã hóa"),
    (["IMPACT"], 26, "Mã hóa thứ bậc"),
    (AF, 26, "Biến đổi log, giữ cờ bằng 0"),
    (["SIFT", "PolyPhen"], 27, "Mã hóa và đánh dấu thiếu có cấu trúc"),
    (["BLOSUM62"], 27, "Đọc như thứ bậc, khôi phục điểm 0"),
    (["CADD_PHRED", "LoFtool"], 27, "Chuẩn hóa và đánh dấu thiếu"),
    (["CADD_RAW"], 27, "Loại vì trùng CADD_PHRED"),
    (["CLNDN", "CLNDISDB"], 27, "Tách và đánh dấu thiếu ẩn, xem nguy cơ rò rỉ"),
    (["CLNDNINCL", "CLNDISDBINCL", "CLNSIGINCL"], 27, "Loại vì gần như trống"),
    (["CLNHGVS", "SSR", "DISTANCE", *MOTIFS], 28, "Loại"),
    (["CLNVI"], 28, "Tách nguồn, xem nguy cơ rò rỉ và nguyên nhân thiếu"),
    (["ORIGIN"], 28, "Giải mã bitmask"),
    (["BAM_EDIT"], 28, "Tìm hiểu thêm ý nghĩa"),
)
OTHER_PROPOSALS = (
    (["Feature", "CLNDISDB"], "Thảo · notebook cell 28–30 · Loại khỏi df_model"),
    (["SYMBOL"], "Thảo · notebook cell 30 · Đại diện gene; Dương · §4.2 · Gộp mức hiếm"),
    (["EXON", "INTRON"], "Toàn · phần VEP · Nêu là tín hiệu; Thảo · tr. 10 · Thiếu theo vùng; "
     "Dương · §4.2 · Giữ cờ, tách cú pháp hợp lệ"),
    (["SIFT", "PolyPhen", "BLOSUM62"], "Toàn · phần VEP · Nêu nhóm điểm dự đoán; Thảo · tr. 11, cell 51–52 "
     "· Đo thiếu theo missense; Dương · §4.2 · Thử điểm kèm cờ thiếu"),
    (["CADD_PHRED", "CADD_RAW", "LoFtool"], "Toàn · phần VEP · Nêu nhóm điểm dự đoán; "
     "Dương · §4.2 · Thử điểm kèm cờ thiếu"),
    (["CHROM", "ORIGIN", "SSR", "STRAND"], "Dương · §4.2 · Đọc như phân loại hoặc mã cờ"),
    (AF, "Thảo · tr. 9–10 · Thống kê giá trị 0; Dương · §4.2 · Giữ 0 và thêm cờ"),
    (["CLNDN", "Allele"], "Thảo · tr. 9–10 · Kiểm tra thiếu ẩn; Dương · §4.2 · "
     "Chuẩn hóa bệnh chưa rõ, giữ dấu - của Allele"),
    (["CLNSIGINCL", "CLNVI", "CLNHGVS"], "Dương · §2 · Loại trường rò rỉ hoặc mã thô khỏi baseline"),
    (list(MOTIFS), "Dương · §4.2 · Loại cột motif gần như trống"),
    (["CLASS"], "Dương · §1, §4.3 · Dùng làm nhãn phân tầng"),
)


def decision_table(raw, observed, out):
    lookup = {col: info for cols, *info in DECISIONS for col in cols}
    require(set(lookup) == set(raw) and len(lookup) == 46, "Bảng cần đủ 46 cột gốc.")
    vy = {col: proposal for cols, proposal in VY_PROPOSALS for col in cols}
    son = {col: f"Sơn · Bảng 10 tr. {page} · {proposal}"
           for cols, page, proposal in SON_PROPOSALS for col in cols}
    others = {col: proposal for cols, proposal in OTHER_PROPOSALS for col in cols}
    lines = [
        "# Bảng quyết định cho 46 cột ClinVar", "",
        f"Độ phủ tính trên {len(raw):,} dòng train trước khi điền thiếu. Các cột số yêu cầu giá trị hữu hạn.",
        "Dấu **Giữ dù thiếu nhiều** ghi các đề xuất cụ thể trong báo cáo Week 1. "
        "Nguồn và điểm bất đồng được tổng hợp trong báo cáo kết quả.", "",
        "| Cột | Nguồn trong workspace | Lý do | Có dữ liệu trên train | Mã hóa hiện tại | Quyết định thử | Đề xuất của Vy | Đề xuất của Sơn | Ý kiến khác trong Week 1 |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    for col in raw:
        source, reason, current, decision = lookup[col]
        count = int(observed[col].sum())
        priority = " · **Giữ dù thiếu nhiều**" if col in (*PROTEIN, "INTRON") else ""
        proposal = vy.get(col, "tr. 7 · CLASS là nhãn")
        lines.append(f"| `{col}` | {source} | {reason} | {count:,} ({count / len(raw):.1%}) | "
                     f"{current} | {decision} | Vy · {proposal}{priority} | "
                     f"{son.get(col, 'Sơn · §6 · CLASS là nhãn')} | "
                     f"{others.get(col, 'Chưa thấy đề xuất riêng cho cột này')} |")
    (out / "quyet_dinh_46_cot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_heatmap(values, counts, columns, labels, out_path, *, correlation):
    vmin = -1 if correlation else 0
    cmap = ("RdBu_r" if correlation else "Blues")
    fig, ax = plt.subplots(figsize=(max(12, len(columns) * 0.8), max(8, len(labels) * 0.51)),
                           layout="constrained")
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("#e4e7eb")
    img = ax.imshow(values, aspect="auto", cmap=colors, vmin=vmin, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i, j in np.ndindex(values.shape):
        val, n = values[i, j], counts[i, j]
        formatted = (f"{val:.2f}" if correlation else f"{val:.0%}") if np.isfinite(val) else "—"
        mark = "*" if n < 30 else ""
        foreground = "white" if np.isfinite(val) and (abs(val) > 0.72) else "#19232d"
        ax.text(j, i, f"{formatted}\n{n:,}{mark}", ha="center", va="center",
                fontsize=6 if correlation else 5.8, color=foreground)
    title = "Tương quan Spearman trên train" if correlation else "Độ phủ chú giải theo hệ quả trên train"
    subtitle = ("Mỗi ô ghi ρ và số cặp cùng quan sát. * dưới 30 cặp. — hệ số không xác định."
                if correlation else
                "Mỗi ô ghi tỷ lệ và số dòng có dữ liệu. * dưới 30 dòng có chú giải. Một dòng có thể thuộc nhiều hệ quả.")
    ax.set_title(title + "\n" + subtitle, loc="left", pad=12, fontsize=12)
    fig.colorbar(img, ax=ax, shrink=0.70, label="Spearman ρ" if correlation else "Tỷ lệ có dữ liệu")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def audit(raw, baseline, y, ids, out):
    numeric = raw[NUMERIC_AUDIT].apply(number)
    for col in AF + ["CADD_PHRED"]:
        numeric[col] = baseline[col]
    observed = raw.notna()
    observed[numeric.columns] = numeric.notna()
    decision_table(raw, observed, out)
    rho = numeric.corr(method="spearman", min_periods=3)
    present = numeric.notna().astype(np.int64)
    pairs = present.T @ present
    rho.to_csv(out / "spearman.csv")
    pairs.to_csv(out / "spearman_pair_counts.csv")
    draw_heatmap(rho.to_numpy(), pairs.to_numpy(), rho.columns, rho.index,
                 out / "spearman.png", correlation=True)

    terms = raw.Consequence.map(_terms)
    term_counts = terms.explode().value_counts()
    masks = {term: terms.map(lambda parts: term in parts) for term in term_counts.index}
    if (~terms.astype(bool)).any():
        masks["__MISSING__"] = ~terms.astype(bool)
    _, valid_positions = position_features(raw)
    coverage = []
    categories = []
    categ_cols = [*PROTEIN[:2], "MC", "SYMBOL", "CLNVC", "Consequence", "IMPACT", "BIOTYPE",
                  "Feature_type", "STRAND", "BAM_EDIT", "HIGH_INF_POS"]
    scopes = {"all": pd.Series(True, index=raw.index), **masks}
    for group, mask in scopes.items():
        for label in ("all", 0, 1):
            use = mask if label == "all" else mask & y.eq(label)
            for col, count in observed.loc[use].sum().items():
                n = int(use.sum())
                coverage.append(dict(consequence=group, label=label, column=col, n=n, observed=count,
                                     coverage=count / n if n else np.nan))
            for col in categ_cols:
                values = raw.loc[use, col].fillna("__MISSING__").value_counts()
                categories.extend(dict(consequence=group, label=label, column=col, level=level, n=n)
                                  for level, n in values.items())
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(out / "coverage_by_consequence_and_class.csv", index=False)
    pd.DataFrame(categories).to_csv(out / "categorical_levels.csv", index=False)
    heat = coverage.query("label == 'all' and consequence != 'all'")
    levels = [f"{group} (n={mask.sum():,})" for group, mask in masks.items()]
    fractions = heat.pivot(index="consequence", columns="column", values="coverage").loc[list(masks), COVERAGE_COLUMNS]
    counts = heat.pivot(index="consequence", columns="column", values="observed").loc[list(masks), COVERAGE_COLUMNS]
    draw_heatmap(fractions.to_numpy(), counts.to_numpy(), COVERAGE_COLUMNS, levels,
                 out / "coverage.png", correlation=False)

    comparison = raw.MC.map(lambda value: set() if pd.isna(value) else {
        term.split("|", 1)[-1] for term in value.split(",")
    })
    # Chuẩn hóa tên đồng nghĩa trước khi đối chiếu hai nguồn.
    comparison = comparison.map(lambda s: {"stop_gained" if t == "nonsense" else t for t in s})
    both = terms.astype(bool) & comparison.astype(bool)
    equal = comparison[both].eq(terms[both])
    common = [bool(a & b) for a, b in zip(comparison[both], terms[both])]
    diagnostics = [
        "# Kiểm tra cú pháp và chú giải trên train", "",
        f"- MC và Consequence cùng có dữ liệu ở {both.sum():,} dòng. Tập hệ quả trùng hoàn toàn "
        f"ở {equal.sum():,} dòng và giao nhau ở {sum(common):,} dòng sau khi quy `nonsense` về `stop_gained`.",
        f"- ALT có dấu phẩy ở {ids.ALT.str.contains(',', regex=False).sum():,} dòng. Các dòng này bật cờ ALT nhiều alen "
        "và bỏ độ dài ALT lẫn chênh lệch. Độ dài REF vẫn xác định.",
        f"- BLOSUM62 có {(numeric.BLOSUM62 < 0).sum():,} điểm âm hợp lệ.", "",
        f"- Khôi phục được {recover_blosum(raw, numeric.BLOSUM62).sum():,} điểm BLOSUM62 bằng 0 "
        "trên missense có cặp axit amin đơn khớp điểm 0 theo ma trận NCBI. "
        "Heatmap và cờ BLOSUM62 giữ trạng thái ghi nhận gốc của CSV. "
        "[Ma trận NCBI](https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62)", "",
        "| Trường | Có dữ liệu | Phân tích được | Giá trị lỗi/không rõ | Ví dụ đơn hoặc phân số | Ví dụ khoảng | Ví dụ lỗi |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for col in [*COUNTS, *POSITIONS]:
        value = raw[col]
        good = valid_positions[col]
        invalid = value.notna() & ~good
        ranges = good & value.str.contains("-", regex=False, na=False)
        examples = [", ".join(value[mask].drop_duplicates().head(3)) or "—"
                    for mask in (good & ~ranges, ranges, invalid)]
        diagnostics.append(f"| {col} | {value.notna().sum():,} | {good.sum():,} | {invalid.sum():,} | "
                           + " | ".join(examples) + " |")
    diagnostics.extend([
        "", "Tỷ lệ EXON/INTRON dùng mẫu số tổng ghi trong chính trường đó. CDS/protein chỉ cho phép "
        "tọa độ dương đơn hoặc khoảng tăng, và không tính vị trí tương đối.", "",
        "Cờ hiện diện đo việc trường có giá trị trong CSV. Với SIFT/PolyPhen/BLOSUM62, các hệ quả "
        "không ghi nhận thay thế axit amin là ngữ cảnh dự kiến ít áp dụng. Trường hợp thiếu trên "
        "missense và các ngữ cảnh chưa rõ vẫn được ghi là thiếu, chưa xác định nguyên nhân. "
        "Giá trị PolyPhen `unknown` là một mức được ghi nhận.", "",
        "Các điểm VEP, CADD và tần số quần thể được dùng theo snapshot của bộ dữ liệu này. Việc chúng "
        "khả dụng ở thời điểm diễn giải ban đầu chưa được kiểm chứng. Hồ sơ bệnh và ý nghĩa lâm sàng "
        "thành phần vẫn ở ngoài đầu vào.", "",
    ])
    (out / "chan_doan_du_lieu.md").write_text("\n".join(diagnostics), encoding="utf-8")
