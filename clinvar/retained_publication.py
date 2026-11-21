"""Sinh báo cáo công bố từ bảng sạch và kết quả ClinVar đã lưu."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from feature_study import OUTPUT, ROOT
from preprocess import MISSING, require
from retained_features import EXTRA_POSITIONS, PROTEIN_VALUES, parse_coordinate


PAGE = "<!-- pagebreak -->"
LR, RF = "LogisticRegression", "RandomForest"
LABEL = {LR: "Logistic Regression", RF: "Random Forest"}
GROUPS = [
    ("Điểm chú giải", "CADD_RAW LoFtool".split()),
    ("Vị trí exon và intron", "EXON_index EXON_total EXON_fraction INTRON_index INTRON_total INTRON_fraction transcript_region".split()),
    ("Vị trí CDS và protein", "CDS_position_start CDS_position_span Protein_position_start Protein_position_span".split()),
    ("Loại biến thể và hệ quả", "CLNVC IMPACT BIOTYPE Consequence".split()),
    ("Có chú giải trong CSV", "has_LoFtool has_SIFT has_PolyPhen has_BLOSUM62".split()),
]
AF_LR = "log_AF_ESP log_AF_EXAC log_AF_TGP zero_AF_ESP zero_AF_EXAC zero_AF_TGP".split()
AF_RF = "AF_ESP AF_EXAC AF_TGP".split()
TRIAL_NAMES = {
    "af_zero": "AF gốc + cờ 0", "af_log": "Log AF", "af_log+af_zero": "Log AF + cờ 0",
    "cadd_present": "Cờ có CADD_RAW", "protein_confidence": "Giá trị protein",
    "positions": "Tọa độ bổ sung", "consequence_rare": "Hệ quả nhóm hiếm", "impact_order": "IMPACT thứ bậc",
    "chrom": "CHROM one-hot", "af_log+af_zero+consequence_rare": "Log AF + cờ 0 + hệ quả nhóm hiếm",
}


def integer(value):
    return f"{int(value):,}".replace(",", ".")


def decimal(value, places=4, signed=False):
    return f"{value:{'+' if signed else ''}.{places}f}".replace(".", ",").replace("-", "−")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def utr_counts(out, audit):
    cols = ["source_row", "cDNA_position_raw", "CDS_position_raw", "Consequence_terms"]
    x = pd.read_csv(out / "clean_train.csv", usecols=cols, keep_default_na=False)
    labels = pd.read_csv(out / "y_train.csv")
    require(x.source_row.is_unique and x.source_row.equals(labels.source_row), "Bảng sạch lệch mã dòng train.")
    cdna = ~x.cDNA_position_raw.str.strip().str.upper().isin(MISSING)
    cds = ~x.CDS_position_raw.str.strip().str.upper().isin(MISSING)
    target = cdna & ~cds
    groups = [x.Consequence_terms.str.split("&").map(lambda terms: term in terms)
              for term in ("5_prime_UTR_variant", "3_prime_UTR_variant")]
    records = [("Toàn bộ train", len(x), int(target.sum()))]
    for label, term, mask in zip(("5′ UTR", "3′ UTR"), ("5_prime_UTR_variant", "3_prime_UTR_variant"), groups):
        record = (label, int(mask.sum()), int((mask & target).sum()))
        require(record[1:] == (audit[term, "n"], audit[term, "with_cDNA_without_CDS"]), "Số đếm UTR lệch audit.")
        records.append(record)
    require(records[0][2] == audit["cDNA_position", "observed_without_CDS"], "Tổng cDNA thiếu CDS lệch audit.")
    both = groups[0] & groups[1]
    outside = target & ~(groups[0] | groups[1])
    return records, int(both.sum()), int((both & target).sum()), int(outside.sum())


def figures(out, missing, rho, counts, audit):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(data, labels, rows, phrases, name, coverage=False):
        fig, ax = plt.subplots(figsize=(6.45, 1.85 if coverage else 3.40), layout="constrained")
        hm = ax.imshow(data, aspect="auto", cmap="Blues" if coverage else "RdBu_r", vmin=0 if coverage else -1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=0 if coverage else 27,
                      ha="center" if coverage else "right", fontsize=8)
        ax.set_yticks(range(len(rows)), rows, fontsize=8)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        for i, j in np.ndindex(data.shape):
            ax.text(j, i, phrases[i][j], va="center", ha="center", fontsize=8,
                    color="white" if abs(data[i, j]) > .72 else "#192832")
        color = fig.colorbar(hm, ax=ax, fraction=.045, pad=.03, shrink=.92)
        if coverage:
            color.set_ticks([0, .25, .5, .75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
        color.ax.tick_params(labelsize=8)
        fig.savefig(out / name, dpi=220, facecolor="white")
        plt.close(fig)

    a = ["EXON", "INTRON", "SIFT", "PolyPhen", "CADD_PHRED", "CADD_RAW"]
    values = missing.loc[a, a].to_numpy()
    draw(values, a, a, [[decimal(x, 3) for x in row] for row in values], "hinh_1_co_thieu.png")
    b = ["AF_ESP", "AF_EXAC", "AF_TGP", "CADD_PHRED", "CADD_RAW", "LoFtool", "BLOSUM62"]
    values = rho.loc[b, b].to_numpy()
    n = counts.loc[b, b].to_numpy()
    draw(values, b, b, [[f"{decimal(x,2)}\n{integer(k)}" for x,k in zip(row, nr)] for row,nr in zip(values,n)],
         "hinh_2_spearman.png")
    protein = ["SIFT", "PolyPhen", "BLOSUM62"]
    missense = sum(audit["BLOSUM62", x] for x in ("observed_missense", "missing_missense"))
    total = sum(audit["transcript_region", x] for x in ("exon", "intron", "exon_and_intron", "unresolved"))
    groups = [("missense", missense), ("non_missense", total - missense)]
    observed = np.array([[audit[col, "observed_" + group] for col in protein] for group, _ in groups])
    rates = observed / np.array([n for _, n in groups])[:, None]
    draw(rates, protein, [f"Missense (n={integer(groups[0][1])})", f"Ngoài missense (n={integer(groups[1][1])})"],
         [[f"{decimal(100 * value, 2)}%\n{integer(k)}" for value,k in zip(rate,row)] for rate,row in zip(rates,observed)],
         "hinh_3_do_phu.png", coverage=True)


def verdict(group, winning, min_gain, max_loss):
    if len(group) != 4 or group.folds_valid.ne(5).any():
        return "Chưa đủ phép đo"
    variant = group.loc["variant", "all"]
    failed = []
    if variant.delta_ap < min_gain:
        failed.append("Δ dưới ngưỡng")
    if variant.folds_up < 4:
        failed.append("dưới 4 fold tăng")
    if group.delta_ap.min() < -max_loss:
        failed.append("giảm quá giới hạn")
    return ", ".join(failed) if failed else "Đã chọn" if winning else "Đạt, xếp sau"


def endpoint_report(out, selected, add, tbl, page, source):
    from study_retained import MAX_LOSS, MIN_GAIN

    review = out / "endpoint_review"
    if not (review / "selection.json").exists():
        return
    decision = load_json(review / "selection.json")
    if any(decision[m]["baseline"] != selected[m]["configuration"] for m in LABEL):
        return
    result = pd.read_csv(review / "cv_summary.csv").set_index(["model", "config", "scheme", "subgroup"]).sort_index()
    partial = pd.read_csv(review / "partial_oof.csv")
    operating = pd.read_csv(review / "operating_metrics.csv").set_index("model")
    page("Thử riêng từng mút tọa độ")
    add("A là cấu hình cuối hiện tại của mỗi mô hình. B giữ start đã biết trong cột CDS_position_start hoặc "
        "Protein_position_start hiện có. C giữ cách xử lý của B và thêm hai cột số CDS_position_partial_end, "
        "Protein_position_partial_end cùng hai cột phân loại CDS_position_partial_endpoint, "
        "Protein_position_partial_endpoint. Các mức phân loại là start_only, end_only hoặc other. "
        "span của khoảng thiếu một mút tiếp tục để thiếu trước nội suy.")
    add("CDS và protein cùng có khoảng partial trên 35 biến thể train, gồm 8 mẫu dương. B khôi phục start trên 25 dòng. "
        "C có end quan sát trên 10 dòng và ghi trạng thái mút trên cả 35 dòng. So B với A để đo riêng việc giữ start. So C với B để đo bước bổ sung "
        "end và trạng thái. Cả ba chạy trên cùng fold, cùng tham số và cùng quy trình fit.")
    tbl(("Mô hình và nhánh", "AP biến thể", "Δ so với A", "Δ so với B", "AP theo gene"), [
        (f"{LABEL[m]} · {arm}", decimal(result.loc[(m, arm, "variant", "all"), "ap"]),
         decimal(result.loc[(m, arm, "variant", "all"), "delta_ap"], 5, True),
         "—" if arm in ("A", "B") else decimal(result.loc[(m, arm, "variant", "all"), "ap"]
                                                 - result.loc[(m, "B", "variant", "all"), "ap"], 5, True),
         decimal(result.loc[(m, arm, "gene", "all"), "ap"]))
        for m in LABEL for arm in ("A", "B", "C")
    ])
    add("AP lớp 1 bình quân 5 fold toàn tập train. So với A bằng tiêu chí ở phần trước. "
        + ". ".join(f"{LABEL[m]} chọn {decision[m]['selected_arm']}" for m in LABEL) + ". "
        + source(review / "cv_summary.csv", "AP và số fold tăng cho bốn tổ hợp") + " · "
        + source(review / "cv_folds.csv", "AP từng fold"))
    for m in LABEL:
        reasons = [f"Nhánh {arm} — {verdict(result.loc[(m, arm)], decision[m]['selected_arm'] == arm, MIN_GAIN, MAX_LOSS)}"
                   for arm in ("B", "C")]
        add(f"**{LABEL[m]}.** {'. '.join(reasons)}.")
    examples = []
    for m in LABEL:
        data = partial[partial.model.eq(m)]
        require(len(data) == 35 and data.CLASS.sum() == 8 and data.source_row.is_unique, "Bảng OOF partial sai phạm vi.")
        for old, new in (("A", "B"), ("B", "C")):
            delta = (data[f"variant_{new}"] - data[f"variant_{old}"]).abs()
            examples.append((f"{LABEL[m]} · {new} so với {old}", int(delta.gt(1e-12).sum()),
                             decimal(delta.mean(), 5), decimal(delta.max(), 5)))
    tbl(("So sánh trên 35 dòng", "Số xác suất đổi", "TB độ đổi tuyệt đối",
         "Độ đổi lớn nhất"), examples)
    add("Độ thay đổi tuyệt đối của xác suất ngoài fold theo biến thể. Mỗi so sánh có 35 dòng. "
        "Ngưỡng đếm thay đổi là 10⁻¹². Phân tích nhóm nhỏ mang tính thăm dò vì chỉ có 8 mẫu dương. "
        + source(review / "partial_oof.csv", "Xác suất theo từng dòng ở cả hai cách chia"))
    page("AP và ngưỡng quyết định lớp")
    add("AP lớp 1 dùng để chọn feature. Sau khi cố định cấu hình, chọn một ngưỡng cho mỗi mô hình bằng cách tối đa hóa "
        "F1 lớp 1 trên xác suất ngoài fold của toàn bộ train. Khi nhiều ngưỡng có cùng F1, lấy ngưỡng cao nhất. "
        "F1 trên tập xác suất dùng để chọn ngưỡng được ghi riêng.")
    tbl(("Mô hình", "AP CV", "Ngưỡng OOF", "F1 chọn ngưỡng", "Precision ngoài", "Recall ngoài", "F1 ngoài"), [
        (LABEL[m], *[decimal(operating.loc[m, col]) for col in
                     ("ap_mean", "final_threshold", "tuning_f1", "precision", "recall", "f1")])
        for m in LABEL
    ])
    add("AP là trung bình 5 fold theo biến thể. Precision, recall và F1 ngoài được tính bằng cách gộp dự đoán nhị phân "
        "của 52.150 dòng validation, gồm 13.147 mẫu dương. Với mỗi fold ngoài, chọn ngưỡng trên xác suất ngoài fold "
        "của bốn fold bên trong phần fit, rồi áp lên fold ngoài. Vì vậy ba cột hiệu năng ngoài dùng ngưỡng riêng "
        "từng fold, còn cột ngưỡng OOF là giá trị chốt từ toàn bộ train. "
        + source(review / "operating_metrics.csv", "Tổng hợp AP và F1") + " · "
        + source(review / "threshold_folds.csv", "Ngưỡng và hiệu năng từng fold"))
    add("Đây là đánh giá nội bộ với cấu hình đã cố định. Ngưỡng và kết quả test của bộ 24 cột lịch sử thuộc cấu hình "
        "lịch sử. Kết quả cho hai cấu hình hiện tại dùng train. "
        + source(ROOT / "study_endpoints.py", "Mã tái lập"))


def oof_report(out, add, tbl, page, source):
    review = out / "endpoint_review"
    if not (review / "cv_by_split_comparison.csv").exists():
        return
    errors = pd.read_csv(review / "error_by_subgroup.csv")
    comparison = pd.read_csv(review / "cv_by_split_comparison.csv").set_index(["model", "scheme"])
    sites = pd.read_csv(review / "site_cv_summary.csv").set_index("model")
    groupings = {
        "transcript_region": "Vùng transcript",
        "missense_protein": "Chú giải trong missense",
        "allele_frequency": "Tần số alen",
        "variant_type": "Loại CLNVC",
    }
    group_names = {
        "exon": "Chỉ exon", "intron": "Chỉ intron", "exon_and_intron": "Cả exon và intron",
        "unresolved": "Chưa xác định", "present": "Có chú giải protein",
        "missing": "Thiếu", "all_zero": "Cả ba AF bằng 0", "any_positive": "Có AF dương",
        "unclassified": "AF chưa phân loại",
    }

    def label(row):
        name = ("Thiếu cả ba chú giải protein" if row.grouping == "missense_protein" and row.group == "missing"
                else group_names.get(row.group, row.group))
        rate = decimal(100 * row.miss_rate, 1) + "%" if pd.notna(row.miss_rate) else "chưa xác định"
        limited = " †" if row.positive < 30 else ""
        return f"{name} — {integer(row.FN)}/{integer(row.positive)} ({rate}){limited}"

    page("Phân tích các biến thể bị bỏ sót")
    records = []
    for key, grouping in groupings.items():
        for model, name in LABEL.items():
            group = errors.query("model == @model and grouping == @key")
            largest_rate = group.sort_values(["miss_rate", "FN", "group"],
                                            ascending=[False, False, True]).iloc[0]
            largest_count = group.sort_values(["FN", "positive", "group"],
                                             ascending=[False, False, True]).iloc[0]
            records.append((grouping, name, label(largest_rate), label(largest_count)))
    tbl(("Phép phân nhóm", "Mô hình", "Tỷ lệ bỏ sót cao nhất", "Nhiều ca bỏ sót nhất"), records)
    add("Mỗi ô ghi FN/số mẫu CLASS=1 trong nhóm và tỷ lệ bỏ sót FN/(TP+FN). "
        "Dự đoán ngoài fold theo biến thể gồm 52.150 dòng mỗi mô hình, trong đó có 13.147 mẫu dương. "
        "Phép so sánh chú giải protein chỉ dùng 25.892 dòng train mang token missense_variant, "
        "và dùng cờ có điểm SIFT, PolyPhen hoặc BLOSUM62 trong CSV gốc. Các phép phân nhóm có thể chồng lấn, "
        "nên đọc riêng từng phép so sánh. Dấu † ghi nhóm có dưới 30 mẫu dương. "
        "Trong CSV, NA biểu thị chỉ số có mẫu số bằng 0. "
        + source(review / "error_by_subgroup.csv", "Ma trận nhầm lẫn theo nhóm") + " · "
        + source(review / "oof_predictions.csv", "Dự đoán ngoài fold từng biến thể"))

    page("Đánh giá trên vị trí nhiễm sắc thể mới")
    tbl(("Mô hình", "Cách chia", "AP", "Precision", "Recall", "F1"), [
        (label_name, split_name, *[
            f"{decimal(comparison.loc[(model, scheme), metric])} ({decimal(comparison.loc[(model, scheme), metric+'_std'])})"
            for metric in ("ap", "precision", "recall", "f1")
        ])
        for model, label_name in LABEL.items()
        for scheme, split_name in [("variant", "Biến thể"), ("site", "Vị trí CHROM + POS")]
    ])
    add(f"Cả hai cách chia dùng 52.150 dòng train gồm 13.147 mẫu dương. Phép chia theo vị trí có "
        f"{integer(sites.sites.iloc[0])} khóa CHROM + POS. Mọi alen cùng vị trí vào chung một fold. "
        "Mỗi fold có cả hai lớp và không có vị trí giao giữa fit và validation. Ngưỡng F1 học ở vòng trong "
        "trên các nhóm vị trí thuộc phần fit. Vị trí mới nghĩa là chưa gặp tổ hợp CHROM + POS trong phần fit, "
        "và cùng gene vẫn có thể xuất hiện ở hai phía. "
        + source(review / "site_fold_assignments.csv", "Phân bổ vị trí") + " · "
        + source(review / "site_cv_folds.csv", "Kết quả từng fold và kiểm tra trùng vị trí"))
    add("AP là trung bình không trọng số của 5 fold. Precision, recall và F1 tính từ các dự đoán nhị phân gộp. "
        "Số trong ngoặc là độ lệch chuẩn mẫu của chỉ số qua 5 fold, dùng để mô tả độ dao động. "
        "Hai cách chia có thành phần fold khác nhau nên so chênh lệch tổng thể. "
        + source(review / "cv_by_split_comparison.csv", "So sánh đầy đủ hai cách chia") + " · "
        + source(review / "site_cv_summary.csv", "Tổng hợp theo vị trí") + " · "
        + source(review / "site_oof_predictions.csv", "Dự đoán ngoài fold theo vị trí"))
    for model, name in LABEL.items():
        delta = comparison.loc[(model, "site")] - comparison.loc[(model, "variant")]
        add(f"**{name}.** Chia theo vị trí so với chia theo biến thể cho ΔAP "
            f"{decimal(delta.ap, 5, True)} và ΔF1 {decimal(delta.f1, 5, True)}.")


def write_report(out, summary, selected):
    from study_retained import LOCKED, MAX_LOSS, MIN_GAIN, qualified

    out = Path(out)
    aud = pd.read_csv(out / "audit_train.csv").set_index(["column_or_group", "state"]).n.to_dict()
    rows, both, overlap, outside = utr_counts(out, aud)
    missing = pd.read_csv(out / "missing_indicator_corr_train.csv", index_col=0)
    rho = pd.read_csv(OUTPUT / "spearman.csv", index_col=0)
    counts = pd.read_csv(OUTPUT / "spearman_pair_counts.csv", index_col=0)
    data = load_json(ROOT / "data" / "report.json")
    classes = pd.read_csv(out / "y_train.csv").CLASS
    test_classes = pd.read_csv(out / "y_test.csv").CLASS
    schema = {m: load_json(out / m / "input_columns.json") for m in LABEL}
    encoded = {m: load_json(out / m / "feature_names.json") for m in LABEL}
    shared = [col for _, cols in GROUPS for col in cols]
    for model, af in ((LR, AF_LR), (RF, AF_RF)):
        actual = schema[model]
        require(set(actual) == set(shared + af) and len(actual) == selected[model]["input_columns"]
                and len(encoded[model]) == len(set(encoded[model])) == selected[model]["encoded_columns"],
                "Schema thay đổi, cần rà lại nhóm cột trong báo cáo.")
    figures(out, missing, rho, counts, aud)
    metrics = summary.set_index(["model", "config", "scheme", "subgroup"]).sort_index()
    exon = aud["transcript_region", "exon"]
    intron = aud["transcript_region", "intron"]
    exon_intron = aud["transcript_region", "exon_and_intron"]
    unresolved = aud["transcript_region", "unresolved"]

    def source(path, label):
        relative = os.path.relpath(path, start=out).replace("\\", "/")
        return f"[{label}]({relative})"

    def ap(model, config=LOCKED, scheme="variant", group="all"):
        return metrics.loc[(model, config, scheme, group), "ap"]

    def benchmark(model, trial, scheme="variant", group="all"):
        return f"{decimal(ap(model, LOCKED, scheme, group))} → {decimal(ap(model, trial, scheme, group))}"

    def model_group(model, name):
        return metrics.loc[(model, name)].reindex(
            [("variant", "all"), ("variant", "missense"), ("gene", "all"), ("gene", "missense")]
        ).dropna(subset=["folds_valid"])

    lines = []

    def add(*items):
        lines.extend([*items, ""])

    def tbl(header, records):
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for entry in records:
            lines.append("| " + " | ".join(map(str, entry)) + " |")
        lines.append("")

    def page(title):
        add(PAGE, "", f"## {title}")

    add("# Tiền xử lý và đánh giá đặc trưng ClinVar")
    add("Mục tiêu là dự đoán CLASS=1, tức diễn giải mâu thuẫn giữa các đánh giá ClinVar. "
        "Average precision (AP) đo khả năng xếp các biến thể lớp 1 lên đầu theo xác suất dự đoán.")
    add("### Cách gán nhãn CLASS và trường hợp VUS")
    add("Năm mức ý nghĩa lâm sàng khi diễn giải biến thể theo ACMG/AMP là gây bệnh (pathogenic), có khả năng gây bệnh "
        "(likely pathogenic), chưa rõ ý nghĩa (VUS), có khả năng lành tính (likely benign) và lành tính (benign). "
        "Các mức này mô tả kết luận về biến thể. CLASS ghi nhận tình trạng mâu thuẫn theo quy tắc của bộ dữ liệu. "
        "[Hướng dẫn ACMG/AMP](https://pmc.ncbi.nlm.nih.gov/articles/PMC4544753/)")
    add("Tác giả khởi tạo CLASS=0 và gán CLASS=1 khi CLNSIGCONF khác null. Bộ dữ liệu chỉ giữ hai giá trị CLNREVSTAT "
        "lịch sử gồm criteria_provided,_multiple_submitters,_no_conflicts và "
        "criteria_provided,_conflicting_interpretations. Tác giả bỏ CLNSIG, CLNSIGCONF và CLNREVSTAT trước khi xuất CSV. "
        "[Mã tạo bộ dữ liệu, dòng 89–109](https://github.com/arvkevi/clinvar-kaggle/blob/master/process_clinvar.py#L89-L109)")
    add("**VUS đi đâu?** Riêng mức VUS chưa đủ để suy ra CLASS. Trong phạm vi đã lọc, bản ghi có CLNSIGCONF khác null "
        "nhận lớp 1, còn trường null nhận lớp 0. Quy tắc cũng áp dụng cho pathogenic và benign. CLASS=0 nghĩa là "
        "không ghi nhận mâu thuẫn theo tiêu chí của bộ dữ liệu. Muốn đếm VUS theo lớp cần đối chiếu đúng bản VCF nguồn "
        "vì các trường cần thiết đã được bỏ khỏi CSV. "
        "[Mã tạo nhãn](https://github.com/arvkevi/clinvar-kaggle/blob/master/process_clinvar.py#L89-L109)")
    tbl(("Phạm vi", "Biến thể", "CLASS=1"), [
        ("Toàn bộ", integer(len(classes) + len(test_classes)), integer(classes.sum() + test_classes.sum())),
        ("Train", integer(len(classes)), integer(classes.sum())),
        ("Test lịch sử", integer(len(test_classes)), integer(test_classes.sum())),
    ])
    add("Số dòng đếm theo source_row duy nhất. Cột CLASS=1 đếm nhãn bằng 1 trong từng tập. "
        + source(out / "y_train.csv", "Nhãn train") + " · " + source(out / "y_test.csv", "Nhãn test"))
    add("### Hai cấu hình đã chọn")
    tbl(("Mô hình", "Thay đổi so với mốc 24 đầu vào", "Đầu vào → sau mã hóa", "AP mốc → cuối"), [
        (LABEL[LR], "Log AF, ba cờ AF bằng 0, multi-hot hệ quả có nhóm hiếm", "27 → 56",
         benchmark(LR, selected[LR]["configuration"])),
        (LABEL[RF], "Giữ AF gốc, multi-hot hệ quả có nhóm hiếm", "24 → 53",
         benchmark(RF, selected[RF]["configuration"])),
    ])
    add("AP cho CLASS=1 là trung bình không trọng số trên cùng 5 fold theo biến thể từ 52.150 dòng train. "
        "Bộ mốc locked_24 có 24 đầu vào và 83 cột sau mã hóa. "
        + source(out / "cv_summary.csv", "Tổng hợp CV") + " · " + source(out / "selected.json", "Cấu hình đã xuất"))
    add("Trong Logistic Regression, lợi ích chính đến từ log AF kết hợp ba cờ bằng 0. "
        "Khi thêm multi-hot có nhóm hiếm sau bước AF, AP đổi "
        + decimal(ap(LR, selected[LR]["configuration"]) - ap(LR, "af_log+af_zero"), 5, True)
        + " và số cột sau mã hóa giảm từ 86 xuống 56. Random Forest giữ AF gốc theo kết quả CV. "
        + source(out / "cv_summary.csv", "Kết quả từng cấu hình"))
    page("Danh sách đầu vào cuối")
    tbl(("Nhóm", "Tên cột đầu vào", "Mô hình"), [
        *[(label, ", ".join(cols), "Cả hai") for label, cols in GROUPS],
        ("AF biến đổi", ", ".join(AF_LR), LABEL[LR]),
        ("AF gốc", ", ".join(AF_RF), LABEL[RF]),
    ])
    add("Danh sách có 21 cột chung. Sáu dẫn xuất AF của Logistic Regression cho 27 đầu vào, mã hóa thành 56 cột. "
        "Ba AF gốc của Random Forest cho 24 đầu vào, mã hóa thành 53 cột. Số cột là độ dài schema đã fit trên toàn tập train. "
        + source(out / LR / "input_columns.json", "Đầu vào Logistic Regression") + " · "
        + source(out / RF / "input_columns.json", "Đầu vào Random Forest") + " · "
        + source(out / LR / "feature_names.json", "56 cột sau mã hóa") + " · "
        + source(out / RF / "feature_names.json", "53 cột sau mã hóa"))
    add("Giữ tỷ lệ và nhóm vùng của EXON/INTRON vì hai trường bổ sung nhau. Giữ điểm và cờ LoFtool. "
        "Các giá trị SIFT, PolyPhen, BLOSUM62 và tọa độ cDNA được bảo toàn trong bảng sạch. "
        "Đầu vào cuối của ba chú giải protein dùng cờ có giá trị trong CSV gốc.")
    page("Tương quan cờ thiếu và tương quan điểm")
    add("![Tương quan cờ thiếu trên train](hinh_1_co_thieu.png)",
        "**Hình 1. Các cột nào thường thiếu cùng nhau?** Tương quan Pearson giữa cờ thiếu 0/1 trên 52.150 dòng train. "
        "EXON/INTRON = " + decimal(missing.loc["EXON", "INTRON"], 3)
        + ", SIFT/PolyPhen = " + decimal(missing.loc["SIFT", "PolyPhen"], 3)
        + " và hai cờ thiếu CADD = " + decimal(missing.loc["CADD_PHRED", "CADD_RAW"], 3)
        + f". Bốn nhóm vùng có {integer(exon)} chỉ exon, {integer(intron)} chỉ intron, "
        f"{integer(exon_intron)} có cả hai và {integer(unresolved)} chưa xác định. "
        "Hai trường bổ sung nhau nên giữ cả hai. Hình của Sơn dùng 16 cột trên toàn bộ 65.188 dòng. "
        + source(out / "missing_indicator_corr_train.csv", "Ma trận cờ thiếu train")
        + " · [Hình 12 của Sơn, mục 5.2](https://drive.google.com/drive/folders/1T5GGkzTAMNXZzxXAzhSQZfCwRPwkpaAt)")
    add("![Spearman và số cặp trên train](hinh_2_spearman.png)",
        "**Hình 2. Hai thang điểm nào có thứ hạng gần trùng?** Mỗi ô ghi Spearman của giá trị điểm và số cặp cùng quan sát. "
        "CADD_PHRED/CADD_RAW có ρ ≈ " + decimal(rho.loc["CADD_PHRED", "CADD_RAW"], 6)
        + " trên " + integer(counts.loc["CADD_PHRED", "CADD_RAW"])
        + " cặp từ train. Các ô hiển thị hệ số làm tròn. Ô thiếu ở một trong hai trường nằm ngoài phép tính. "
        + source(OUTPUT / "spearman.csv", "Giá trị Spearman") + " · "
        + source(OUTPUT / "spearman_pair_counts.csv", "Số cặp"))
    page("Chú giải protein và cách dùng CADD")
    add("![Độ phủ chú giải protein trên train](hinh_3_do_phu.png)",
        "**Hình 3. Chú giải protein có mặt ở nhóm nào?** Tử số là số dòng có chú giải gốc, mẫu số là số dòng của nhóm train. "
        "Missense được xác định khi Consequence có từ missense_variant. Điểm BLOSUM62 khôi phục nằm ngoài tử số. "
        + source(out / "audit_train.csv", "Số đếm độ phủ") + " · "
        + source(OUTPUT / "coverage.png", "Độ phủ theo hệ quả"))
    add(f"Trên train, khôi phục {integer(aud['BLOSUM62','recovered_missense'])} điểm BLOSUM62 bằng 0 trong "
        f"{integer(aud['BLOSUM62','missing_missense'])} ô thiếu missense khi cặp axit amin hợp lệ có điểm ma trận bằng 0. "
        f"Còn {integer(aud['BLOSUM62','missing_missense']-aud['BLOSUM62','recovered_missense'])} ô thiếu. "
        "Bảng sạch giữ điểm âm, chú giải ngoài missense, PolyPhen unknown và mức tin cậy SIFT. "
        + source(out / "audit_train.csv", "Kiểm tra protein"))
    add("### CADD theo từng giai đoạn")
    median = data["preprocessors"]["scaled"]["train_imputation_values"]["CADD_PHRED"]
    phred = pd.read_csv(out / "clean_train.csv", usecols=["CADD_PHRED"]).CADD_PHRED
    require(phred.notna().sum() == counts.loc["CADD_PHRED", "CADD_RAW"] and phred.median() == median,
            "Trung vị PHRED hoặc mẫu số cần đối chiếu lại.")
    tbl(("Giai đoạn", "Thang điểm và xử lý thiếu"), [
        ("Bộ cơ sở lịch sử, 8 đầu vào", f"CADD_PHRED. Trung vị {decimal(median,2)} học trên các giá trị PHRED hợp lệ trong toàn bộ 52.150 dòng train ở bản xuất lịch sử."),
        ("Bảng sạch trước nội suy", "Giữ CADD_PHRED và CADD_RAW trước điền thiếu."),
        ("Bộ mốc mở rộng và hai cấu hình cuối", "CADD_RAW. Bảo toàn cả điểm RAW âm hữu hạn."),
        ("Cross-validation", "Học trung vị cột số từ phần huấn luyện của từng fold. Preprocessor xuất học trên toàn bộ train."),
    ])
    add("Trung vị lịch sử 14,06 dùng 51.293 giá trị PHRED hợp lệ của train. Hình 1 đo sự đồng thời thiếu của hai thang. "
        "Hình 2 đo tương quan thứ hạng của các giá trị cùng có điểm. "
        + source(ROOT / "data" / "report.json", "Thống kê bộ cơ sở") + " · "
        + source(OUTPUT / "pipeline" / "input_columns.json", "Schema mốc") + " · "
        + source(ROOT / "preprocess.py", "Quy tắc điền thiếu"))
    page("Tọa độ thiếu và độ phủ của cDNA")
    examples = ["73", "73-75", "1476-?", "?-73", "?", "", "75-73"]
    parsed = parse_coordinate(pd.Series(examples))
    tbl(("Giá trị gốc", "start", "end", "span", "status"), [
        (example or "Ô trống", *["NaN" if pd.isna(getattr(entry, key)) else str(int(getattr(entry, key)))
                                for key in ("start", "end", "span")], entry.status)
        for example, entry in zip(examples, parsed.itertuples())
    ])
    add("Ví dụ tính bằng cùng quy tắc cho CDS, protein và cDNA. Tọa độ đơn có span = 1. "
        "Khoảng đầy đủ có span = end − start + 1. NaN trong bộ nhớ được ghi thành ô trống trong CSV. "
        + source(ROOT / "retained_features.py", "Hàm tách tọa độ"))
    add("**1476-? qua ba bước.** Bảng sạch lưu start=1476, end=NaN, span=NaN và status=partial. "
        "Khi chuẩn bị CDS/protein, cả hai bộ cuối đặt start của dòng partial về NaN. span vẫn là NaN, "
        "còn end nằm ngoài schema cuối. Khi biến đổi, start và span thiếu nhận trung vị của cột tương ứng từ phần fit. "
        + source(ROOT / "study_retained.py", "Hàm chuẩn bị X") + " · "
        + source(ROOT / "preprocess.py", "Bộ điền thiếu"))
    add("cDNA được giữ trong bảng sạch và thử trong nhánh tọa độ với start, end, span, status và cờ có cDNA khi thiếu CDS. "
        "Cả hai bộ cuối dùng start và span của CDS/protein. 1476 là tọa độ trên CDS. Muốn tính tỷ lệ chiều dài "
        "cần chiều dài CDS của đúng transcript làm mẫu số.")
    add("### Đếm dòng có cDNA và thiếu CDS trên train")
    tbl(("Nhóm hệ quả", "Tổng dòng trong nhóm", "Có cDNA và thiếu CDS"),
        [(label, integer(n), integer(k)) for label, n, k in rows])
    add(f"Đếm source_row duy nhất trong {integer(rows[0][1])} dòng train. Tử số lọc cDNA_position có giá trị và "
        "CDS_position thiếu theo quy ước ký hiệu thiếu. Mẫu số 519 và 345 là số dòng train chứa hệ quả 5′ UTR và 3′ UTR "
        f"tương ứng. Hai nhóm giao {integer(both)} dòng trên train và {integer(overlap)} dòng trong tử số. "
        f"Có {integer(outside)} dòng thỏa điều kiện bên ngoài hai nhóm. "
        + source(out / "clean_train.csv", "Bảng sạch train") + " · " + source(out / "audit_train.csv", "Số đếm kiểm tra"))
    page("Bằng chứng chọn và loại thử nghiệm")
    add("So từng mô hình với chính mô hình đó dùng locked_24. Quy tắc yêu cầu đủ 5 fold hợp lệ ở cả bốn tổ hợp "
        "biến thể/gene × toàn tập/missense. ΔAP toàn tập theo biến thể phải đạt ít nhất +0,002 và tăng ở ít nhất 4/5 fold. "
        "Không tổ hợp nào được có ΔAP dưới −0,002. Các điều kiện được tính trên số chưa làm tròn. "
        "Trong các cấu hình đạt, chọn ΔAP toàn tập theo biến thể cao nhất. "
        + source(ROOT / "study_retained.py", "Quy tắc chọn"))
    evidence = [(LR, "positions"), (LR, "chrom"), (RF, selected[RF]["configuration"])]
    tbl(("Mô hình và thử nghiệm", "AP mốc → thử", "ΔAP", "Fold tăng", "Căn cứ"), [
        (f"{LABEL[m]} · {TRIAL_NAMES[name]}", benchmark(m, name),
         decimal(metrics.loc[(m,name,"variant","all"),"delta_ap"], 5, True),
         f"{int(metrics.loc[(m,name,'variant','all'),'folds_up'])}/5",
         "Dưới ngưỡng +0,002" if name == "positions" else
         "ΔAP toàn tập theo gene −0,01070" if name == "chrom" else
         "Đủ 5 fold ở cả bốn tổ hợp, đạt mọi ngưỡng")
        for m, name in evidence
    ])
    add("AP và ΔAP trong bảng đo trên toàn tập, chia theo biến thể. ΔAP là trung bình chênh lệch AP thử trừ mốc "
        "theo từng fold. Cột fold tăng đếm fold có ΔAP > 0. "
        + source(out / "cv_summary.csv", "Tổng hợp thử nghiệm") + " · " + source(out / "cv_folds.csv", "Kết quả ghép cặp"))
    add("Nhánh tọa độ tăng ở 4/5 fold cho Logistic Regression, nhưng mức tăng trung bình +0,00077 thấp hơn +0,002. "
        "CHROM đạt mức tăng và số fold theo biến thể, nhưng mức giảm −0,01070 trên toàn tập theo gene vượt giới hạn. "
        "Random Forest với hệ quả nhóm hiếm đạt +0,01106 trên toàn tập theo biến thể, +0,00954 trên missense theo biến thể, "
        "+0,01027 trên toàn tập theo gene và +0,00368 trên missense theo gene. "
        + source(out / "cv_summary.csv", "ΔAP theo bốn tổ hợp"))
    tbl(("Nhóm CV trên train", LABEL[LR], LABEL[RF]), [
        (caption, *[benchmark(m, selected[m]["configuration"], scheme, group) for m in LABEL])
        for scheme, group, caption in [("variant","missense","Missense theo biến thể"),
                                        ("gene","all","Toàn tập theo gene"), ("gene","missense","Missense theo gene")]
    ])
    add("AP mốc → cuối, trung bình không trọng số của 5 fold trong từng nhóm. Mỗi dòng thuộc đúng một fold trong mỗi cách chia. "
        + source(out / "fold_assignments.csv", "Phân bổ fold") + " · " + source(out / "cv_summary.csv", "AP theo nhóm"))
    page("CHROM và tần suất gene SYMBOL")
    add("Quy tắc để chọn feature yêu cầu đủ 5 fold trong cả bốn tổ hợp biến thể/gene × toàn tập/missense, "
        "ΔAP toàn tập theo biến thể ≥ +0,002, ít nhất 4/5 fold tăng và mọi ΔAP trung bình ≥ −0,002. "
        "Dung sai giảm áp dụng riêng cho từng tổ hợp. "
        + source(ROOT / "study_retained.py", "Quy tắc chọn"))
    symbol = pd.read_csv(OUTPUT / "gene_fold_metrics.csv").query("config == 'symbol_frequency' and subgroup == 'all'")
    symbol = symbol.groupby(["model", "scheme"]).delta_ap.mean()
    chrom = model_group(LR, "chrom").delta_ap
    tbl(("Thử nghiệm", "Bộ mốc", "ΔAP biến thể", "ΔAP theo gene"), [
        ("CHROM · Logistic Regression", "locked_24", decimal(chrom["variant", "all"], 4, True),
         decimal(chrom["gene", "all"], 4, True)),
        *[(f"Tần suất SYMBOL · {LABEL[m]}", "Baseline one-hot", decimal(symbol[m, "variant"], 4, True),
           decimal(symbol[m, "gene"], 4, True)) for m in LABEL],
    ])
    add("Mỗi ô là trung bình ΔAP trên toàn tập ở 5 fold train, theo đúng bộ mốc của hàng đó. "
        + source(out / "cv_summary.csv", "Kết quả CHROM") + " · "
        + source(OUTPUT / "gene_fold_metrics.csv", "Kết quả SYMBOL lịch sử"))
    add("CHROM vượt giới hạn giảm trên gene chưa gặp. Với Logistic Regression lịch sử, mức giảm toàn tập theo gene của "
        "tần suất SYMBOL nằm trong dung sai. Cần thử lại trên đúng bộ mốc hiện tại và kiểm tra cả missense trước khi "
        "quyết định thêm feature. Với Random Forest lịch sử, mức giảm toàn tập theo gene vượt dung sai. "
        "Tần suất của gene chưa xuất hiện trong phần fit được mã hóa thành 0. "
        "Trong cấu hình hiện tại, SYMBOL phục vụ chia fold theo gene. "
        + source(ROOT / "feature_study.py", "Mã hóa tần suất"))
    endpoint_report(out, selected, add, tbl, page, source)
    oof_report(out, add, tbl, page, source)
    for model in LABEL:
        page(f"Phụ lục thử nghiệm {LABEL[model]}")
        trial_rows = []
        for name in TRIAL_NAMES:
            group = model_group(model, name)
            v = group.loc["variant", "all"]
            why = verdict(group, name == selected[model]["configuration"], MIN_GAIN, MAX_LOSS)
            if why == "Đã chọn":
                require(qualified(summary, model)[0] == name, "Bảng báo cáo không khớp quy tắc chọn.")
            trial_rows.append((TRIAL_NAMES[name], benchmark(model, name), decimal(v.delta_ap, 5, True),
                               f"{int(v.folds_up)}/5", decimal(group.delta_ap.min(), 5, True), why))
        tbl(("Thay đổi", "AP mốc → thử", "ΔAP", "Fold tăng", "ΔAP thấp nhất", "Quyết định"), trial_rows)
        add("AP và số fold tăng tính trên toàn tập theo biến thể. ΔAP thấp nhất lấy trong các tổ hợp có kết quả. "
            "Các dòng “Chưa đủ phép đo” thiếu ít nhất một tổ hợp 5 fold và không được xếp hạng. "
            "Các điều kiện bị vi phạm cho biết lý do không đạt. “Đạt, xếp sau” là cấu hình qua ngưỡng nhưng có AP toàn tập "
            "theo biến thể thấp hơn bộ cuối. Tra từng fold bằng config, model, scheme, subgroup và fold. "
            + source(out / "cv_summary.csv", "Tổng hợp") + " · " + source(out / "cv_folds.csv", "Chỉ số theo fold"))
        if model == LR:
            add("### Giải nghĩa nhánh thử")
            add("**Tọa độ bổ sung.** Thêm " + ", ".join(EXTRA_POSITIONS)
                + ". Giữ mút start đã biết của khoảng partial trong nhánh này.")
            add("**Giá trị protein.** Thêm " + ", ".join(PROTEIN_VALUES)
                + " vào bộ mốc đã có cờ hiện diện.")
            add("**Các nhánh khác.** Log AF thay ba AF gốc. Cờ AF và cờ có CADD_RAW thêm các chỉ báo tương ứng. "
                "IMPACT thứ bậc thay one-hot bằng mã 0–3. CHROM thêm one-hot. Hệ quả nhóm hiếm thay one-hot cả chuỗi "
                "bằng multi-hot theo từng từ. "
                + source(ROOT / "study_retained.py", "Định nghĩa các nhánh"))
        else:
            add("Hệ quả có dưới 50 dòng fit hoặc chỉ gặp lúc validation bật nhóm hiếm. Hệ quả trống có cờ riêng. "
                "Trung vị, chuẩn hóa và từ vựng được fit lại ở từng fold. Các mô tả cột thử ở phụ lục Logistic Regression "
                "áp dụng cho cả hai mô hình. " + source(ROOT / "consequence_encoder.py", "Bộ mã hóa hệ quả"))
    page("Phụ lục kiểm chứng và test lịch sử")
    history = pd.read_csv(OUTPUT / "holdout_threshold_metrics.csv")
    tbl(("Mô hình", "AP cơ sở 8 trường", "AP bộ 24 đã khóa"), [
        (label, *[decimal(history.loc[history.model.eq(m) & history.config.eq(c) & history.subgroup.eq("all"), "ap"].item())
                   for c in ("baseline_onehot", "selected")]) for m, label in LABEL.items()
    ])
    add("AP CLASS=1 đo một lần trên test lịch sử 13.038 dòng, gồm 3.287 mẫu dương. "
        "Bộ cơ sở one-hot dùng 8 trường gốc. Bộ mở rộng lịch sử dùng 24 đầu vào, 83 cột sau mã hóa. "
        "Test của lần cập nhật mới phục vụ kiểm tra phép biến đổi. "
        + source(OUTPUT / "holdout_threshold_metrics.csv", "Chỉ số test lịch sử"))
    add("### Nghiệm thu bảng sạch")
    r = [aud["transcript_region", state] for state in ("exon", "intron", "exon_and_intron", "unresolved")]
    add(f"Trên 52.150 dòng train, đếm theo hai trường phân số hợp lệ cho {integer(r[0])} chỉ exon, "
        f"{integer(r[1])} chỉ intron, {integer(r[2])} có cả hai và {integer(r[3])} chưa xác định. "
        "Nhóm chưa xác định gồm dòng thiếu cả hai trường hợp lệ. "
        + source(out / "audit_train.csv", "Số đếm vùng và lỗi cú pháp"))
    add("Cả 22 kiểm thử đều đạt trong lần bàn giao. Bốn ma trận khớp từng phần tử khi đọc lại CSV và nạp lại preprocessor. "
        "Mã dòng và nhãn căn đúng thứ tự, khóa biến thể giữa train và test tách biệt, tên cột duy nhất và dữ liệu ma trận hữu hạn. "
        + source(ROOT / "study_retained.py", "Kiểm tra đầu ra") + " · "
        + source(out / "column_dictionary.csv", "Từ điển cột"))
    add("ORIGIN được lưu để kiểm tra, với các bit quan sát khớp header GRCh37 tham chiếu. Phiên bản VCF gốc và thời điểm "
        "ORIGIN có sẵn khi dự đoán cần xác minh. CLNDN được giữ trong bảng sạch với trạng thái chỉ placeholder, "
        "có cả placeholder và tên bệnh, hoặc có tên bệnh. SYMBOL dùng để chia fold theo gene. "
        + source(out / "origin_train.csv", "Đối chiếu mã ORIGIN") + " · "
        + source(out / "QUYET_DINH_46_COT.md", "Phụ lục quyết định đủ 46 cột"))
    add("Tái lập dữ liệu và CV bằng `python study_retained.py`. Chạy phép thử mút tọa độ và ngưỡng với "
        "`python study_endpoints.py`. Tái sinh báo cáo bằng `python retained_publication.py`. "
        "Kiểm thử bằng `python -m unittest -q`. "
        + source(ROOT / "README.md", "Hướng dẫn workspace"))
    page("Phụ lục đối chiếu 46 trường gốc")
    from feature_documentation import REQUIRED, TRIAL

    fields = pd.read_csv(ROOT / "data" / "raw_column_profile.csv").column.tolist()
    require(len(fields) == len(set(fields)) == 46, "Danh sách trường gốc không hợp lệ.")

    def role(col):
        if col in REQUIRED:
            return "Giữ theo đề xuất"
        return "Ứng viên" if col in TRIAL else "Truy vết / kiểm tra"

    tbl(("Trường gốc", "Vai trò", "Trường gốc", "Vai trò"), [
        (a, role(a), b, role(b)) for a, b in zip(fields[:23], fields[23:])
    ])
    add("Các vai trò phản ánh quyết định lưu trong bảng sạch và phạm vi thử. "
        "Danh sách ở trang 2 ghi chính xác những cột đi vào mô hình cuối. "
        + source(out / "QUYET_DINH_46_COT.md", "Nguồn đề xuất, cách xử lý và trạng thái của từng trường"))
    (out / "BAO_CAO.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "feature_retained")
    directory = parser.parse_args().out.resolve()
    write_report(directory, pd.read_csv(directory / "cv_summary.csv", float_precision="round_trip"),
                 load_json(directory / "selected.json"))
