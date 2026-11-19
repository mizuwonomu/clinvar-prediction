"""Biểu diễn ClinVar trước nội suy, với trạng thái thiếu và nguồn của từng giá trị."""

import numpy as np
import pandas as pd

from consequence_encoder import _terms
from feature_study import COUNTS, number, prepare_features, recover_blosum
from preprocess import AF, KEY, MISSING


POSITIONS = ("CDS_position", "Protein_position", "cDNA_position")
PROTEIN_VALUES = ("SIFT_call", "SIFT_low_confidence", "PolyPhen", "BLOSUM62_score", "BLOSUM62_source")
EXTRA_POSITIONS = (
    "CDS_position_end", "CDS_position_status", "Protein_position_end", "Protein_position_status",
    "cDNA_position_start", "cDNA_position_end", "cDNA_position_span", "cDNA_position_status",
    "has_cDNA_without_CDS",
)
IMPACT_ORDER = ("MODIFIER", "LOW", "MODERATE", "HIGH")
ORIGIN_BITS = {
    1: "germline", 2: "somatic", 4: "inherited", 8: "paternal", 16: "maternal", 32: "de_novo",
    64: "biparental", 128: "uniparental", 256: "not_tested", 512: "tested_inconclusive",
    1073741824: "other",
}
KEEP_RAW = (*AF, *COUNTS, *POSITIONS, "SIFT", "PolyPhen", "BLOSUM62", "Amino_acids",
            "CLNVC", "IMPACT", "Consequence", "ORIGIN", "CADD_RAW", "CADD_PHRED", "LoFtool",
            "SYMBOL", "BIOTYPE", "CLNDN")


def clean_token(value):
    if pd.isna(value):
        return None
    text = value.strip()
    return None if text.upper() in MISSING else text


def parse_coordinate(values):
    """Các mút chưa biết để trống, kể cả mút thiếu của một khoảng."""
    text = values.str.strip()
    tokens = text.str.extract(r"^([1-9]\d*|\?)(?:-([1-9]\d*|\?))?$")
    first, last = tokens[0], tokens[1]
    start = number(first)
    is_single = start.notna() & last.isna()
    end = number(last).mask(is_single, start)
    partial = last.notna() & start.notna().ne(end.notna())
    unknown = first.eq("?") & (last.isna() | last.eq("?"))
    valid_pair = last.notna() & start.notna() & end.ge(start)
    missing = text.isna() | text.str.upper().isin(MISSING - {"?"})
    status = np.select(
        [missing, unknown, is_single, valid_pair, partial],
        ["missing", "unknown", "single", "range", "partial"], default="invalid",
    )
    keep = is_single | valid_pair | partial
    return pd.DataFrame({
        "start": start.where(keep), "end": end.where(keep),
        "span": (end - start + 1).where(is_single | valid_pair), "status": status,
    }, index=values.index)


def allele_frequency(values):
    text = values.map(clean_token)
    numeric = number(text)
    valid = numeric.between(0, 1)
    state = np.select(
        [text.isna(), numeric.isna(), ~valid, numeric.eq(0)],
        ["missing", "invalid_numeric", "out_of_range", "zero"], default="positive",
    )
    af = numeric.where(valid)
    return af, np.log1p(af * 1_000_000) / np.log(10), af.eq(0).astype("int8"), state


def disease_status(value):
    if pd.isna(value):
        return "missing"
    items = {item for token in str(value).split("|") if (item := clean_token(token))}
    placeholder = {item for item in items if item.casefold() in {"not_specified", "not_provided"}}
    if not items:
        return "missing"
    if not placeholder:
        return "named"
    return "placeholder_only" if items == placeholder else "mixed"


def origin_details(value):
    if not (text := clean_token(value)):
        return "", "missing"
    if not text.isascii() or not text.isdecimal():
        return "", "invalid"
    value = int(text)
    if value == 0:
        return "unknown", "unknown"
    flags = [label for bit, label in ORIGIN_BITS.items() if value & bit]
    if value & ~sum(ORIGIN_BITS):
        return "|".join(flags), "unmapped_bits"
    return "|".join(flags), "combined" if len(flags) > 1 else "single"


def build_clean(source, raw, baseline, ids):
    """Ghép phần đã kiểm chứng với các biểu diễn mới, theo cùng source_row."""
    base = prepare_features(raw, baseline, ids)
    clean = ids[KEY].copy()
    clean = clean.join(source.loc[raw.index, list(KEEP_RAW)].add_suffix("_raw"))
    reusable = [
        *AF, *(f"{prefix}_{col}" for col in AF for prefix in ("zero", "log")),
        "CLNVC", "Consequence", "IMPACT", "BIOTYPE", "CADD_RAW", "CADD_PHRED", "LoFtool",
        "has_LoFtool", "has_CADD_RAW", "has_SIFT", "has_PolyPhen", "has_BLOSUM62",
        "ref_length", "alt_length", "alt_minus_ref", "multiple_alt", "transcript_region",
        *(f"{col}_{part}" for col in COUNTS for part in ("index", "total", "fraction")),
    ]
    clean = clean.join(base[reusable])
    for col in AF:
        clean[col], clean[f"log_{col}"], clean[f"zero_{col}"], clean[f"{col}_status"] = allele_frequency(raw[col])
    for col in COUNTS:
        clean[f"{col}_status"] = np.select(
            [raw[col].isna(), base[f"{col}_index"].notna()], ["missing", "valid"], default="invalid",
        )
    for col in POSITIONS:
        parsed = parse_coordinate(source.loc[raw.index, col]).add_prefix(f"{col}_")
        clean = clean.join(parsed)
    clean["has_cDNA_without_CDS"] = (raw.cDNA_position.notna() & raw.CDS_position.isna()).astype("int8")

    clean["SIFT_call"] = raw.SIFT.str.replace(r"_low_confidence$", "", regex=True)
    clean["SIFT_low_confidence"] = raw.SIFT.str.endswith("_low_confidence").map({True: 1., False: 0.})
    clean["PolyPhen"] = raw.PolyPhen
    clean["Amino_acids"] = raw.Amino_acids
    clean["has_Amino_acids"] = raw.Amino_acids.notna().astype("int8")
    clean["BLOSUM62"] = base.BLOSUM62
    recovered = recover_blosum(raw, base.BLOSUM62) & raw.BLOSUM62.isna()
    clean["BLOSUM62_score"] = base.BLOSUM62.mask(recovered, 0)
    clean["BLOSUM62_source"] = np.select(
        [base.BLOSUM62.notna(), recovered], ["observed", "recovered"], default="missing",
    )
    clean["Consequence_terms"] = raw.Consequence.map(lambda value: "&".join(sorted(_terms(value))))
    clean["IMPACT_rank"] = raw.IMPACT.map(dict(zip(IMPACT_ORDER, range(len(IMPACT_ORDER)))))
    clean["SYMBOL"] = raw.SYMBOL
    clean["ORIGIN_flags"], clean["ORIGIN_status"] = zip(*raw.ORIGIN.map(origin_details))
    clean["CLNDN_status"] = raw.CLNDN.map(disease_status)
    clean["CLNDN_unknown_only"] = clean.CLNDN_status.eq("placeholder_only").astype("int8")
    return clean, base
