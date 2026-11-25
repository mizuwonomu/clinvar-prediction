"""Quy tắc dễ sai khi giữ chú giải ClinVar trước nội suy."""

import unittest

import numpy as np
import pandas as pd

from consequence_encoder import RareConsequence
from feature_study import load_rows, read_source
from retained_features import allele_frequency, build_clean, disease_status, origin_details, parse_coordinate
from study_retained import LOCKED, model_inputs


class RetainedFeatureTests(unittest.TestCase):
    def test_coordinates_keep_known_endpoint(self):
        parsed = parse_coordinate(pd.Series(["5", "4-6", "1476-?", "?-73", "?", "?-?", "", "6-4", "0", "a-9"]))
        self.assertEqual(parsed.status.tolist(),
                         ["single", "range", "partial", "partial", "unknown", "unknown",
                          "missing", "invalid", "invalid", "invalid"])
        np.testing.assert_allclose(parsed.start[:4], [5, 4, 1476, np.nan], equal_nan=True)
        np.testing.assert_allclose(parsed.end[:4], [5, 6, np.nan, 73], equal_nan=True)
        np.testing.assert_allclose(parsed.span[:4], [1, 3, np.nan, np.nan], equal_nan=True)
        self.assertTrue(parsed.iloc[4:, :3].isna().all().all())

    def test_af_keeps_zero_distinct_from_unavailable(self):
        af, log, flag, state = allele_frequency(pd.Series(["0", "1e-6", "1", "-0.1", "1.1", "bad", "NaN", ""]))
        self.assertEqual(state.tolist(), ["zero", "positive", "positive", "out_of_range", "out_of_range",
                                          "invalid_numeric", "missing", "missing"])
        self.assertEqual(flag.tolist(), [1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(af.iloc[:3], [0, 1e-6, 1])
        np.testing.assert_allclose(log.iloc[:3], np.log10([1, 2, 1_000_001]))
        self.assertTrue(af.iloc[3:].isna().all() and log.iloc[3:].isna().all())

    def test_disease_tokens_and_origin_bitmask(self):
        self.assertEqual(disease_status("not_specified|not_provided"), "placeholder_only")
        self.assertEqual(disease_status("Limb-girdle,_type_2|NOT_PROVIDED"), "mixed")
        self.assertEqual(disease_status("Condition_not_provided_subtype"), "named")
        self.assertEqual(disease_status(" . | NA "), "missing")
        self.assertEqual(disease_status(pd.NA), "missing")
        self.assertEqual(origin_details("33"), ("germline|de_novo", "combined"))
        self.assertEqual(origin_details("513"), ("germline|tested_inconclusive", "combined"))
        self.assertEqual(origin_details("0"), ("unknown", "unknown"))
        self.assertEqual(origin_details("1025"), ("germline", "unmapped_bits"))

    def test_rare_terms_count_rows_and_learn_only_from_fit(self):
        encoder = RareConsequence(min_count=2).fit(pd.DataFrame({"c": ["a&b&b", "a", "c", np.nan]}))
        self.assertEqual(encoder.get_feature_names_out().tolist(),
                         ["Consequence_term_a", "Consequence_rare_or_unseen", "Consequence_missing"])
        sample = pd.DataFrame({"c": ["a&b&unseen", "unseen", np.nan, "a"]})
        np.testing.assert_array_equal(encoder.transform(sample).toarray(),
                                      [[1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])

    def test_source_snapshot_and_locked_projection(self):
        source = read_source()
        raw, baseline, y, ids = load_rows(source, "train")
        clean, base = build_clean(source, raw, baseline, ids)
        self.assertEqual(clean.transcript_region.value_counts().to_dict(),
                         {"exon": 44988, "intron": 7035, "exon_and_intron": 31, "unresolved": 96})
        self.assertEqual(int(clean.has_cDNA_without_CDS.sum()), 860)
        recovered = clean.BLOSUM62_source.eq("recovered")
        self.assertEqual(int(recovered.sum()), 5442)
        self.assertTrue(clean.loc[recovered, "BLOSUM62_score"].eq(0).all())
        self.assertTrue(clean.loc[recovered, "has_BLOSUM62"].eq(0).all())
        observed = clean.BLOSUM62_source.eq("observed")
        pd.testing.assert_series_equal(clean.loc[observed, "BLOSUM62_score"],
                                       clean.loc[observed, "BLOSUM62"], check_names=False)
        pd.testing.assert_series_equal(clean.PolyPhen, raw.PolyPhen)
        self.assertEqual(int(clean.PolyPhen.eq("unknown").sum()), 4)
        columns = ["CDS_position_start", "Protein_position_start"]
        pd.testing.assert_frame_equal(model_inputs(clean, columns, LOCKED), base[columns], check_exact=True)
        self.assertTrue(clean.loc[clean.CDS_position_raw.eq("1476-?"), "CDS_position_start"].eq(1476).all())
        self.assertEqual(clean.index.tolist(), y.index.tolist())


if __name__ == "__main__":
    unittest.main()
