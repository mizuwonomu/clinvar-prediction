"""Shared repository paths and imports for the existing ClinVar pipeline."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CLINVAR = ROOT / "clinvar"
FEATURE_RETAINED = CLINVAR / "feature_retained"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"

clinvar_path = str(CLINVAR)
if clinvar_path not in sys.path:
    sys.path.insert(0, clinvar_path)
