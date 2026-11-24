"""Shared repository paths and ClinVar import setup."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CLINVAR = ROOT / "clinvar"
FEATURES = CLINVAR / "feature_retained"
if str(CLINVAR) not in sys.path:
    sys.path.insert(0, str(CLINVAR))
