"""
Map PTB-XL SCP diagnostic codes to the 5 standard superclasses.

PTB-XL records carry fine-grained SCP codes (71 possible) with confidence
scores. For training we group them into 5 clinically meaningful superclasses
and produce a multi-hot label vector per record.

Reference:
    Wagner et al. (2020). PTB-XL, a large publicly available
    electrocardiography dataset. Scientific Data.
    https://doi.org/10.1038/s41597-020-0495-6

Usage:
    from src.data.labels import load_metadata, SUPERCLASSES

    df = load_metadata()
    # df now has columns: NORM, MI, STTC, CD, HYP (each 0 or 1)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from src.utils.config import get_config

# -----------------------------------------------------------------------------
# SCP code -> superclass mapping
# -----------------------------------------------------------------------------
# Every SCP code that has a diagnostic meaning maps to one of the 5 superclasses.
# Codes not in this dict (e.g., "SR" for sinus rhythm, "ABQRS" for abnormal QRS)
# are non-diagnostic metadata and are ignored.
#
# The mapping follows the official PTB-XL diagnostic hierarchy.
# Note: some codes map to the same superclass as their sub-codes; that's fine.
# -----------------------------------------------------------------------------

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

SCP_TO_SUPERCLASS: dict[str, str] = {
    # --- NORM ---
    "NORM": "NORM",
    # --- Myocardial Infarction (MI) ---
    "IMI": "MI",  # Inferior MI
    "AMI": "MI",  # Anterior MI
    "LMI": "MI",  # Lateral MI
    "PMI": "MI",  # Posterior MI
    "INJAS": "MI",  # Injury, anterior subepicardial
    "INJAL": "MI",  # Injury, anterolateral
    "INJIN": "MI",  # Injury, inferior
    "INJLA": "MI",  # Injury, lateral
    "INJIL": "MI",  # Injury, inferolateral
    "IPMI": "MI",  # Ischemia, posterior MI
    # --- ST/T Change (STTC) ---
    "STTC": "STTC",
    "NST_": "STTC",  # Non-specific ST changes
    "NSTT": "STTC",
    "DIG": "STTC",  # Digitalis effect
    "LNGQT": "STTC",  # Long QT
    "ISC_": "STTC",  # Ischemia
    "ISCAL": "STTC",  # Ischemia, anterolateral
    "ISCAN": "STTC",  # Ischemia, anterior
    "ISCIN": "STTC",  # Ischemia, inferior
    "ISCLA": "STTC",  # Ischemia, lateral
    "ISCIL": "STTC",  # Ischemia, inferolateral
    "ISCA": "STTC",  # Ischemia, anteroseptal
    "STD_": "STTC",  # ST depression
    "STE_": "STTC",  # ST elevation
    "NDT": "STTC",  # Non-diagnostic T abnormalities
    "NSTT": "STTC",
    "TAB_": "STTC",  # T-wave abnormality
    "TAB": "STTC",
    "TV1": "STTC",  # T-wave in V1
    "TAB1": "STTC",
    "TAB2": "STTC",
    "LOWT": "STTC",  # Low T-waves
    "NT_": "STTC",  # Non-specific T changes
    # --- Conduction Disturbance (CD) ---
    "CD": "CD",
    "AVB": "CD",  # AV block
    "1AVB": "CD",  # First-degree AV block
    "2AVB": "CD",  # Second-degree AV block
    "3AVB": "CD",  # Third-degree AV block
    "CLBBB": "CD",  # Complete LBBB
    "CRBBB": "CD",  # Complete RBBB
    "ILBBB": "CD",  # Incomplete LBBB
    "IRBBB": "CD",  # Incomplete RBBB
    "IVCD": "CD",  # Non-specific IVCD
    "LAFB": "CD",  # Left anterior fascicular block
    "LPFB": "CD",  # Left posterior fascicular block
    "WPW": "CD",  # Wolff-Parkinson-White
    "LPR": "CD",  # Prolonged PR
    "ABQRS": "CD",  # Abnormal QRS
    "PVC": "CD",  # Premature ventricular contraction
    "SVARR": "CD",  # Supraventricular arrhythmia
    "BIGU": "CD",  # Bigeminy
    "TRIGU": "CD",  # Trigeminy
    # --- Hypertrophy (HYP) ---
    "HYP": "HYP",
    "LVH": "HYP",  # Left ventricular hypertrophy
    "RVH": "HYP",  # Right ventricular hypertrophy
    "LAO/LAE": "HYP",  # Left atrial overload/enlargement
    "RAO/RAE": "HYP",  # Right atrial overload/enlargement
    "SEHYP": "HYP",  # Septal hypertrophy
    "VCLVH": "HYP",  # Voltage criteria LVH
    "HVOLT": "HYP",  # High voltage
}


# -----------------------------------------------------------------------------
# Metadata loading
# -----------------------------------------------------------------------------


def load_metadata(config_path: Path | None = None) -> pd.DataFrame:
    """
    Load PTB-XL metadata CSV and attach 5 multi-hot superclass labels.

    Returns a DataFrame with:
    - One row per ECG record
    - Original metadata columns (age, sex, scp_codes, filename_hr, ...)
    - 5 new binary columns: NORM, MI, STTC, CD, HYP

    Records that have zero positives across all 5 superclasses are dropped,
    since there's no signal to train on.
    """
    cfg = get_config(config_path)
    metadata_path = cfg.paths.data_raw / cfg.dataset.metadata_file

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"PTB-XL metadata not found at {metadata_path}. "
            f"Did you run the download step?"
        )

    df = pd.read_csv(metadata_path, index_col="ecg_id")

    # `scp_codes` is stored as a Python-dict string. Parse it safely.
    df["scp_codes"] = df["scp_codes"].apply(_parse_scp_codes)

    # Initialize superclass columns to 0
    for sc in SUPERCLASSES:
        df[sc] = 0

    # Fill in positives
    for ecg_id, codes in df["scp_codes"].items():
        for code, likelihood in codes.items():
            if likelihood < cfg.labels.min_likelihood:
                continue
            sc = SCP_TO_SUPERCLASS.get(code)
            if sc is not None:
                df.at[ecg_id, sc] = 1

    # Drop records with no positive superclass
    mask = df[SUPERCLASSES].sum(axis=1) > 0
    dropped = (~mask).sum()
    if dropped:
        print(f"[labels] dropped {dropped} records with no superclass label")
    df = df[mask].copy()

    return df


def _parse_scp_codes(raw: str) -> dict[str, float]:
    """
    Parse PTB-XL's `scp_codes` field, which is stored as a string
    representation of a Python dict.

    Example input:  "{'NORM': 100.0, 'SR': 0.0}"
    Example output: {"NORM": 100.0, "SR": 0.0}
    """
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def get_label_distribution(df: pd.DataFrame) -> pd.Series:
    """Return per-superclass positive counts. Useful for sanity checks."""
    return df[SUPERCLASSES].sum().astype(int)


if __name__ == "__main__":
    # Smoke test: `python -m src.data.labels`
    print("Loading PTB-XL metadata...")
    df = load_metadata()
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns[:10])}...")
    print()
    print("Superclass distribution (positive counts):")
    dist = get_label_distribution(df)
    for sc in SUPERCLASSES:
        pct = 100 * dist[sc] / len(df)
        print(f"  {sc:5s}: {dist[sc]:6d}  ({pct:5.1f}%)")
    print()
    print("Multi-label stats:")
    label_count = df[SUPERCLASSES].sum(axis=1)
    print(f"  Records with 1 label: {(label_count == 1).sum()}")
    print(f"  Records with 2 labels:{(label_count == 2).sum()}")
    print(f"  Records with 3+ labels:{(label_count >= 3).sum()}")
