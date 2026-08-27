"""AIRR-like input normalization and donor-aware quality control."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
CHAIN_ALIASES = {
    "TRA": "TRA", "TCRA": "TRA", "ALPHA": "TRA",
    "TRB": "TRB", "TCRB": "TRB", "BETA": "TRB",
    "TRG": "TRG", "TCRG": "TRG", "GAMMA": "TRG",
    "TRD": "TRD", "TCRD": "TRD", "DELTA": "TRD",
    "IGH": "IGH", "IGK": "IGK", "IGL": "IGL",
}

@dataclass(frozen=True)
class AIRRSchema:
    sample_id: str = "sample_id"
    donor_id: str = "donor_id"
    sequence_aa: str = "cdr3_aa"
    chain: str = "chain"
    count: str = "count"
    productive: str = "productive"
    v_gene: str = "v_gene"
    j_gene: str = "j_gene"
    batch: str = "batch"
    label: str = "label"

@dataclass(frozen=True)
class QCConfig:
    min_unique_cdr3: int = 300
    min_total_count: int = 1000
    min_productive_fraction: float = 0.80
    min_cdr3_length: int = 5
    max_cdr3_length: int = 40
    require_productive: bool = True
    collapse_alleles: bool = True


def _to_bool(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x.fillna(False)
    return x.astype(str).str.lower().isin({"true", "t", "1", "yes", "productive"})


def _base_gene(x: pd.Series) -> pd.Series:
    return x.fillna("").astype(str).str.split("*").str[0]


def infer_donor_groups(sample_ids: Iterable[str], donor_ids: Optional[Iterable[str]] = None) -> np.ndarray:
    """Return leakage-safe group IDs; strips common technical replicate suffixes.

    Explicit donor IDs always win. If absent, sample IDs such as X_r, X_r2,
    X-rep1 and X.tech2 are conservatively grouped together.
    """
    samples = pd.Series(list(sample_ids), dtype="string")
    if donor_ids is not None:
        donors = pd.Series(list(donor_ids), dtype="string")
        if len(donors) != len(samples):
            raise ValueError("donor_ids and sample_ids must have equal length")
        missing = donors.isna() | donors.str.strip().eq("")
        donors.loc[missing] = samples.loc[missing]
    else:
        donors = samples.copy()
    pattern = r"(?i)(?:[_\-.](?:r\d*|rep(?:licate)?\d*|tech(?:nical)?\d*))$"
    return donors.str.replace(pattern, "", regex=True).astype(str).to_numpy()


def validate_airr(
    frame: pd.DataFrame,
    schema: AIRRSchema = AIRRSchema(),
    config: QCConfig = QCConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and normalize an AIRR-like clone table.

    Returns (clean_clone_table, sample_qc). Invalid sequences are removed, but
    every exclusion is reflected in per-sample QC. `qc_pass` is intentionally
    conservative and can be used to gate model inference.
    """
    required = {schema.sample_id, schema.sequence_aa, schema.chain, schema.count}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required AIRR columns: {sorted(missing)}")

    df = frame.copy()
    df[schema.sample_id] = df[schema.sample_id].astype(str).str.strip()
    df[schema.sequence_aa] = df[schema.sequence_aa].astype(str).str.upper().str.strip()
    raw_chain = df[schema.chain].astype(str).str.upper().str.strip()
    df[schema.chain] = raw_chain.map(CHAIN_ALIASES)
    df[schema.count] = pd.to_numeric(df[schema.count], errors="coerce")
    df["_valid_count"] = df[schema.count].notna() & (df[schema.count] > 0)
    df["_valid_chain"] = df[schema.chain].notna()
    lengths = df[schema.sequence_aa].str.len()
    df["_valid_sequence"] = (
        df[schema.sequence_aa].map(lambda x: bool(AA_RE.fullmatch(x)))
        & lengths.between(config.min_cdr3_length, config.max_cdr3_length)
    )
    if schema.productive in df:
        df["_productive"] = _to_bool(df[schema.productive])
    else:
        df["_productive"] = True

    pre = df.groupby(schema.sample_id, sort=False).agg(
        input_rows=(schema.sequence_aa, "size"),
        total_count_input=(schema.count, lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()),
        productive_fraction=("_productive", "mean"),
        invalid_sequence_fraction=("_valid_sequence", lambda x: 1.0 - float(x.mean())),
        invalid_chain_fraction=("_valid_chain", lambda x: 1.0 - float(x.mean())),
    )

    keep = df["_valid_count"] & df["_valid_chain"] & df["_valid_sequence"]
    if config.require_productive:
        keep &= df["_productive"]
    clean = df.loc[keep].copy()
    if config.collapse_alleles:
        for col in (schema.v_gene, schema.j_gene):
            if col in clean:
                clean[col] = _base_gene(clean[col])

    key = [schema.sample_id, schema.sequence_aa, schema.chain]
    for optional in (schema.v_gene, schema.j_gene):
        if optional in clean:
            key.append(optional)
    clean = clean.groupby(key, dropna=False, as_index=False)[schema.count].sum()

    post = clean.groupby(schema.sample_id, sort=False).agg(
        unique_cdr3=(schema.sequence_aa, "nunique"),
        total_count=(schema.count, "sum"),
        n_chains=(schema.chain, "nunique"),
    )
    qc = pre.join(post, how="left").fillna({"unique_cdr3": 0, "total_count": 0, "n_chains": 0})
    qc["depth_confidence"] = np.minimum(1.0, qc["unique_cdr3"] / max(config.min_unique_cdr3, 1))
    qc["qc_pass"] = (
        (qc["unique_cdr3"] >= config.min_unique_cdr3)
        & (qc["total_count"] >= config.min_total_count)
        & (qc["productive_fraction"] >= config.min_productive_fraction)
        & (qc["invalid_chain_fraction"] == 0)
    )
    qc["qc_reason"] = "PASS"
    qc.loc[qc["unique_cdr3"] < config.min_unique_cdr3, "qc_reason"] = "LOW_UNIQUE_CDR3"
    qc.loc[qc["total_count"] < config.min_total_count, "qc_reason"] = "LOW_TOTAL_COUNT"
    qc.loc[qc["productive_fraction"] < config.min_productive_fraction, "qc_reason"] = "LOW_PRODUCTIVE_FRACTION"
    qc.loc[qc["invalid_chain_fraction"] > 0, "qc_reason"] = "INVALID_CHAIN"
    return clean.reset_index(drop=True), qc.reset_index()
