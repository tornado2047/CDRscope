"""Versioned reference-panel manifests and independence checks."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def hash_ids(values: Iterable[str]):
    text = "\n".join(sorted({str(v) for v in values}))
    return hashlib.sha256(text.encode()).hexdigest()

@dataclass
class PanelManifest:
    panel_id: str
    version: str
    chain: str
    embedding_model: str
    embedding_layer: int
    pooling: str
    n_prototypes: int
    assignment_metric: str = "euclidean"
    training_sources: List[str] = field(default_factory=list)
    training_sample_hash: str = ""
    training_sequence_hash: str = ""
    centroid_sha256: str = ""
    random_seed: int = 42
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    applicability: Dict[str, object] = field(default_factory=dict)

    def validate(self):
        if self.chain not in {"TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"}:
            raise ValueError(f"unsupported chain: {self.chain}")
        if self.n_prototypes < 2:
            raise ValueError("n_prototypes must be >= 2")
        if not self.panel_id or not self.version or not self.centroid_sha256:
            raise ValueError("panel_id, version and centroid_sha256 are required")
        return self

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()


def verify_panel_independence(manifest: PanelManifest, evaluation_sample_ids: Iterable[str],
                              known_training_sample_ids: Optional[Iterable[str]] = None):
    """Fail closed when panel/sample independence cannot be demonstrated.

    Raw training IDs are optional for public manifests. When supplied, exact
    overlap is tested; otherwise the function explicitly returns `unverified`
    rather than silently claiming independence from a one-way hash.
    """
    eval_ids = {str(x) for x in evaluation_sample_ids}
    if known_training_sample_ids is None:
        return {"status": "unverified", "overlap": None,
                "reason": "training sample IDs not supplied; hash alone cannot prove non-membership"}
    train_ids = {str(x) for x in known_training_sample_ids}
    overlap = sorted(eval_ids.intersection(train_ids))
    if overlap:
        raise ValueError(f"Evaluation samples occur in panel training data: {overlap[:10]}")
    return {"status": "independent", "overlap": 0,
            "evaluation_sample_hash": hash_ids(eval_ids)}
