"""Content-addressed embedding cache for million-sequence workflows."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Callable, Iterable, Sequence
import numpy as np


class EmbeddingCache:
    def __init__(self, cache_dir, model_id, dtype="float32"):
        self.root = Path(cache_dir) / hashlib.sha256(model_id.encode()).hexdigest()[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.dtype = np.dtype(dtype)

    def _path(self, sequence):
        key = hashlib.sha256(sequence.encode()).hexdigest()
        return self.root / key[:2] / f"{key}.npy"

    def get_many(self, sequences: Sequence[str], embed_fn: Callable[[Sequence[str]], np.ndarray], batch_size=256):
        """Read cached embeddings and compute each missing unique sequence once."""
        sequences = [str(x) for x in sequences]
        unique = list(dict.fromkeys(sequences))
        missing = [s for s in unique if not self._path(s).is_file()]
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            vectors = np.asarray(embed_fn(batch), dtype=self.dtype)
            if len(vectors) != len(batch):
                raise ValueError("embed_fn returned the wrong number of embeddings")
            for seq, vector in zip(batch, vectors):
                path = self._path(seq); path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.npy")
                np.save(tmp, vector)
                tmp.replace(path)
        lookup = {s: np.load(self._path(s), allow_pickle=False) for s in unique}
        return np.stack([lookup[s] for s in sequences])

    def manifest(self):
        return {"model_id": self.model_id, "dtype": str(self.dtype), "cache_dir": str(self.root)}
