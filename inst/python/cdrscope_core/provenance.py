"""Reproducible run manifests: configuration, code, inputs and environment."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
from typing import Dict, Iterable, Optional

from .panel import sha256_file


def _json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

@dataclass
class RunTracker:
    output_dir: str
    config: Dict[str, object]
    repo_dir: Optional[str] = None

    def start(self, inputs: Iterable[str] = ()): 
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        input_hashes = {str(Path(p).resolve()): sha256_file(p) for p in inputs if Path(p).is_file()}
        commit = _git(self.repo_dir or Path.cwd(), "rev-parse", "HEAD")
        dirty = _git(self.repo_dir or Path.cwd(), "status", "--porcelain")
        payload = {
            "run_id": _json_hash({"config": self.config, "inputs": input_hashes, "commit": commit})[:16],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "config_sha256": _json_hash(self.config),
            "inputs": input_hashes,
            "git_commit": commit,
            "git_dirty": bool(dirty),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "executable": sys.executable,
            },
        }
        path = out / "run_manifest.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self.manifest_ = payload
        self.path_ = str(path)
        return payload

    def finish(self, results: Dict[str, object]):
        if not hasattr(self, "manifest_"):
            raise RuntimeError("start must be called before finish")
        payload = dict(self.manifest_)
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        payload["results"] = results
        Path(self.path_).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return payload
