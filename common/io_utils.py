"""
io_utils.py — Shared filesystem helpers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and any parents) if it does not exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> None:
    """Serialise *data* to a pretty-printed JSON file at *path*."""
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
