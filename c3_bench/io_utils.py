"""Filesystem and JSON helpers shared by inference and evaluation."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def sanitize_model_id(model_id: str) -> str:
    """Turn a model id into a filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def model_basename(model_id: str) -> str:
    """Return the final component of a HuggingFace id (``org/name`` -> ``name``)."""
    return model_id.rsplit("/", 1)[-1]


def resolve_image(path_str: str, project_root: Path) -> Path:
    """Resolve a possibly-relative image path against ``project_root``."""
    path = Path(path_str)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def load_items(json_path: Path) -> list[dict[str, Any]]:
    """Load a JSON file and normalise it to a list of records."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def infer_domain_topic(item: dict[str, Any]) -> tuple[str, str]:
    """Determine ``(domain, topic)`` for an item.

    Prefers explicit ``domain``/``topic`` fields and otherwise infers them from
    an image path of the form ``data/<domain>/<topic>/...``. Falls back to
    ``"unknown"`` for any component that cannot be determined.
    """
    domain = item.get("domain")
    topic = item.get("topic")
    if domain and topic:
        return domain, topic
    for key in ("image_t0_path", "image_t1_path"):
        path = item.get(key)
        if not path:
            continue
        parts = Path(path).as_posix().split("/")
        if len(parts) >= 3 and parts[0] == "data":
            return parts[1], parts[2]
    return domain or "unknown", topic or "unknown"


def atomic_json_dump(
    obj: Any,
    out_path: str | Path,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    """Write ``obj`` as JSON atomically via a temp file and ``os.replace``.

    The temp file is created in the destination directory so that the final
    ``os.replace`` is atomic on the same filesystem.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".tmp", dir=str(out_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
            f.flush()
            os.fsync(f.fileno())  # force bytes to disk before the swap
        os.replace(tmp_path, out_path)  # atomic swap
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
