"""Atomic writes for individual index and checkpoint artifacts."""
import json
import os
from pathlib import Path
import tempfile


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Refusing to write through an artifact symlink")
    staging: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".cartographer-", delete=False) as stream:
            staging = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        if staging is not None and staging.exists():
            staging.unlink()
