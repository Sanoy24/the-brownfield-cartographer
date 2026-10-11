"""Bounded local file inventory shared by all analysis stages."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path, PureWindowsPath

from pathspec import GitIgnoreSpec

SOURCE_EXTENSIONS = {".py", ".sql", ".yml", ".yaml", ".js", ".ts"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cartography",
             ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox", "build", "dist",
             ".agents", ".codex", ".aws"}


@dataclass(frozen=True)
class ScanSettings:
    max_file_bytes: int = 2 * 1024 * 1024
    max_total_bytes: int = 100 * 1024 * 1024
    max_files: int = 20_000
    max_entries: int = 100_000
    max_depth: int = 128
    dbt_manifest: str | None = None

    def __post_init__(self) -> None:
        if min(self.max_file_bytes, self.max_total_bytes, self.max_files, self.max_entries, self.max_depth) <= 0:
            raise ValueError("Scanner limits must be positive")
        if self.dbt_manifest is not None:
            path = Path(self.dbt_manifest)
            if (not self.dbt_manifest or len(self.dbt_manifest) > 1000 or "\\" in self.dbt_manifest
                    or path.is_absolute() or PureWindowsPath(self.dbt_manifest).drive
                    or ".." in path.parts or path.suffix != ".json"
                    or any(part in {".git", ".aws", ".agents", ".codex", ".cartography"}
                           or part.startswith(".env") for part in path.parts)):
                raise ValueError("dbt manifest must be a repository-relative JSON path outside protected directories")


@dataclass(frozen=True)
class ScanDiagnostic:
    path: str
    reason: str
    incomplete: bool = False


@dataclass
class ScanResult:
    root: Path
    files: tuple[Path, ...] = ()
    fingerprints: dict[str, str] = field(default_factory=dict)
    diagnostics: list[ScanDiagnostic] = field(default_factory=list)
    complete: bool = True
    dbt_manifest: Path | None = None
    dbt_manifest_content: str | None = field(default=None, repr=False)

    def report(self) -> dict:
        return {"complete": self.complete, "files": len(self.files),
                "diagnostics": [vars(d) for d in self.diagnostics],
                "dbt_manifest": self.dbt_manifest.relative_to(self.root).as_posix() if self.dbt_manifest else None}


def _link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def scan_repository(root: Path, settings: ScanSettings | None = None) -> ScanResult:
    """Respect nested git/tool ignores without executing repository code.

    Symlinks/junctions are never followed. Ignore files participate in the
    fingerprint so policy edits invalidate previous inventories. This is a
    static filesystem boundary, not an OS sandbox against concurrent mutation.
    """
    settings = settings or ScanSettings()
    if _link(root):
        raise ValueError("Repository root must not be a symlink or junction")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Repository root must be a directory")
    result = ScanResult(root=root)
    files: list[Path] = []
    visited = total_bytes = 0

    def diagnostic(path: Path, reason: str, incomplete: bool = False) -> None:
        if len(result.diagnostics) < 1000:
            result.diagnostics.append(ScanDiagnostic(path.relative_to(root).as_posix(), reason, incomplete))
        if incomplete:
            result.complete = False

    def read(path: Path) -> bytes | None:
        nonlocal total_bytes
        try:
            if _link(path) or not path.resolve().is_relative_to(root):
                diagnostic(path, "symlink", True)
                return None
            if path.stat().st_size > settings.max_file_bytes:
                diagnostic(path, "oversized", True)
                return None
            with path.open("rb") as stream:
                data = stream.read(settings.max_file_bytes + 1)
            if len(data) > settings.max_file_bytes:
                diagnostic(path, "oversized", True)
                return None
            total_bytes += len(data)
            if total_bytes > settings.max_total_bytes:
                diagnostic(path, "byte_limit", True)
                return None
            if b"\0" in data:
                diagnostic(path, "binary", True)
                return None
            data.decode("utf-8")
            return data
        except (OSError, UnicodeError):
            diagnostic(path, "unreadable_or_non_utf8", True)
            return None

    def walk(directory: Path, rules: list[tuple[Path, GitIgnoreSpec]]) -> None:
        nonlocal visited
        if len(directory.relative_to(root).parts) > settings.max_depth:
            diagnostic(directory, "depth_limit", True)
            return
        local_rules = list(rules)
        for name in (".gitignore", ".cartographerignore"):
            policy = directory / name
            if policy.exists() or policy.is_symlink():
                data = read(policy)
                if data is not None:
                    result.fingerprints[policy.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
                    local_rules.append((directory, GitIgnoreSpec.from_lines(data.decode().splitlines())))
        try:
            # Bound enumeration as well as accepted files; prune before descent.
            with os.scandir(directory) as entries:
                paths = []
                for entry in entries:
                    visited += 1
                    if visited > settings.max_entries:
                        diagnostic(directory, "entry_limit", True)
                        return
                    paths.append(Path(entry.path))
            for path in sorted(paths):
                if visited > settings.max_entries or total_bytes > settings.max_total_bytes:
                    return
                if _link(path):
                    diagnostic(path, "symlink", True)
                    continue
                is_dir = path.is_dir()
                if is_dir and (path.name in SKIP_DIRS or path.name.endswith(".egg-info")):
                    continue
                ignored = False
                for base, spec in local_rules:
                    decision = spec.check_file(path.relative_to(base).as_posix() + ("/" if is_dir else ""))
                    if decision.include is not None:
                        ignored = decision.include
                if ignored:
                    continue
                if is_dir:
                    walk(path, local_rules)
                    continue
                if path.name in {".gitignore", ".cartographerignore"}:
                    continue
                if (path.name.startswith(".env") or path.suffix.lower() in {".pem", ".key", ".p12"}
                        or path.name.lower() in {"credentials", "id_rsa", "id_ed25519"}):
                    diagnostic(path, "secret_filename")
                    continue
                if path.suffix.lower() not in SOURCE_EXTENSIONS:
                    continue
                if len(files) >= settings.max_files:
                    diagnostic(path, "file_limit", True)
                    return
                data = read(path)
                if data is not None:
                    files.append(path)
                    result.fingerprints[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
        except OSError:
            diagnostic(directory, "unreadable_directory", True)

    walk(root, [])
    # A selected generated artifact is an explicit exception to ignore rules.
    # It still obeys repository scope, ancestor-link and resource boundaries.
    if settings.dbt_manifest is not None:
        manifest = root / settings.dbt_manifest
        result.dbt_manifest = manifest
        ancestors = (manifest, *manifest.parents)
        if any(_link(path) for path in ancestors if path.is_relative_to(root)):
            diagnostic(manifest, "symlink", True)
        elif len(manifest.relative_to(root).parts) > settings.max_depth:
            diagnostic(manifest, "depth_limit", True)
        elif len(files) >= settings.max_files:
            diagnostic(manifest, "file_limit", True)
        else:
            data = read(manifest)
            if data is not None:
                result.dbt_manifest_content = data.decode("utf-8")
                result.fingerprints[manifest.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
    result.files = tuple(sorted(files))
    result.fingerprints = dict(sorted(result.fingerprints.items()))
    return result
