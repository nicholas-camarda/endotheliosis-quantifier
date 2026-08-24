"""Portable path contract for runtime-owned generated artifacts."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

GENERATED_MASK_RELEASE_SCHEMA_VERSION = 'eq.medsam.generated-mask-release/1'
RUNTIME_RELATIVE_POSIX_ENCODING = 'runtime-relative-posix'


class ArtifactProvenanceError(ValueError):
    """Raised when generated-artifact provenance is unsafe or stale."""


def serialize_runtime_relative_path(
    path: Path | str, *, runtime_root: Path | str
) -> str:
    """Encode an in-root runtime path as a portable POSIX relative path."""
    root = _runtime_root(runtime_root)
    raw_path = Path(path).expanduser()
    _reject_traversal(raw_path, value=str(path))
    resolved_path = (
        raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
    )
    try:
        return resolved_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ArtifactProvenanceError(
            f'Path is outside runtime_root: {resolved_path} (runtime_root={root})'
        ) from exc


def resolve_runtime_relative_path(
    value: str,
    *,
    runtime_root: Path | str,
    require_exists: bool = False,
    artifact_label: str = 'artifact',
) -> Path:
    """Resolve a serialized runtime-relative POSIX path without trusting process CWD."""
    root = _runtime_root(runtime_root)
    path_value = str(value or '').strip()
    if not path_value:
        raise ArtifactProvenanceError('Runtime-relative path is required')
    if (
        Path(path_value).is_absolute()
        or PurePosixPath(path_value).is_absolute()
        or PureWindowsPath(path_value).is_absolute()
    ):
        raise ArtifactProvenanceError(
            f'Stale absolute path is not permitted in runtime-relative provenance: {path_value}'
        )
    if '\\' in path_value:
        raise ArtifactProvenanceError(
            f'Runtime-relative path must use POSIX separators: {path_value}'
        )
    encoded_path = PurePosixPath(path_value)
    _reject_traversal(encoded_path, value=path_value)
    resolved_path = (root / Path(*encoded_path.parts)).resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise ArtifactProvenanceError(
            f'Path is outside runtime_root: {resolved_path} (runtime_root={root})'
        ) from exc
    if require_exists and not resolved_path.is_file():
        raise ArtifactProvenanceError(f'missing {artifact_label}: {resolved_path}')
    return resolved_path


def _runtime_root(runtime_root: Path | str) -> Path:
    if runtime_root is None:
        raise ArtifactProvenanceError('runtime_root must be explicit')
    return Path(runtime_root).expanduser().resolve()


def _reject_traversal(path: Path | PurePosixPath, *, value: str) -> None:
    if '..' in path.parts:
        raise ArtifactProvenanceError(
            f'Runtime-relative path traversal is not permitted: {value}'
        )
