"""Portable path contract for runtime-owned generated artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

GENERATED_MASK_RELEASE_SCHEMA_VERSION = 'eq.medsam.generated-mask-release/1'
RUNTIME_RELATIVE_POSIX_ENCODING = 'runtime-relative-posix'

_RELEASE_ROW_PATH_FIELDS = (
    'source_image_path',
    'reference_mask_path',
    'generated_mask_path',
    'checkpoint_path',
    'provenance_path',
    'release_manifest_path',
    'metric_evidence_path',
)
_GENERATED_RELEASE_REQUIRED_PATH_FIELDS = frozenset(
    {
        'source_image_path',
        'generated_mask_path',
        'checkpoint_path',
        'provenance_path',
    }
)


class ArtifactProvenanceError(ValueError):
    """Raised when generated-artifact provenance is unsafe or stale."""


def serialize_runtime_relative_path(
    path: Path | str, *, runtime_root: Path | str
) -> str:
    """Encode an in-root runtime path as a portable POSIX relative path."""
    root = _runtime_root(runtime_root)
    raw_value = str(path)
    raw_path = Path(raw_value).expanduser()
    if not raw_path.is_absolute() and PureWindowsPath(raw_value).is_absolute():
        raise ArtifactProvenanceError(
            f'Windows absolute paths are not permitted in runtime-relative provenance: {raw_value}'
        )
    if '\\' in raw_value and not raw_path.is_absolute():
        raw_path = Path(*PureWindowsPath(raw_value).parts)
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


def validate_generated_mask_artifact_paths(
    record: Mapping[str, Any], *, runtime_root: Path | str
) -> dict[str, Path]:
    """Validate every serialized artifact path in one release or registry row."""
    resolved: dict[str, Path] = {}
    for field in _RELEASE_ROW_PATH_FIELDS:
        value = str(record.get(field) or '').strip()
        if value:
            resolved[field] = resolve_runtime_relative_path(
                value,
                runtime_root=runtime_root,
                require_exists=True,
                artifact_label=field,
            )

    status = str(record.get('generation_status') or '').strip().lower()
    required_fields = (
        _GENERATED_RELEASE_REQUIRED_PATH_FIELDS if status == 'generated' else set()
    )
    if 'release_manifest_path' in record:
        required_fields = set(required_fields) | {'release_manifest_path'}
    missing = sorted(
        field for field in required_fields if not str(record.get(field) or '').strip()
    )
    if missing:
        raise ArtifactProvenanceError(
            f'Generated-mask provenance is stale or incomplete; missing path fields: {missing}'
        )

    provenance_path = resolved.get('provenance_path')
    if provenance_path is not None:
        _validate_provenance_document(provenance_path, runtime_root=runtime_root)
    return resolved


def _validate_provenance_document(
    provenance_path: Path, *, runtime_root: Path | str
) -> None:
    try:
        payload = json.loads(provenance_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactProvenanceError(
            f'Generated-mask provenance document is unreadable: {provenance_path}'
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactProvenanceError(
            f'Generated-mask provenance document must be an object: {provenance_path}'
        )
    if payload.get('schema_version') != GENERATED_MASK_RELEASE_SCHEMA_VERSION:
        raise ArtifactProvenanceError(
            'Generated-mask provenance document has unsupported schema_version'
        )
    if payload.get('path_encoding') != RUNTIME_RELATIVE_POSIX_ENCODING:
        raise ArtifactProvenanceError(
            'Generated-mask provenance document has unsupported path_encoding'
        )
    for field in ('checkpoint_path', 'manifest_path', 'metric_evidence_path'):
        if field == 'metric_evidence_path' and not str(payload.get(field) or '').strip():
            continue
        resolve_runtime_relative_path(
            str(payload.get(field) or ''),
            runtime_root=runtime_root,
            require_exists=True,
            artifact_label=field,
        )


def _runtime_root(runtime_root: Path | str) -> Path:
    if runtime_root is None:
        raise ArtifactProvenanceError('runtime_root must be explicit')
    return Path(runtime_root).expanduser().resolve()


def _reject_traversal(path: Path | PurePosixPath, *, value: str) -> None:
    if '..' in path.parts:
        raise ArtifactProvenanceError(
            f'Runtime-relative path traversal is not permitted: {value}'
        )
