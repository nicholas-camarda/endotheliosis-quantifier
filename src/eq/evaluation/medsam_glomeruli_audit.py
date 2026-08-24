"""Deterministic cohort and split audits for MedSAM glomerular evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

_STRONG_GROUP_KEYS = ('subject_id', 'specimen_id', 'source_sample_id', 'split_group_id')
_SOURCE_GROUP_KEYS = ('source_sample_id', 'split_group_id', 'subject_id', 'specimen_id')
_ROW_COLUMNS = (
    'manifest_row_id',
    'split',
    'cohort_id',
    'lane_assignment',
    'strongest_grouping_key',
    'strongest_grouping_value',
    'source_group_value',
    'image_path',
    'mask_path',
    'image_sha256',
    'mask_sha256',
    'mask_foreground_pixels',
    'mask_status',
)


def _text(value: Any) -> str:
    return str(value or '').strip()


def _runtime_path(value: Any, runtime_root: Path) -> tuple[Path | None, str | None]:
    raw = _text(value)
    if not raw:
        return None, 'missing_path_value'
    candidate = Path(raw).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (runtime_root / candidate).resolve()
    try:
        resolved.relative_to(runtime_root)
    except ValueError:
        return None, 'path_outside_runtime_root'
    return resolved, None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _first_available_key(row: dict[str, Any], keys: Iterable[str]) -> tuple[str, str]:
    for key in keys:
        value = _text(row.get(key))
        if value:
            return key, value
    return '', ''


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        text = _text(value) or '<missing>'
        counts[text] += 1
    return {key: counts[key] for key in sorted(counts)}


def _overlap(values_by_split: dict[str, set[str]]) -> dict[str, Any]:
    overlaps = [
        {'value': value, 'splits': sorted(splits)}
        for value, splits in values_by_split.items()
        if len(splits) > 1
    ]
    overlaps.sort(key=lambda item: item['value'])
    return {'values': [item['value'] for item in overlaps], 'details': overlaps}


def _issue(
    issues: list[dict[str, Any]],
    *,
    reason_code: str,
    row_id: str = '',
    field: str = '',
    value: str = '',
) -> None:
    item: dict[str, Any] = {'reason_code': reason_code}
    if row_id:
        item['manifest_row_id'] = row_id
    if field:
        item['field'] = field
    if value:
        item['value'] = value
    issues.append(item)


def audit_medsam_evaluation_cohort(
    rows: pd.DataFrame | Iterable[dict[str, Any]], runtime_root: Path | str
) -> dict[str, Any]:
    """Audit the exact rows used by a MedSAM evaluation run.

    The result is JSON serializable and intentionally treats path, duplicate,
    and cross-split failures as promotion-blocking evidence. A lack of empty
    masks is reported as a distribution limitation rather than manufactured
    negative evidence.
    """
    root = Path(runtime_root).expanduser().resolve()
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    frame = frame.fillna('')
    records = frame.to_dict(orient='records')
    issues: list[dict[str, Any]] = []
    for record in records:
        row_id = _text(record.get('manifest_row_id'))
        if not row_id:
            _issue(issues, reason_code='missing_manifest_row_id')
        if not _text(record.get('split')):
            _issue(issues, reason_code='missing_split_metadata', row_id=row_id)

    strongest_key = next(
        (
            key
            for key in _STRONG_GROUP_KEYS
            if any(_text(record.get(key)) for record in records)
        ),
        'manifest_row_id',
    )
    source_key = next(
        (
            key
            for key in _SOURCE_GROUP_KEYS
            if any(_text(record.get(key)) for record in records)
        ),
        'manifest_row_id',
    )

    image_hash_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    mask_hash_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    split_values: dict[str, dict[str, set[str]]] = {
        'strongest_grouping_key': defaultdict(set),
        'source_groups': defaultdict(set),
        'images': defaultdict(set),
        'manifest_rows': defaultdict(set),
    }
    row_audit: list[dict[str, Any]] = []
    foreground_pixels: list[int] = []
    positive_rows = 0
    background_rows = 0

    for record in records:
        row_id = _text(record.get('manifest_row_id'))
        split = _text(record.get('split'))
        strongest_value = _text(record.get(strongest_key))
        source_value = _text(record.get(source_key))
        if not strongest_value:
            _issue(
                issues,
                reason_code='missing_strongest_grouping_value',
                row_id=row_id,
                field=strongest_key,
            )

        image_value = _text(record.get('image_path_resolved')) or _text(
            record.get('image_path')
        )
        mask_value = _text(record.get('mask_path_resolved')) or _text(
            record.get('mask_path')
        )
        image_path, image_path_issue = _runtime_path(image_value, root)
        mask_path, mask_path_issue = _runtime_path(mask_value, root)
        if image_path_issue:
            _issue(
                issues,
                reason_code=image_path_issue,
                row_id=row_id,
                field='image_path',
                value=image_value,
            )
        if mask_path_issue:
            _issue(
                issues,
                reason_code=mask_path_issue,
                row_id=row_id,
                field='mask_path',
                value=mask_value,
            )

        image_sha256 = ''
        mask_sha256 = ''
        mask_status = 'unavailable'
        foreground_count: int | None = None
        if image_path is not None:
            if not image_path.is_file():
                _issue(
                    issues,
                    reason_code='missing_image_file',
                    row_id=row_id,
                    field='image_path',
                    value=str(image_path),
                )
            else:
                try:
                    image_sha256 = _hash_file(image_path)
                except OSError:
                    _issue(
                        issues,
                        reason_code='unreadable_image_file',
                        row_id=row_id,
                        field='image_path',
                        value=str(image_path),
                    )
        if mask_path is not None:
            if not mask_path.is_file():
                _issue(
                    issues,
                    reason_code='missing_mask_file',
                    row_id=row_id,
                    field='mask_path',
                    value=str(mask_path),
                )
            else:
                try:
                    mask_sha256 = _hash_file(mask_path)
                    mask = np.asarray(Image.open(mask_path).convert('L'))
                    foreground_count = int(np.count_nonzero(mask))
                    foreground_pixels.append(foreground_count)
                    if foreground_count > 0:
                        positive_rows += 1
                        mask_status = 'positive'
                    else:
                        background_rows += 1
                        mask_status = 'background'
                except (OSError, ValueError):
                    _issue(
                        issues,
                        reason_code='unreadable_mask_file',
                        row_id=row_id,
                        field='mask_path',
                        value=str(mask_path),
                    )

        if image_sha256:
            image_hash_rows[image_sha256].append(
                {'manifest_row_id': row_id, 'path': str(image_path)}
            )
        if mask_sha256:
            mask_hash_rows[mask_sha256].append(
                {'manifest_row_id': row_id, 'path': str(mask_path)}
            )
        if split:
            if strongest_value:
                split_values['strongest_grouping_key'][strongest_value].add(split)
            if source_value:
                split_values['source_groups'][source_value].add(split)
            if image_path is not None:
                split_values['images'][str(image_path)].add(split)
            if row_id:
                split_values['manifest_rows'][row_id].add(split)
        row_audit.append(
            {
                'manifest_row_id': row_id,
                'split': split,
                'cohort_id': _text(record.get('cohort_id')),
                'lane_assignment': _text(record.get('lane_assignment')),
                'strongest_grouping_key': strongest_key,
                'strongest_grouping_value': strongest_value,
                'source_group_value': source_value,
                'image_path': str(image_path) if image_path is not None else image_value,
                'mask_path': str(mask_path) if mask_path is not None else mask_value,
                'image_sha256': image_sha256,
                'mask_sha256': mask_sha256,
                'mask_foreground_pixels': foreground_count,
                'mask_status': mask_status,
            }
        )

    duplicate_images = [
        {'sha256': digest, 'row_ids': sorted(item['manifest_row_id'] for item in items), 'paths': sorted({item['path'] for item in items})}
        for digest, items in image_hash_rows.items()
        if len(items) > 1
    ]
    duplicate_masks = [
        {'sha256': digest, 'row_ids': sorted(item['manifest_row_id'] for item in items), 'paths': sorted({item['path'] for item in items})}
        for digest, items in mask_hash_rows.items()
        if len(items) > 1
    ]
    duplicate_images.sort(key=lambda item: item['sha256'])
    duplicate_masks.sort(key=lambda item: item['sha256'])
    for item in duplicate_images:
        _issue(issues, reason_code='duplicate_image_content_hash', value=item['sha256'])
    for item in duplicate_masks:
        _issue(issues, reason_code='duplicate_mask_content_hash', value=item['sha256'])

    overlap = {
        key: _overlap(values) for key, values in split_values.items()
    }
    overlap_reason_codes = {
        'strongest_grouping_key': 'cross_split_strong_group_leakage',
        'source_groups': 'cross_split_source_group_leakage',
        'images': 'cross_split_image_leakage',
        'manifest_rows': 'cross_split_manifest_row_leakage',
    }
    for key, reason_code in overlap_reason_codes.items():
        for value in overlap[key]['values']:
            _issue(issues, reason_code=reason_code, value=value)

    if foreground_pixels and background_rows == 0:
        _issue(issues, reason_code='positive_only_evidence_distribution_limitation')
    if not records:
        _issue(issues, reason_code='empty_evaluation_cohort')

    blocking_codes = {
        issue['reason_code']
        for issue in issues
        if issue['reason_code'] != 'positive_only_evidence_distribution_limitation'
    }
    issues.sort(
        key=lambda item: (
            item.get('reason_code', ''),
            item.get('manifest_row_id', ''),
            item.get('field', ''),
            item.get('value', ''),
        )
    )
    if blocking_codes:
        independence_status = 'blocked'
    elif any(
        issue['reason_code'] == 'positive_only_evidence_distribution_limitation'
        for issue in issues
    ):
        independence_status = 'independent_with_distribution_limitation'
    else:
        independence_status = 'independent'

    payload: dict[str, Any] = {
        'row_counts': {
            'total': len(records),
            'by_split': _counts(record.get('split') for record in records),
            'by_cohort': _counts(record.get('cohort_id') for record in records),
            'by_lane': _counts(record.get('lane_assignment') for record in records),
        },
        'unique_counts': {
            'manifest_rows': len({_text(record.get('manifest_row_id')) for record in records if _text(record.get('manifest_row_id'))}),
            'images': len({item['image_path'] for item in row_audit if item['image_path']}),
            'masks': len({item['mask_path'] for item in row_audit if item['mask_path']}),
            'source_groups': len({_text(record.get(source_key)) for record in records if _text(record.get(source_key))}),
            'strongest_grouping_key': strongest_key,
            'strongest_grouping_values': len({_text(record.get(strongest_key)) for record in records if _text(record.get(strongest_key))}),
        },
        'duplicate_content_hashes': {
            'images': duplicate_images,
            'masks': duplicate_masks,
        },
        'mask_composition': {
            'positive_rows': positive_rows,
            'background_rows': background_rows,
            'foreground_pixel_quantiles': {
                label: float(np.quantile(foreground_pixels, quantile))
                if foreground_pixels
                else None
                for label, quantile in (
                    ('p0', 0.0),
                    ('p25', 0.25),
                    ('p50', 0.5),
                    ('p75', 0.75),
                    ('p100', 1.0),
                )
            },
        },
        'cross_split_overlap': overlap,
        'independence_status': independence_status,
        'promotion_blocked': bool(blocking_codes),
        'blocking_reason_codes': sorted(blocking_codes),
        'issues': issues,
        'row_audit': sorted(row_audit, key=lambda item: item['manifest_row_id']),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    payload['audit_sha256'] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return payload


def write_medsam_evaluation_cohort_audit(
    rows: pd.DataFrame | Iterable[dict[str, Any]],
    runtime_root: Path | str,
    evaluation_dir: Path | str,
) -> dict[str, Any]:
    """Write deterministic JSON and CSV audit artifacts beneath one run root."""
    output_dir = Path(evaluation_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_medsam_evaluation_cohort(rows, runtime_root)
    json_path = output_dir / 'dataset_split_audit.json'
    csv_path = output_dir / 'dataset_split_audit.csv'
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding='utf-8')
    row_frame = pd.DataFrame(audit['row_audit'], columns=_ROW_COLUMNS)
    row_frame.to_csv(csv_path, index=False)
    audit['artifacts'] = {'json': str(json_path), 'csv': str(csv_path)}
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding='utf-8')
    return audit
