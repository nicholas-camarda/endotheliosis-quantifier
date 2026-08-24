"""Deterministic cohort and split audits for MedSAM glomerular evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

_STRONG_GROUP_KEYS = ('subject_id', 'specimen_id', 'source_sample_id', 'split_group_id')
_SOURCE_GROUP_KEYS = ('source_sample_id', 'split_group_id', 'subject_id', 'specimen_id')
_EXPECTED_SPLITS = frozenset({'train', 'validation', 'test'})
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
        split = _text(record.get('split'))
        if not split:
            _issue(issues, reason_code='missing_split_metadata', row_id=row_id)
        elif split not in _EXPECTED_SPLITS:
            _issue(
                issues,
                reason_code='invalid_split_label',
                row_id=row_id,
                value=split,
            )

    observed_splits = {_text(record.get('split')) for record in records}
    for split in sorted(_EXPECTED_SPLITS - observed_splits):
        _issue(issues, reason_code='missing_required_split', value=split)

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
        'split_groups': defaultdict(set),
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
            split_group_value = _text(record.get('split_group_id'))
            if split_group_value:
                split_values['split_groups'][split_group_value].add(split)
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
        'split_groups': 'cross_split_split_group_leakage',
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


def _metric_value(metrics: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = pd.to_numeric(metrics.get(key), errors='coerce')
    if pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def _box_frame(box: dict[str, Any] | None, image_shape: tuple[int, int]) -> tuple[float, float, float, float] | None:
    if not isinstance(box, dict):
        return None
    try:
        values = tuple(float(box[key]) for key in ('bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1'))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    height, width = (int(image_shape[0]), int(image_shape[1]))
    x0, y0, x1, y1 = values
    if height <= 0 or width <= 0 or x1 <= x0 or y1 <= y0:
        return None
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    return values


def box_to_encoder_frame(
    box: dict[str, Any], image_shape: tuple[int, int], *, encoder_size: int = 1024
) -> dict[str, float]:
    """Map an original-image half-open XYXY box to a square encoder frame."""
    values = _box_frame(box, image_shape)
    if values is None:
        raise ValueError('Box is outside image bounds or has invalid coordinates')
    height, width = (float(image_shape[0]), float(image_shape[1]))
    scale_x = float(encoder_size) / width
    scale_y = float(encoder_size) / height
    x0, y0, x1, y1 = values
    return {
        'bbox_x0': x0 * scale_x,
        'bbox_y0': y0 * scale_y,
        'bbox_x1': x1 * scale_x,
        'bbox_y1': y1 * scale_y,
    }


def box_from_encoder_frame(
    box: dict[str, Any], image_shape: tuple[int, int], *, encoder_size: int = 1024
) -> dict[str, int]:
    """Map a square-encoder half-open XYXY box back to image pixels."""
    try:
        encoded = tuple(
            float(box[key]) for key in ('bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1')
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError('Encoder box has invalid coordinates') from None
    if not all(math.isfinite(value) for value in encoded):
        raise ValueError('Encoder box has invalid coordinates')
    height, width = (float(image_shape[0]), float(image_shape[1]))
    if any(value < 0 or value > float(encoder_size) for value in encoded):
        raise ValueError('Encoder box is outside encoder bounds')
    x0, y0, x1, y1 = encoded
    restored = {
        'bbox_x0': int(round(x0 * width / float(encoder_size))),
        'bbox_y0': int(round(y0 * height / float(encoder_size))),
        'bbox_x1': int(round(x1 * width / float(encoder_size))),
        'bbox_y1': int(round(y1 * height / float(encoder_size))),
    }
    if _box_frame(restored, (int(height), int(width))) is None:
        raise ValueError('Restored box is outside image bounds or has invalid coordinates')
    return restored


def check_box_round_trip(
    box: dict[str, Any],
    image_shape: tuple[int, int],
    *,
    encoder_size: int = 1024,
    tolerance: int = 1,
) -> dict[str, Any]:
    """Return a JSON-safe coordinate-frame round-trip invariant result."""
    encoded = box_to_encoder_frame(box, image_shape, encoder_size=encoder_size)
    restored = box_from_encoder_frame(encoded, image_shape, encoder_size=encoder_size)
    errors = {
        key: abs(int(restored[key]) - int(box[key]))
        for key in ('bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1')
    }
    return {
        'passed': bool(max(errors.values()) <= int(tolerance)),
        'max_integer_pixel_error': int(max(errors.values())),
        'coordinate_frame': 'original_image_pixels_xyxy',
        'encoder_frame': 'square_encoder_pixels_xyxy',
        'image_shape': [int(image_shape[0]), int(image_shape[1])],
        'encoder_size': int(encoder_size),
    }


def box_diagnostics_for_row(
    gt_mask: np.ndarray,
    automatic_box: dict[str, Any] | list[dict[str, Any]] | None,
    image_shape: tuple[int, int],
    oracle_metrics: dict[str, Any] | None,
    automatic_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Separate automatic proposal geometry from paired segmentation metrics."""
    gt = np.asarray(gt_mask) > 0
    y_indices, x_indices = np.where(gt)
    oracle_dice = _metric_value(oracle_metrics, 'dice')
    oracle_jaccard = _metric_value(oracle_metrics, 'jaccard')
    automatic_dice = _metric_value(automatic_metrics, 'dice')
    automatic_jaccard = _metric_value(automatic_metrics, 'jaccard')
    result: dict[str, Any] = {
        'proposal_status': 'missing' if automatic_box is None else 'invalid',
        'proposal_count': 0,
        'box_iou': None,
        'gt_mask_coverage': None,
        'proposal_to_gt_area_ratio': None,
        'normalized_center_displacement': None,
        'automatic_metric_status': 'available'
        if automatic_dice is not None and automatic_jaccard is not None
        else 'unavailable',
        'oracle_metric_status': 'available'
        if oracle_dice is not None and oracle_jaccard is not None
        else 'unavailable',
        'automatic_dice': automatic_dice,
        'automatic_jaccard': automatic_jaccard,
        'oracle_dice': oracle_dice,
        'oracle_jaccard': oracle_jaccard,
    }
    if automatic_box is None or not len(y_indices):
        return result
    raw_boxes = automatic_box if isinstance(automatic_box, list) else [automatic_box]
    if not raw_boxes:
        return result
    frames = [_box_frame(box, image_shape) for box in raw_boxes]
    if any(frame is None for frame in frames):
        return result
    proposal_mask = np.zeros(
        (int(image_shape[0]), int(image_shape[1])), dtype=bool
    )
    for frame in frames:
        assert frame is not None
        x0, y0, x1, y1 = frame
        proposal_mask[int(y0) : int(y1), int(x0) : int(x1)] = True
    gt_x0, gt_x1 = int(x_indices.min()), int(x_indices.max()) + 1
    gt_y0, gt_y1 = int(y_indices.min()), int(y_indices.max()) + 1
    gt_box_mask = np.zeros_like(proposal_mask)
    gt_box_mask[gt_y0:gt_y1, gt_x0:gt_x1] = True
    intersection = int(np.count_nonzero(proposal_mask & gt_box_mask))
    box_area = int(np.count_nonzero(proposal_mask))
    gt_area = float((gt_x1 - gt_x0) * (gt_y1 - gt_y0))
    union = box_area + gt_area - intersection
    covered = int(np.count_nonzero(gt & proposal_mask))
    proposal_y, proposal_x = np.where(proposal_mask)
    gt_center = ((gt_x0 + gt_x1) / 2.0, (gt_y0 + gt_y1) / 2.0)
    box_center = (
        (float(proposal_x.min()) + float(proposal_x.max()) + 1.0) / 2.0,
        (float(proposal_y.min()) + float(proposal_y.max()) + 1.0) / 2.0,
    )
    diagonal = float(np.hypot(float(image_shape[1]), float(image_shape[0])))
    result.update(
        {
            'proposal_status': 'valid',
            'proposal_count': int(len(frames)),
            'box_iou': float(intersection / union) if union else 0.0,
            'gt_mask_coverage': float(covered / float(np.count_nonzero(gt))),
            'proposal_to_gt_area_ratio': float(box_area / gt_area),
            'normalized_center_displacement': float(
                np.hypot(box_center[0] - gt_center[0], box_center[1] - gt_center[1])
                / diagonal
            ),
        }
    )
    return result


def write_medsam_box_diagnostics(
    fixed_rows: pd.DataFrame,
    *,
    proposal_boxes_path: Path | str,
    automatic_metrics_path: Path | str,
    oracle_metrics_path: Path | str,
    output_dir: Path | str,
    selected_candidate_family: str = '',
    selected_threshold: float | None = None,
) -> dict[str, Any]:
    """Write one automatic-box diagnostic row for every locked evaluation row."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    proposals = (
        pd.read_csv(proposal_boxes_path).fillna('')
        if Path(proposal_boxes_path).is_file()
        else pd.DataFrame()
    )
    automatic = (
        pd.read_csv(automatic_metrics_path).fillna('')
        if Path(automatic_metrics_path).is_file()
        else pd.DataFrame()
    )
    oracle = (
        pd.read_csv(oracle_metrics_path).fillna('')
        if Path(oracle_metrics_path).is_file()
        else pd.DataFrame()
    )
    proposal_by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proposal_status_by_row: dict[str, str] = {}
    if not proposals.empty and 'manifest_row_id' in proposals.columns:
        for row in proposals.to_dict(orient='records'):
            if selected_candidate_family and str(row.get('candidate_family') or '').strip() != selected_candidate_family:
                continue
            if selected_threshold is not None and not np.isclose(
                float(row.get('threshold', selected_threshold)), float(selected_threshold)
            ):
                continue
            row_id = str(row['manifest_row_id'])
            decision = str(row.get('decision') or '').strip()
            if decision == 'generated':
                proposal_by_row[row_id].append(row)
            elif row_id not in proposal_status_by_row:
                proposal_status_by_row[row_id] = (
                    'overflow' if decision == 'overflow' else 'rejected'
                )

    def _metrics_by_row(
        frame: pd.DataFrame,
    ) -> tuple[dict[str, dict[str, Any]], set[str]]:
        if frame.empty or 'manifest_row_id' not in frame.columns:
            return {}, set()
        mapping: dict[str, dict[str, Any]] = {}
        duplicates: set[str] = set()
        for row in frame.to_dict(orient='records'):
            row_id = str(row['manifest_row_id'])
            if row_id in mapping:
                duplicates.add(row_id)
            mapping[row_id] = row
        return mapping, duplicates

    automatic_by_row, automatic_duplicates = _metrics_by_row(automatic)
    oracle_by_row, oracle_duplicates = _metrics_by_row(oracle)
    output_rows: list[dict[str, Any]] = []
    for record in fixed_rows.fillna('').sort_values('manifest_row_id').to_dict(orient='records'):
        row_id = str(record['manifest_row_id'])
        mask_path = Path(str(record.get('mask_path_resolved') or record.get('mask_path', '')))
        image_path = Path(str(record.get('image_path_resolved') or record.get('image_path', '')))
        if mask_path.is_file():
            gt_mask = np.asarray(Image.open(mask_path).convert('L'))
        else:
            gt_mask = np.zeros((1, 1), dtype=np.uint8)
        image_shape = Image.open(image_path).size[::-1] if image_path.is_file() else gt_mask.shape
        row_result = box_diagnostics_for_row(
            gt_mask,
            proposal_by_row.get(row_id) or None,
            (int(image_shape[0]), int(image_shape[1])),
            oracle_by_row.get(row_id),
            automatic_by_row.get(row_id),
        )
        if not proposal_by_row.get(row_id) and row_id in proposal_status_by_row:
            row_result['proposal_status'] = proposal_status_by_row[row_id]
        if row_id in automatic_duplicates:
            row_result.update(
                automatic_metric_status='invalid',
                automatic_dice=None,
                automatic_jaccard=None,
            )
        if row_id in oracle_duplicates:
            row_result.update(
                oracle_metric_status='invalid',
                oracle_dice=None,
                oracle_jaccard=None,
            )
        output_rows.append(
            {
                'manifest_row_id': row_id,
                'cohort_id': str(record.get('cohort_id', '')),
                'lane_assignment': str(record.get('lane_assignment', '')),
                **row_result,
            }
        )
    by_status = dict(pd.Series([row['proposal_status'] for row in output_rows]).value_counts().sort_index())
    by_stratum: dict[str, int] = defaultdict(int)
    for row in output_rows:
        if row['proposal_status'] == 'missing':
            stratum = 'missing'
        elif row['proposal_status'] != 'valid':
            stratum = row['proposal_status']
        elif float(row['box_iou']) >= 0.75:
            stratum = 'high_iou'
        elif float(row['box_iou']) >= 0.50:
            stratum = 'medium_iou'
        else:
            stratum = 'low_iou'
        by_stratum[stratum] += 1
    csv_path = output / 'box_diagnostics.csv'
    json_path = output / 'box_diagnostics.json'
    pd.DataFrame(output_rows).to_csv(csv_path, index=False)
    summary = {
        'row_count': len(output_rows),
        'by_proposal_status': {str(key): int(value) for key, value in by_status.items()},
        'by_quality_stratum': {key: int(value) for key, value in sorted(by_stratum.items())},
        'metrics_by_quality_stratum': {},
        'coordinate_metadata': {
            'source_frame': 'original_image_pixels_xyxy',
            'encoder_frame': 'square_encoder_pixels_xyxy',
            'encoder_size': 1024,
            'round_trip_tolerance_pixels': 1,
        },
    }
    for row in output_rows:
        if row['proposal_status'] == 'missing':
            stratum = 'missing'
        elif row['proposal_status'] != 'valid':
            stratum = row['proposal_status']
        elif float(row['box_iou']) >= 0.75:
            stratum = 'high_iou'
        elif float(row['box_iou']) >= 0.50:
            stratum = 'medium_iou'
        else:
            stratum = 'low_iou'
        bucket = summary['metrics_by_quality_stratum'].setdefault(
            stratum,
            {
                'row_count': 0,
                'automatic_metric_count': 0,
                'oracle_metric_count': 0,
                'mean_automatic_dice': None,
                'mean_automatic_jaccard': None,
                'mean_oracle_dice': None,
                'mean_oracle_jaccard': None,
            },
        )
        bucket['row_count'] += 1
        automatic_values = bucket.setdefault('_automatic_values', [])
        oracle_values = bucket.setdefault('_oracle_values', [])
        if row['automatic_dice'] is not None and row['automatic_jaccard'] is not None:
            bucket['automatic_metric_count'] += 1
            automatic_values.append((row['automatic_dice'], row['automatic_jaccard']))
        if row['oracle_dice'] is not None and row['oracle_jaccard'] is not None:
            bucket['oracle_metric_count'] += 1
            oracle_values.append((row['oracle_dice'], row['oracle_jaccard']))
    for bucket in summary['metrics_by_quality_stratum'].values():
        automatic_values = bucket.pop('_automatic_values')
        oracle_values = bucket.pop('_oracle_values')
        if automatic_values:
            bucket['mean_automatic_dice'] = float(np.mean([value[0] for value in automatic_values]))
            bucket['mean_automatic_jaccard'] = float(np.mean([value[1] for value in automatic_values]))
        if oracle_values:
            bucket['mean_oracle_dice'] = float(np.mean([value[0] for value in oracle_values]))
            bucket['mean_oracle_jaccard'] = float(np.mean([value[1] for value in oracle_values]))
    json_path.write_text(
        json.dumps({'summary': summary, 'rows': output_rows}, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    return {
        'summary': summary,
        'artifacts': {
            'csv': str(csv_path),
            'json': str(json_path),
            'sha256': digest,
        },
    }
