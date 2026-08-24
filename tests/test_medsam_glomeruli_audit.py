import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eq.evaluation.medsam_glomeruli_audit import audit_medsam_evaluation_cohort


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        Image.fromarray(array.astype(np.uint8)).save(path)
    else:
        Image.fromarray(array.astype(np.uint8), mode='RGB').save(path)


def _cohort_fixture(runtime_root: Path) -> pd.DataFrame:
    rows = [
        {
            'manifest_row_id': 'row-1',
            'split': 'train',
            'cohort_id': 'cohort-a',
            'lane_assignment': 'manual_mask_core',
            'subject_id': 'subject-1',
            'source_sample_id': 'sample-1',
            'split_group_id': 'group-1',
            'image_path': 'raw/images/image-1.png',
            'mask_path': 'raw/masks/mask-1.png',
        },
        {
            'manifest_row_id': 'row-2',
            'split': 'train',
            'cohort_id': 'cohort-a',
            'lane_assignment': 'manual_mask_external',
            'subject_id': 'subject-2',
            'source_sample_id': 'sample-2',
            'split_group_id': 'group-2',
            'image_path': 'raw/images/image-2.png',
            'mask_path': 'raw/masks/mask-2.png',
        },
        {
            'manifest_row_id': 'row-3',
            'split': 'validation',
            'cohort_id': 'cohort-b',
            'lane_assignment': 'manual_mask_core',
            'subject_id': 'subject-3',
            'source_sample_id': 'sample-3',
            'split_group_id': 'group-3',
            'image_path': 'raw/images/image-3.png',
            'mask_path': 'raw/masks/mask-3.png',
        },
        {
            'manifest_row_id': 'row-4',
            'split': 'test',
            'cohort_id': 'cohort-b',
            'lane_assignment': 'manual_mask_external',
            'subject_id': 'subject-4',
            'source_sample_id': 'sample-4',
            'split_group_id': 'group-4',
            'image_path': 'raw/images/image-4.png',
            'mask_path': 'raw/masks/mask-4.png',
        },
    ]
    image_arrays = [
        np.full((5, 5, 3), 10, dtype=np.uint8),
        np.full((5, 5, 3), 20, dtype=np.uint8),
        np.full((5, 5, 3), 30, dtype=np.uint8),
        np.full((5, 5, 3), 40, dtype=np.uint8),
    ]
    mask_arrays = []
    for foreground_pixels in (4, 9, 0, 16):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask.flat[:foreground_pixels] = 255
        mask_arrays.append(mask)
    for row, image, mask in zip(rows, image_arrays, mask_arrays):
        _write_png(runtime_root / row['image_path'], image)
        _write_png(runtime_root / row['mask_path'], mask)
    return pd.DataFrame(rows)


def test_audit_reports_locked_split_composition_and_mask_quantiles(
    tmp_path: Path,
) -> None:
    rows = _cohort_fixture(tmp_path)

    audit = audit_medsam_evaluation_cohort(rows, tmp_path)

    assert audit['row_counts']['by_split'] == {
        'test': 1,
        'train': 2,
        'validation': 1,
    }
    assert audit['row_counts']['by_cohort'] == {'cohort-a': 2, 'cohort-b': 2}
    assert audit['row_counts']['by_lane'] == {
        'manual_mask_core': 2,
        'manual_mask_external': 2,
    }
    assert audit['unique_counts'] == {
        'manifest_rows': 4,
        'images': 4,
        'masks': 4,
        'source_groups': 4,
        'strongest_grouping_key': 'subject_id',
        'strongest_grouping_values': 4,
    }
    assert audit['mask_composition']['positive_rows'] == 3
    assert audit['mask_composition']['background_rows'] == 1
    assert audit['mask_composition']['foreground_pixel_quantiles'] == {
        'p0': 0.0,
        'p25': 3.0,
        'p50': 6.5,
        'p75': 10.75,
        'p100': 16.0,
    }
    assert audit['cross_split_overlap']['strongest_grouping_key']['values'] == []
    assert audit['independence_status'] == 'independent'
    assert audit['issues'] == []
    json.dumps(audit)


def test_audit_reports_duplicate_content_hashes_with_row_identities(
    tmp_path: Path,
) -> None:
    rows = _cohort_fixture(tmp_path)
    rows.loc[1, 'image_path'] = rows.loc[0, 'image_path']
    rows.loc[1, 'mask_path'] = rows.loc[0, 'mask_path']

    audit = audit_medsam_evaluation_cohort(rows, tmp_path)

    assert len(audit['duplicate_content_hashes']['images']) == 1
    assert audit['duplicate_content_hashes']['images'][0]['row_ids'] == [
        'row-1',
        'row-2',
    ]
    assert len(audit['duplicate_content_hashes']['masks']) == 1
    assert {
        issue['reason_code'] for issue in audit['issues']
    } >= {'duplicate_image_content_hash', 'duplicate_mask_content_hash'}
    assert audit['promotion_blocked'] is True


def test_audit_marks_positive_only_evidence_as_distribution_limitation(
    tmp_path: Path,
) -> None:
    rows = _cohort_fixture(tmp_path)
    rows = rows[rows['mask_path'].ne('raw/masks/mask-3.png')].reset_index(drop=True)
    (tmp_path / 'raw/masks/mask-3.png').unlink()

    audit = audit_medsam_evaluation_cohort(rows, tmp_path)

    assert audit['mask_composition']['positive_rows'] == 3
    assert audit['mask_composition']['background_rows'] == 0
    assert audit['independence_status'] == 'independent_with_distribution_limitation'
    assert any(
        issue['reason_code'] == 'positive_only_evidence_distribution_limitation'
        for issue in audit['issues']
    )
    assert audit['promotion_blocked'] is False


def test_audit_fails_closed_for_cross_split_strong_group_and_source_overlap(
    tmp_path: Path,
) -> None:
    rows = _cohort_fixture(tmp_path)
    rows.loc[3, 'subject_id'] = rows.loc[0, 'subject_id']
    rows.loc[3, 'source_sample_id'] = rows.loc[0, 'source_sample_id']
    rows.loc[3, 'split_group_id'] = rows.loc[0, 'split_group_id']

    audit = audit_medsam_evaluation_cohort(rows, tmp_path)

    assert audit['independence_status'] == 'blocked'
    assert audit['promotion_blocked'] is True
    assert audit['cross_split_overlap']['strongest_grouping_key']['values'] == [
        'subject-1'
    ]
    assert audit['cross_split_overlap']['source_groups']['values'] == ['sample-1']
    assert {
        issue['reason_code'] for issue in audit['issues']
    } >= {'cross_split_strong_group_leakage', 'cross_split_source_group_leakage'}


def test_audit_records_missing_and_outside_runtime_paths(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                'manifest_row_id': 'row-missing',
                'split': 'test',
                'cohort_id': 'cohort-a',
                'lane_assignment': 'manual_mask_core',
                'subject_id': 'subject-missing',
                'image_path': 'raw/images/missing.png',
                'mask_path': 'raw/masks/missing.png',
            },
            {
                'manifest_row_id': 'row-outside',
                'split': 'test',
                'cohort_id': 'cohort-a',
                'lane_assignment': 'manual_mask_core',
                'subject_id': 'subject-outside',
                'image_path': str(tmp_path.parent / 'outside.png'),
                'mask_path': 'raw/masks/missing.png',
            },
        ]
    )

    audit = audit_medsam_evaluation_cohort(rows, tmp_path)

    reason_codes = {issue['reason_code'] for issue in audit['issues']}
    assert {
        'missing_image_file',
        'missing_mask_file',
        'path_outside_runtime_root',
    } <= reason_codes
    assert audit['promotion_blocked'] is True
    json.dumps(audit)
