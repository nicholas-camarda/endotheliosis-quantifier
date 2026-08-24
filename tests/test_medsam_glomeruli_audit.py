import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eq.evaluation.medsam_glomeruli_audit import (
    audit_medsam_evaluation_cohort,
    box_diagnostics_for_row,
    box_from_encoder_frame,
    box_to_encoder_frame,
    check_box_round_trip,
    write_medsam_box_diagnostics,
)


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
    positive_mask = np.zeros((5, 5), dtype=np.uint8)
    positive_mask[0, 0] = 255
    _write_png(tmp_path / 'raw/masks/mask-3.png', positive_mask)

    audit = audit_medsam_evaluation_cohort(rows, tmp_path)

    assert audit['mask_composition']['positive_rows'] == 4
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
    } >= {
        'cross_split_strong_group_leakage',
        'cross_split_source_group_leakage',
        'cross_split_split_group_leakage',
    }


def test_audit_checks_split_group_even_when_other_grouping_fields_differ(
    tmp_path: Path,
) -> None:
    rows = _cohort_fixture(tmp_path)
    rows.loc[3, 'subject_id'] = 'subject-5'
    rows.loc[3, 'source_sample_id'] = 'sample-5'
    rows.loc[3, 'split_group_id'] = rows.loc[0, 'split_group_id']

    audit = audit_medsam_evaluation_cohort(rows, tmp_path)

    assert audit['promotion_blocked'] is True
    assert any(
        issue['reason_code'] == 'cross_split_split_group_leakage'
        for issue in audit['issues']
    )


def test_audit_blocks_duplicate_manifest_row_ids_within_one_split(
    tmp_path: Path,
) -> None:
    rows = _cohort_fixture(tmp_path)
    rows.loc[1, 'manifest_row_id'] = rows.loc[0, 'manifest_row_id']

    audit = audit_medsam_evaluation_cohort(rows, tmp_path)

    assert audit['duplicate_manifest_rows_by_split'] == {'train': ['row-1']}
    assert audit['promotion_blocked'] is True
    assert any(
        issue['reason_code'] == 'duplicate_manifest_row_id_within_split'
        and issue['manifest_row_id'] == 'row-1'
        for issue in audit['issues']
    )


def test_audit_rejects_invalid_or_incomplete_split_composition(tmp_path: Path) -> None:
    rows = _cohort_fixture(tmp_path)
    rows.loc[3, 'split'] = 'holdout'

    audit = audit_medsam_evaluation_cohort(rows, tmp_path)

    reason_codes = {issue['reason_code'] for issue in audit['issues']}
    assert {'invalid_split_label', 'missing_required_split'} <= reason_codes
    assert audit['independence_status'] == 'blocked'
    assert audit['promotion_blocked'] is True


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


def _diagnostic_metrics() -> tuple[dict[str, float], dict[str, float]]:
    return (
        {'dice': 0.81, 'jaccard': 0.70},
        {'dice': 0.93, 'jaccard': 0.87},
    )


def test_box_diagnostics_reports_perfect_proposal_and_paired_metrics() -> None:
    gt_mask = np.zeros((10, 10), dtype=np.uint8)
    gt_mask[2:6, 3:7] = 1
    automatic_metrics, oracle_metrics = _diagnostic_metrics()

    diagnostics = box_diagnostics_for_row(
        gt_mask,
        {'bbox_x0': 3, 'bbox_y0': 2, 'bbox_x1': 7, 'bbox_y1': 6},
        (10, 10),
        oracle_metrics,
        automatic_metrics,
    )

    assert diagnostics['proposal_status'] == 'valid'
    assert diagnostics['box_iou'] == pytest.approx(1.0)
    assert diagnostics['gt_mask_coverage'] == pytest.approx(1.0)
    assert diagnostics['proposal_to_gt_area_ratio'] == pytest.approx(1.0)
    assert diagnostics['normalized_center_displacement'] == pytest.approx(0.0)
    assert diagnostics['automatic_dice'] == pytest.approx(0.81)
    assert diagnostics['automatic_jaccard'] == pytest.approx(0.70)
    assert diagnostics['oracle_dice'] == pytest.approx(0.93)
    assert diagnostics['oracle_jaccard'] == pytest.approx(0.87)
    json.dumps(diagnostics, allow_nan=False)


def test_box_diagnostics_reports_partial_and_rejects_clipped_or_invalid_boxes() -> None:
    gt_mask = np.zeros((10, 10), dtype=np.uint8)
    gt_mask[2:6, 3:7] = 1
    automatic_metrics, oracle_metrics = _diagnostic_metrics()

    partial = box_diagnostics_for_row(
        gt_mask,
        {'bbox_x0': 3, 'bbox_y0': 2, 'bbox_x1': 5, 'bbox_y1': 6},
        (10, 10),
        oracle_metrics,
        automatic_metrics,
    )
    clipped = box_diagnostics_for_row(
        gt_mask,
        {'bbox_x0': -1, 'bbox_y0': 2, 'bbox_x1': 7, 'bbox_y1': 6},
        (10, 10),
        oracle_metrics,
        automatic_metrics,
    )
    invalid = box_diagnostics_for_row(
        gt_mask,
        {'bbox_x0': 7, 'bbox_y0': 2, 'bbox_x1': 3, 'bbox_y1': 6},
        (10, 10),
        oracle_metrics,
        automatic_metrics,
    )

    assert partial['proposal_status'] == 'valid'
    assert partial['box_iou'] == pytest.approx(0.5)
    assert partial['gt_mask_coverage'] == pytest.approx(0.5)
    assert partial['proposal_to_gt_area_ratio'] == pytest.approx(0.5)
    assert partial['normalized_center_displacement'] == pytest.approx(
        1.0 / np.sqrt(200.0)
    )
    for failed in (clipped, invalid):
        assert failed['proposal_status'] == 'invalid'
        assert failed['box_iou'] is None
        assert failed['gt_mask_coverage'] is None
        assert failed['proposal_to_gt_area_ratio'] is None
        assert failed['normalized_center_displacement'] is None
        assert failed['automatic_dice'] == pytest.approx(0.81)
        assert failed['oracle_dice'] == pytest.approx(0.93)


def test_box_diagnostics_keeps_missing_proposal_distinct_from_metric_failure() -> None:
    gt_mask = np.zeros((10, 10), dtype=np.uint8)
    gt_mask[2:6, 3:7] = 1

    diagnostics = box_diagnostics_for_row(
        gt_mask,
        None,
        (10, 10),
        {'dice': 0.93, 'jaccard': 0.87},
        None,
    )

    assert diagnostics['proposal_status'] == 'missing'
    assert diagnostics['automatic_metric_status'] == 'unavailable'
    assert diagnostics['oracle_metric_status'] == 'available'
    assert diagnostics['automatic_dice'] is None
    assert diagnostics['automatic_jaccard'] is None
    assert diagnostics['oracle_dice'] == pytest.approx(0.93)
    assert diagnostics['oracle_jaccard'] == pytest.approx(0.87)
    json.dumps(diagnostics, allow_nan=False)


def test_box_diagnostics_uses_all_accepted_boxes_as_one_proposal_union() -> None:
    gt_mask = np.zeros((10, 10), dtype=np.uint8)
    gt_mask[2:6, 3:7] = 1

    diagnostics = box_diagnostics_for_row(
        gt_mask,
        [
            {'bbox_x0': 3, 'bbox_y0': 2, 'bbox_x1': 5, 'bbox_y1': 6},
            {'bbox_x0': 5, 'bbox_y0': 2, 'bbox_x1': 7, 'bbox_y1': 6},
        ],
        (10, 10),
        {'dice': 0.93, 'jaccard': 0.87},
        {'dice': 0.81, 'jaccard': 0.70},
    )

    assert diagnostics['proposal_status'] == 'valid'
    assert diagnostics['proposal_count'] == 2
    assert diagnostics['box_iou'] == pytest.approx(1.0)
    assert diagnostics['gt_mask_coverage'] == pytest.approx(1.0)


@pytest.mark.parametrize('image_shape', [(1024, 1024), (2448, 2048)])
def test_box_coordinate_round_trip_preserves_integer_bounds_within_one_pixel(
    image_shape: tuple[int, int],
) -> None:
    height, width = image_shape
    box = {
        'bbox_x0': 137,
        'bbox_y0': 211,
        'bbox_x1': min(1701, width - 1),
        'bbox_y1': min(1899, height - 1),
    }

    encoded = box_to_encoder_frame(box, image_shape, encoder_size=1024)
    restored = box_from_encoder_frame(encoded, image_shape, encoder_size=1024)
    invariant = check_box_round_trip(
        box, image_shape, encoder_size=1024, tolerance=1
    )

    assert all(
        abs(restored[key] - box[key]) <= 1
        for key in ('bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1')
    )
    assert invariant['passed'] is True
    assert invariant['max_integer_pixel_error'] <= 1
    assert invariant['coordinate_frame'] == 'original_image_pixels_xyxy'
    assert invariant['encoder_frame'] == 'square_encoder_pixels_xyxy'


def test_box_coordinate_helpers_reject_out_of_frame_boxes() -> None:
    with pytest.raises(ValueError, match='outside image bounds'):
        box_to_encoder_frame(
            {'bbox_x0': -1, 'bbox_y0': 2, 'bbox_x1': 7, 'bbox_y1': 6},
            (10, 10),
        )


def test_box_diagnostics_writer_is_deterministic_and_emits_one_row_per_locked_sample(
    tmp_path: Path,
) -> None:
    mask_a = tmp_path / 'mask-a.png'
    mask_b = tmp_path / 'mask-b.png'
    image_a = tmp_path / 'image-a.png'
    image_b = tmp_path / 'image-b.png'
    gt_a = np.zeros((10, 10), dtype=np.uint8)
    gt_a[2:6, 3:7] = 1
    gt_b = np.zeros((10, 10), dtype=np.uint8)
    gt_b[1:5, 1:5] = 1
    _write_png(mask_a, gt_a)
    _write_png(mask_b, gt_b)
    _write_png(image_a, np.zeros((10, 10, 3), dtype=np.uint8))
    _write_png(image_b, np.zeros((10, 10, 3), dtype=np.uint8))
    fixed_rows = pd.DataFrame(
        [
            {
                'manifest_row_id': 'row-a',
                'cohort_id': 'demo',
                'lane_assignment': 'manual_mask_core',
                'image_path_resolved': str(image_a),
                'mask_path_resolved': str(mask_a),
            },
            {
                'manifest_row_id': 'row-b',
                'cohort_id': 'demo',
                'lane_assignment': 'manual_mask_core',
                'image_path_resolved': str(image_b),
                'mask_path_resolved': str(mask_b),
            },
        ]
    )
    proposal_path = tmp_path / 'proposal_boxes.csv'
    pd.DataFrame(
        [
            {
                'manifest_row_id': 'row-a',
                'candidate_family': 'transfer',
                'threshold': 0.35,
                'decision': 'generated',
                'bbox_x0': 3,
                'bbox_y0': 2,
                'bbox_x1': 7,
                'bbox_y1': 6,
            },
            {
                'manifest_row_id': 'row-b',
                'candidate_family': 'transfer',
                'threshold': 0.35,
                'decision': 'rejected',
                'bbox_x0': '',
                'bbox_y0': '',
                'bbox_x1': '',
                'bbox_y1': '',
            },
        ]
    ).to_csv(proposal_path, index=False)
    automatic_path = tmp_path / 'automatic_metrics.csv'
    pd.DataFrame(
        [
            {
                'manifest_row_id': 'row-a',
                'dice': 0.81,
                'jaccard': 0.70,
            }
        ]
    ).to_csv(automatic_path, index=False)
    oracle_path = tmp_path / 'oracle_metrics.csv'
    pd.DataFrame(
        [
            {'manifest_row_id': 'row-a', 'dice': 0.93, 'jaccard': 0.87},
            {'manifest_row_id': 'row-b', 'dice': 0.90, 'jaccard': 0.82},
        ]
    ).to_csv(oracle_path, index=False)

    first = write_medsam_box_diagnostics(
        fixed_rows,
        proposal_boxes_path=proposal_path,
        automatic_metrics_path=automatic_path,
        oracle_metrics_path=oracle_path,
        output_dir=tmp_path / 'diagnostics',
        selected_candidate_family='transfer',
        selected_threshold=0.35,
    )
    csv_bytes = Path(first['artifacts']['csv']).read_bytes()
    json_bytes = Path(first['artifacts']['json']).read_bytes()
    second = write_medsam_box_diagnostics(
        fixed_rows,
        proposal_boxes_path=proposal_path,
        automatic_metrics_path=automatic_path,
        oracle_metrics_path=oracle_path,
        output_dir=tmp_path / 'diagnostics',
        selected_candidate_family='transfer',
        selected_threshold=0.35,
    )

    rows = pd.read_csv(first['artifacts']['csv']).fillna('')
    assert rows['manifest_row_id'].tolist() == ['row-a', 'row-b']
    assert rows.loc[0, 'proposal_status'] == 'valid'
    assert rows.loc[1, 'proposal_status'] == 'rejected'
    assert rows.loc[0, 'automatic_dice'] == pytest.approx(0.81)
    assert rows.loc[1, 'automatic_metric_status'] == 'unavailable'
    assert first['summary']['by_proposal_status'] == {'rejected': 1, 'valid': 1}
    assert first['summary']['by_quality_stratum'] == {'high_iou': 1, 'rejected': 1}
    assert first['summary']['metrics_by_quality_stratum']['high_iou'][
        'mean_automatic_dice'
    ] == pytest.approx(0.81)
    assert len(first['artifacts']['sha256']) == 64
    assert csv_bytes == Path(second['artifacts']['csv']).read_bytes()
    assert json_bytes == Path(second['artifacts']['json']).read_bytes()
    json.dumps(first, allow_nan=False)
