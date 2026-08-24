import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow import (
    _admitted_manual_rows,
    _baseline_metric_rows,
    _build_adaptation_command,
    _build_checkpoint_provenance,
    _build_deterministic_splits,
    _build_finetuned_comparison_summary,
    _classify_generated_mask_adoption,
    _configured_oracle_metrics_path,
    _load_metric_evidence,
    _package_generated_mask_release,
    _portable_report_value,
    _preflight_training_dependencies,
    _prepare_medsam_npy_data,
    _promotion_evidence_summary,
    _run_external_fixed_split_baselines,
    _run_finetuned_fixed_split_evaluation,
    _training_command_for_config,
    _trivial_baseline_aggregate,
    _update_generated_mask_registry,
    _validate_release_manifest_paths,
    _validate_split_manifest,
    run_medsam_glomeruli_fine_tuning_workflow,
)
from eq.utils.artifact_provenance import ArtifactProvenanceError
from eq.utils.paths import ensure_not_under_runtime_raw_data


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr > 0).astype(np.uint8) * 255).save(path)


def _metric_evidence_context(manifest_row_ids: list[str]) -> dict[str, object]:
    return {
        'run_id': 'run-1',
        'fixed_split': 'test',
        'fixed_manifest_row_ids': manifest_row_ids,
    }


def _write_metric_evidence_csv(
    path: Path, *, manifest_row_ids: list[str], method: str, dice: float, jaccard: float
) -> None:
    pd.DataFrame(
        [
            {
                'manifest_row_id': manifest_row_id,
                'method': method,
                'dice': dice,
                'jaccard': jaccard,
            }
            for manifest_row_id in manifest_row_ids
        ]
    ).to_csv(path, index=False)


def test_configured_oracle_metrics_path_is_selected_when_external_baselines_are_disabled(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / 'runtime'
    configured = {
        'external_baselines': False,
        'oracle_metrics_csv': 'output/evaluation/baseline/oracle_metrics.csv',
    }

    assert _configured_oracle_metrics_path(configured, runtime_root) == (
        runtime_root / 'output/evaluation/baseline/oracle_metrics.csv'
    )
    assert _configured_oracle_metrics_path(
        {'external_baselines': False}, runtime_root
    ) is None


def test_promotion_report_marks_external_paths_explicitly(tmp_path: Path) -> None:
    encoded = _portable_report_value(
        '/outside/source/config.yaml',
        runtime_root=tmp_path / 'runtime',
        key='config_path',
    )

    assert encoded == {
        'path': '/outside/source/config.yaml',
        'path_encoding': 'external-absolute',
        'reason': 'outside_runtime_root',
    }


def test_promotion_evidence_is_bound_to_run_split_and_checkpoint(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / 'runtime'
    evidence_path = runtime_root / 'evidence/overlay.json'
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(
            {
                'schema_version': 'eq.medsam.promotion-evidence/1',
                'evidence_kind': 'overlay_review',
                'status': 'passed',
                'evaluation_run_id': 'run-1',
                'fixed_split': 'test',
                'checkpoint_sha256': 'c' * 64,
            }
        ),
        encoding='utf-8',
    )

    config = {
        'promotion_evidence': {
            'overlay_review': {
                'status': 'passed',
                'artifact_path': 'evidence/overlay.json',
            }
        }
    }
    valid = _promotion_evidence_summary(
        config,
        runtime_root,
        expected_run_id='run-1',
        expected_fixed_split='test',
        expected_checkpoint_sha256='c' * 64,
    )
    foreign = _promotion_evidence_summary(
        config,
        runtime_root,
        expected_run_id='run-2',
        expected_fixed_split='test',
        expected_checkpoint_sha256='c' * 64,
    )

    assert valid['overlay_review']['reason_code'] == ''
    assert foreign['overlay_review']['reason_code'] == (
        'overlay_review_artifact_identity_invalid'
    )


def test_fixed_split_generation_keeps_source_groups_together():
    rows = pd.DataFrame(
        [
            {
                'manifest_row_id': f'row-{idx}',
                'cohort_id': 'vegfri_dox',
                'lane_assignment': 'manual_mask_external',
                'source_sample_id': group,
                'split_group_id': group,
                'image_path': f'image-{idx}.png',
                'mask_path': f'mask-{idx}.png',
            }
            for idx, group in enumerate(['a', 'a', 'b', 'c', 'd', 'e'], start=1)
        ]
    )

    split = _build_deterministic_splits(
        rows, train_fraction=0.5, validation_fraction=0.25
    )

    assert split.groupby('split_group_id')['split'].nunique().max() == 1
    assert set(split['split']) == {'train', 'validation', 'test'}
    assert 'selection_rank' in split.columns
    assert 'selection_reason' in split.columns


def test_audit_admission_view_retains_blank_artifact_paths(tmp_path: Path):
    manifest = pd.DataFrame(
        [
            {
                'manifest_row_id': 'row-missing',
                'admission_status': 'admitted',
                'lane_assignment': 'manual_mask_core',
                'cohort_id': 'demo',
                'source_sample_id': 'sample-missing',
                'image_path': '',
                'mask_path': '',
            }
        ]
    )

    audit_view = _admitted_manual_rows(
        manifest,
        tmp_path,
        lane_assignments=['manual_mask_core'],
        include_missing_files=True,
    )
    split_view = _admitted_manual_rows(
        manifest, tmp_path, lane_assignments=['manual_mask_core']
    )

    assert audit_view['manifest_row_id'].tolist() == ['row-missing']
    assert split_view.empty


def test_explicit_split_manifest_validation_hashes_required_paths(tmp_path: Path):
    runtime_root = tmp_path
    image = np.zeros((6, 6), dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=np.uint8)
    _write_png(runtime_root / 'raw_data/cohorts/demo/images/img.png', image)
    _write_png(runtime_root / 'raw_data/cohorts/demo/masks/mask.png', mask)
    split_path = tmp_path / 'split.csv'
    pd.DataFrame(
        [
            {
                'manifest_row_id': 'row-1',
                'cohort_id': 'demo',
                'lane_assignment': 'manual_mask_core',
                'source_sample_id': 'sample-1',
                'split_group_id': 'sample-1',
                'image_path': 'raw_data/cohorts/demo/images/img.png',
                'mask_path': 'raw_data/cohorts/demo/masks/mask.png',
                'split': 'validation',
                'selection_rank': 1,
                'selection_reason': 'test',
            }
        ]
    ).to_csv(split_path, index=False)

    digest = _validate_split_manifest(split_path, runtime_root)

    assert len(digest) == 64


def test_generated_mask_release_rejects_raw_data_paths(tmp_path: Path):
    with pytest.raises(ValueError, match='raw_data'):
        ensure_not_under_runtime_raw_data(
            tmp_path / 'raw_data/cohorts/demo/masks/generated', tmp_path
        )


def test_trivial_baseline_rows_have_metric_schema():
    manual = np.zeros((4, 4), dtype=np.uint8)
    manual[1:3, 1:3] = 1

    rows = _baseline_metric_rows(
        manual_mask=manual,
        manifest_row_id='row-1',
        cohort_id='demo',
        lane_assignment='manual_mask_core',
    )

    assert {row['method'] for row in rows} == {
        'trivial_all_background',
        'trivial_all_foreground',
    }
    assert all('dice' in row and 'jaccard' in row for row in rows)
    assert rows[0]['candidate_artifact'] == 'trivial_baseline'


def test_trivial_baseline_aggregate_fail_closed_for_empty_csv(tmp_path: Path) -> None:
    grouped_csv = tmp_path / 'trivial_grouped.csv'
    grouped_csv.write_text('', encoding='utf-8')

    aggregate = _trivial_baseline_aggregate(
        {'trivial_baseline_metric_by_source': str(grouped_csv)}
    )

    assert aggregate == {'dice': 0.0, 'jaccard': 0.0}


def test_trivial_baseline_aggregate_fail_closed_for_missing_metric_columns(
    tmp_path: Path,
) -> None:
    grouped_csv = tmp_path / 'trivial_grouped.csv'
    pd.DataFrame([{'source_sample_id': 's1', 'foo': '1.0'}]).to_csv(
        grouped_csv, index=False
    )

    aggregate = _trivial_baseline_aggregate(
        {'trivial_baseline_metric_by_source': str(grouped_csv)}
    )

    assert aggregate == {'dice': 0.0, 'jaccard': 0.0}


def test_trivial_baseline_aggregate_fail_closed_for_malformed_csv(
    tmp_path: Path,
) -> None:
    grouped_csv = tmp_path / 'trivial_grouped.csv'
    grouped_csv.write_text('dice,jaccard\n"0.5,0.4\n', encoding='utf-8')

    aggregate = _trivial_baseline_aggregate(
        {'trivial_baseline_metric_by_source': str(grouped_csv)}
    )

    assert aggregate == {'dice': 0.0, 'jaccard': 0.0}


def test_finetuned_comparison_uses_selected_oracle_metrics_csv_when_external_baselines_skipped(
    tmp_path: Path,
) -> None:
    oracle_metrics_csv = tmp_path / 'oracle_metrics.csv'
    finetuned_metrics_csv = tmp_path / 'finetuned_metrics.csv'
    manifest_row_ids = [f'row-{index}' for index in range(42)]
    _write_metric_evidence_csv(
        oracle_metrics_csv,
        manifest_row_ids=manifest_row_ids,
        method='medsam_oracle',
        dice=0.921631295838860,
        jaccard=0.86,
    )
    _write_metric_evidence_csv(
        finetuned_metrics_csv,
        manifest_row_ids=manifest_row_ids,
        method='medsam_finetuned_automatic',
        dice=0.815511056209316,
        jaccard=0.72,
    )

    comparison = _build_finetuned_comparison_summary(
        finetuned_evaluation={
            'status': 'completed',
            'metrics_csv': str(finetuned_metrics_csv),
            'overlay_review_status': 'passed',
            'prompt_failure_count': 0,
            **_metric_evidence_context(manifest_row_ids),
        },
        external_baselines_summary={
            'status': 'skipped_config',
            'oracle_medsam_metrics_csv': str(oracle_metrics_csv),
            **_metric_evidence_context(manifest_row_ids),
        },
        baseline_outputs={},
        adoption_gates={},
    )

    assert comparison['oracle_dice_gap'] == pytest.approx(0.106120239629544)
    assert comparison['metric_evidence']['oracle']['artifact_path'] == str(
        oracle_metrics_csv
    )
    assert comparison['metric_evidence']['fine_tuned']['artifact_path'] == str(
        finetuned_metrics_csv
    )
    assert (
        comparison['metric_evidence']['oracle']['artifact_path']
        != comparison['metric_evidence']['fine_tuned']['artifact_path']
    )
    assert comparison['metric_evidence']['oracle']['sample_count'] == 42
    assert comparison['metric_evidence']['fine_tuned']['sample_count'] == 42
    assert len(comparison['metric_evidence']['oracle']['sha256']) == 64
    assert len(comparison['metric_evidence']['fine_tuned']['sha256']) == 64


def test_finetuned_comparison_blocks_mismatched_fixed_split_metric_evidence(
    tmp_path: Path,
) -> None:
    finetuned_metrics_csv = tmp_path / 'finetuned_metrics.csv'
    oracle_metrics_csv = tmp_path / 'oracle_metrics.csv'
    finetuned_ids = ['row-1', 'row-2']
    oracle_ids = ['other-1', 'other-2']
    _write_metric_evidence_csv(
        finetuned_metrics_csv,
        manifest_row_ids=finetuned_ids,
        method='medsam_finetuned_automatic',
        dice=0.91,
        jaccard=0.84,
    )
    _write_metric_evidence_csv(
        oracle_metrics_csv,
        manifest_row_ids=oracle_ids,
        method='medsam_oracle',
        dice=0.94,
        jaccard=0.86,
    )

    comparison = _build_finetuned_comparison_summary(
        finetuned_evaluation={
            'status': 'completed',
            'metrics_csv': str(finetuned_metrics_csv),
            'overlay_review_status': 'passed',
            'prompt_failure_count': 0,
            **_metric_evidence_context(finetuned_ids),
        },
        external_baselines_summary={
            'oracle_medsam_metrics_csv': str(oracle_metrics_csv),
            **_metric_evidence_context(oracle_ids),
        },
        baseline_outputs={},
        adoption_gates={},
    )

    assert comparison['status'] == 'blocked'
    assert 'oracle_metric_evidence_provenance_mismatch' in comparison['reason_codes']


def test_finetuned_comparison_blocks_shared_oracle_automatic_artifact_and_wrong_method(
    tmp_path: Path,
) -> None:
    shared_metrics_csv = tmp_path / 'oracle_metrics.csv'
    finetuned_metrics_csv = tmp_path / 'finetuned_metrics.csv'
    manifest_row_ids = ['row-1', 'row-2']
    _write_metric_evidence_csv(
        shared_metrics_csv,
        manifest_row_ids=manifest_row_ids,
        method='medsam_oracle',
        dice=0.94,
        jaccard=0.86,
    )
    _write_metric_evidence_csv(
        finetuned_metrics_csv,
        manifest_row_ids=manifest_row_ids,
        method='medsam_finetuned_automatic',
        dice=0.91,
        jaccard=0.84,
    )

    comparison = _build_finetuned_comparison_summary(
        finetuned_evaluation={
            'status': 'completed',
            'metrics_csv': str(finetuned_metrics_csv),
            'overlay_review_status': 'passed',
            'prompt_failure_count': 0,
            **_metric_evidence_context(manifest_row_ids),
        },
        external_baselines_summary={
            'oracle_medsam_metrics_csv': str(shared_metrics_csv),
            'automatic_medsam_metrics_csv': str(shared_metrics_csv),
            **_metric_evidence_context(manifest_row_ids),
        },
        baseline_outputs={},
        adoption_gates={},
    )

    assert comparison['status'] == 'blocked'
    assert 'automatic_oracle_metric_artifact_not_distinct' in comparison['reason_codes']
    assert 'automatic_medsam_metric_method_mismatch' in comparison['reason_codes']


@pytest.mark.parametrize(
    ('rows', 'aggregation', 'expected_methods'),
    [
        pytest.param(
            [
                {
                    'manifest_row_id': 'row-1',
                    'method': 'medsam_oracle',
                    'dice': 0.95,
                    'jaccard': 0.90,
                },
                {
                    'manifest_row_id': 'row-1',
                    'method': 'medsam_oracle',
                    'dice': 0.05,
                    'jaccard': 0.03,
                },
                {
                    'manifest_row_id': 'row-2',
                    'method': 'medsam_oracle',
                    'dice': 0.90,
                    'jaccard': 0.82,
                },
            ],
            'mean',
            {'medsam_oracle'},
            id='duplicate-manifest-row-id',
        ),
        pytest.param(
            [
                {
                    'manifest_row_id': 'row-1',
                    'method': 'trivial_all_background',
                    'dice': 0.10,
                    'jaccard': 0.05,
                },
                {
                    'manifest_row_id': 'row-1',
                    'method': 'trivial_all_background',
                    'dice': 0.90,
                    'jaccard': 0.82,
                },
                {
                    'manifest_row_id': 'row-1',
                    'method': 'trivial_all_foreground',
                    'dice': 0.20,
                    'jaccard': 0.12,
                },
                {
                    'manifest_row_id': 'row-2',
                    'method': 'trivial_all_background',
                    'dice': 0.10,
                    'jaccard': 0.05,
                },
                {
                    'manifest_row_id': 'row-2',
                    'method': 'trivial_all_foreground',
                    'dice': 0.20,
                    'jaccard': 0.12,
                },
            ],
            'max_method_mean',
            {'trivial_all_background', 'trivial_all_foreground'},
            id='duplicate-method-row-combination',
        ),
    ],
)
def test_metric_evidence_rejects_duplicate_evaluation_rows(
    tmp_path: Path,
    rows: list[dict[str, object]],
    aggregation: str,
    expected_methods: set[str],
) -> None:
    metrics_csv = tmp_path / 'metrics.csv'
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    provenance = _metric_evidence_context(['row-1', 'row-2'])

    evidence = _load_metric_evidence(
        metrics_csv,
        label='test_lane',
        aggregation=aggregation,
        provenance_source=provenance,
        expected_provenance=provenance,
        expected_methods=expected_methods,
    )

    assert evidence['status'] == 'unavailable'
    assert evidence['reason_code'] == 'test_lane_metric_evidence_duplicate_rows'
    assert 'metrics' not in evidence


def test_finetuned_comparison_blocks_when_oracle_metric_evidence_is_unavailable(
    tmp_path: Path,
) -> None:
    finetuned_metrics_csv = tmp_path / 'finetuned_metrics.csv'
    pd.DataFrame([{'dice': 0.815511056209316, 'jaccard': 0.72}] * 42).to_csv(
        finetuned_metrics_csv, index=False
    )

    comparison = _build_finetuned_comparison_summary(
        finetuned_evaluation={
            'status': 'completed',
            'metrics_csv': str(finetuned_metrics_csv),
            'overlay_review_status': 'passed',
            'prompt_failure_count': 0,
        },
        external_baselines_summary={'status': 'skipped_config'},
        baseline_outputs={},
        adoption_gates={},
    )

    assert comparison['adoption_tier'] == 'blocked'
    assert 'oracle_metric_evidence_unavailable' in comparison['reason_codes']
    assert 'oracle_dice_gap' not in comparison
    assert comparison['metric_evidence']['oracle']['status'] == 'unavailable'


def test_adoption_gate_distinguishes_oracle_level_candidate_and_blocked():
    oracle = _classify_generated_mask_adoption(
        fine_tuned_metrics={'dice': 0.91, 'jaccard': 0.84},
        oracle_metrics={'dice': 0.94, 'jaccard': 0.86},
        current_auto_metrics={'dice': 0.78, 'jaccard': 0.64},
        current_segmenter_metrics={'dice': 0.72, 'jaccard': 0.58},
        trivial_baseline_metrics={'dice': 0.20, 'jaccard': 0.12},
        prompt_failure_count=0,
        gates={'min_dice': 0.90, 'min_jaccard': 0.82, 'max_oracle_dice_gap': 0.05},
        overlay_review_status='passed',
    )
    assert oracle['adoption_tier'] == 'oracle_level_preferred'
    assert oracle['recommended_generated_mask_source'] == 'medsam_finetuned_glomeruli'
    assert oracle['oracle_dice_gap'] == pytest.approx(0.03)

    improved = _classify_generated_mask_adoption(
        fine_tuned_metrics={'dice': 0.86, 'jaccard': 0.76},
        oracle_metrics={'dice': 0.94, 'jaccard': 0.86},
        current_auto_metrics={'dice': 0.78, 'jaccard': 0.64},
        current_segmenter_metrics={'dice': 0.72, 'jaccard': 0.58},
        trivial_baseline_metrics={'dice': 0.20, 'jaccard': 0.12},
        prompt_failure_count=0,
        gates={'min_dice': 0.90, 'min_jaccard': 0.82, 'max_oracle_dice_gap': 0.05},
        overlay_review_status='passed',
    )
    assert improved['adoption_tier'] == 'improved_candidate_not_oracle'
    assert improved['recommended_generated_mask_source'] == ''

    blocked = _classify_generated_mask_adoption(
        fine_tuned_metrics={'dice': 0.75, 'jaccard': 0.60},
        oracle_metrics={'dice': 0.94, 'jaccard': 0.86},
        current_auto_metrics={'dice': 0.78, 'jaccard': 0.64},
        current_segmenter_metrics={'dice': 0.72, 'jaccard': 0.58},
        trivial_baseline_metrics={'dice': 0.20, 'jaccard': 0.12},
        prompt_failure_count=0,
        gates={'min_dice': 0.90, 'min_jaccard': 0.82, 'max_oracle_dice_gap': 0.05},
        overlay_review_status='passed',
    )
    assert blocked['adoption_tier'] == 'blocked'
    assert blocked['failure_mode'] == 'oracle_gap'


def test_adoption_gate_blocks_shape_overcoverage_even_when_metrics_clear():
    result = _classify_generated_mask_adoption(
        fine_tuned_metrics={'dice': 0.91, 'jaccard': 0.84},
        oracle_metrics={'dice': 0.94, 'jaccard': 0.86},
        current_auto_metrics={'dice': 0.78, 'jaccard': 0.64},
        current_segmenter_metrics={'dice': 0.72, 'jaccard': 0.58},
        trivial_baseline_metrics={'dice': 0.20, 'jaccard': 0.12},
        prompt_failure_count=0,
        gates={
            'min_dice': 0.90,
            'min_jaccard': 0.82,
            'max_oracle_dice_gap': 0.05,
            'max_prompt_failures': 0,
            'min_prediction_foreground_fraction': 0.0,
            'max_prediction_foreground_fraction': 0.50,
            'min_prediction_to_truth_area_ratio': 0.50,
            'max_prediction_to_truth_area_ratio': 1.50,
        },
        overlay_review_status='passed',
        fine_tuned_shape_metrics={
            'status': 'available',
            'metrics': {
                'prediction_foreground_fraction': {'min': 0.05, 'max': 0.10},
                'area_ratio': {'min': 0.8, 'max': 2.1},
            },
        },
    )

    assert result['adoption_tier'] == 'blocked'
    assert result['shape_gate_passed'] is False
    assert 'prediction_to_truth_area_ratio_above_gate' in result['reason_codes']


def _package_portable_release(runtime_root: Path) -> tuple[dict[str, Path], Path]:
    run_mask = runtime_root / 'run/masks/row-1.png'
    _write_png(run_mask, np.ones((4, 4), dtype=np.uint8))
    source_image = runtime_root / 'raw_data/cohorts/demo/images/img.png'
    reference_mask = runtime_root / 'raw_data/cohorts/demo/masks/mask.png'
    _write_png(source_image, np.ones((4, 4), dtype=np.uint8))
    _write_png(reference_mask, np.ones((4, 4), dtype=np.uint8))
    checkpoint = runtime_root / 'models/medsam_glomeruli/ckpt-1/model.pth'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'checkpoint')
    metric_evidence = runtime_root / 'output/segmentation_evaluation/metrics.csv'
    metric_evidence.parent.mkdir(parents=True, exist_ok=True)
    metric_evidence.write_text('manifest_row_id,dice,jaccard\nrow-1,0.9,0.8\n')
    release_dir = (
        runtime_root
        / 'derived_data/generated_masks/glomeruli/medsam_finetuned/release-1'
    )
    registry = runtime_root / 'derived_data/generated_masks/glomeruli/manifest.csv'

    release = _package_generated_mask_release(
        runtime_root=runtime_root,
        release_dir=release_dir,
        mask_release_id='release-1',
        run_id='run-1',
        checkpoint_id='ckpt-1',
        checkpoint_path=checkpoint,
        proposal_source='transfer',
        proposal_threshold=0.35,
        adoption_tier='improved_candidate_not_oracle',
        checkpoint_sha256='c' * 64,
        metric_evidence_path=str(metric_evidence),
        metric_evidence_sha256='m' * 64,
        generated_masks=[
            {
                'manifest_row_id': 'row-1',
                'cohort_id': 'demo',
                'lane_assignment': 'manual_mask_core',
                'source_sample_id': 'sample-1',
                'source_image_path': str(source_image),
                'reference_mask_path': str(reference_mask),
                'generated_mask_path': str(run_mask),
                'generation_status': 'generated',
            }
        ],
    )
    _update_generated_mask_registry(
        registry, release['manifest'], runtime_root=runtime_root
    )
    return release, registry


def test_generated_mask_release_uses_portable_runtime_relative_provenance(
    tmp_path: Path,
):
    release_a, registry_a = _package_portable_release(tmp_path / 'runtime-a')
    release_b, registry_b = _package_portable_release(tmp_path / 'runtime-b')

    manifest_a = pd.read_csv(release_a['manifest']).fillna('')
    manifest_b = pd.read_csv(release_b['manifest']).fillna('')
    registry_a_frame = pd.read_csv(registry_a).fillna('')
    registry_b_frame = pd.read_csv(registry_b).fillna('')
    provenance_a = json.loads(release_a['provenance'].read_text(encoding='utf-8'))
    provenance_b = json.loads(release_b['provenance'].read_text(encoding='utf-8'))

    assert manifest_a.to_dict(orient='records') == manifest_b.to_dict(orient='records')
    assert registry_a_frame.to_dict(orient='records') == registry_b_frame.to_dict(
        orient='records'
    )
    assert provenance_a == provenance_b
    assert manifest_a.loc[0, 'schema_version'] == 'eq.medsam.generated-mask-release/1'
    assert manifest_a.loc[0, 'path_encoding'] == 'runtime-relative-posix'
    assert (
        registry_a_frame.loc[0, 'schema_version']
        == 'eq.medsam.generated-mask-release/1'
    )
    assert registry_a_frame.loc[0, 'path_encoding'] == 'runtime-relative-posix'
    assert manifest_a.loc[0, 'generated_mask_path'] == (
        'derived_data/generated_masks/glomeruli/medsam_finetuned/release-1/masks/row-1.png'
    )
    assert registry_a_frame.loc[0, 'release_manifest_path'] == (
        'derived_data/generated_masks/glomeruli/medsam_finetuned/release-1/manifest.csv'
    )
    assert provenance_a['manifest_path'] == (
        'derived_data/generated_masks/glomeruli/medsam_finetuned/release-1/manifest.csv'
    )
    assert manifest_a.loc[0, 'mask_source'] == 'medsam_finetuned_glomeruli'
    assert manifest_a.loc[0, 'adoption_tier'] == 'improved_candidate_not_oracle'
    assert manifest_a.loc[0, 'checkpoint_sha256'] == 'c' * 64
    assert manifest_a.loc[0, 'metric_evidence_path'] == (
        'output/segmentation_evaluation/metrics.csv'
    )
    assert provenance_a['checkpoint_sha256'] == 'c' * 64
    assert provenance_a['metric_evidence_path'] == (
        'output/segmentation_evaluation/metrics.csv'
    )
    assert provenance_a['metric_evidence_sha256'] == 'm' * 64
    assert (release_a['index']).exists()
    assert (release_a['provenance']).exists()


def test_runtime_relative_path_contract_rejects_unsafe_and_missing_artifacts(
    tmp_path: Path,
):
    from eq.utils.artifact_provenance import (
        ArtifactProvenanceError,
        resolve_runtime_relative_path,
        serialize_runtime_relative_path,
    )

    runtime_root = tmp_path / 'runtime'
    existing = runtime_root / 'derived_data/generated_masks/glomeruli/mask.png'
    _write_png(existing, np.ones((2, 2), dtype=np.uint8))

    with pytest.raises(ArtifactProvenanceError, match='absolute'):
        resolve_runtime_relative_path(
            '/stale/runtime/mask.png', runtime_root=runtime_root
        )
    with pytest.raises(ArtifactProvenanceError, match='Windows absolute'):
        serialize_runtime_relative_path(
            r'C:\\stale\\runtime\\mask.png', runtime_root=runtime_root
        )
    assert serialize_runtime_relative_path(
        r'derived_data\generated_masks\glomeruli\mask.png',
        runtime_root=runtime_root,
    ) == 'derived_data/generated_masks/glomeruli/mask.png'
    with pytest.raises(ArtifactProvenanceError, match='traversal'):
        resolve_runtime_relative_path('../outside.png', runtime_root=runtime_root)
    with pytest.raises(ArtifactProvenanceError, match='outside runtime_root'):
        serialize_runtime_relative_path(
            tmp_path / 'outside.png', runtime_root=runtime_root
        )
    with pytest.raises(ArtifactProvenanceError, match='missing generated artifact'):
        resolve_runtime_relative_path(
            'derived_data/generated_masks/glomeruli/missing.png',
            runtime_root=runtime_root,
            require_exists=True,
            artifact_label='generated artifact',
        )


_SERIALIZED_RELEASE_PATH_FIELDS = (
    'source_image_path',
    'reference_mask_path',
    'generated_mask_path',
    'checkpoint_path',
    'provenance_path',
    'manifest_path',
    'release_manifest_path',
)


def _mutate_serialized_release_path(
    *, release: dict[str, Path], registry: Path, field: str, value: str
) -> None:
    if field == 'manifest_path':
        provenance = json.loads(release['provenance'].read_text(encoding='utf-8'))
        provenance[field] = value
        release['provenance'].write_text(
            json.dumps(provenance, indent=2), encoding='utf-8'
        )
        return
    target = registry if field == 'release_manifest_path' else release['manifest']
    frame = pd.read_csv(target).fillna('')
    frame.loc[0, field] = value
    frame.to_csv(target, index=False)


@pytest.mark.parametrize('field', _SERIALIZED_RELEASE_PATH_FIELDS)
@pytest.mark.parametrize(
    ('unsafe_kind', 'expected_error'),
    (
        ('absolute', 'absolute'),
        ('traversal', 'traversal'),
        ('outside_root', 'outside runtime_root'),
    ),
)
def test_release_registry_rejects_unsafe_encoding_for_every_artifact_path(
    tmp_path: Path, field: str, unsafe_kind: str, expected_error: str
):
    runtime_root = tmp_path / 'runtime'
    release, registry = _package_portable_release(runtime_root)
    outside_dir = tmp_path / 'outside'
    outside_file = outside_dir / 'artifact.bin'
    outside_file.parent.mkdir(parents=True, exist_ok=True)
    outside_file.write_bytes(b'artifact')
    if unsafe_kind == 'absolute':
        unsafe_value = str(outside_file)
    elif unsafe_kind == 'traversal':
        unsafe_value = '../outside/artifact.bin'
    else:
        escape = runtime_root / 'escape'
        escape.symlink_to(outside_dir, target_is_directory=True)
        unsafe_value = 'escape/artifact.bin'
    _mutate_serialized_release_path(
        release=release, registry=registry, field=field, value=unsafe_value
    )

    with pytest.raises(ArtifactProvenanceError, match=expected_error):
        _validate_release_manifest_paths(
            pd.read_csv(
                registry if field == 'release_manifest_path' else release['manifest']
            ).fillna(''),
            runtime_root=runtime_root,
        )


@pytest.mark.parametrize('field', _SERIALIZED_RELEASE_PATH_FIELDS)
def test_release_registry_rejects_missing_file_for_every_required_artifact_path(
    tmp_path: Path, field: str
):
    runtime_root = tmp_path / 'runtime'
    release, registry = _package_portable_release(runtime_root)
    _mutate_serialized_release_path(
        release=release,
        registry=registry,
        field=field,
        value=f'missing/{field}.bin',
    )

    with pytest.raises(ArtifactProvenanceError, match='missing'):
        _validate_release_manifest_paths(
            pd.read_csv(
                registry if field == 'release_manifest_path' else release['manifest']
            ).fillna(''),
            runtime_root=runtime_root,
        )


@pytest.mark.parametrize(
    'missing_field', ('source_image_path', 'checkpoint_path')
)
def test_generated_mask_release_writer_requires_existing_input_artifacts(
    tmp_path: Path, missing_field: str
):
    runtime_root = tmp_path / 'runtime'
    run_mask = runtime_root / 'run/masks/row-1.png'
    source_image = runtime_root / 'raw_data/cohorts/demo/images/img.png'
    reference_mask = runtime_root / 'raw_data/cohorts/demo/masks/mask.png'
    checkpoint = runtime_root / 'models/medsam_glomeruli/ckpt-1/model.pth'
    _write_png(run_mask, np.ones((4, 4), dtype=np.uint8))
    _write_png(source_image, np.ones((4, 4), dtype=np.uint8))
    _write_png(reference_mask, np.ones((4, 4), dtype=np.uint8))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'checkpoint')
    paths = {
        'source_image_path': source_image,
        'reference_mask_path': reference_mask,
        'checkpoint_path': checkpoint,
    }
    paths[missing_field].unlink()

    with pytest.raises(ArtifactProvenanceError, match=f'missing {missing_field}'):
        _package_generated_mask_release(
            runtime_root=runtime_root,
            release_dir=runtime_root
            / 'derived_data/generated_masks/glomeruli/medsam_finetuned/release-1',
            mask_release_id='release-1',
            run_id='run-1',
            checkpoint_id='ckpt-1',
            checkpoint_path=checkpoint,
            proposal_source='transfer',
            proposal_threshold=0.35,
            adoption_tier='improved_candidate_not_oracle',
            generated_masks=[
                {
                    'source_image_path': str(source_image),
                    'reference_mask_path': str(reference_mask),
                    'generated_mask_path': str(run_mask),
                    'generation_status': 'generated',
                }
            ],
        )


def test_generated_mask_release_allows_missing_reference_mask(tmp_path: Path):
    runtime_root = tmp_path / 'runtime'
    run_mask = runtime_root / 'run/masks/row-1.png'
    source_image = runtime_root / 'raw_data/cohorts/demo/images/img.png'
    checkpoint = runtime_root / 'models/medsam_glomeruli/ckpt-1/model.pth'
    _write_png(run_mask, np.ones((4, 4), dtype=np.uint8))
    _write_png(source_image, np.ones((4, 4), dtype=np.uint8))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'checkpoint')

    release = _package_generated_mask_release(
        runtime_root=runtime_root,
        release_dir=runtime_root
        / 'derived_data/generated_masks/glomeruli/medsam_finetuned/release-1',
        mask_release_id='release-1',
        run_id='run-1',
        checkpoint_id='ckpt-1',
        checkpoint_path=checkpoint,
        proposal_source='transfer',
        proposal_threshold=0.35,
        adoption_tier='improved_candidate_not_oracle',
        generated_masks=[
            {
                'source_image_path': str(source_image),
                'reference_mask_path': '',
                'generated_mask_path': str(run_mask),
                'generation_status': 'generated',
            }
        ],
    )

    manifest = pd.read_csv(release['manifest']).fillna('')
    assert manifest.loc[0, 'reference_mask_path'] == ''


def test_preflight_and_provenance_fail_closed_without_success_status(tmp_path: Path):
    config = {
        'medsam': {
            'python': str(tmp_path / 'missing-python'),
            'repo': str(tmp_path / 'missing-repo'),
            'base_checkpoint': str(tmp_path / 'missing.pth'),
            'adaptation_mode': 'frozen_image_encoder_mask_decoder',
            'frozen_components': ['image_encoder', 'prompt_encoder'],
            'trainable_components': ['mask_decoder'],
        },
        'training': {'entrypoint': 'train_one_gpu.py'},
    }
    with pytest.raises(FileNotFoundError, match='MedSAM Python interpreter'):
        _preflight_training_dependencies(config)

    provenance = _build_checkpoint_provenance(
        training_command=['python', 'train_one_gpu.py'],
        environment={'PYTHON': 'python'},
        medsam_repo=tmp_path / 'MedSAM',
        base_checkpoint=tmp_path / 'base.pth',
        split_manifest_paths={'train': tmp_path / 'train.csv'},
        split_manifest_hashes={'train': 'abc'},
        adaptation_mode='frozen_image_encoder_mask_decoder',
        frozen_components=['image_encoder', 'prompt_encoder'],
        trainable_components=['mask_decoder'],
        hyperparameters={'epochs': 1},
        checkpoint_files=[],
        training_status='failed_dependency_preflight',
        local_feasibility_status='requires_external_accelerator',
    )

    assert provenance['training_status'] == 'failed_dependency_preflight'
    assert provenance['checkpoint_files'] == []
    assert provenance['supported_checkpoint'] is False


def test_adaptation_command_wraps_upstream_training_without_vendoring(tmp_path: Path):
    command = _build_adaptation_command(
        medsam_python=tmp_path / 'env/bin/python',
        entrypoint=tmp_path / 'MedSAM/train_one_gpu.py',
        train_npy_root=tmp_path / 'npy_data',
        work_dir=tmp_path / 'models/medsam_glomeruli/ckpt-1',
        base_checkpoint=tmp_path / 'medsam_vit_b.pth',
        task_name='eq_glomeruli',
        epochs=3,
        batch_size=1,
        learning_rate=0.0001,
        device='mps',
    )

    assert command[:2] == [
        str(tmp_path / 'env/bin/python'),
        str(tmp_path / 'MedSAM/train_one_gpu.py'),
    ]
    assert '-i' in command
    assert '-pretrain_model_path' in command
    assert 'mps' in command


def test_training_command_eq_native_adapter_invokes_in_repo_adapter(
    tmp_path: Path,
) -> None:
    config = {'medsam': {'device': 'mps'}}
    preflight = {
        'medsam_python': str(tmp_path / 'py'),
        'medsam_repo': str(tmp_path / 'repo'),
        'base_checkpoint': str(tmp_path / 'ckpt.pth'),
        'training_backend': 'eq_native_adapter',
        'entrypoint': '',
    }
    (tmp_path / 'py').write_bytes(b'')
    cmd = _training_command_for_config(
        config=config,
        preflight=preflight,
        train_npy_root=tmp_path / 'npy',
        work_dir=tmp_path / 'wd',
        training_cfg={'epochs': 2, 'batch_size': 1},
    )
    assert cmd[1] == '-m'
    assert 'eq.evaluation.medsam_glomeruli_adapter' in cmd
    assert '--lr-scheduler' in cmd
    assert cmd[cmd.index('--lr-scheduler') + 1] == 'none'


def test_training_command_eq_native_adapter_passes_val_npy_when_set(
    tmp_path: Path,
) -> None:
    config = {'medsam': {'device': 'mps'}}
    preflight = {
        'medsam_python': str(tmp_path / 'py'),
        'medsam_repo': str(tmp_path / 'repo'),
        'base_checkpoint': str(tmp_path / 'ckpt.pth'),
        'training_backend': 'eq_native_adapter',
        'entrypoint': '',
    }
    (tmp_path / 'py').write_bytes(b'')
    val_root = tmp_path / 'val_npy'
    cmd = _training_command_for_config(
        config=config,
        preflight=preflight,
        train_npy_root=tmp_path / 'npy',
        work_dir=tmp_path / 'wd',
        training_cfg={'epochs': 2, 'batch_size': 1, 'val_max_examples': 8},
        val_npy_root=val_root,
    )
    i = cmd.index('--val-npy-root')
    assert cmd[i + 1] == str(val_root)
    j = cmd.index('--val-max-examples')
    assert cmd[j + 1] == '8'


def test_prepare_medsam_npy_data_writes_paired_normalized_arrays(tmp_path: Path):
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[..., 0] = 128
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[2:6, 3:9] = 1
    image_path = tmp_path / 'raw_data/cohorts/demo/images/img.png'
    mask_path = tmp_path / 'raw_data/cohorts/demo/masks/mask.png'
    _write_png(image_path, image[..., 0])
    Image.fromarray(image).save(image_path)
    _write_png(mask_path, mask)
    split = pd.DataFrame(
        [
            {
                'manifest_row_id': 'row-1',
                'cohort_id': 'demo',
                'lane_assignment': 'manual_mask_core',
                'source_sample_id': 'sample-1',
                'split_group_id': 'sample-1',
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'split': 'train',
                'selection_rank': 1,
                'selection_reason': 'test',
            }
        ]
    )

    summary = _prepare_medsam_npy_data(
        train_rows=split,
        output_root=tmp_path / 'derived_data/medsam_glomeruli/npy_data',
        image_size=32,
    )

    img = np.load(summary['image_files'][0])
    gt = np.load(summary['mask_files'][0])
    assert summary['prepared_count'] == 1
    assert img.shape == (32, 32, 3)
    assert gt.shape == (32, 32)
    assert img.min() >= 0.0
    assert img.max() <= 1.0
    assert set(np.unique(gt).tolist()) == {0, 1}


def test_external_fixed_split_baselines_skips_when_fixed_rows_empty(tmp_path: Path):
    cfg = {
        'baseline_evaluation': {},
        'proposal': {},
        'tiling': {},
        'current_segmenter': {},
        'medsam': {},
    }
    out = _run_external_fixed_split_baselines(
        fixed_rows=pd.DataFrame(),
        config=cfg,
        baseline_dir=tmp_path / 'baseline_metrics',
        _runtime_root=tmp_path,
    )
    assert out['status'] == 'skipped_empty_fixed_split'


def test_external_fixed_split_oracle_only_invokes_medsam_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / 'MedSAM'
    repo.mkdir()
    (repo / 'MedSAM_Inference.py').write_text('# stub\n', encoding='utf-8')
    fake_python = tmp_path / 'fake_python'
    fake_python.write_bytes(b'')
    checkpoint = tmp_path / 'medsam.pth'
    checkpoint.write_bytes(b'')

    cfg = {
        'baseline_evaluation': {
            'external_baselines': True,
            'automatic_medsam': False,
            'oracle_prompt_medsam': True,
            'current_segmenter': False,
        },
        'proposal': {'min_component_area': 1, 'padding': 0},
        'tiling': {},
        'current_segmenter': {},
        'medsam': {
            'python': str(fake_python),
            'repo': str(repo),
            'base_checkpoint': str(checkpoint),
            'device': 'cpu',
        },
    }

    image_path = tmp_path / 'img.png'
    mask_path = tmp_path / 'msk.png'
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    _write_png(mask_path, mask)

    fixed_rows = pd.DataFrame(
        [
            {
                'manifest_row_id': 'row-1',
                'cohort_id': 'c1',
                'lane_assignment': 'manual_mask_core',
                'source_sample_id': 's1',
                'image_path': 'img.png',
                'mask_path': 'msk.png',
                'image_path_resolved': str(image_path),
                'mask_path_resolved': str(mask_path),
                'selection_rank': 1,
                'selection_reason': 'test',
            }
        ]
    )

    calls: list = []

    def fake_batch(**kwargs):
        calls.append(kwargs)
        for item in kwargs['items']:
            Path(item['output_path']).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ones((16, 16), dtype=np.uint8) * 255).save(
                item['output_path']
            )
        return []

    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._run_medsam_batch',
        fake_batch,
    )

    out = _run_external_fixed_split_baselines(
        fixed_rows=fixed_rows,
        config=cfg,
        baseline_dir=tmp_path / 'baseline_metrics',
        _runtime_root=tmp_path,
    )

    assert len(calls) == 1
    assert out['status'] == 'completed'
    assert 'oracle_medsam_metrics_csv' in out


@pytest.mark.parametrize(
    ('comparison_override', 'expected_packaging_status', 'expect_release'),
    [
        pytest.param(
            None,
            'skipped_comparison_not_eligible',
            False,
            id='blocked-missing-evidence',
        ),
        pytest.param(
            {
                'status': 'completed',
                'reason': '',
                'reason_codes': [],
                'failed_gates': [],
                'adoption_tier': 'blocked',
                'primary_generated_mask_transition_status': 'blocked',
                'recommended_generated_mask_source': '',
                'failure_mode': 'oracle_gap',
                'oracle_dice_gap': 0.19,
            },
            'skipped_comparison_not_eligible',
            False,
            id='completed-blocked-adoption',
        ),
        pytest.param(
            {
                'status': 'completed',
                'reason': '',
                'reason_codes': [],
                'failed_gates': [],
                'adoption_tier': 'improved_candidate_not_oracle',
                'primary_generated_mask_transition_status': (
                    'improved_candidate_not_oracle'
                ),
                'recommended_generated_mask_source': '',
                'failure_mode': 'oracle_gap',
                'oracle_dice_gap': 0.08,
                'oracle_level_gates_passed': False,
                'improves_current_auto': True,
                'improves_current_segmenter': True,
                'beats_trivial_baseline': True,
            },
            'completed',
            True,
            id='completed-reliable-improved-candidate',
        ),
    ],
)
def test_workflow_summary_packages_only_eligible_completed_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    comparison_override: dict[str, object] | None,
    expected_packaging_status: str,
    expect_release: bool,
) -> None:
    runtime_root = tmp_path
    split_rows = []
    for index, split in enumerate(('train', 'validation', 'test'), start=1):
        image_path = runtime_root / f'raw_data/cohorts/demo/images/img-{split}.png'
        mask_path = runtime_root / f'raw_data/cohorts/demo/masks/mask-{split}.png'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.full((12, 12, 3), index, dtype=np.uint8)
        ).save(image_path)
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2 + index : 8 + index, 3:9] = 1
        _write_png(mask_path, mask)
        split_rows.append(
            {
                'manifest_row_id': f'row-{split}',
                'cohort_id': 'demo',
                'lane_assignment': 'manual_mask_core',
                'source_sample_id': f'sample-{split}',
                'split_group_id': f'group-{split}',
                'image_path': str(image_path.relative_to(runtime_root)),
                'mask_path': str(mask_path.relative_to(runtime_root)),
                'split': split,
                'selection_rank': 1,
                'selection_reason': 'test',
            }
        )
    split_dir = runtime_root / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_row in split_rows:
        pd.DataFrame([split_row]).to_csv(
            split_dir / f"{split_row['split']}.csv", index=False
        )

    manifest_path = runtime_root / 'raw_data/cohorts/manifest.csv'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_rows).to_csv(manifest_path, index=False)

    base_checkpoint = runtime_root / 'models/medsam/base.pth'
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base_checkpoint.write_bytes(b'base')
    medsam_repo = runtime_root / 'vendor/MedSAM'
    medsam_repo.mkdir(parents=True, exist_ok=True)
    (medsam_repo / 'MedSAM_Inference.py').write_text('# stub\n', encoding='utf-8')
    medsam_python = runtime_root / 'bin/python'
    medsam_python.parent.mkdir(parents=True, exist_ok=True)
    medsam_python.write_text('#!/usr/bin/env python\n', encoding='utf-8')

    checkpoint_root = runtime_root / 'models/medsam_glomeruli/test-run'
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / 'finetuned_epoch_1.pth').write_bytes(b'finetuned')
    transfer_model = runtime_root / 'models/current/transfer.pkl'
    scratch_model = runtime_root / 'models/current/scratch.pkl'
    transfer_model.parent.mkdir(parents=True, exist_ok=True)
    transfer_model.write_bytes(b'transfer')
    scratch_model.write_bytes(b'scratch')

    class _FakeLearner:
        class _Model:
            def eval(self) -> None:
                return None

        model = _Model()

    monkeypatch.setattr(
        'eq.data_management.model_loading.load_model_safely',
        lambda *_args, **_kwargs: _FakeLearner(),
    )
    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._predict_probability',
        lambda **_kwargs: (np.ones((12, 12), dtype=np.float32), {}),
    )

    def _fake_batch(**kwargs):
        for item in kwargs['items']:
            Path(item['output_path']).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ones((12, 12), dtype=np.uint8) * 255).save(
                item['output_path']
            )
        failures_path = Path(kwargs['output_dir']) / 'medsam_batch_failures.json'
        failures_path.write_text('[]', encoding='utf-8')
        return []

    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._run_medsam_batch',
        _fake_batch,
    )

    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._preflight_training_dependencies',
        lambda _cfg: {
            'medsam_python': str(medsam_python),
            'medsam_repo': str(medsam_repo),
            'base_checkpoint': str(base_checkpoint),
            'entrypoint': '',
            'adaptation_mode': 'frozen_image_encoder_mask_decoder',
            'training_backend': 'eq_native_adapter',
        },
    )
    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._training_command_for_config',
        lambda **_kwargs: ['python', '-m', 'eq.evaluation.medsam_glomeruli_adapter'],
    )
    if comparison_override is not None:
        monkeypatch.setattr(
            'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._build_finetuned_comparison_summary',
            lambda **_kwargs: comparison_override,
        )

    config_path = runtime_root / 'config.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'workflow: medsam_glomeruli_fine_tuning',
                'run:',
                '  name: test-run',
                f'  runtime_root_default: {runtime_root}',
                'medsam:',
                f'  python: {medsam_python}',
                f'  repo: {medsam_repo}',
                f'  base_checkpoint: {base_checkpoint}',
                '  device: cpu',
                '  adaptation_mode: frozen_image_encoder_mask_decoder',
                '  frozen_components: [image_encoder, prompt_encoder]',
                '  trainable_components: [mask_decoder]',
                'inputs:',
                f'  manifest_path: {manifest_path.relative_to(runtime_root)}',
                '  lane_assignments: [manual_mask_core]',
                '  split_manifests:',
                f'    train: {(split_dir / "train.csv").relative_to(runtime_root)}',
                f'    validation: {(split_dir / "validation.csv").relative_to(runtime_root)}',
                f'    test: {(split_dir / "test.csv").relative_to(runtime_root)}',
                'outputs:',
                '  evaluation_dir: output/test-eval',
                f'  checkpoint_root: {checkpoint_root.relative_to(runtime_root)}',
                f'  generated_mask_release_root: {(runtime_root / "derived_data/generated_masks/glomeruli/medsam_finetuned/test-run").relative_to(runtime_root)}',
                f'  generated_mask_registry_path: {(runtime_root / "derived_data/generated_masks/glomeruli/manifest.csv").relative_to(runtime_root)}',
                'baseline_evaluation:',
                '  external_baselines: false',
                'current_segmenter:',
                f'  transfer_model_path: {transfer_model}',
                f'  scratch_model_path: {scratch_model}',
                'training:',
                '  image_size: 32',
                '  run_training: false',
                '  local_feasibility_required: false',
                'adoption_gates: {}',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    result = run_medsam_glomeruli_fine_tuning_workflow(config_path, dry_run=False)
    summary = json.loads(result['summary'].read_text(encoding='utf-8'))
    finetuned = dict(summary['finetuned_evaluation'])
    comparison = dict(summary['finetuned_comparison'])

    assert finetuned['status'] == 'completed'
    assert 'metric_rows' in finetuned
    assert 'expected_metric_rows' in finetuned
    assert 'metrics_csv' in finetuned
    assert 'masks_dir' in finetuned
    assert 'overlays_dir' in finetuned
    assert 'proposal_boxes_csv' in finetuned
    assert 'proposal_recall_csv' in finetuned
    assert 'prompt_failure_count' in finetuned
    assert 'prompt_failures_csv' in finetuned
    assert 'prompt_failures_json' in finetuned
    assert finetuned['overlay_review_status'] == 'not_reviewed'
    assert Path(finetuned['metrics_csv']).exists()
    assert Path(finetuned['prompt_failures_csv']).exists()
    assert Path(finetuned['proposal_boxes_csv']).exists()
    assert Path(finetuned['proposal_recall_csv']).exists()
    assert Path(finetuned['overlays_dir']).exists()
    assert Path(finetuned['masks_dir']).exists()
    expected_adoption_tier = (
        'blocked'
        if comparison_override is None
        else str(comparison_override['adoption_tier'])
    )
    assert comparison['adoption_tier'] == expected_adoption_tier
    if comparison_override is None:
        assert 'oracle_dice_gap' not in comparison
        assert comparison['status'] == 'blocked'
        assert comparison['failure_mode'] == 'metric_evidence_unavailable'
        assert 'oracle_metric_evidence_unavailable' in comparison['reason_codes']
        assert comparison['failed_gates']
    else:
        assert comparison['status'] == 'completed'
        assert comparison['oracle_dice_gap'] == pytest.approx(
            comparison_override['oracle_dice_gap']
        )
    assert summary['generated_mask_packaging']['status'] == expected_packaging_status
    if comparison_override is not None and comparison_override['adoption_tier'] == (
        'improved_candidate_not_oracle'
    ):
        assert summary['promotion_status'] == 'blocked'
        assert 'scientific_promotion_not_eligible' in summary[
            'promotion_reason_codes'
        ]
        assert summary['promotion_report']['decision'] == 'blocked'
        assert summary['promotion_report']['practical_adoption_decision'] == (
            'package_candidate'
        )
        assert 'scientific_promotion_not_eligible' in summary['promotion_report'][
            'reason_codes'
        ]
        report = summary['promotion_report']
        assert report['path_encoding'] == 'runtime-relative-posix'
        assert not Path(
            report['generated_mask_packaging']['manifest']
        ).is_absolute()
        assert not Path(
            report['box_diagnostics']['artifacts']['csv']
        ).is_absolute()
        assert report['promotion_evidence']['overlay_review']['status'] == (
            'unavailable'
        )
        json.dumps(report, allow_nan=False)
    release_root = (
        runtime_root
        / 'derived_data/generated_masks/glomeruli/medsam_finetuned/test-run'
    )
    registry_path = runtime_root / 'derived_data/generated_masks/glomeruli/manifest.csv'
    assert release_root.exists() is expect_release
    assert registry_path.exists() is expect_release
    if expect_release:
        release_manifest = pd.read_csv(release_root / 'manifest.csv')
        registry = pd.read_csv(registry_path)
        assert set(release_manifest['adoption_tier']) == {
            'improved_candidate_not_oracle'
        }
        assert set(registry['adoption_tier']) == {'improved_candidate_not_oracle'}


def test_workflow_summary_fails_closed_when_finetuned_inference_is_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root = tmp_path
    image_path = runtime_root / 'raw_data/cohorts/demo/images/img.png'
    mask_path = runtime_root / 'raw_data/cohorts/demo/masks/mask.png'
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_path)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 1
    _write_png(mask_path, mask)

    split_row = {
        'manifest_row_id': 'row-1',
        'cohort_id': 'demo',
        'lane_assignment': 'manual_mask_core',
        'source_sample_id': 'sample-1',
        'split_group_id': 'sample-1',
        'image_path': 'raw_data/cohorts/demo/images/img.png',
        'mask_path': 'raw_data/cohorts/demo/masks/mask.png',
        'split': 'train',
        'selection_rank': 1,
        'selection_reason': 'test',
    }
    split_dir = runtime_root / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split_row]).to_csv(split_dir / 'train.csv', index=False)
    pd.DataFrame([{**split_row, 'split': 'validation'}]).to_csv(
        split_dir / 'validation.csv', index=False
    )
    pd.DataFrame([{**split_row, 'split': 'test'}]).to_csv(
        split_dir / 'test.csv', index=False
    )

    manifest_path = runtime_root / 'raw_data/cohorts/manifest.csv'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split_row]).to_csv(manifest_path, index=False)

    base_checkpoint = runtime_root / 'models/medsam/base.pth'
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base_checkpoint.write_bytes(b'base')
    medsam_repo = runtime_root / 'vendor/MedSAM'
    medsam_repo.mkdir(parents=True, exist_ok=True)
    (medsam_repo / 'MedSAM_Inference.py').write_text('# stub\n', encoding='utf-8')
    medsam_python = runtime_root / 'bin/python'
    medsam_python.parent.mkdir(parents=True, exist_ok=True)
    medsam_python.write_text('#!/usr/bin/env python\n', encoding='utf-8')

    checkpoint_root = runtime_root / 'models/medsam_glomeruli/test-run'
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / 'finetuned_epoch_1.pth').write_bytes(b'finetuned')
    transfer_model = runtime_root / 'models/current/transfer.pkl'
    scratch_model = runtime_root / 'models/current/scratch.pkl'
    transfer_model.parent.mkdir(parents=True, exist_ok=True)
    transfer_model.write_bytes(b'transfer')
    scratch_model.write_bytes(b'scratch')

    class _FakeLearner:
        class _Model:
            def eval(self) -> None:
                return None

        model = _Model()

    monkeypatch.setattr(
        'eq.data_management.model_loading.load_model_safely',
        lambda *_args, **_kwargs: _FakeLearner(),
    )
    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._predict_probability',
        lambda **_kwargs: (np.ones((12, 12), dtype=np.float32), {}),
    )

    def _fake_partial_batch(**kwargs):
        failures_path = Path(kwargs['output_dir']) / 'medsam_batch_failures.json'
        failures_path.write_text(
            json.dumps(
                [
                    {
                        'manifest_row_id': 'row-1',
                        'image_path': str(image_path),
                        'failure_reason': 'simulated prompt failure',
                    }
                ]
            ),
            encoding='utf-8',
        )
        return [
            {
                'manifest_row_id': 'row-1',
                'image_path': str(image_path),
                'failure_reason': 'simulated prompt failure',
            }
        ]

    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._run_medsam_batch',
        _fake_partial_batch,
    )
    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._preflight_training_dependencies',
        lambda _cfg: {
            'medsam_python': str(medsam_python),
            'medsam_repo': str(medsam_repo),
            'base_checkpoint': str(base_checkpoint),
            'entrypoint': '',
            'adaptation_mode': 'frozen_image_encoder_mask_decoder',
            'training_backend': 'eq_native_adapter',
        },
    )
    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._training_command_for_config',
        lambda **_kwargs: ['python', '-m', 'eq.evaluation.medsam_glomeruli_adapter'],
    )

    config_path = runtime_root / 'config.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'workflow: medsam_glomeruli_fine_tuning',
                'run:',
                '  name: test-run',
                f'  runtime_root_default: {runtime_root}',
                'medsam:',
                f'  python: {medsam_python}',
                f'  repo: {medsam_repo}',
                f'  base_checkpoint: {base_checkpoint}',
                '  device: cpu',
                '  adaptation_mode: frozen_image_encoder_mask_decoder',
                '  frozen_components: [image_encoder, prompt_encoder]',
                '  trainable_components: [mask_decoder]',
                'inputs:',
                f'  manifest_path: {manifest_path.relative_to(runtime_root)}',
                '  lane_assignments: [manual_mask_core]',
                '  split_manifests:',
                f'    train: {(split_dir / "train.csv").relative_to(runtime_root)}',
                f'    validation: {(split_dir / "validation.csv").relative_to(runtime_root)}',
                f'    test: {(split_dir / "test.csv").relative_to(runtime_root)}',
                'outputs:',
                '  evaluation_dir: output/test-eval',
                f'  checkpoint_root: {checkpoint_root.relative_to(runtime_root)}',
                f'  generated_mask_release_root: {(runtime_root / "derived_data/generated_masks/glomeruli/medsam_finetuned/test-run").relative_to(runtime_root)}',
                f'  generated_mask_registry_path: {(runtime_root / "derived_data/generated_masks/glomeruli/manifest.csv").relative_to(runtime_root)}',
                'baseline_evaluation:',
                '  external_baselines: false',
                'current_segmenter:',
                f'  transfer_model_path: {transfer_model}',
                f'  scratch_model_path: {scratch_model}',
                'training:',
                '  image_size: 32',
                '  run_training: false',
                '  local_feasibility_required: false',
                'adoption_gates: {}',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    result = run_medsam_glomeruli_fine_tuning_workflow(config_path, dry_run=False)
    summary = json.loads(result['summary'].read_text(encoding='utf-8'))

    assert summary['finetuned_evaluation']['status'] == 'failed_no_predictions'
    assert summary['finetuned_evaluation']['prompt_failure_count'] == 1
    assert summary['finetuned_comparison']['adoption_tier'] == 'blocked'
    assert summary['finetuned_comparison']['failure_mode'] == 'failed_no_predictions'


def test_finetuned_comparison_blocks_and_preserves_failed_partial_predictions() -> None:
    comparison = _build_finetuned_comparison_summary(
        finetuned_evaluation={
            'status': 'failed_partial_predictions',
            'metrics_csv': '',
            'overlay_review_status': 'not_run',
            'prompt_failure_count': 0,
        },
        external_baselines_summary={},
        baseline_outputs={},
        adoption_gates={},
    )

    assert comparison['adoption_tier'] == 'blocked'
    assert comparison['primary_generated_mask_transition_status'] == 'blocked'
    assert comparison['recommended_generated_mask_source'] == ''
    assert comparison['failure_mode'] == 'failed_partial_predictions'


def test_finetuned_evaluation_sets_failed_partial_predictions_when_some_masks_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_a = tmp_path / 'raw_data/cohorts/demo/images/img-a.png'
    image_b = tmp_path / 'raw_data/cohorts/demo/images/img-b.png'
    mask_a = tmp_path / 'raw_data/cohorts/demo/masks/mask-a.png'
    mask_b = tmp_path / 'raw_data/cohorts/demo/masks/mask-b.png'
    image_a.parent.mkdir(parents=True, exist_ok=True)
    mask_a.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_a)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_b)
    binary_mask = np.zeros((12, 12), dtype=np.uint8)
    binary_mask[2:10, 2:10] = 1
    _write_png(mask_a, binary_mask)
    _write_png(mask_b, binary_mask)

    transfer_model = tmp_path / 'models/current/transfer.pkl'
    scratch_model = tmp_path / 'models/current/scratch.pkl'
    transfer_model.parent.mkdir(parents=True, exist_ok=True)
    transfer_model.write_bytes(b'transfer')
    scratch_model.write_bytes(b'scratch')

    checkpoint = tmp_path / 'models/medsam_glomeruli/ckpt/final.pth'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'finetuned')

    class _FakeLearner:
        class _Model:
            def eval(self) -> None:
                return None

        model = _Model()

    monkeypatch.setattr(
        'eq.data_management.model_loading.load_model_safely',
        lambda *_args, **_kwargs: _FakeLearner(),
    )
    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._predict_probability',
        lambda **_kwargs: (np.ones((12, 12), dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._preflight_medsam_inference_paths',
        lambda _cfg: {
            'medsam_python': tmp_path / 'bin/python',
            'medsam_repo': tmp_path / 'vendor/MedSAM',
            'checkpoint': tmp_path / 'models/medsam/base.pth',
            'medsam_script': tmp_path / 'vendor/MedSAM/MedSAM_Inference.py',
        },
    )

    def _fake_partial_batch(**kwargs):
        for index, item in enumerate(kwargs['items']):
            if index > 0:
                break
            Path(item['output_path']).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ones((12, 12), dtype=np.uint8) * 255).save(
                item['output_path']
            )
        failures_path = Path(kwargs['output_dir']) / 'medsam_batch_failures.json'
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        failures_path.write_text('[]', encoding='utf-8')
        return []

    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._run_medsam_batch',
        _fake_partial_batch,
    )

    fixed_rows = pd.DataFrame(
        [
            {
                'manifest_row_id': 'row-1',
                'cohort_id': 'demo',
                'lane_assignment': 'manual_mask_core',
                'source_sample_id': 'sample-1',
                'image_path': str(image_a),
                'mask_path': str(mask_a),
                'image_path_resolved': str(image_a),
                'mask_path_resolved': str(mask_a),
                'selection_rank': 1,
                'selection_reason': 'test',
            },
            {
                'manifest_row_id': 'row-2',
                'cohort_id': 'demo',
                'lane_assignment': 'manual_mask_core',
                'source_sample_id': 'sample-2',
                'image_path': str(image_b),
                'mask_path': str(mask_b),
                'image_path_resolved': str(image_b),
                'mask_path_resolved': str(mask_b),
                'selection_rank': 2,
                'selection_reason': 'test',
            },
        ]
    )
    summary = _run_finetuned_fixed_split_evaluation(
        fixed_rows=fixed_rows,
        evaluation_dir=tmp_path / 'output/test-eval',
        config={
            'proposal': {'thresholds': [0.35]},
            'tiling': {},
            'current_segmenter': {
                'transfer_model_path': str(transfer_model),
                'scratch_model_path': str(scratch_model),
            },
            'medsam': {'device': 'cpu'},
        },
        checkpoint_provenance={
            'supported_checkpoint': True,
            'checkpoint_files': [str(checkpoint)],
        },
        dry_run=False,
    )

    assert summary['status'] == 'failed_partial_predictions'
    assert summary['metric_rows'] == 1
    assert summary['expected_metric_rows'] == 2
    assert summary['prompt_failure_count'] == 0


def test_workflow_summary_marks_finetuned_evaluation_skipped_without_supported_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root = tmp_path
    image_path = runtime_root / 'raw_data/cohorts/demo/images/img.png'
    mask_path = runtime_root / 'raw_data/cohorts/demo/masks/mask.png'
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_path)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 1
    _write_png(mask_path, mask)

    split_row = {
        'manifest_row_id': 'row-1',
        'cohort_id': 'demo',
        'lane_assignment': 'manual_mask_core',
        'source_sample_id': 'sample-1',
        'split_group_id': 'sample-1',
        'image_path': 'raw_data/cohorts/demo/images/img.png',
        'mask_path': 'raw_data/cohorts/demo/masks/mask.png',
        'split': 'train',
        'selection_rank': 1,
        'selection_reason': 'test',
    }
    split_dir = runtime_root / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split_row]).to_csv(split_dir / 'train.csv', index=False)
    pd.DataFrame([{**split_row, 'split': 'validation'}]).to_csv(
        split_dir / 'validation.csv', index=False
    )
    pd.DataFrame([{**split_row, 'split': 'test'}]).to_csv(
        split_dir / 'test.csv', index=False
    )

    manifest_path = runtime_root / 'raw_data/cohorts/manifest.csv'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split_row]).to_csv(manifest_path, index=False)

    base_checkpoint = runtime_root / 'models/medsam/base.pth'
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base_checkpoint.write_bytes(b'base')
    medsam_repo = runtime_root / 'vendor/MedSAM'
    medsam_repo.mkdir(parents=True, exist_ok=True)
    medsam_python = runtime_root / 'bin/python'
    medsam_python.parent.mkdir(parents=True, exist_ok=True)
    medsam_python.write_text('#!/usr/bin/env python\n', encoding='utf-8')

    checkpoint_root = runtime_root / 'models/medsam_glomeruli/test-run'
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._preflight_training_dependencies',
        lambda _cfg: {
            'medsam_python': str(medsam_python),
            'medsam_repo': str(medsam_repo),
            'base_checkpoint': str(base_checkpoint),
            'entrypoint': '',
            'adaptation_mode': 'frozen_image_encoder_mask_decoder',
            'training_backend': 'eq_native_adapter',
        },
    )
    monkeypatch.setattr(
        'eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._training_command_for_config',
        lambda **_kwargs: ['python', '-m', 'eq.evaluation.medsam_glomeruli_adapter'],
    )

    config_path = runtime_root / 'config.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'workflow: medsam_glomeruli_fine_tuning',
                'run:',
                '  name: test-run',
                f'  runtime_root_default: {runtime_root}',
                'medsam:',
                f'  python: {medsam_python}',
                f'  repo: {medsam_repo}',
                f'  base_checkpoint: {base_checkpoint}',
                '  device: cpu',
                '  adaptation_mode: frozen_image_encoder_mask_decoder',
                '  frozen_components: [image_encoder, prompt_encoder]',
                '  trainable_components: [mask_decoder]',
                'inputs:',
                f'  manifest_path: {manifest_path.relative_to(runtime_root)}',
                '  lane_assignments: [manual_mask_core]',
                '  split_manifests:',
                f'    train: {(split_dir / "train.csv").relative_to(runtime_root)}',
                f'    validation: {(split_dir / "validation.csv").relative_to(runtime_root)}',
                f'    test: {(split_dir / "test.csv").relative_to(runtime_root)}',
                'outputs:',
                '  evaluation_dir: output/test-eval',
                f'  checkpoint_root: {checkpoint_root.relative_to(runtime_root)}',
                f'  generated_mask_release_root: {(runtime_root / "derived_data/generated_masks/glomeruli/medsam_finetuned/test-run").relative_to(runtime_root)}',
                f'  generated_mask_registry_path: {(runtime_root / "derived_data/generated_masks/glomeruli/manifest.csv").relative_to(runtime_root)}',
                'baseline_evaluation:',
                '  external_baselines: false',
                'training:',
                '  image_size: 32',
                '  run_training: false',
                '  local_feasibility_required: false',
                'adoption_gates: {}',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    result = run_medsam_glomeruli_fine_tuning_workflow(config_path, dry_run=False)
    summary = json.loads(result['summary'].read_text(encoding='utf-8'))

    assert (
        summary['finetuned_evaluation']['status']
        == 'skipped_missing_supported_checkpoint'
    )
    audit_summary = summary['dataset_split_audit']
    assert audit_summary['row_counts']['by_split'] == {
        'test': 1,
        'train': 1,
        'validation': 1,
    }
    assert Path(audit_summary['artifacts']['json']).exists()
    assert Path(audit_summary['artifacts']['csv']).exists()
    assert len(audit_summary['audit_sha256']) == 64
    assert audit_summary['promotion_blocked'] is True
    assert summary['promotion_status'] == 'blocked'
    assert 'cross_split_strong_group_leakage' in summary['promotion_reason_codes']
    assert summary['promotion_report']['decision'] == 'blocked'
    assert any(
        gate['reason_code'] == 'cross_split_strong_group_leakage'
        for gate in summary['promotion_report']['failed_gates']
    )
    box_diagnostics = summary['box_diagnostics']
    assert Path(box_diagnostics['artifacts']['csv']).exists()
    assert Path(box_diagnostics['artifacts']['json']).exists()
    assert box_diagnostics['summary']['row_count'] == 2
    promotion_report = summary['promotion_report']
    assert Path(summary['promotion_report_path']).exists()
    assert promotion_report['decision'] == 'blocked'
    assert promotion_report['evaluation']['run_id'] == 'test-run'
    assert promotion_report['checkpoint']['checkpoint_id'] == 'test-run'
    assert promotion_report['dataset_audit']['id'] == audit_summary['audit_sha256']
    assert promotion_report['box_diagnostics']['artifacts']['sha256']
    assert 'scientific_promotion_not_eligible' in promotion_report['reason_codes']
