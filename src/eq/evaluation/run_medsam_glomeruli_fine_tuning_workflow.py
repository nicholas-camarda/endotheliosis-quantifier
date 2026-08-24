"""Run MedSAM glomeruli fine-tuning workflow from YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

from eq.evaluation.medsam_glomeruli_audit import (
    write_medsam_box_diagnostics,
    write_medsam_evaluation_cohort_audit,
)
from eq.evaluation.medsam_glomeruli_workflow import (
    _file_hash,
    _runtime_path,
    _runtime_root,
    _write_csv,
    ensure_evaluation_output_path,
    load_binary_mask,
    metric_row,
)
from eq.evaluation.medsam_torch_runtime import (
    medsam_subprocess_extra_env,
    resolve_medsam_torch_python,
    training_backend,
)
from eq.evaluation.run_medsam_automatic_glomeruli_prompts_workflow import (
    AUTOMATIC_MASK_SOURCE,
    AUTOMATIC_METRIC_FIELDS,
    PROPOSAL_BOX_FIELDS,
    _automatic_metric_row,
    _candidate_paths,
    _predict_probability,
    _select_best_setting,
    derive_proposal_boxes,
    proposal_recall_row,
)
from eq.evaluation.run_medsam_manual_glomeruli_comparison_workflow import (
    _run_medsam_batch,
    derive_oracle_boxes,
)
from eq.utils.artifact_provenance import (
    GENERATED_MASK_RELEASE_SCHEMA_VERSION,
    RUNTIME_RELATIVE_POSIX_ENCODING,
    ArtifactProvenanceError,
    resolve_runtime_relative_path,
    serialize_runtime_relative_path,
    validate_generated_mask_artifact_paths,
)
from eq.utils.execution_logging import current_execution_log_context
from eq.utils.paths import (
    ensure_not_under_runtime_raw_data,
    get_runtime_generated_masks_glomeruli_manifest_path,
    get_runtime_medsam_finetuned_release_path,
    get_runtime_medsam_glomeruli_checkpoint_path,
)

WORKFLOW_ID = 'medsam_glomeruli_fine_tuning'
DEFAULT_RUN_ID = 'pilot_medsam_glomeruli_fine_tuning'
DEFAULT_OUTPUT_DIR = 'output/segmentation_evaluation/medsam_glomeruli_fine_tuning'
DEFAULT_MANIFEST_PATH = 'raw_data/cohorts/manifest.csv'
MASK_SOURCE = 'medsam_finetuned_glomeruli'
LOGGER = logging.getLogger('eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow')
DEFAULT_ADOPTION_GATES = {
    'min_dice': 0.90,
    'min_jaccard': 0.82,
    'max_oracle_dice_gap': 0.05,
    'max_prompt_failures': 0,
    'min_prediction_foreground_fraction': 0.0,
    'max_prediction_foreground_fraction': 0.50,
    'min_prediction_to_truth_area_ratio': 0.50,
    'max_prediction_to_truth_area_ratio': 1.50,
}


def _emit(message: str) -> None:
    LOGGER.info('%s', message)


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f'Config does not exist: {config_path}')
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Config must be a mapping: {config_path}')
    if payload.get('workflow') != WORKFLOW_ID:
        raise ValueError(f'Fine-tuning config must use `workflow: {WORKFLOW_ID}`.')
    return payload


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f'{key} must be a mapping')
    return value


def _effective_adoption_gates(gates: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit effective gate values recorded in promotion evidence."""
    return {**DEFAULT_ADOPTION_GATES, **gates}


def _portable_report_value(
    value: Any, *, runtime_root: Path, key: str = ''
) -> Any:
    """Encode report artifact paths relative to the declared runtime root."""
    path_keys = {
        'artifact_path',
        'artifact_paths',
        'base_checkpoint',
        'checkpoint_path',
        'checkpoint_files',
        'config_path',
        'generated_mask_path',
        'image_path',
        'index',
        'inference_checkpoint_path',
        'manifest',
        'mask_path',
        'masks_dir',
        'metrics_csv',
        'overlays_dir',
        'paths',
        'prompt_failures_csv',
        'prompt_failures_json',
        'proposal_boxes_csv',
        'proposal_recall_csv',
        'provenance',
        'provenance_path',
        'reference_mask_path',
        'release_dir',
        'release_manifest_path',
        'source_image_path',
        'split_manifest_paths',
    }
    if isinstance(value, dict):
        return {
            str(child_key): _portable_report_value(
                child_value, runtime_root=runtime_root, key=str(child_key)
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            _portable_report_value(item, runtime_root=runtime_root, key=key)
            for item in value
        ]
    is_path_key = (
        key in path_keys
        or key.endswith(('_path', '_dir', '_csv', '_json'))
        or key in {'csv', 'json', 'txt'}
    )
    if isinstance(value, str) and is_path_key:
        if not value:
            return ''
        path = Path(value)
        try:
            return path.resolve().relative_to(runtime_root.resolve()).as_posix()
        except ValueError:
            return value
    return value


def _resolve_output_paths(
    config: dict[str, Any], runtime_root: Path
) -> dict[str, Path]:
    run_cfg = _mapping(config, 'run')
    outputs = _mapping(config, 'outputs')
    run_id = str(run_cfg.get('name') or DEFAULT_RUN_ID)
    mask_release_id = str(outputs.get('mask_release_id') or run_id)
    checkpoint_id = str(outputs.get('checkpoint_id') or run_id)
    evaluation_dir = ensure_evaluation_output_path(
        runtime_root, outputs.get('evaluation_dir', f'{DEFAULT_OUTPUT_DIR}/{run_id}')
    )
    checkpoint_root = _runtime_path(
        runtime_root,
        outputs.get(
            'checkpoint_root',
            str(
                get_runtime_medsam_glomeruli_checkpoint_path(
                    checkpoint_id, runtime_root
                )
            ),
        ),
    )
    generated_release_root = _runtime_path(
        runtime_root,
        outputs.get(
            'generated_mask_release_root',
            str(
                get_runtime_medsam_finetuned_release_path(mask_release_id, runtime_root)
            ),
        ),
    )
    generated_registry_path = _runtime_path(
        runtime_root,
        outputs.get(
            'generated_mask_registry_path',
            str(get_runtime_generated_masks_glomeruli_manifest_path(runtime_root)),
        ),
    )
    ensure_not_under_runtime_raw_data(generated_release_root, runtime_root)
    ensure_not_under_runtime_raw_data(generated_registry_path, runtime_root)
    return {
        'evaluation_dir': evaluation_dir,
        'checkpoint_root': checkpoint_root,
        'generated_release_root': generated_release_root,
        'generated_registry_path': generated_registry_path,
        'run_id': Path(run_id),
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _strongest_group_id(row: pd.Series) -> str:
    for key in ('source_sample_id', 'subject_id', 'animal_id', 'manifest_row_id'):
        value = str(row.get(key, '')).strip()
        if value:
            return value
    return ''


def _admitted_manual_rows(
    manifest: pd.DataFrame,
    runtime_root: Path,
    *,
    lane_assignments: list[str],
    include_missing_files: bool = False,
) -> pd.DataFrame:
    frame = manifest.fillna('').copy()
    eligible = frame[
        (frame['admission_status'].astype(str) == 'admitted')
        & frame['lane_assignment'].astype(str).isin(lane_assignments)
    ].copy()
    eligible['image_path_resolved'] = eligible['image_path'].map(
        lambda p: str(_runtime_path(runtime_root, p)) if str(p).strip() else ''
    )
    eligible['mask_path_resolved'] = eligible['mask_path'].map(
        lambda p: str(_runtime_path(runtime_root, p)) if str(p).strip() else ''
    )
    if not include_missing_files:
        eligible = eligible[
            eligible['image_path'].astype(str).str.strip().ne('')
            & eligible['mask_path'].astype(str).str.strip().ne('')
            & eligible['image_path_resolved'].map(lambda p: Path(p).exists())
            & eligible['mask_path_resolved'].map(lambda p: Path(p).exists())
        ].copy()
    eligible['split_group_id'] = eligible.apply(_strongest_group_id, axis=1)
    eligible = eligible.sort_values(
        ['cohort_id', 'split_group_id', 'manifest_row_id'], kind='mergesort'
    ).reset_index(drop=True)
    return eligible


def _build_deterministic_splits(
    rows: pd.DataFrame, *, train_fraction: float, validation_fraction: float
) -> pd.DataFrame:
    group_order = (
        rows[['split_group_id']]
        .drop_duplicates()
        .sort_values(['split_group_id'], kind='mergesort')
        .reset_index(drop=True)
    )
    total_groups = len(group_order)
    train_cut = max(1, int(total_groups * train_fraction))
    validation_cut = max(
        train_cut + 1, int(total_groups * (train_fraction + validation_fraction))
    )
    assignments: dict[str, str] = {}
    for index, group in enumerate(group_order['split_group_id'].tolist()):
        if index < train_cut:
            assignments[group] = 'train'
        elif index < validation_cut:
            assignments[group] = 'validation'
        else:
            assignments[group] = 'test'
    output = rows.copy()
    output['split'] = output['split_group_id'].map(assignments).fillna('test')
    output['selection_rank'] = output.groupby('split').cumcount() + 1
    output['selection_reason'] = 'deterministic_grouped_assignment'
    return output


def _required_split_fields() -> list[str]:
    return [
        'manifest_row_id',
        'cohort_id',
        'lane_assignment',
        'source_sample_id',
        'split_group_id',
        'image_path',
        'mask_path',
        'split',
        'selection_rank',
        'selection_reason',
    ]


def _write_split_manifests(
    frame: pd.DataFrame, evaluation_dir: Path
) -> dict[str, Path]:
    split_dir = evaluation_dir / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ('train', 'validation', 'test'):
        target = split_dir / f'{split}.csv'
        subset = frame[frame['split'] == split].copy()
        subset[_required_split_fields()].to_csv(target, index=False)
        paths[split] = target
    return paths


def _validate_split_manifest(
    path: Path, runtime_root: Path, *, expected_split: str | None = None
) -> str:
    if not path.exists():
        raise FileNotFoundError(f'Configured split manifest does not exist: {path}')
    frame = pd.read_csv(path).fillna('')
    missing = [
        field for field in _required_split_fields() if field not in frame.columns
    ]
    if missing:
        raise ValueError(f'Split manifest is missing required fields {missing}: {path}')
    if frame.empty:
        raise ValueError(f'Split manifest must not be empty: {path}')
    if expected_split is not None and set(frame['split'].astype(str)) != {expected_split}:
        raise ValueError(
            f'Split manifest contains invalid split labels for {expected_split}: {path}'
        )
    for record in frame.to_dict(orient='records'):
        image_path = _runtime_path(runtime_root, record['image_path'])
        mask_path = _runtime_path(runtime_root, record['mask_path'])
        if not image_path.exists():
            raise FileNotFoundError(f'Split image path missing: {image_path}')
        if not mask_path.exists():
            raise FileNotFoundError(f'Split mask path missing: {mask_path}')
    return _hash_file(path)


def _fixed_evaluation_rows(
    split_paths: dict[str, str], runtime_root: Path
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ('validation', 'test'):
        path_text = split_paths.get(split, '')
        if not path_text:
            continue
        frame = pd.read_csv(path_text).fillna('')
        frame['split'] = split
        frame['image_path_resolved'] = frame['image_path'].map(
            lambda value: str(_runtime_path(runtime_root, value))
        )
        frame['mask_path_resolved'] = frame['mask_path'].map(
            lambda value: str(_runtime_path(runtime_root, value))
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=_required_split_fields())
    return pd.concat(frames, ignore_index=True)


def _split_rows(
    split_paths: dict[str, str], runtime_root: Path, split: str
) -> pd.DataFrame:
    path_text = split_paths.get(split, '')
    if not path_text:
        return pd.DataFrame(columns=_required_split_fields())
    frame = pd.read_csv(path_text).fillna('')
    frame['split'] = split
    frame['image_path_resolved'] = frame['image_path'].map(
        lambda value: str(_runtime_path(runtime_root, value))
    )
    frame['mask_path_resolved'] = frame['mask_path'].map(
        lambda value: str(_runtime_path(runtime_root, value))
    )
    return frame


def _write_fixed_baseline_metrics(
    *, fixed_rows: pd.DataFrame, output_dir: Path
) -> dict[str, Any]:
    metrics_dir = output_dir / 'baseline_metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in fixed_rows.to_dict(orient='records'):
        manual_mask = load_binary_mask(Path(str(record['mask_path_resolved'])))
        rows.extend(
            _baseline_metric_rows(
                manual_mask=manual_mask,
                manifest_row_id=str(record['manifest_row_id']),
                cohort_id=str(record['cohort_id']),
                lane_assignment=str(record['lane_assignment']),
            )
        )
    metrics_path = metrics_dir / 'trivial_baseline_metrics.csv'
    if rows:
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        grouped = (
            pd.DataFrame(rows)
            .groupby('method', dropna=False)[['dice', 'jaccard', 'precision', 'recall']]
            .mean()
            .reset_index()
        )
    else:
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        grouped = pd.DataFrame(
            columns=['method', 'dice', 'jaccard', 'precision', 'recall']
        )
    grouped_path = metrics_dir / 'trivial_baseline_metric_by_source.csv'
    grouped.to_csv(grouped_path, index=False)
    return {
        'metric_rows': len(rows),
        'trivial_baseline_metrics': str(metrics_path),
        'trivial_baseline_metric_by_source': str(grouped_path),
    }


def _baseline_execution_plan(
    *, fixed_rows: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    proposal = _mapping(config, 'proposal')
    medsam = _mapping(config, 'medsam')
    current_segmenter = _mapping(config, 'current_segmenter')
    return {
        'fixed_evaluation_rows': int(len(fixed_rows)),
        'automatic_medsam': {
            'status': 'configured',
            'mask_source': AUTOMATIC_MASK_SOURCE,
            'proposal_thresholds': proposal.get('thresholds', []),
            'proposal_source': 'current_segmenter_candidates',
            'medsam_checkpoint': str(medsam.get('base_checkpoint', '')),
        },
        'oracle_prompt_medsam': {
            'status': 'configured_if_enabled',
            'reference': 'manual_mask_component_boxes',
            'medsam_checkpoint': str(medsam.get('base_checkpoint', '')),
        },
        'current_segmenter': {
            'status': 'configured',
            'comparison_threshold': current_segmenter.get('comparison_threshold', ''),
            'candidate_paths': {
                'transfer': str(current_segmenter.get('transfer_model_path', '')),
                'scratch': str(current_segmenter.get('scratch_model_path', '')),
            },
        },
        'trivial_baselines': {
            'status': 'computed_from_fixed_manual_masks',
            'methods': ['trivial_all_background', 'trivial_all_foreground'],
        },
    }


def _load_rgb_resized(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert('RGB')
    image = image.resize((int(image_size), int(image_size)), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def _load_mask_resized(path: Path, image_size: int) -> np.ndarray:
    mask = Image.open(path).convert('L')
    mask = mask.resize((int(image_size), int(image_size)), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def _prepare_medsam_npy_data(
    *, train_rows: pd.DataFrame, output_root: Path, image_size: int
) -> dict[str, Any]:
    imgs_dir = output_root / 'imgs'
    gts_dir = output_root / 'gts'
    imgs_dir.mkdir(parents=True, exist_ok=True)
    gts_dir.mkdir(parents=True, exist_ok=True)
    image_files: list[str] = []
    mask_files: list[str] = []
    skipped: list[dict[str, str]] = []
    manifest_rows: list[dict[str, Any]] = []
    for record in train_rows.fillna('').to_dict(orient='records'):
        row_id = str(record.get('manifest_row_id', '')).strip()
        image_path = Path(
            str(record.get('image_path_resolved') or record['image_path'])
        )
        mask_path = Path(str(record.get('mask_path_resolved') or record['mask_path']))
        stem = row_id or image_path.stem
        target_name = f'{stem}.npy'
        if not image_path.exists() or not mask_path.exists():
            skipped.append(
                {
                    'manifest_row_id': row_id,
                    'reason': 'missing_image_or_mask',
                    'image_path': str(image_path),
                    'mask_path': str(mask_path),
                }
            )
            continue
        image = _load_rgb_resized(image_path, image_size)
        mask = _load_mask_resized(mask_path, image_size)
        if int(mask.sum()) <= 0:
            skipped.append(
                {
                    'manifest_row_id': row_id,
                    'reason': 'empty_mask_after_resize',
                    'image_path': str(image_path),
                    'mask_path': str(mask_path),
                }
            )
            continue
        image_target = imgs_dir / target_name
        mask_target = gts_dir / target_name
        np.save(image_target, image)
        np.save(mask_target, mask)
        image_files.append(str(image_target))
        mask_files.append(str(mask_target))
        manifest_rows.append(
            {
                'manifest_row_id': row_id,
                'source_image_path': str(image_path),
                'source_mask_path': str(mask_path),
                'npy_image_path': str(image_target),
                'npy_mask_path': str(mask_target),
                'image_size': int(image_size),
                'mask_foreground_pixels': int(mask.sum()),
            }
        )
    manifest_path = output_root / 'manifest.csv'
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    skipped_path = output_root / 'skipped.csv'
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)
    return {
        'output_root': str(output_root),
        'prepared_count': len(image_files),
        'skipped_count': len(skipped),
        'manifest': str(manifest_path),
        'skipped': str(skipped_path),
        'image_files': image_files,
        'mask_files': mask_files,
    }


def _baseline_metric_rows(
    *,
    manual_mask: np.ndarray,
    manifest_row_id: str,
    cohort_id: str,
    lane_assignment: str,
) -> list[dict[str, Any]]:
    manual = (np.asarray(manual_mask) > 0).astype(np.uint8)
    return [
        metric_row(
            method='trivial_all_background',
            candidate_artifact='trivial_baseline',
            manifest_row_id=manifest_row_id,
            cohort_id=cohort_id,
            lane_assignment=lane_assignment,
            manual_mask=manual,
            predicted_mask=np.zeros_like(manual, dtype=np.uint8),
        ),
        metric_row(
            method='trivial_all_foreground',
            candidate_artifact='trivial_baseline',
            manifest_row_id=manifest_row_id,
            cohort_id=cohort_id,
            lane_assignment=lane_assignment,
            manual_mask=manual,
            predicted_mask=np.ones_like(manual, dtype=np.uint8),
        ),
    ]


_FIXED_SPLIT_ROW_FIELDS = [
    'manifest_row_id',
    'cohort_id',
    'lane_assignment',
    'source_sample_id',
    'image_path',
    'mask_path',
    'image_path_resolved',
    'mask_path_resolved',
    'selection_rank',
    'selection_reason',
]


def _preflight_medsam_inference_paths(config: dict[str, Any]) -> dict[str, Path]:
    medsam = _mapping(config, 'medsam')
    python_path = resolve_medsam_torch_python(config)
    repo_path = Path(str(medsam.get('repo', '')).strip()).expanduser()
    checkpoint = Path(str(medsam.get('base_checkpoint', '')).strip()).expanduser()
    script_path = repo_path / 'MedSAM_Inference.py'
    bundle = {
        'medsam_python': python_path,
        'medsam_repo': repo_path,
        'checkpoint': checkpoint,
        'medsam_script': script_path,
    }
    for label, path in bundle.items():
        if not path.exists():
            raise FileNotFoundError(
                f'MedSAM inference preflight missing {label}: {path}'
            )
    return bundle


def _baseline_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {'rows': 0}
    frame = pd.DataFrame(rows)
    return {
        'rows': int(len(frame)),
        'mean_dice': float(frame['dice'].mean()),
        'mean_jaccard': float(frame['jaccard'].mean()),
        'mean_precision': float(frame['precision'].mean()),
        'mean_recall': float(frame['recall'].mean()),
    }


def _run_external_fixed_split_baselines(
    *,
    fixed_rows: pd.DataFrame,
    config: dict[str, Any],
    baseline_dir: Path,
    _runtime_root: Path,
) -> dict[str, Any]:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_eval_cfg = _mapping(config, 'baseline_evaluation')
    proposal_cfg = _mapping(config, 'proposal')
    tiling_cfg = _mapping(config, 'tiling')
    current_segmenter_cfg = _mapping(config, 'current_segmenter')
    medsam_cfg = _mapping(config, 'medsam')

    run_auto = bool(baseline_eval_cfg.get('automatic_medsam', True))
    run_oracle = bool(baseline_eval_cfg.get('oracle_prompt_medsam', True))
    run_current = bool(baseline_eval_cfg.get('current_segmenter', True))

    medsam_inference = bool(run_auto or run_oracle)
    need_segmenters = bool(run_auto or run_current)

    summary: dict[str, Any] = {
        'status': 'completed',
        'automatic_medsam': {'enabled': run_auto},
        'oracle_prompt_medsam': {'enabled': run_oracle},
        'current_segmenter': {'enabled': run_current},
    }

    if fixed_rows.empty:
        summary['status'] = 'skipped_empty_fixed_split'
        return summary

    bundle: dict[str, Path] = {}
    if medsam_inference:
        bundle = _preflight_medsam_inference_paths(config)

    tile_size = int(tiling_cfg.get('tile_size', 512))
    stride = int(tiling_cfg.get('stride', 512))
    expected_size = int(tiling_cfg.get('expected_size', 256))
    thresholds = [
        float(value) for value in proposal_cfg.get('thresholds', [0.2, 0.35, 0.5])
    ]
    min_area = int(proposal_cfg.get('min_component_area', 2000))
    max_area = int(proposal_cfg.get('max_component_area', 750000))
    padding = int(proposal_cfg.get('padding', 16))
    merge_iou = float(proposal_cfg.get('merge_iou', 0.25))
    max_boxes = int(proposal_cfg.get('max_boxes_per_image', 20))
    comparison_threshold = float(
        current_segmenter_cfg.get('comparison_threshold', 0.75)
    )

    recall_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    proposal_by_setting: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    current_metrics: list[dict[str, Any]] = []
    manual_masks: dict[str, np.ndarray] = {}

    current_models: dict[str, Any] = {}
    candidate_paths = _candidate_paths(current_segmenter_cfg)
    if need_segmenters:
        from eq.data_management.model_loading import load_model_safely

        for _, model_path in candidate_paths.items():
            if not model_path.exists():
                raise FileNotFoundError(
                    f'Missing current segmenter artifact: {model_path}'
                )
        for family, model_path in candidate_paths.items():
            learner = load_model_safely(str(model_path), model_type='glomeruli')
            learner.model.eval()
            current_models[family] = learner.model

    for record in fixed_rows.fillna('').to_dict(orient='records'):
        row_id = str(record['manifest_row_id'])
        image_path = Path(str(record['image_path_resolved']))
        manual_mask = load_binary_mask(Path(str(record['mask_path_resolved'])))
        manual_masks[row_id] = manual_mask

        base_record = {
            field: record.get(field, '') for field in _FIXED_SPLIT_ROW_FIELDS
        }

        if need_segmenters:
            for family, model in current_models.items():
                model_path = candidate_paths[family]
                probability, _audit = _predict_probability(
                    model=model,
                    image_path=image_path,
                    tile_size=tile_size,
                    stride=stride,
                    expected_size=expected_size,
                )
                if run_current:
                    current_mask = (probability >= comparison_threshold).astype(
                        np.uint8
                    )
                    current_metrics.append(
                        metric_row(
                            method=f'current_segmenter_{family}',
                            candidate_artifact=str(model_path),
                            manifest_row_id=row_id,
                            cohort_id=str(record['cohort_id']),
                            lane_assignment=str(record['lane_assignment']),
                            manual_mask=manual_mask,
                            predicted_mask=current_mask,
                        )
                    )
                if run_auto:
                    for threshold in thresholds:
                        boxes, decisions = derive_proposal_boxes(
                            probability,
                            threshold=threshold,
                            min_component_area=min_area,
                            max_component_area=max_area,
                            padding=padding,
                            merge_iou=merge_iou,
                            max_boxes=max_boxes,
                        )
                        proposal_by_setting[(row_id, family, threshold)] = boxes
                        overflow_count = sum(
                            1
                            for decision in decisions
                            if decision.get('decision') == 'overflow'
                        )
                        for decision in decisions:
                            proposal_rows.append(
                                {
                                    **base_record,
                                    'candidate_family': family,
                                    'candidate_artifact': str(model_path),
                                    'threshold': threshold,
                                    **decision,
                                }
                            )
                        recall_row = proposal_recall_row(
                            manual_mask=manual_mask,
                            proposal_boxes=boxes,
                            manifest_row_id=row_id,
                            cohort_id=str(record['cohort_id']),
                            lane_assignment=str(record['lane_assignment']),
                            candidate_family=family,
                            candidate_artifact=str(model_path),
                            threshold=threshold,
                            min_component_area=min_area,
                        )
                        recall_row['overflow_count'] = int(overflow_count)
                        recall_rows.append(recall_row)

    if proposal_rows:
        pd.DataFrame(proposal_rows).reindex(columns=PROPOSAL_BOX_FIELDS).to_csv(
            baseline_dir / 'proposal_boxes.csv', index=False
        )
        summary['proposal_boxes_csv'] = str(baseline_dir / 'proposal_boxes.csv')
    if recall_rows:
        pd.DataFrame(recall_rows).to_csv(
            baseline_dir / 'proposal_recall.csv', index=False
        )
        summary['proposal_recall_csv'] = str(baseline_dir / 'proposal_recall.csv')

    if current_metrics:
        path = baseline_dir / 'current_segmenter_metrics.csv'
        pd.DataFrame(current_metrics).to_csv(path, index=False)
        summary['current_segmenter_metrics_csv'] = str(path)
        summary['current_segmenter'].update(_baseline_metric_summary(current_metrics))

    device = str(medsam_cfg.get('device', 'cpu'))
    checkpoint_str = str(bundle.get('checkpoint', ''))

    prompt_failures: list[dict[str, Any]] = []
    automatic_metrics: list[dict[str, Any]] = []

    if run_auto:
        if not recall_rows:
            summary['automatic_medsam']['status'] = 'skipped_no_proposals'
        else:
            best_family, best_threshold = _select_best_setting(recall_rows)
            summary['automatic_medsam']['selected_candidate_family'] = best_family
            summary['automatic_medsam']['selected_threshold'] = float(best_threshold)
            auto_mask_root = baseline_dir / 'automatic_medsam_masks' / best_family
            auto_mask_root.mkdir(parents=True, exist_ok=True)
            medsam_auto_items: list[dict[str, Any]] = []
            for record in fixed_rows.fillna('').to_dict(orient='records'):
                row_id = str(record['manifest_row_id'])
                manual_mask = manual_masks[row_id]
                boxes = proposal_by_setting.get(
                    (row_id, best_family, best_threshold), []
                )
                medsam_auto_items.append(
                    {
                        'manifest_row_id': row_id,
                        'image_path': str(record['image_path_resolved']),
                        'height': int(manual_mask.shape[0]),
                        'width': int(manual_mask.shape[1]),
                        'boxes': boxes,
                        'output_path': str(
                            auto_mask_root
                            / f'{row_id}_{best_family}_{best_threshold:g}_medsam_auto.png'
                        ),
                    }
                )
            batch_dir = baseline_dir / 'medsam_batch_automatic'
            batch_dir.mkdir(parents=True, exist_ok=True)
            auto_failures = _run_medsam_batch(
                medsam_python=bundle['medsam_python'],
                medsam_repo=bundle['medsam_repo'],
                checkpoint=bundle['checkpoint'],
                device=device,
                items=medsam_auto_items,
                output_dir=batch_dir,
            )
            for failure in auto_failures:
                failure['baseline_prompt_lane'] = 'automatic_medsam'
            prompt_failures.extend(auto_failures)

            items_by_id = {item['manifest_row_id']: item for item in medsam_auto_items}
            for record in fixed_rows.fillna('').to_dict(orient='records'):
                row_id = str(record['manifest_row_id'])
                manual_mask = manual_masks[row_id]
                auto_path = Path(items_by_id[row_id]['output_path'])
                if auto_path.exists():
                    automatic_metrics.append(
                        _automatic_metric_row(
                            prompt_mode='automatic_current_segmenter_boxes',
                            proposal_threshold=float(best_threshold),
                            candidate_family=str(best_family),
                            mask_source=AUTOMATIC_MASK_SOURCE,
                            method='medsam_automatic',
                            candidate_artifact=checkpoint_str,
                            manifest_row_id=row_id,
                            cohort_id=str(record['cohort_id']),
                            lane_assignment=str(record['lane_assignment']),
                            manual_mask=manual_mask,
                            predicted_mask=load_binary_mask(auto_path),
                        )
                    )

            if automatic_metrics:
                path = baseline_dir / 'automatic_medsam_metrics.csv'
                _write_csv(path, automatic_metrics, fieldnames=AUTOMATIC_METRIC_FIELDS)
                summary['automatic_medsam_metrics_csv'] = str(path)
                summary['automatic_medsam'].update(
                    _baseline_metric_summary(automatic_metrics)
                )

    oracle_metrics: list[dict[str, Any]] = []
    if run_oracle:
        oracle_rows_csv: list[dict[str, Any]] = []
        oracle_mask_dir = baseline_dir / 'oracle_medsam_masks'
        oracle_mask_dir.mkdir(parents=True, exist_ok=True)
        medsam_oracle_items: list[dict[str, Any]] = []
        for record in fixed_rows.fillna('').to_dict(orient='records'):
            row_id = str(record['manifest_row_id'])
            manual_mask = manual_masks[row_id]
            base_record = {
                field: record.get(field, '') for field in _FIXED_SPLIT_ROW_FIELDS
            }
            boxes, skipped = derive_oracle_boxes(
                manual_mask, min_component_area=min_area, padding=padding
            )
            for box in boxes:
                oracle_rows_csv.append({**base_record, **box, 'skip_reason': ''})
            for skip in skipped:
                oracle_rows_csv.append({**base_record, **skip})
            medsam_oracle_items.append(
                {
                    'manifest_row_id': row_id,
                    'image_path': str(record['image_path_resolved']),
                    'height': int(manual_mask.shape[0]),
                    'width': int(manual_mask.shape[1]),
                    'boxes': boxes,
                    'output_path': str(
                        oracle_mask_dir / f'{row_id}_medsam_oracle_union.png'
                    ),
                }
            )
        pd.DataFrame(oracle_rows_csv).to_csv(
            baseline_dir / 'oracle_boxes.csv', index=False
        )
        summary['oracle_boxes_csv'] = str(baseline_dir / 'oracle_boxes.csv')

        oracle_batch_dir = baseline_dir / 'medsam_batch_oracle'
        oracle_batch_dir.mkdir(parents=True, exist_ok=True)
        oracle_failures = _run_medsam_batch(
            medsam_python=bundle['medsam_python'],
            medsam_repo=bundle['medsam_repo'],
            checkpoint=bundle['checkpoint'],
            device=device,
            items=medsam_oracle_items,
            output_dir=oracle_batch_dir,
        )
        for failure in oracle_failures:
            failure['baseline_prompt_lane'] = 'oracle_prompt_medsam'
        prompt_failures.extend(oracle_failures)

        for record in fixed_rows.fillna('').to_dict(orient='records'):
            row_id = str(record['manifest_row_id'])
            manual_mask = manual_masks[row_id]
            oracle_path = oracle_mask_dir / f'{row_id}_medsam_oracle_union.png'
            if oracle_path.exists():
                oracle_metrics.append(
                    metric_row(
                        method='medsam_oracle',
                        candidate_artifact=checkpoint_str,
                        manifest_row_id=row_id,
                        cohort_id=str(record['cohort_id']),
                        lane_assignment=str(record['lane_assignment']),
                        manual_mask=manual_mask,
                        predicted_mask=load_binary_mask(oracle_path),
                    )
                )

        if oracle_metrics:
            path = baseline_dir / 'oracle_medsam_metrics.csv'
            pd.DataFrame(oracle_metrics).to_csv(path, index=False)
            summary['oracle_medsam_metrics_csv'] = str(path)
            summary['oracle_prompt_medsam'].update(
                _baseline_metric_summary(oracle_metrics)
            )

    if prompt_failures:
        path = baseline_dir / 'prompt_failures.csv'
        pd.DataFrame(prompt_failures).to_csv(path, index=False)
        summary['prompt_failures_csv'] = str(path)
        summary['prompt_failure_count'] = int(len(prompt_failures))

    return summary


def _mean_metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    return 0.0 if value in ('', None) else float(value)


def _metric_aggregate(mean_dice: Any, mean_jaccard: Any) -> dict[str, float]:
    return {
        'dice': 0.0 if mean_dice in ('', None) else float(mean_dice),
        'jaccard': 0.0 if mean_jaccard in ('', None) else float(mean_jaccard),
    }


def _trivial_baseline_aggregate(baseline_outputs: dict[str, Any]) -> dict[str, float]:
    grouped_path_text = str(
        baseline_outputs.get('trivial_baseline_metric_by_source', '')
    ).strip()
    if not grouped_path_text:
        return {'dice': 0.0, 'jaccard': 0.0}
    grouped_path = Path(grouped_path_text)
    if not grouped_path.exists() or not grouped_path.is_file():
        return {'dice': 0.0, 'jaccard': 0.0}
    try:
        grouped = pd.read_csv(grouped_path).fillna('')
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return {'dice': 0.0, 'jaccard': 0.0}
    if (
        grouped.empty
        or 'dice' not in grouped.columns
        or 'jaccard' not in grouped.columns
    ):
        return {'dice': 0.0, 'jaccard': 0.0}
    dice = pd.to_numeric(grouped['dice'], errors='coerce').fillna(0.0)
    jaccard = pd.to_numeric(grouped['jaccard'], errors='coerce').fillna(0.0)
    return {'dice': float(dice.max()), 'jaccard': float(jaccard.max())}


def _finetuned_metric_aggregate(
    finetuned_evaluation: dict[str, Any],
) -> dict[str, float]:
    metrics_csv_text = str(finetuned_evaluation.get('metrics_csv', '')).strip()
    if not metrics_csv_text:
        return {'dice': 0.0, 'jaccard': 0.0}
    metrics_csv = Path(metrics_csv_text)
    if not metrics_csv.exists() or not metrics_csv.is_file():
        return {'dice': 0.0, 'jaccard': 0.0}
    try:
        frame = pd.read_csv(metrics_csv).fillna('')
    except pd.errors.EmptyDataError:
        return {'dice': 0.0, 'jaccard': 0.0}
    if frame.empty or 'dice' not in frame.columns or 'jaccard' not in frame.columns:
        return {'dice': 0.0, 'jaccard': 0.0}
    dice = pd.to_numeric(frame['dice'], errors='coerce').fillna(0.0)
    jaccard = pd.to_numeric(frame['jaccard'], errors='coerce').fillna(0.0)
    return {'dice': float(dice.mean()), 'jaccard': float(jaccard.mean())}


def _unavailable_metric_evidence(
    *,
    label: str,
    reason: str,
    artifact_path: str = '',
    reason_code: str = '',
    artifact_sha256: str = '',
    sample_count: int | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        'status': 'unavailable',
        'reason': reason,
        'reason_code': reason_code or f'{label}_metric_evidence_unavailable',
        'artifact_path': artifact_path,
    }
    if artifact_sha256:
        evidence['sha256'] = artifact_sha256
    if sample_count is not None:
        evidence['sample_count'] = sample_count
    return evidence


def _shape_metric_evidence(frame: pd.DataFrame, *, label: str) -> dict[str, Any]:
    """Summarize mask-shape evidence without turning missing columns into zeros."""
    fields = (
        'manual_foreground_fraction',
        'prediction_foreground_fraction',
        'area_ratio',
    )
    missing = sorted(field for field in fields if field not in frame.columns)
    if missing:
        return {
            'status': 'unavailable',
            'reason': 'Metric evidence lacks required mask-shape columns.',
            'reason_code': f'{label}_shape_metric_columns_missing',
            'missing_columns': missing,
        }
    values = frame[list(fields)].apply(pd.to_numeric, errors='coerce')
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        return {
            'status': 'unavailable',
            'reason': 'Metric evidence contains missing or non-finite mask-shape values.',
            'reason_code': f'{label}_shape_metric_values_invalid',
        }
    return {
        'status': 'available',
        'reason': '',
        'reason_code': '',
        'sample_count': int(len(values)),
        'metrics': {
            field: {
                'min': float(values[field].min()),
                'max': float(values[field].max()),
                'mean': float(values[field].mean()),
            }
            for field in fields
        },
    }


def _metric_evidence_provenance(source: dict[str, Any]) -> dict[str, Any] | None:
    run_id = str(source.get('run_id', '')).strip()
    fixed_split = str(source.get('fixed_split', '')).strip()
    row_ids_value = source.get('fixed_manifest_row_ids')
    if (
        not run_id
        or not fixed_split
        or not isinstance(row_ids_value, list)
        or not row_ids_value
    ):
        return None
    row_ids = [str(value).strip() for value in row_ids_value]
    if any(not value for value in row_ids) or len(set(row_ids)) != len(row_ids):
        return None
    return {
        'run_id': run_id,
        'fixed_split': fixed_split,
        'fixed_manifest_row_ids': sorted(row_ids),
    }


def _load_metric_evidence(
    artifact_path: Any,
    *,
    label: str,
    aggregation: str,
    provenance_source: dict[str, Any],
    expected_provenance: dict[str, Any] | None,
    expected_methods: set[str] | None = None,
    expected_method_prefix: str = '',
) -> dict[str, Any]:
    artifact_text = str(artifact_path or '').strip()
    if not artifact_text:
        return _unavailable_metric_evidence(
            label=label, reason='No metric CSV artifact was selected.'
        )
    path = Path(artifact_text)
    if not path.exists() or not path.is_file():
        return _unavailable_metric_evidence(
            label=label,
            reason='Selected metric CSV artifact does not exist.',
            artifact_path=str(path),
        )
    try:
        frame = pd.read_csv(path)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return _unavailable_metric_evidence(
            label=label,
            reason='Selected metric CSV artifact cannot be read.',
            artifact_path=str(path),
        )
    required_columns = {'manifest_row_id', 'method', 'dice', 'jaccard'}
    if frame.empty or not required_columns.issubset(frame.columns):
        return _unavailable_metric_evidence(
            label=label,
            reason=(
                'Selected metric CSV artifact lacks usable method, row identity, '
                'or Dice/Jaccard rows.'
            ),
            artifact_path=str(path),
        )
    metrics = frame[['dice', 'jaccard']].apply(pd.to_numeric, errors='coerce')
    if metrics.isna().any().any() or not np.isfinite(metrics.to_numpy()).all():
        return _unavailable_metric_evidence(
            label=label,
            reason='Selected metric CSV artifact contains missing or non-finite Dice/Jaccard values.',
            artifact_path=str(path),
        )
    artifact_sha256 = _hash_file(path)
    sample_count = int(len(metrics))
    normalized_frame = frame.assign(
        manifest_row_id=frame['manifest_row_id'].map(lambda value: str(value).strip()),
        method=frame['method'].map(lambda value: str(value).strip()),
        dice=metrics['dice'],
        jaccard=metrics['jaccard'],
    )
    if normalized_frame.duplicated(
        subset=['method', 'manifest_row_id'], keep=False
    ).any():
        return _unavailable_metric_evidence(
            label=label,
            reason=(
                'Selected metric CSV artifact contains duplicate method and '
                'manifest-row evidence.'
            ),
            reason_code=f'{label}_metric_evidence_duplicate_rows',
            artifact_path=str(path),
            artifact_sha256=artifact_sha256,
            sample_count=sample_count,
        )
    actual_provenance = _metric_evidence_provenance(provenance_source)
    artifact_row_ids = sorted(set(normalized_frame['manifest_row_id'].tolist()))
    if expected_provenance is None or actual_provenance is None or not artifact_row_ids:
        return _unavailable_metric_evidence(
            label=label,
            reason='Metric evidence run and fixed-split provenance is unavailable.',
            reason_code=f'{label}_metric_evidence_provenance_unavailable',
            artifact_path=str(path),
            artifact_sha256=artifact_sha256,
            sample_count=sample_count,
        )
    if (
        actual_provenance != expected_provenance
        or artifact_row_ids != actual_provenance['fixed_manifest_row_ids']
    ):
        return _unavailable_metric_evidence(
            label=label,
            reason='Metric evidence does not match the evaluation run and fixed split.',
            reason_code=f'{label}_metric_evidence_provenance_mismatch',
            artifact_path=str(path),
            artifact_sha256=artifact_sha256,
            sample_count=sample_count,
        )
    methods = set(normalized_frame['method'].tolist())
    method_matches = True
    if expected_methods is not None:
        method_matches = methods == expected_methods
    elif expected_method_prefix:
        method_matches = bool(methods) and all(
            method.startswith(expected_method_prefix) for method in methods
        )
    if not method_matches:
        return _unavailable_metric_evidence(
            label=label,
            reason='Metric evidence method provenance does not match its comparison lane.',
            reason_code=f'{label}_metric_method_mismatch',
            artifact_path=str(path),
            artifact_sha256=artifact_sha256,
            sample_count=sample_count,
        )
    if aggregation == 'mean':
        aggregate = metrics.mean()
    elif aggregation == 'max_method_mean':
        aggregate = (
            normalized_frame.groupby('method', dropna=False)[['dice', 'jaccard']]
            .mean()
            .max()
        )
    else:
        raise ValueError(f'Unsupported metric aggregation: {aggregation}')
    return {
        'status': 'available',
        'reason': '',
        'reason_code': '',
        'artifact_path': str(path),
        'sha256': artifact_sha256,
        'sample_count': sample_count,
        'provenance': actual_provenance,
        'methods': sorted(methods),
        'metrics': {
            'dice': float(aggregate['dice']),
            'jaccard': float(aggregate['jaccard']),
        },
        'shape_metrics': _shape_metric_evidence(normalized_frame, label=label),
    }


def _build_finetuned_comparison_summary(
    *,
    finetuned_evaluation: dict[str, Any],
    external_baselines_summary: dict[str, Any],
    baseline_outputs: dict[str, Any],
    adoption_gates: dict[str, Any],
) -> dict[str, Any]:
    expected_provenance = _metric_evidence_provenance(finetuned_evaluation)
    metric_evidence = {
        'fine_tuned': _load_metric_evidence(
        finetuned_evaluation.get('metrics_csv', ''),
            label='fine_tuned',
            aggregation='mean',
            provenance_source=finetuned_evaluation,
            expected_provenance=expected_provenance,
            expected_methods={'medsam_finetuned_automatic'},
        ),
        'oracle': _load_metric_evidence(
            external_baselines_summary.get('oracle_medsam_metrics_csv', ''),
            label='oracle',
            aggregation='mean',
            provenance_source=external_baselines_summary,
            expected_provenance=expected_provenance,
            expected_methods={'medsam_oracle'},
        ),
        'automatic_medsam': _load_metric_evidence(
            external_baselines_summary.get('automatic_medsam_metrics_csv', ''),
            label='automatic_medsam',
            aggregation='mean',
            provenance_source=external_baselines_summary,
            expected_provenance=expected_provenance,
            expected_methods={'medsam_automatic'},
        ),
        'current_segmenter': _load_metric_evidence(
            external_baselines_summary.get('current_segmenter_metrics_csv', ''),
            label='current_segmenter',
            aggregation='mean',
            provenance_source=external_baselines_summary,
            expected_provenance=expected_provenance,
            expected_method_prefix='current_segmenter_',
        ),
        'trivial_baseline': _load_metric_evidence(
            baseline_outputs.get('trivial_baseline_metrics', ''),
            label='trivial_baseline',
            aggregation='max_method_mean',
            provenance_source=baseline_outputs,
            expected_provenance=expected_provenance,
            expected_methods={'trivial_all_background', 'trivial_all_foreground'},
        ),
    }
    evidence_failures = [
        evidence
        for evidence in metric_evidence.values()
        if evidence['status'] != 'available'
    ]
    oracle_evidence = metric_evidence['oracle']
    automatic_evidence = metric_evidence['automatic_medsam']
    if (
        oracle_evidence.get('artifact_path')
        and automatic_evidence.get('artifact_path')
        and (
            Path(str(oracle_evidence['artifact_path'])).resolve()
            == Path(str(automatic_evidence['artifact_path'])).resolve()
            or (
                oracle_evidence.get('sha256')
                and oracle_evidence.get('sha256') == automatic_evidence.get('sha256')
            )
        )
    ):
        evidence_failures.append(
            {
                'status': 'unavailable',
                'reason': (
                    'Automatic and oracle metric evidence must use distinct artifacts.'
                ),
                'reason_code': 'automatic_oracle_metric_artifact_not_distinct',
            }
        )
    finetuned_status = str(finetuned_evaluation.get('status', ''))
    if finetuned_status != 'completed':
        return {
            'status': 'blocked',
            'reason': 'Fine-tuned inference did not complete.',
            'reason_codes': ['finetuned_inference_not_completed'],
            'failed_gates': [
                {
                    'gate': 'finetuned_inference',
                    'reason_code': 'finetuned_inference_not_completed',
                    'reason': 'Fine-tuned inference did not complete.',
                }
            ],
            'metric_evidence': metric_evidence,
            'adoption_tier': 'blocked',
            'primary_generated_mask_transition_status': 'blocked',
            'recommended_generated_mask_source': '',
            'failure_mode': finetuned_status or 'finetuned_inference_not_run',
        }
    if evidence_failures:
        reason_codes = list(
            dict.fromkeys(
                str(evidence['reason_code']) for evidence in evidence_failures
            )
        )
        comparison: dict[str, Any] = {
            'status': 'blocked',
            'reason': 'Required metric evidence is unavailable.',
            'reason_codes': reason_codes,
            'failed_gates': [
                {
                    'gate': 'metric_evidence',
                    'reason_code': str(evidence['reason_code']),
                    'reason': str(evidence['reason']),
                }
                for evidence in evidence_failures
            ],
            'metric_evidence': metric_evidence,
            'adoption_tier': 'blocked',
            'primary_generated_mask_transition_status': 'blocked',
            'recommended_generated_mask_source': '',
            'failure_mode': 'metric_evidence_unavailable',
        }
        if (
            metric_evidence['fine_tuned']['status'] == 'available'
            and metric_evidence['oracle']['status'] == 'available'
        ):
            comparison['oracle_dice_gap'] = max(
                0.0,
                metric_evidence['oracle']['metrics']['dice']
                - metric_evidence['fine_tuned']['metrics']['dice'],
            )
        return comparison
    comparison = _classify_generated_mask_adoption(
        fine_tuned_metrics=metric_evidence['fine_tuned']['metrics'],
        oracle_metrics=metric_evidence['oracle']['metrics'],
        current_auto_metrics=metric_evidence['automatic_medsam']['metrics'],
        current_segmenter_metrics=metric_evidence['current_segmenter']['metrics'],
        trivial_baseline_metrics=metric_evidence['trivial_baseline']['metrics'],
        prompt_failure_count=int(
            finetuned_evaluation.get('prompt_failure_count', 0) or 0
        ),
        gates=adoption_gates,
        fine_tuned_shape_metrics=metric_evidence['fine_tuned'].get(
            'shape_metrics'
        ),
        overlay_review_status='passed'
        if str(finetuned_evaluation.get('overlay_review_status', '')) == 'passed'
        else 'not_run',
    )
    comparison.update(
        {
            'status': 'completed',
            'reason': '',
            'metric_evidence': metric_evidence,
        }
    )
    return comparison


def _classify_generated_mask_adoption(
    *,
    fine_tuned_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
    current_auto_metrics: dict[str, Any],
    current_segmenter_metrics: dict[str, Any],
    trivial_baseline_metrics: dict[str, Any],
    prompt_failure_count: int,
    gates: dict[str, Any],
    overlay_review_status: str,
    fine_tuned_shape_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fine_dice = _mean_metric(fine_tuned_metrics, 'dice')
    fine_jaccard = _mean_metric(fine_tuned_metrics, 'jaccard')
    oracle_dice = _mean_metric(oracle_metrics, 'dice')
    oracle_gap = max(0.0, oracle_dice - fine_dice) if oracle_dice else 0.0
    improves_auto = fine_dice > _mean_metric(current_auto_metrics, 'dice')
    improves_segmenter = fine_dice > _mean_metric(current_segmenter_metrics, 'dice')
    beats_trivial = fine_dice > _mean_metric(trivial_baseline_metrics, 'dice')
    reliability_passed = (
        int(prompt_failure_count) <= int(gates.get('max_prompt_failures', 0))
        and str(overlay_review_status) == 'passed'
        and beats_trivial
    )
    shape_gate_reasons: list[str] = []
    shape_gate_observed: dict[str, Any] = {}
    shape_gates_configured = any(
        key in gates
        for key in (
            'min_prediction_foreground_fraction',
            'max_prediction_foreground_fraction',
            'min_prediction_to_truth_area_ratio',
            'max_prediction_to_truth_area_ratio',
        )
    )
    if shape_gates_configured:
        if not isinstance(fine_tuned_shape_metrics, dict) or str(
            fine_tuned_shape_metrics.get('status', '')
        ) != 'available':
            shape_gate_reasons.append('fine_tuned_shape_metric_evidence_unavailable')
        else:
            shape_values = fine_tuned_shape_metrics.get('metrics', {})
            shape_gate_observed = dict(shape_values)
            checks = (
                (
                    'prediction_foreground_fraction',
                    'min_prediction_foreground_fraction',
                    0.0,
                    'prediction_foreground_fraction_below_gate',
                    'min',
                    '>=',
                ),
                (
                    'prediction_foreground_fraction',
                    'max_prediction_foreground_fraction',
                    1.0,
                    'prediction_foreground_fraction_above_gate',
                    'max',
                    '<=',
                ),
                (
                    'area_ratio',
                    'min_prediction_to_truth_area_ratio',
                    0.5,
                    'prediction_to_truth_area_ratio_below_gate',
                    'min',
                    '>=',
                ),
                (
                    'area_ratio',
                    'max_prediction_to_truth_area_ratio',
                    1.5,
                    'prediction_to_truth_area_ratio_above_gate',
                    'max',
                    '<=',
                ),
            )
            for metric_name, gate_name, default, reason_code, observed_key, comparator in checks:
                metric_record = shape_values.get(metric_name, {})
                if not isinstance(metric_record, dict) or observed_key not in metric_record:
                    shape_gate_reasons.append(
                        f'fine_tuned_{metric_name}_evidence_unavailable'
                    )
                    continue
                observed = float(metric_record[observed_key])
                threshold = float(gates.get(gate_name, default))
                passed = observed >= threshold if comparator == '>=' else observed <= threshold
                if not passed:
                    shape_gate_reasons.append(reason_code)
    shape_gate_passed = not shape_gate_reasons
    reliability_passed = reliability_passed and shape_gate_passed
    oracle_level = (
        fine_dice >= float(gates.get('min_dice', 0.90))
        and fine_jaccard >= float(gates.get('min_jaccard', 0.82))
        and oracle_gap <= float(gates.get('max_oracle_dice_gap', 0.05))
    )
    if oracle_level and improves_auto and improves_segmenter and reliability_passed:
        adoption_tier = 'oracle_level_preferred'
        recommended_source = MASK_SOURCE
        failure_mode = 'none_detected'
    elif improves_auto and improves_segmenter and reliability_passed:
        adoption_tier = 'improved_candidate_not_oracle'
        recommended_source = ''
        failure_mode = 'oracle_gap'
    else:
        adoption_tier = 'blocked'
        recommended_source = ''
        failure_mode = 'downstream_integration'
        if not oracle_level:
            failure_mode = 'oracle_gap'
        if int(prompt_failure_count) > int(gates.get('max_prompt_failures', 0)):
            failure_mode = 'prompt_geometry'
        if not beats_trivial:
            failure_mode = 'training_quality'
        if shape_gate_reasons:
            failure_mode = 'prediction_shape'
    reason_codes: list[str] = []
    failed_gates: list[dict[str, Any]] = []
    if not oracle_level:
        reason_codes.append('oracle_level_gates_failed')
        failed_gates.append(
            {
                'gate': 'oracle_level_metrics',
                'reason_code': 'oracle_level_gates_failed',
                'reason': 'Fine-tuned automatic metrics did not clear oracle-level thresholds.',
            }
        )
    for reason_code in shape_gate_reasons:
        reason_codes.append(reason_code)
        failed_gates.append(
            {
                'gate': 'prediction_shape',
                'reason_code': reason_code,
                'reason': 'Fine-tuned automatic mask-shape evidence did not clear configured gates.',
            }
        )
    return {
        'adoption_tier': adoption_tier,
        'primary_generated_mask_transition_status': adoption_tier,
        'recommended_generated_mask_source': recommended_source,
        'oracle_dice_gap': oracle_gap,
        'oracle_level_gates_passed': bool(oracle_level),
        'improves_current_auto': bool(improves_auto),
        'improves_current_segmenter': bool(improves_segmenter),
        'beats_trivial_baseline': bool(beats_trivial),
        'shape_gate_passed': bool(shape_gate_passed),
        'shape_gate_observed': shape_gate_observed,
        'failure_mode': failure_mode,
        'reason_codes': reason_codes,
        'failed_gates': failed_gates,
    }


def _promotion_evidence_summary(
    config: dict[str, Any], runtime_root: Path
) -> dict[str, Any]:
    """Resolve explicit human/downstream evidence used by scientific promotion."""
    evidence_cfg = _mapping(config, 'promotion_evidence')
    records: dict[str, Any] = {}
    for key in (
        'overlay_review',
        'downstream_grading_stability',
        'nondegenerate_prediction_review',
    ):
        raw_record = evidence_cfg.get(key, {})
        record = raw_record if isinstance(raw_record, dict) else {}
        status = str(record.get('status', 'unavailable')).strip() or 'unavailable'
        raw_path = str(record.get('artifact_path', '')).strip()
        resolved_path = _runtime_path(runtime_root, raw_path) if raw_path else None
        reason_code = ''
        reason = ''
        if status != 'passed':
            reason_code = f'{key}_not_passed'
            reason = f'{key} status is {status!r}, not passed.'
        elif resolved_path is None or not resolved_path.is_file():
            reason_code = f'{key}_artifact_unavailable'
            reason = f'{key} requires a readable artifact path.'
        artifact_hash = _file_hash(resolved_path) if resolved_path is not None else ''
        records[key] = {
            'status': status,
            'artifact_path': str(resolved_path) if resolved_path is not None else '',
            'sha256': artifact_hash,
            'reason': reason,
            'reason_code': reason_code,
        }
    return records


def _release_manifest_fields() -> list[str]:
    return [
        'schema_version',
        'path_encoding',
        'generated_mask_id',
        'mask_release_id',
        'mask_source',
        'adoption_tier',
        'cohort_id',
        'lane_assignment',
        'source_sample_id',
        'source_image_path',
        'reference_mask_path',
        'generated_mask_path',
        'checkpoint_id',
        'checkpoint_path',
        'proposal_source',
        'proposal_threshold',
        'run_id',
        'generation_status',
        'provenance_path',
    ]


def _package_generated_mask_release(
    *,
    runtime_root: Path,
    release_dir: Path,
    mask_release_id: str,
    run_id: str,
    checkpoint_id: str,
    checkpoint_path: Path,
    proposal_source: str,
    proposal_threshold: float,
    adoption_tier: str,
    generated_masks: list[dict[str, Any]],
) -> dict[str, Path]:
    runtime_root = Path(runtime_root).expanduser().resolve()
    ensure_not_under_runtime_raw_data(release_dir, runtime_root)
    checkpoint_path_serialized = _serialize_required_runtime_path(
        checkpoint_path, runtime_root=runtime_root, artifact_label='checkpoint_path'
    )
    for item in generated_masks:
        _serialize_required_runtime_path(
            item.get('source_image_path', ''),
            runtime_root=runtime_root,
            artifact_label='source_image_path',
        )
    masks_dir = release_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = release_dir / 'provenance.json'
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(generated_masks, start=1):
        source_mask = Path(str(item['generated_mask_path']))
        target_mask = masks_dir / source_mask.name
        if not source_mask.is_file():
            raise ArtifactProvenanceError(f'Missing generated artifact: {source_mask}')
        shutil.copy2(source_mask, target_mask)
        status = str(item.get('generation_status') or 'generated')
        rows.append(
            {
                'schema_version': GENERATED_MASK_RELEASE_SCHEMA_VERSION,
                'path_encoding': RUNTIME_RELATIVE_POSIX_ENCODING,
                'generated_mask_id': f'{mask_release_id}:{index:05d}',
                'mask_release_id': mask_release_id,
                'mask_source': MASK_SOURCE,
                'adoption_tier': adoption_tier,
                'cohort_id': str(item.get('cohort_id', '')),
                'lane_assignment': str(item.get('lane_assignment', '')),
                'source_sample_id': str(item.get('source_sample_id', '')),
                'source_image_path': _serialize_optional_runtime_path(
                    item.get('source_image_path', ''), runtime_root=runtime_root
                ),
                'reference_mask_path': _serialize_optional_runtime_path(
                    item.get('reference_mask_path', ''), runtime_root=runtime_root
                ),
                'generated_mask_path': serialize_runtime_relative_path(
                    target_mask, runtime_root=runtime_root
                ),
                'checkpoint_id': checkpoint_id,
                'checkpoint_path': checkpoint_path_serialized,
                'proposal_source': proposal_source,
                'proposal_threshold': float(proposal_threshold),
                'run_id': run_id,
                'generation_status': status,
                'provenance_path': serialize_runtime_relative_path(
                    provenance_path, runtime_root=runtime_root
                ),
            }
        )
    manifest_path = release_dir / 'manifest.csv'
    pd.DataFrame(rows, columns=_release_manifest_fields()).to_csv(
        manifest_path, index=False
    )
    provenance = {
        'schema_version': GENERATED_MASK_RELEASE_SCHEMA_VERSION,
        'path_encoding': RUNTIME_RELATIVE_POSIX_ENCODING,
        'mask_release_id': mask_release_id,
        'mask_source': MASK_SOURCE,
        'adoption_tier': adoption_tier,
        'run_id': run_id,
        'checkpoint_id': checkpoint_id,
        'checkpoint_path': checkpoint_path_serialized,
        'proposal_source': proposal_source,
        'proposal_threshold': float(proposal_threshold),
        'manifest_path': serialize_runtime_relative_path(
            manifest_path, runtime_root=runtime_root
        ),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    _validate_release_manifest_paths(
        pd.DataFrame(rows, columns=_release_manifest_fields()),
        runtime_root=runtime_root,
    )
    (release_dir / 'INDEX.md').write_text(
        '\n'.join(
            [
                f'# MedSAM Fine-Tuned Glomeruli Release: {mask_release_id}',
                '',
                f'- Mask source: `{MASK_SOURCE}`',
                f'- Adoption tier: `{adoption_tier}`',
                f'- Manifest: `{provenance["manifest_path"]}`',
                f'- Provenance: `{serialize_runtime_relative_path(provenance_path, runtime_root=runtime_root)}`',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    return {
        'release_dir': release_dir,
        'manifest': manifest_path,
        'provenance': provenance_path,
        'index': release_dir / 'INDEX.md',
    }


def _serialize_optional_runtime_path(value: Any, *, runtime_root: Path) -> str:
    return (
        serialize_runtime_relative_path(str(value), runtime_root=runtime_root)
        if str(value or '').strip()
        else ''
    )


def _serialize_required_runtime_path(
    value: Any, *, runtime_root: Path, artifact_label: str
) -> str:
    encoded = serialize_runtime_relative_path(str(value), runtime_root=runtime_root)
    resolve_runtime_relative_path(
        encoded,
        runtime_root=runtime_root,
        require_exists=True,
        artifact_label=artifact_label,
    )
    return encoded


def _update_generated_mask_registry(
    registry_path: Path, release_manifest_path: Path, *, runtime_root: Path
) -> None:
    runtime_root = Path(runtime_root).expanduser().resolve()
    release_frame = pd.read_csv(release_manifest_path).fillna('')
    _validate_release_manifest_paths(release_frame, runtime_root=runtime_root)
    release_frame['release_manifest_path'] = serialize_runtime_relative_path(
        release_manifest_path, runtime_root=runtime_root
    )
    columns = [
        'schema_version',
        'path_encoding',
        'generated_mask_id',
        'mask_release_id',
        'mask_source',
        'adoption_tier',
        'cohort_id',
        'lane_assignment',
        'source_sample_id',
        'source_image_path',
        'reference_mask_path',
        'generated_mask_path',
        'release_manifest_path',
        'checkpoint_id',
        'checkpoint_path',
        'proposal_source',
        'proposal_threshold',
        'run_id',
        'generation_status',
        'provenance_path',
    ]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        existing = pd.read_csv(registry_path).fillna('')
        _validate_release_manifest_paths(existing, runtime_root=runtime_root)
        combined = pd.concat([existing, release_frame[columns]], ignore_index=True)
        combined = combined.drop_duplicates(subset=['generated_mask_id'], keep='last')
    else:
        combined = release_frame[columns]
    combined.to_csv(registry_path, index=False)


def _validate_release_manifest_paths(
    frame: pd.DataFrame, *, runtime_root: Path
) -> None:
    required_metadata = {'schema_version', 'path_encoding', 'generated_mask_path'}
    missing = required_metadata.difference(frame.columns)
    if missing:
        raise ArtifactProvenanceError(
            f'Generated-mask manifest is stale or incomplete; missing columns: {sorted(missing)}'
        )
    for row in frame.to_dict(orient='records'):
        if row.get('schema_version') != GENERATED_MASK_RELEASE_SCHEMA_VERSION:
            raise ArtifactProvenanceError(
                'Generated-mask manifest has unsupported schema_version'
            )
        if row.get('path_encoding') != RUNTIME_RELATIVE_POSIX_ENCODING:
            raise ArtifactProvenanceError(
                'Generated-mask manifest has unsupported path_encoding'
            )
        validate_generated_mask_artifact_paths(row, runtime_root=runtime_root)


def _build_checkpoint_provenance(
    *,
    training_command: list[str],
    environment: dict[str, str],
    medsam_repo: Path,
    base_checkpoint: Path,
    split_manifest_paths: dict[str, Path],
    split_manifest_hashes: dict[str, str],
    adaptation_mode: str,
    frozen_components: list[str],
    trainable_components: list[str],
    hyperparameters: dict[str, Any],
    checkpoint_files: list[Path],
    training_status: str,
    local_feasibility_status: str,
    reuse_existing_checkpoints: bool = False,
) -> dict[str, Any]:
    checkpoint_file_strings = [str(path) for path in checkpoint_files if path.exists()]
    supported = bool(
        training_status == 'completed' and checkpoint_file_strings
    ) or bool(reuse_existing_checkpoints and checkpoint_file_strings)
    return {
        'training_command': training_command,
        'environment': environment,
        'medsam_repo': str(medsam_repo),
        'base_checkpoint': str(base_checkpoint),
        'base_checkpoint_hash': _file_hash(base_checkpoint),
        'split_manifest_paths': {
            key: str(value) for key, value in split_manifest_paths.items()
        },
        'split_manifest_hashes': split_manifest_hashes,
        'adaptation_mode': adaptation_mode,
        'frozen_components': frozen_components,
        'trainable_components': trainable_components,
        'hyperparameters': hyperparameters,
        'checkpoint_files': checkpoint_file_strings,
        'training_status': training_status,
        'local_feasibility_status': local_feasibility_status,
        'reuse_existing_checkpoints': bool(reuse_existing_checkpoints),
        'supported_checkpoint': supported,
    }


def _run_finetuned_fixed_split_evaluation(
    *,
    fixed_rows: pd.DataFrame,
    evaluation_dir: Path,
    config: dict[str, Any],
    checkpoint_provenance: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    finetuned_dir = evaluation_dir / 'finetuned_evaluation'
    masks_dir = finetuned_dir / 'masks'
    overlays_dir = finetuned_dir / 'overlays'
    batch_dir = finetuned_dir / 'medsam_batch'
    metrics_csv = finetuned_dir / 'metrics.csv'
    prompt_failures_csv = finetuned_dir / 'prompt_failures.csv'
    prompt_failures_json = batch_dir / 'medsam_batch_failures.json'
    proposal_boxes_csv = finetuned_dir / 'proposal_boxes.csv'
    proposal_recall_csv = finetuned_dir / 'proposal_recall.csv'
    summary: dict[str, Any] = {
        'status': 'skipped_missing_supported_checkpoint',
        'metric_rows': 0,
        'metrics_csv': str(metrics_csv),
        'masks_dir': str(masks_dir),
        'overlays_dir': str(overlays_dir),
        'prompt_failures_csv': str(prompt_failures_csv),
        'prompt_failures_json': str(prompt_failures_json),
        'proposal_boxes_csv': str(proposal_boxes_csv),
        'proposal_recall_csv': str(proposal_recall_csv),
        'prompt_failure_count': 0,
        'overlay_review_status': 'not_run',
    }
    if not bool(checkpoint_provenance.get('supported_checkpoint', False)):
        return summary

    def _resolve_inference_checkpoint(path: Path) -> Path:
        """Normalize adapter checkpoints to SAM-compatible state dicts for inference."""
        try:
            import torch

            payload = torch.load(path, map_location='cpu')
        except Exception:
            return path
        if not isinstance(payload, dict):
            return path
        model_state = payload.get('model')
        if not isinstance(model_state, dict):
            return path
        converted = finetuned_dir / f'{path.stem}_sam_state_dict.pth'
        torch.save(model_state, converted)
        return converted

    for directory in (finetuned_dir, masks_dir, overlays_dir, batch_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics_csv, [], fieldnames=AUTOMATIC_METRIC_FIELDS)
    _write_csv(
        prompt_failures_csv,
        [],
        fieldnames=['manifest_row_id', 'image_path', 'failure_reason'],
    )
    _write_csv(proposal_boxes_csv, [], fieldnames=PROPOSAL_BOX_FIELDS)
    _write_csv(proposal_recall_csv, [], fieldnames=[])
    if dry_run:
        summary['status'] = 'skipped_dry_run'
        return summary
    if fixed_rows.empty:
        summary['status'] = 'skipped_empty_fixed_split'
        return summary

    medsam_bundle = _preflight_medsam_inference_paths(config)
    checkpoint_candidates = [
        Path(str(path))
        for path in checkpoint_provenance.get('checkpoint_files', [])
        if str(path).strip()
    ]
    checkpoint_candidates = [path for path in checkpoint_candidates if path.exists()]
    if not checkpoint_candidates:
        summary['status'] = 'failed_missing_supported_checkpoint_artifact'
        return summary
    best_val_paths = [
        path
        for path in checkpoint_candidates
        if path.name == 'medsam_glomeruli_best_val_dice.pth'
    ]
    if best_val_paths:
        chosen_checkpoint = best_val_paths[0]
    else:
        checkpoint_candidates.sort(key=lambda path: (path.stat().st_mtime, str(path)))
        chosen_checkpoint = checkpoint_candidates[-1]
    finetuned_checkpoint = _resolve_inference_checkpoint(chosen_checkpoint)

    proposal_cfg = _mapping(config, 'proposal')
    tiling_cfg = _mapping(config, 'tiling')
    current_segmenter_cfg = _mapping(config, 'current_segmenter')
    tile_size = int(tiling_cfg.get('tile_size', 512))
    stride = int(tiling_cfg.get('stride', 512))
    expected_size = int(tiling_cfg.get('expected_size', 256))
    thresholds = [
        float(value) for value in proposal_cfg.get('thresholds', [0.2, 0.35, 0.5])
    ]
    min_area = int(proposal_cfg.get('min_component_area', 2000))
    max_area = int(proposal_cfg.get('max_component_area', 750000))
    padding = int(proposal_cfg.get('padding', 16))
    merge_iou = float(proposal_cfg.get('merge_iou', 0.25))
    max_boxes = int(proposal_cfg.get('max_boxes_per_image', 20))

    from eq.data_management.model_loading import load_model_safely

    candidate_paths = _candidate_paths(current_segmenter_cfg)
    current_models: dict[str, Any] = {}
    for _, model_path in candidate_paths.items():
        if not model_path.exists():
            raise FileNotFoundError(f'Missing current segmenter artifact: {model_path}')
    for family, model_path in candidate_paths.items():
        learner = load_model_safely(str(model_path), model_type='glomeruli')
        learner.model.eval()
        current_models[family] = learner.model

    recall_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    proposal_by_setting: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    manual_masks: dict[str, np.ndarray] = {}
    fixed_records = fixed_rows.fillna('').to_dict(orient='records')
    for record in fixed_records:
        row_id = str(record['manifest_row_id'])
        image_path = Path(str(record['image_path_resolved']))
        manual_mask = load_binary_mask(Path(str(record['mask_path_resolved'])))
        manual_masks[row_id] = manual_mask
        base_record = {
            field: record.get(field, '') for field in _FIXED_SPLIT_ROW_FIELDS
        }
        for family, model in current_models.items():
            model_path = candidate_paths[family]
            probability, _audit = _predict_probability(
                model=model,
                image_path=image_path,
                tile_size=tile_size,
                stride=stride,
                expected_size=expected_size,
            )
            for threshold in thresholds:
                boxes, decisions = derive_proposal_boxes(
                    probability,
                    threshold=threshold,
                    min_component_area=min_area,
                    max_component_area=max_area,
                    padding=padding,
                    merge_iou=merge_iou,
                    max_boxes=max_boxes,
                )
                proposal_by_setting[(row_id, family, threshold)] = boxes
                overflow_count = sum(
                    1
                    for decision in decisions
                    if decision.get('decision') == 'overflow'
                )
                for decision in decisions:
                    proposal_rows.append(
                        {
                            **base_record,
                            'candidate_family': family,
                            'candidate_artifact': str(model_path),
                            'threshold': threshold,
                            **decision,
                        }
                    )
                recall_row = proposal_recall_row(
                    manual_mask=manual_mask,
                    proposal_boxes=boxes,
                    manifest_row_id=row_id,
                    cohort_id=str(record['cohort_id']),
                    lane_assignment=str(record['lane_assignment']),
                    candidate_family=family,
                    candidate_artifact=str(model_path),
                    threshold=threshold,
                    min_component_area=min_area,
                )
                recall_row['overflow_count'] = int(overflow_count)
                recall_rows.append(recall_row)

    if proposal_rows:
        pd.DataFrame(proposal_rows).reindex(columns=PROPOSAL_BOX_FIELDS).to_csv(
            proposal_boxes_csv, index=False
        )
    if recall_rows:
        pd.DataFrame(recall_rows).to_csv(proposal_recall_csv, index=False)
    if not recall_rows:
        summary['status'] = 'skipped_no_proposals'
        return summary

    best_family, best_threshold = _select_best_setting(recall_rows)
    medsam_items: list[dict[str, Any]] = []
    for record in fixed_records:
        row_id = str(record['manifest_row_id'])
        manual_mask = manual_masks[row_id]
        boxes = proposal_by_setting.get((row_id, best_family, best_threshold), [])
        medsam_items.append(
            {
                'manifest_row_id': row_id,
                'image_path': str(record['image_path_resolved']),
                'height': int(manual_mask.shape[0]),
                'width': int(manual_mask.shape[1]),
                'boxes': boxes,
                'output_path': str(
                    masks_dir
                    / f'{row_id}_{best_family}_{best_threshold:g}_medsam_finetuned_auto.png'
                ),
            }
        )
    prompt_failures = _run_medsam_batch(
        medsam_python=medsam_bundle['medsam_python'],
        medsam_repo=medsam_bundle['medsam_repo'],
        checkpoint=finetuned_checkpoint,
        device=str(_mapping(config, 'medsam').get('device', 'cpu')),
        items=medsam_items,
        output_dir=batch_dir,
    )
    _write_csv(
        prompt_failures_csv,
        prompt_failures,
        fieldnames=['manifest_row_id', 'image_path', 'failure_reason'],
    )

    rows: list[dict[str, Any]] = []
    item_by_id = {item['manifest_row_id']: item for item in medsam_items}
    for record in fixed_records:
        row_id = str(record['manifest_row_id'])
        manual_mask = manual_masks[row_id]
        predicted_path = Path(item_by_id[row_id]['output_path'])
        if not predicted_path.exists():
            continue
        predicted_mask = load_binary_mask(predicted_path)
        rows.append(
            _automatic_metric_row(
                prompt_mode='automatic_current_segmenter_boxes',
                proposal_threshold=float(best_threshold),
                candidate_family=str(best_family),
                mask_source=MASK_SOURCE,
                method='medsam_finetuned_automatic',
                candidate_artifact=str(finetuned_checkpoint),
                manifest_row_id=row_id,
                cohort_id=str(record['cohort_id']),
                lane_assignment=str(record['lane_assignment']),
                manual_mask=manual_mask,
                predicted_mask=predicted_mask,
            )
        )
        image = Image.open(Path(str(record['image_path_resolved']))).convert('RGB')
        overlay = image.copy().convert('RGBA')
        for mask, color in (
            (manual_mask, (0, 255, 0, 90)),
            (predicted_mask, (255, 0, 0, 90)),
        ):
            alpha = (np.asarray(mask) > 0).astype(np.uint8) * color[3]
            color_arr = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
            color_arr[..., 0] = color[0]
            color_arr[..., 1] = color[1]
            color_arr[..., 2] = color[2]
            color_arr[..., 3] = alpha
            overlay = Image.alpha_composite(
                overlay, Image.fromarray(color_arr, mode='RGBA')
            )
        draw = ImageDraw.Draw(overlay)
        for box in proposal_by_setting.get((row_id, best_family, best_threshold), []):
            draw.rectangle(
                [box['bbox_x0'], box['bbox_y0'], box['bbox_x1'], box['bbox_y1']],
                outline=(0, 0, 255, 255),
                width=3,
            )
        draw.text(
            (8, 8),
            'manual=green finetuned=red proposal_box=blue',
            fill=(255, 255, 255, 255),
        )
        overlay.convert('RGB').save(overlays_dir / f'{row_id}_overlay.png')

    _write_csv(metrics_csv, rows, fieldnames=AUTOMATIC_METRIC_FIELDS)
    expected_rows = int(len(fixed_records))
    produced_rows = int(len(rows))
    prompt_failure_count = int(len(prompt_failures))
    status = 'completed'
    if produced_rows < expected_rows:
        status = 'failed_partial_predictions'
    if prompt_failure_count > 0:
        status = 'failed_prompt_failures'
    if produced_rows == 0:
        status = 'failed_no_predictions'

    summary.update(
        {
            'status': status,
            'selected_candidate_family': str(best_family),
            'selected_threshold': float(best_threshold),
            'checkpoint_path': str(finetuned_checkpoint),
            'expected_metric_rows': expected_rows,
            'metric_rows': produced_rows,
            'metrics_csv': str(metrics_csv),
            'masks_dir': str(masks_dir),
            'overlays_dir': str(overlays_dir),
            'prompt_failures_csv': str(prompt_failures_csv),
            'prompt_failures_json': str(prompt_failures_json),
            'proposal_boxes_csv': str(proposal_boxes_csv),
            'proposal_recall_csv': str(proposal_recall_csv),
            'prompt_failure_count': prompt_failure_count,
            'overlay_review_status': 'not_reviewed'
            if status == 'completed'
            else 'not_run',
        }
    )
    return summary


def _build_adaptation_command(
    *,
    medsam_python: Path,
    entrypoint: Path,
    train_npy_root: Path,
    work_dir: Path,
    base_checkpoint: Path,
    task_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> list[str]:
    return [
        str(medsam_python),
        str(entrypoint),
        '-i',
        str(train_npy_root),
        '-task_name',
        str(task_name),
        '-model_type',
        'vit_b',
        '-checkpoint',
        str(base_checkpoint),
        '--load_pretrain',
        'True',
        '-pretrain_model_path',
        str(base_checkpoint),
        '-work_dir',
        str(work_dir),
        '-num_epochs',
        str(int(epochs)),
        '-batch_size',
        str(int(batch_size)),
        '-lr',
        str(float(learning_rate)),
        '--device',
        str(device),
    ]


def _training_command_for_config(
    *,
    config: dict[str, Any],
    preflight: dict[str, str],
    train_npy_root: Path,
    work_dir: Path,
    training_cfg: dict[str, Any],
    val_npy_root: Path | None = None,
) -> list[str]:
    medsam = _mapping(config, 'medsam')
    device = str(medsam.get('device', 'cpu'))
    backend = preflight.get('training_backend', 'upstream_medsam')
    python_path = Path(preflight['medsam_python'])
    repo = Path(preflight['medsam_repo'])
    ckpt = Path(preflight['base_checkpoint'])
    epochs = int(training_cfg.get('epochs', 10))
    batch_size = int(training_cfg.get('batch_size', 2))
    lr = float(training_cfg.get('learning_rate', 0.0001))
    if backend == 'eq_native_adapter':
        cmd = [
            str(python_path),
            '-m',
            'eq.evaluation.medsam_glomeruli_adapter',
            '--medsam-repo',
            str(repo),
            '--train-npy-root',
            str(train_npy_root),
            '--checkpoint',
            str(ckpt),
            '--work-dir',
            str(work_dir),
            '--model-type',
            str(training_cfg.get('model_type', 'vit_b')),
            '--epochs',
            str(epochs),
            '--batch-size',
            str(batch_size),
            '--learning-rate',
            str(lr),
            '--weight-decay',
            str(float(training_cfg.get('weight_decay', 0.01))),
            '--bbox-shift',
            str(int(training_cfg.get('bbox_shift', 20))),
            '--device',
            device,
            '--max-examples',
            str(int(training_cfg.get('max_examples', 0))),
            '--seed',
            str(int(training_cfg.get('seed', 2023))),
            '--lr-scheduler',
            str(training_cfg.get('lr_scheduler', 'none')),
            '--min-lr',
            str(float(training_cfg.get('min_lr', 0.0))),
        ]
        if val_npy_root is not None:
            cmd.extend(['--val-npy-root', str(val_npy_root)])
            val_cap = int(training_cfg.get('val_max_examples', 0))
            if val_cap > 0:
                cmd.extend(['--val-max-examples', str(val_cap)])
        return cmd
    entrypoint = Path(preflight['entrypoint'])
    return _build_adaptation_command(
        medsam_python=python_path,
        entrypoint=entrypoint,
        train_npy_root=train_npy_root,
        work_dir=work_dir,
        base_checkpoint=ckpt,
        task_name=str(training_cfg.get('task_name', 'eq_glomeruli')),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        device=device,
    )


def _run_local_feasibility_smoke(
    *,
    command: list[str],
    env: dict[str, str],
    timeout_seconds: int,
    image_size: int,
    batch_size: int,
) -> dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        memory_failure = (
            'memory' in (completed.stderr or '').lower()
            or 'oom' in (completed.stderr or '').lower()
        )
        status = (
            'local_feasible'
            if completed.returncode == 0 and not memory_failure
            else 'requires_external_accelerator'
        )
        return {
            'local_feasibility_status': status,
            'returncode': int(completed.returncode),
            'stdout': completed.stdout[-4000:],
            'stderr': completed.stderr[-4000:],
            'elapsed_seconds': time.time() - started,
            'device': env.get('PYTORCH_DEVICE', ''),
            'image_size': int(image_size),
            'batch_size': int(batch_size),
            'memory_failure': bool(memory_failure),
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            'local_feasibility_status': 'requires_external_accelerator',
            'returncode': '',
            'stdout': '',
            'stderr': str(exc),
            'elapsed_seconds': time.time() - started,
            'device': env.get('PYTORCH_DEVICE', ''),
            'image_size': int(image_size),
            'batch_size': int(batch_size),
            'memory_failure': 'memory' in str(exc).lower(),
        }


def _preflight_training_dependencies(config: dict[str, Any]) -> dict[str, str]:
    medsam = _mapping(config, 'medsam')
    training = _mapping(config, 'training')
    python_path = resolve_medsam_torch_python(config)
    repo_path = Path(str(medsam.get('repo', '')).strip()).expanduser()
    base_checkpoint = Path(str(medsam.get('base_checkpoint', '')).strip()).expanduser()
    backend = training_backend(config)
    entrypoint = str(training.get('entrypoint', 'train_one_gpu.py')).strip()
    if not repo_path.exists():
        raise FileNotFoundError(f'Configured MedSAM repo does not exist: {repo_path}')
    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f'Configured MedSAM base checkpoint does not exist: {base_checkpoint}'
        )
    if backend == 'upstream_medsam':
        entrypoint_path = (repo_path / entrypoint).resolve()
        if not entrypoint_path.exists():
            raise FileNotFoundError(
                f'Configured MedSAM entrypoint does not exist: {entrypoint_path}'
            )
        entrypoint_str = str(entrypoint_path)
    else:
        entrypoint_str = ''
    adaptation_mode = str(medsam.get('adaptation_mode', '')).strip()
    if not adaptation_mode:
        raise ValueError('medsam.adaptation_mode is required')
    frozen_components = medsam.get('frozen_components', [])
    trainable_components = medsam.get('trainable_components', [])
    if not isinstance(frozen_components, list) or not isinstance(
        trainable_components, list
    ):
        raise ValueError(
            'medsam.frozen_components and medsam.trainable_components must be lists'
        )
    if not trainable_components:
        raise ValueError(
            'medsam.trainable_components must include at least one component'
        )
    return {
        'medsam_python': str(python_path),
        'medsam_repo': str(repo_path),
        'base_checkpoint': str(base_checkpoint),
        'entrypoint': entrypoint_str,
        'adaptation_mode': adaptation_mode,
        'training_backend': backend,
    }


def _full_training_skip_reason(
    *, dry_run: bool, training_cfg: dict[str, Any], local_feasibility: dict[str, Any]
) -> str:
    if dry_run:
        return 'dry_run'
    if not bool(training_cfg.get('run_training', True)):
        return 'run_training_disabled'
    if bool(training_cfg.get('local_feasibility_required', False)) and bool(
        training_cfg.get('run_local_feasibility_smoke', False)
    ):
        if (
            str(local_feasibility.get('local_feasibility_status', ''))
            != 'local_feasible'
        ):
            return 'local_feasibility_gate'
    return ''


def run_medsam_glomeruli_fine_tuning_workflow(
    config_path: Path, *, dry_run: bool = False
) -> dict[str, Path]:
    config = _load_config(config_path)
    runtime_root = _runtime_root(config)
    log_context = current_execution_log_context()
    if log_context is not None:
        _emit(f'EXECUTION_LOG={log_context.log_path}')
    _emit(f'WORKFLOW={WORKFLOW_ID}')
    _emit('PHASE=load_config status=started')
    medsam = _mapping(config, 'medsam')
    inputs = _mapping(config, 'inputs')
    split_cfg = _mapping(inputs, 'split_manifests')
    paths = _resolve_output_paths(config, runtime_root)
    outputs = _mapping(config, 'outputs')
    run_cfg = _mapping(config, 'run')
    lane_assignments = [
        str(v)
        for v in inputs.get(
            'lane_assignments', ['manual_mask_core', 'manual_mask_external']
        )
    ]
    manifest_path = _runtime_path(
        runtime_root, inputs.get('manifest_path', DEFAULT_MANIFEST_PATH)
    )
    paths['evaluation_dir'].mkdir(parents=True, exist_ok=True)
    generated_splits = False
    split_hashes: dict[str, str] = {}
    split_paths: dict[str, str] = {}
    audit_extra_rows = pd.DataFrame()
    _emit('PHASE=resolve_splits status=started')
    explicit_train = str(split_cfg.get('train', '')).strip()
    explicit_validation = str(split_cfg.get('validation', '')).strip()
    explicit_test = str(split_cfg.get('test', '')).strip()
    if explicit_train and explicit_validation and explicit_test:
        for key, raw in (
            ('train', explicit_train),
            ('validation', explicit_validation),
            ('test', explicit_test),
        ):
            resolved = _runtime_path(runtime_root, raw)
            split_hashes[key] = _validate_split_manifest(
                resolved, runtime_root, expected_split=key
            )
            split_paths[key] = str(resolved)
    else:
        manifest = pd.read_csv(manifest_path)
        admitted_for_audit = _admitted_manual_rows(
            manifest,
            runtime_root,
            lane_assignments=lane_assignments,
            include_missing_files=True,
        )
        admitted = _admitted_manual_rows(
            manifest, runtime_root, lane_assignments=lane_assignments
        )
        split_frame = _build_deterministic_splits(
            admitted,
            train_fraction=float(inputs.get('train_fraction', 0.70)),
            validation_fraction=float(inputs.get('validation_fraction', 0.15)),
        )
        manifest_outputs = _write_split_manifests(split_frame, paths['evaluation_dir'])
        for key, manifest_file in manifest_outputs.items():
            split_hashes[key] = _hash_file(manifest_file)
            split_paths[key] = str(manifest_file)
        valid_ids = set(admitted['manifest_row_id'].astype(str))
        audit_extra_rows = admitted_for_audit[
            ~admitted_for_audit['manifest_row_id'].astype(str).isin(valid_ids)
        ].copy()
        if not audit_extra_rows.empty:
            audit_extra_rows['split'] = 'excluded_missing_artifact'
        generated_splits = True
    _emit(
        'PHASE=resolve_splits status=completed generated_splits=%s'
        % str(generated_splits)
    )
    _emit('PHASE=dependency_preflight status=started')
    preflight = _preflight_training_dependencies(config)
    _emit(
        'PHASE=dependency_preflight status=completed backend=%s'
        % str(preflight.get('training_backend', ''))
    )
    training_cfg = _mapping(config, 'training')
    fixed_rows = _fixed_evaluation_rows(split_paths, runtime_root)
    fixed_split_names = sorted(
        {str(value).strip() for value in fixed_rows.get('split', pd.Series(dtype=str))}
    )
    metric_evidence_context = {
        'run_id': str(run_cfg.get('name') or DEFAULT_RUN_ID),
        'fixed_split': '+'.join(fixed_split_names),
        'fixed_manifest_row_ids': sorted(
            str(value).strip()
            for value in fixed_rows.get('manifest_row_id', pd.Series(dtype=str))
        ),
    }
    train_rows = _split_rows(split_paths, runtime_root, 'train')
    audit_rows = pd.concat(
        [train_rows, fixed_rows, audit_extra_rows], ignore_index=True
    )
    dataset_split_audit = write_medsam_evaluation_cohort_audit(
        audit_rows,
        runtime_root=runtime_root,
        evaluation_dir=paths['evaluation_dir'],
    )
    train_npy_root = _runtime_path(
        runtime_root,
        training_cfg.get(
            'preprocessed_npy_root', 'derived_data/medsam_glomeruli/npy_data'
        ),
    )
    training_data = _prepare_medsam_npy_data(
        train_rows=train_rows,
        output_root=train_npy_root,
        image_size=int(training_cfg.get('image_size', 1024)),
    )
    val_npy_root_for_training: Path | None = None
    validation_npy_data: dict[str, Any]
    if dry_run:
        validation_npy_data = {'status': 'skipped_dry_run'}
    elif str(preflight.get('training_backend', '')) != 'eq_native_adapter':
        validation_npy_data = {'status': 'skipped', 'reason': 'not_eq_native_adapter'}
    else:
        val_rows = _split_rows(split_paths, runtime_root, 'validation')
        if val_rows.empty:
            validation_npy_data = {
                'status': 'skipped',
                'reason': 'empty_validation_split',
            }
        else:
            val_rel = str(training_cfg.get('val_preprocessed_npy_root', '')).strip()
            val_output = (
                _runtime_path(runtime_root, val_rel)
                if val_rel
                else train_npy_root.parent / 'npy_data_validation'
            )
            validation_npy_data = _prepare_medsam_npy_data(
                train_rows=val_rows,
                output_root=val_output,
                image_size=int(training_cfg.get('image_size', 1024)),
            )
            if int(validation_npy_data.get('prepared_count', 0) or 0) > 0:
                val_npy_root_for_training = val_output
            else:
                validation_npy_data['status'] = 'skipped_empty_after_prepare'
    baseline_outputs = _write_fixed_baseline_metrics(
        fixed_rows=fixed_rows, output_dir=paths['evaluation_dir']
    )
    baseline_outputs.update(metric_evidence_context)
    baseline_plan = _baseline_execution_plan(fixed_rows=fixed_rows, config=config)
    baseline_eval_cfg = _mapping(config, 'baseline_evaluation')
    external_allowed = bool(baseline_eval_cfg.get('external_baselines', True))
    external_baselines_summary: dict[str, Any]
    if dry_run:
        external_baselines_summary = {
            'status': 'skipped_dry_run',
            'reason': 'External MedSAM and current-segmenter baselines are skipped during dry-run.',
        }
    elif not external_allowed:
        external_baselines_summary = {
            'status': 'skipped_config',
            'reason': 'baseline_evaluation.external_baselines=false',
        }
    else:
        _emit('PHASE=baseline_evaluation status=started')
        external_baselines_summary = _run_external_fixed_split_baselines(
            fixed_rows=fixed_rows,
            config=config,
            baseline_dir=paths['evaluation_dir'] / 'baseline_metrics',
            _runtime_root=runtime_root,
        )
        _emit('PHASE=baseline_evaluation status=completed')
    external_baselines_summary.update(metric_evidence_context)
    training_command = _training_command_for_config(
        config=config,
        preflight=preflight,
        train_npy_root=train_npy_root,
        work_dir=paths['checkpoint_root'],
        training_cfg=training_cfg,
        val_npy_root=val_npy_root_for_training,
    )
    local_feasibility = {
        'local_feasibility_status': 'requires_external_accelerator'
        if dry_run
        else 'not_run',
        'dry_run': bool(dry_run),
        'reason': 'Dry run records the feasibility contract without launching MedSAM training.'
        if dry_run
        else 'Local feasibility smoke command not requested.',
    }
    if (not dry_run) and bool(training_cfg.get('run_local_feasibility_smoke', False)):
        _emit('PHASE=local_feasibility status=started')
        backend = preflight.get('training_backend', 'upstream_medsam')
        smoke_env = {
            **medsam_subprocess_extra_env(device=str(medsam.get('device', 'cpu'))),
            'PYTORCH_DEVICE': str(medsam.get('device', 'cpu')),
        }
        if backend == 'eq_native_adapter':
            smoke_training_cfg = dict(training_cfg)
            smoke_training_cfg['epochs'] = int(training_cfg.get('smoke_epochs', 1))
            smoke_training_cfg['batch_size'] = int(
                training_cfg.get('smoke_batch_size', 1)
            )
            smoke_training_cfg['max_examples'] = int(
                training_cfg.get('smoke_examples', 2)
            )
            smoke_command = _training_command_for_config(
                config=config,
                preflight=preflight,
                train_npy_root=train_npy_root,
                work_dir=paths['checkpoint_root'],
                training_cfg=smoke_training_cfg,
                val_npy_root=None,
            )
        else:
            smoke_command = training_command + [
                '-num_epochs',
                str(int(training_cfg.get('smoke_epochs', 1))),
                '-batch_size',
                str(int(training_cfg.get('smoke_batch_size', 1))),
            ]
        local_feasibility = _run_local_feasibility_smoke(
            command=smoke_command,
            env=smoke_env,
            timeout_seconds=int(training_cfg.get('smoke_timeout_seconds', 300)),
            image_size=int(training_cfg.get('image_size', 1024)),
            batch_size=int(training_cfg.get('smoke_batch_size', 1)),
        )
        _emit(
            'PHASE=local_feasibility status=completed local_feasibility_status=%s'
            % str(local_feasibility.get('local_feasibility_status', ''))
        )
    skip_reason = _full_training_skip_reason(
        dry_run=dry_run, training_cfg=training_cfg, local_feasibility=local_feasibility
    )
    paths['checkpoint_root'].mkdir(parents=True, exist_ok=True)
    training_process_exit: int | None = None
    if dry_run:
        training_status = 'not_started_dry_run'
    elif skip_reason == 'run_training_disabled':
        training_status = 'skipped_run_training_disabled'
    elif skip_reason == 'local_feasibility_gate':
        training_status = 'skipped_local_feasibility_gate'
    elif skip_reason == 'dry_run':
        training_status = 'not_started_dry_run'
    elif skip_reason:
        training_status = 'skipped'
    else:
        train_env = os.environ.copy()
        train_env.update(
            medsam_subprocess_extra_env(device=str(medsam.get('device', 'cpu')))
        )
        train_env['PYTORCH_DEVICE'] = str(medsam.get('device', 'cpu'))
        _emit('PHASE=training status=started')
        completed = subprocess.run(
            training_command,
            env=train_env,
            check=False,
            text=True,
            cwd=str(runtime_root),
        )
        training_process_exit = int(completed.returncode)
        _emit(
            'PHASE=training status=finished returncode=%s' % str(completed.returncode)
        )
        training_status = 'completed' if completed.returncode == 0 else 'failed'

    checkpoint_files = sorted(
        (path for path in paths['checkpoint_root'].glob('*.pth') if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    checkpoint_provenance = _build_checkpoint_provenance(
        training_command=training_command,
        environment={
            **medsam_subprocess_extra_env(device=str(medsam.get('device', 'cpu'))),
            'PYTORCH_DEVICE': str(medsam.get('device', 'cpu')),
        },
        medsam_repo=Path(preflight['medsam_repo']),
        base_checkpoint=Path(preflight['base_checkpoint']),
        split_manifest_paths={key: Path(value) for key, value in split_paths.items()},
        split_manifest_hashes=split_hashes,
        adaptation_mode=preflight['adaptation_mode'],
        frozen_components=[str(value) for value in medsam.get('frozen_components', [])],
        trainable_components=[
            str(value) for value in medsam.get('trainable_components', [])
        ],
        hyperparameters=training_cfg,
        checkpoint_files=checkpoint_files,
        training_status=training_status,
        local_feasibility_status=str(
            local_feasibility.get('local_feasibility_status', '')
        ),
        reuse_existing_checkpoints=skip_reason == 'run_training_disabled',
    )
    checkpoint_provenance_path = paths['checkpoint_root'] / 'provenance.json'
    checkpoint_provenance_path.write_text(
        json.dumps(checkpoint_provenance, indent=2), encoding='utf-8'
    )
    finetuned_evaluation = _run_finetuned_fixed_split_evaluation(
        fixed_rows=fixed_rows,
        evaluation_dir=paths['evaluation_dir'],
        config=config,
        checkpoint_provenance=checkpoint_provenance,
        dry_run=dry_run,
    )
    finetuned_evaluation.update(metric_evidence_context)
    promotion_evidence = _promotion_evidence_summary(config, runtime_root)
    if (
        str(promotion_evidence['overlay_review'].get('status', '')) == 'passed'
        and str(promotion_evidence['overlay_review'].get('sha256', ''))
    ):
        finetuned_evaluation['overlay_review_status'] = 'passed'
    box_diagnostics = write_medsam_box_diagnostics(
        fixed_rows,
        proposal_boxes_path=finetuned_evaluation.get('proposal_boxes_csv', ''),
        automatic_metrics_path=finetuned_evaluation.get('metrics_csv', ''),
        oracle_metrics_path=external_baselines_summary.get(
            'oracle_medsam_metrics_csv', ''
        ),
        output_dir=paths['evaluation_dir'],
        selected_candidate_family=str(
            finetuned_evaluation.get('selected_candidate_family', '')
        ),
        selected_threshold=finetuned_evaluation.get('selected_threshold'),
    )
    adoption_gates = _effective_adoption_gates(_mapping(config, 'adoption_gates'))
    finetuned_comparison = _build_finetuned_comparison_summary(
        finetuned_evaluation=finetuned_evaluation,
        external_baselines_summary=external_baselines_summary,
        baseline_outputs=baseline_outputs,
        adoption_gates=adoption_gates,
    )
    generated_mask_packaging: dict[str, Any] = {'status': 'skipped'}
    packaging_requested = bool(outputs.get('package_generated_mask_release', True))
    packaging_inputs_ready = (
        (not dry_run)
        and str(finetuned_evaluation.get('status', '')) == 'completed'
        and int(finetuned_evaluation.get('metric_rows', 0) or 0) > 0
        and packaging_requested
    )
    comparison_eligible = str(
        finetuned_comparison.get('status', '')
    ) == 'completed' and str(finetuned_comparison.get('adoption_tier', '')) in {
        'oracle_level_preferred',
        'improved_candidate_not_oracle',
    } and not bool(dataset_split_audit.get('promotion_blocked', False))
    if packaging_inputs_ready and not comparison_eligible:
        generated_mask_packaging = {
            'status': 'skipped_comparison_not_eligible',
            'reason': str(
                finetuned_comparison.get(
                    'reason', 'Fine-tuned comparison is not eligible for packaging.'
                )
            ),
            'reason_codes': list(finetuned_comparison.get('reason_codes', [])),
        }
    elif packaging_inputs_ready and comparison_eligible:
        masks_eval_dir = Path(str(finetuned_evaluation['masks_dir']))
        family = str(finetuned_evaluation.get('selected_candidate_family', ''))
        threshold = float(finetuned_evaluation.get('selected_threshold', 0.0))
        ckpt_infer = Path(str(finetuned_evaluation.get('checkpoint_path', '')))
        mask_release_id = str(
            outputs.get('mask_release_id') or run_cfg.get('name') or ''
        )
        checkpoint_id = str(outputs.get('checkpoint_id') or run_cfg.get('name') or '')
        run_name = str(run_cfg.get('name') or DEFAULT_RUN_ID)
        generated_release_items: list[dict[str, Any]] = []
        for record in fixed_rows.fillna('').to_dict(orient='records'):
            row_id = str(record['manifest_row_id'])
            predicted_path = (
                masks_eval_dir
                / f'{row_id}_{family}_{threshold:g}_medsam_finetuned_auto.png'
            )
            if not predicted_path.exists():
                continue
            generated_release_items.append(
                {
                    'generated_mask_path': str(predicted_path),
                    'cohort_id': str(record.get('cohort_id', '')),
                    'lane_assignment': str(record.get('lane_assignment', '')),
                    'source_sample_id': str(record.get('source_sample_id', '')),
                    'source_image_path': str(record.get('image_path_resolved', '')),
                    'reference_mask_path': str(record.get('mask_path_resolved', '')),
                    'generation_status': 'generated',
                }
            )
        if generated_release_items:
            pkg_paths = _package_generated_mask_release(
                runtime_root=runtime_root,
                release_dir=paths['generated_release_root'],
                mask_release_id=mask_release_id,
                run_id=run_name,
                checkpoint_id=checkpoint_id,
                checkpoint_path=ckpt_infer,
                proposal_source='automatic_current_segmenter_boxes',
                proposal_threshold=threshold,
                adoption_tier=str(finetuned_comparison.get('adoption_tier', '')),
                generated_masks=generated_release_items,
            )
            _update_generated_mask_registry(
                paths['generated_registry_path'],
                pkg_paths['manifest'],
                runtime_root=runtime_root,
            )
            generated_mask_packaging = {
                'status': 'completed',
                'release_dir': str(pkg_paths['release_dir']),
                'manifest': str(pkg_paths['manifest']),
                'provenance': str(pkg_paths['provenance']),
                'index': str(pkg_paths['index']),
                'packaged_masks': len(generated_release_items),
            }
        else:
            generated_mask_packaging = {
                'status': 'skipped_no_prediction_masks',
                'masks_dir': str(masks_eval_dir),
            }
    inference_checkpoint_path = Path(
        str(finetuned_evaluation.get('checkpoint_path', '')).strip()
    )
    inference_checkpoint_hash = _file_hash(inference_checkpoint_path)
    promotion_evidence_failures = [
        {
            'gate': key,
            'reason_code': str(record.get('reason_code', '')),
            'reason': str(record.get('reason', '')),
        }
        for key, record in promotion_evidence.items()
        if record.get('reason_code')
    ]
    if not inference_checkpoint_hash:
        promotion_evidence_failures.append(
            {
                'gate': 'inference_checkpoint',
                'reason_code': 'inference_checkpoint_artifact_unavailable',
                'reason': 'The exact checkpoint used for inference is missing or unreadable.',
            }
        )
    if str(generated_mask_packaging.get('status', '')) != 'completed':
        promotion_evidence_failures.append(
            {
                'gate': 'generated_mask_release',
                'reason_code': 'generated_mask_release_provenance_unavailable',
                'reason': 'Scientific promotion requires a completed generated-mask release.',
            }
        )
    scientific_promotion_eligible = (
        comparison_eligible
        and str(finetuned_comparison.get('adoption_tier', ''))
        == 'oracle_level_preferred'
        and not promotion_evidence_failures
    )
    practical_adoption_decision = (
        'package_candidate'
        if str(generated_mask_packaging.get('status', '')) == 'completed'
        and str(finetuned_comparison.get('adoption_tier', '')) in {
            'oracle_level_preferred',
            'improved_candidate_not_oracle',
        }
        else (
            'candidate_eligible_not_packaged'
            if comparison_eligible
            else 'blocked'
        )
    )
    checkpoint_hashes = {
        str(path): _file_hash(Path(str(path)))
        for path in checkpoint_provenance.get('checkpoint_files', [])
        if str(path).strip() and Path(str(path)).is_file()
    }
    promotion_reason_codes = sorted(
        {
            *dataset_split_audit.get('blocking_reason_codes', []),
            *finetuned_comparison.get('reason_codes', []),
            *[
                str(failure.get('reason_code', ''))
                for failure in promotion_evidence_failures
                if failure.get('reason_code')
            ],
            *([] if comparison_eligible else ['comparison_not_eligible']),
            *(
                []
                if scientific_promotion_eligible
                else ['scientific_promotion_not_eligible']
            ),
        }
    )
    failed_gates = list(finetuned_comparison.get('failed_gates', []))
    failed_gates.extend(
        {
            'gate': 'dataset_split_audit',
            'reason_code': str(issue.get('reason_code', '')),
            'reason': 'Dataset/split audit blocked promotion.',
        }
        for issue in dataset_split_audit.get('issues', [])
        if issue.get('reason_code')
        != 'positive_only_evidence_distribution_limitation'
    )
    if not scientific_promotion_eligible:
        failed_gates.append(
            {
                'gate': 'scientific_promotion',
                'reason_code': 'scientific_promotion_not_eligible',
                'reason': 'Automatic evidence did not satisfy the scientific promotion contract.',
            }
        )
    failed_gates.extend(promotion_evidence_failures)
    base_checkpoint_hash = (
        _file_hash(Path(str(preflight['base_checkpoint'])))
        if Path(str(preflight.get('base_checkpoint', ''))).is_file()
        else ''
    )
    evaluation_created_at = datetime.now().isoformat(timespec='seconds')
    portable_checkpoint_hashes = {
        str(
            _portable_report_value(
                path, runtime_root=runtime_root, key='checkpoint_path'
            )
        ): digest
        for path, digest in checkpoint_hashes.items()
    }
    promotion_report = {
        'schema_version': 'eq.medsam.promotion-report/1',
        'path_encoding': RUNTIME_RELATIVE_POSIX_ENCODING,
        'runtime_root': str(runtime_root),
        'evaluation': {
            'run_id': str(run_cfg.get('name') or DEFAULT_RUN_ID),
            'created_at': evaluation_created_at,
            'config_path': _portable_report_value(
                str(config_path), runtime_root=runtime_root, key='config_path'
            ),
        },
        'checkpoint': {
            'checkpoint_id': str(outputs.get('checkpoint_id') or run_cfg.get('name') or ''),
            'checkpoint_files': _portable_report_value(
                list(checkpoint_provenance.get('checkpoint_files', [])),
                runtime_root=runtime_root,
                key='checkpoint_files',
            ),
            'checkpoint_sha256': portable_checkpoint_hashes,
            'inference_checkpoint_path': _portable_report_value(
                str(inference_checkpoint_path),
                runtime_root=runtime_root,
                key='inference_checkpoint_path',
            ),
            'inference_checkpoint_sha256': inference_checkpoint_hash,
            'base_checkpoint': _portable_report_value(
                str(preflight.get('base_checkpoint', '')),
                runtime_root=runtime_root,
                key='base_checkpoint',
            ),
            'base_checkpoint_sha256': base_checkpoint_hash,
        },
        'split_manifests': {
            'ids': {
                key: f'{key}:{split_hashes.get(key, "")}'
                for key in sorted(split_paths)
            },
            'paths': {
                key: _portable_report_value(
                    value, runtime_root=runtime_root, key='artifact_path'
                )
                for key, value in split_paths.items()
            },
            'sha256': split_hashes,
        },
        'dataset_audit': _portable_report_value(
            {
                'id': str(dataset_split_audit.get('audit_sha256', '')),
                'sha256': str(dataset_split_audit.get('audit_sha256', '')),
                'artifacts': dataset_split_audit.get('artifacts', {}),
            },
            runtime_root=runtime_root,
        ),
        'box_diagnostics': _portable_report_value(
            box_diagnostics, runtime_root=runtime_root
        ),
        'preprocessing_prompt_config': _portable_report_value(
            {
                'training_image_size': training_cfg.get('image_size', ''),
                'tiling': _mapping(config, 'tiling'),
                'proposal': _mapping(config, 'proposal'),
                'selected_candidate_family': finetuned_evaluation.get(
                    'selected_candidate_family', ''
                ),
                'selected_threshold': finetuned_evaluation.get('selected_threshold'),
            },
            runtime_root=runtime_root,
        ),
        'metric_artifacts': _portable_report_value(
            finetuned_comparison.get('metric_evidence', {}),
            runtime_root=runtime_root,
        ),
        'promotion_evidence': _portable_report_value(
            promotion_evidence, runtime_root=runtime_root
        ),
        'thresholds': {
            'adoption_gates': adoption_gates,
            'current_segmenter_comparison_threshold': _mapping(
                config, 'current_segmenter'
            ).get('comparison_threshold'),
        },
        'failed_gates': failed_gates,
        'reason_codes': promotion_reason_codes,
        'decision': 'promote' if scientific_promotion_eligible else 'blocked',
        'practical_adoption_decision': practical_adoption_decision,
        'generated_mask_packaging': _portable_report_value(
            generated_mask_packaging, runtime_root=runtime_root
        ),
    }
    promotion_report_path = paths['evaluation_dir'] / 'promotion_report.json'
    promotion_report_path.write_text(
        json.dumps(
            promotion_report, indent=2, sort_keys=True, allow_nan=False
        ),
        encoding='utf-8',
    )
    summary = {
        'workflow': WORKFLOW_ID,
        'run_id': str(run_cfg.get('name') or DEFAULT_RUN_ID),
        'created_at': evaluation_created_at,
        'config_path': str(config_path),
        'runtime_root': str(runtime_root),
        'log_path': str(log_context.log_path) if log_context is not None else '',
        'manifest_path': str(manifest_path),
        'dry_run': bool(dry_run),
        'medsam_python': preflight['medsam_python'],
        'medsam_repo': preflight['medsam_repo'],
        'base_checkpoint': preflight['base_checkpoint'],
        'base_checkpoint_hash': _file_hash(
            Path(str(medsam.get('base_checkpoint', '')).strip()).expanduser()
        )
        if str(medsam.get('base_checkpoint', '')).strip()
        else '',
        'paths': {
            'evaluation_dir': str(paths['evaluation_dir']),
            'checkpoint_root': str(paths['checkpoint_root']),
            'generated_release_root': str(paths['generated_release_root']),
            'generated_registry_path': str(paths['generated_registry_path']),
        },
        'adoption_gates': adoption_gates,
        'dependency_preflight': preflight,
        'baseline_evaluation': {
            **baseline_plan,
            'outputs': baseline_outputs,
            'external_baselines': external_baselines_summary,
            'dry_run_note': external_baselines_summary.get('reason', '')
            if dry_run
            else '',
        },
        'training_data': training_data,
        'validation_npy_data': validation_npy_data,
        'local_feasibility': local_feasibility,
        'training_skip_reason': skip_reason,
        'training_process_exit_code': training_process_exit,
        'training_command': training_command,
        'checkpoint_provenance': str(checkpoint_provenance_path),
        'checkpoint_provenance_summary': {
            'training_status': checkpoint_provenance['training_status'],
            'supported_checkpoint': checkpoint_provenance['supported_checkpoint'],
            'checkpoint_files': checkpoint_provenance['checkpoint_files'],
        },
        'split_manifests': split_paths,
        'split_manifest_hashes': split_hashes,
        'generated_splits': generated_splits,
        'dataset_split_audit': dataset_split_audit,
        'box_diagnostics': box_diagnostics,
        'promotion_status': (
            'eligible' if scientific_promotion_eligible else 'blocked'
        ),
        'promotion_report_path': str(promotion_report_path),
        'promotion_report': promotion_report,
        'promotion_reason_codes': promotion_reason_codes,
        'finetuned_evaluation': finetuned_evaluation,
        'finetuned_comparison': finetuned_comparison,
        'generated_mask_packaging': generated_mask_packaging,
        'note': 'Fine-tuned checkpoint inference/evaluation tasks run after training produces supported checkpoint artifacts.',
    }
    summary_path = paths['evaluation_dir'] / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _emit(f'ARTIFACT_SUMMARY={summary_path}')
    if not dry_run and bool(outputs.get('create_checkpoint_dir_on_start', False)):
        paths['checkpoint_root'].mkdir(parents=True, exist_ok=True)
    _emit('PHASE=workflow status=completed')
    return {'evaluation_dir': paths['evaluation_dir'], 'summary': summary_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run MedSAM glomeruli fine-tuning workflow.'
    )
    parser.add_argument('--config', default='configs/medsam_glomeruli_fine_tuning.yaml')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_medsam_glomeruli_fine_tuning_workflow(Path(args.config), dry_run=args.dry_run)


if __name__ == '__main__':
    main()
