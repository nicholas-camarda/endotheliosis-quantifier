"""Acceptance artifact generation for the hybrid Label Studio demo workflow."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eq.labelstudio.bootstrap import (
    DEFAULT_CONTAINER_NAME,
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_PROJECT_TITLE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USERNAME,
    BootstrapError,
    BootstrapResult,
    LabelStudioApiClient,
    _prediction_from_mask,
    run_bootstrap,
)
from eq.utils.paths import get_active_runtime_root

COMPARISON_PROOF_METHOD = 'medsam_oracle'
DEFAULT_DEMO_PROJECT_NAME = 'EQ Hybrid Production Proof'
COMPARISON_METRIC_FIELDS = (
    'dice',
    'jaccard',
    'precision',
    'recall',
    'pixel_accuracy',
    'area_ratio',
    'manual_component_count',
    'prediction_component_count',
)


@dataclass(frozen=True)
class ComparisonExample:
    comparison_run_dir: Path
    comparison_run_id: str
    manifest_row_id: str
    cohort_id: str
    lane_assignment: str
    source_sample_id: str
    staged_images_dir: Path
    staged_relative_path: str
    source_image_path: Path
    manual_mask_path: Path
    prediction_mask_path: Path
    overlay_path: Path
    prediction_method: str
    proof_source_id: str
    metrics_by_method: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class HybridDemoResult:
    bootstrap: BootstrapResult
    acceptance_path: Path
    acceptance: dict[str, Any]


@dataclass(frozen=True)
class ResolvedExport:
    annotation_export: Path | None
    project_url: str


def run_hybrid_demo(
    *,
    images_dir: Path | None,
    runtime_root: Path | None = None,
    project_name: str = DEFAULT_DEMO_PROJECT_NAME,
    port: int = DEFAULT_PORT,
    container_name: str = DEFAULT_CONTAINER_NAME,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    api_token: str = 'eq-local-token',
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    dry_run: bool = False,
    config_path: Path | None = None,
    hybrid_mode: str = 'auto',
    acceptance_dir: Path | None = None,
    annotation_export: Path | None = None,
    comparison_run_dir: Path | None = None,
    comparison_manifest_row_id: str | None = None,
    live_export_only: bool = False,
) -> HybridDemoResult:
    """Run the hybrid demo bootstrap and write product-readiness evidence."""
    comparison_example = _prepare_comparison_example(
        runtime_root=runtime_root,
        comparison_run_dir=comparison_run_dir,
        comparison_manifest_row_id=comparison_manifest_row_id,
    )
    effective_images_dir = (
        comparison_example.staged_images_dir
        if comparison_example is not None
        else _require_images_dir(images_dir)
    )
    prediction_override = (
        {
            comparison_example.staged_relative_path: _prediction_from_mask(
                prediction_id='comparison_demo_00001',
                mask_path=comparison_example.prediction_mask_path,
                mask_release_id=comparison_example.proof_source_id,
            )
        }
        if comparison_example is not None
        else None
    )
    if live_export_only and not dry_run:
        raise BootstrapError(
            '--live-export-only must be used together with --dry-run so the current project is not replaced'
        )
    bootstrap = run_bootstrap(
        images_dir=effective_images_dir,
        runtime_root=runtime_root,
        project_name=project_name,
        port=port,
        container_name=container_name,
        docker_image=docker_image,
        username=username,
        password=password,
        api_token=api_token,
        timeout_seconds=timeout_seconds,
        dry_run=dry_run or live_export_only,
        config_path=config_path,
        hybrid_mode=hybrid_mode,
        prediction_by_relative_path_override=prediction_override,
        mask_release_id_override=(
            comparison_example.proof_source_id if comparison_example is not None else ''
        ),
        replace_existing_project=not live_export_only,
    )
    output_dir = (
        Path(acceptance_dir).expanduser().resolve()
        if acceptance_dir is not None
        else bootstrap.plan.runtime_root / 'acceptance'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_export = _resolve_annotation_export(
        annotation_export=annotation_export,
        live_export_only=live_export_only,
        bootstrap=bootstrap,
        output_dir=output_dir,
        api_token=api_token,
    )
    if live_export_only:
        bootstrap = BootstrapResult(
            plan=bootstrap.plan,
            task_manifest_path=bootstrap.task_manifest_path,
            project_url=resolved_export.project_url,
            message=(
                f'Refreshed acceptance evidence from live project '
                f'{bootstrap.plan.project_title!r} at {resolved_export.project_url}. '
                f'Task manifest remains {bootstrap.task_manifest_path}.'
            ),
            companion_status=bootstrap.companion_status,
        )
    acceptance = _build_acceptance_payload(
        bootstrap=bootstrap,
        dry_run=dry_run,
        config_path=config_path,
        annotation_export=resolved_export.annotation_export,
        comparison_example=comparison_example,
        live_export_only=live_export_only,
        project_url_override=resolved_export.project_url,
    )
    acceptance_path = (
        output_dir / f'{_safe_slug(project_name)}_hybrid_demo_acceptance.json'
    )
    acceptance_path.write_text(json.dumps(acceptance, indent=2), encoding='utf-8')
    return HybridDemoResult(
        bootstrap=bootstrap, acceptance_path=acceptance_path, acceptance=acceptance
    )


def _build_acceptance_payload(
    *,
    bootstrap: BootstrapResult,
    dry_run: bool,
    config_path: Path | None,
    annotation_export: Path | None,
    comparison_example: ComparisonExample | None,
    live_export_only: bool,
    project_url_override: str,
) -> dict[str, Any]:
    effective_project_url = project_url_override or bootstrap.project_url
    task_summary = _task_manifest_summary(bootstrap.task_manifest_path)
    payload: dict[str, Any] = {
        'status': 'ready_for_labeling' if not dry_run else 'dry_run_ready',
        'dry_run': bool(dry_run),
        'project_name': bootstrap.plan.project_title,
        'label_studio_url': bootstrap.plan.url,
        'project_url': effective_project_url,
        'login_email': bootstrap.plan.username,
        'task_manifest_path': str(bootstrap.task_manifest_path),
        'task_count': int(bootstrap.plan.task_count),
        'manifest_task_count': int(task_summary['task_count']),
        'preload_prediction_count': int(task_summary['preload_prediction_count']),
        'preload_region_count': int(task_summary['preload_region_count']),
        'preload_status_counts': task_summary['preload_status_counts'],
        'selected_mask_release_id': bootstrap.plan.selected_mask_release_id,
        'hybrid_mode': bootstrap.plan.hybrid_mode,
        'companion_status': bootstrap.companion_status,
        'companion_base_url': bootstrap.plan.companion_base_url,
        'companion_health_path': bootstrap.plan.companion_health_path,
        'companion_required': bool(bootstrap.plan.companion_required),
        'config_path': str(config_path.expanduser().resolve()) if config_path else '',
        'operator_next_steps': _operator_next_steps(
            dry_run=dry_run,
            project_url=effective_project_url,
            live_export_only=live_export_only,
        ),
    }
    if comparison_example is not None:
        payload['comparison_example'] = _comparison_example_payload(comparison_example)
    if annotation_export is not None:
        payload['annotation_export_summary'] = _annotation_export_summary(
            annotation_export
        )
    return payload


def _resolve_annotation_export(
    *,
    annotation_export: Path | None,
    live_export_only: bool,
    bootstrap: BootstrapResult,
    output_dir: Path,
    api_token: str,
) -> ResolvedExport:
    if annotation_export is not None:
        return ResolvedExport(
            annotation_export=Path(annotation_export).expanduser().resolve(),
            project_url=bootstrap.project_url,
        )
    if not live_export_only:
        return ResolvedExport(annotation_export=None, project_url=bootstrap.project_url)
    client = LabelStudioApiClient(bootstrap.plan.url, api_token)
    project_id = client.find_project_id(bootstrap.plan.project_title)
    if project_id is None:
        raise BootstrapError(
            f'Could not find live Label Studio project {bootstrap.plan.project_title!r} '
            f'at {bootstrap.plan.url} for --live-export-only'
        )
    export_payload = client.export_project(project_id)
    export_path = (
        output_dir / f'{_safe_slug(bootstrap.plan.project_title)}-live_export.json'
    )
    export_path.write_text(json.dumps(export_payload, indent=2), encoding='utf-8')
    return ResolvedExport(
        annotation_export=export_path,
        project_url=f'{bootstrap.plan.url}/projects/{project_id}/data',
    )


def _require_images_dir(images_dir: Path | None) -> Path:
    if images_dir is None:
        raise BootstrapError(
            'images_dir is required unless --comparison-run-dir and '
            '--comparison-manifest-row-id are both provided'
        )
    return Path(images_dir).expanduser().resolve()


def _resolve_demo_runtime_root(runtime_root: Path | None) -> Path:
    if runtime_root is not None:
        return Path(runtime_root).expanduser().resolve()
    return get_active_runtime_root() / 'labelstudio'


def _prepare_comparison_example(
    *,
    runtime_root: Path | None,
    comparison_run_dir: Path | None,
    comparison_manifest_row_id: str | None,
) -> ComparisonExample | None:
    if comparison_run_dir is None and not comparison_manifest_row_id:
        return None
    if comparison_run_dir is None or not comparison_manifest_row_id:
        raise BootstrapError(
            'Real-data comparison demo requires both --comparison-run-dir and '
            '--comparison-manifest-row-id'
        )
    run_dir = Path(comparison_run_dir).expanduser().resolve()
    inputs_path = run_dir / 'inputs.csv'
    metrics_path = run_dir / 'metrics.csv'
    if not inputs_path.exists():
        raise BootstrapError(f'Comparison inputs.csv is missing: {inputs_path}')
    if not metrics_path.exists():
        raise BootstrapError(f'Comparison metrics.csv is missing: {metrics_path}')

    manifest_row_id = str(comparison_manifest_row_id).strip()
    input_row = _find_csv_row(inputs_path, 'manifest_row_id', manifest_row_id)
    if input_row is None:
        raise BootstrapError(
            f'Comparison manifest_row_id {manifest_row_id!r} not found in {inputs_path}'
        )
    image_path = _comparison_input_path(
        input_row,
        field='image_path',
        label=f'source image for {manifest_row_id}',
    )
    manual_mask_path = _comparison_input_path(
        input_row,
        field='mask_path',
        label=f'manual mask for {manifest_row_id}',
    )
    prediction_mask_path = _existing_path(
        run_dir
        / 'medsam_masks'
        / f'{manifest_row_id}_{COMPARISON_PROOF_METHOD}_union.png',
        f'MedSAM proof mask for {manifest_row_id}',
    )
    overlay_path = _existing_path(
        run_dir / 'overlays' / f'{manifest_row_id}_overlay.png',
        f'comparison overlay for {manifest_row_id}',
    )
    metrics_by_method = _metrics_for_manifest_row(metrics_path, manifest_row_id)
    if COMPARISON_PROOF_METHOD not in metrics_by_method:
        raise BootstrapError(
            f'Comparison metrics for {manifest_row_id} are missing method '
            f'{COMPARISON_PROOF_METHOD!r} in {metrics_path}'
        )
    if (
        'current_segmenter_transfer' not in metrics_by_method
        or 'current_segmenter_scratch' not in metrics_by_method
    ):
        raise BootstrapError(
            f'Comparison metrics for {manifest_row_id} must include current segmenter rows in {metrics_path}'
        )

    labelstudio_root = _resolve_demo_runtime_root(runtime_root)
    bundle_root = (
        labelstudio_root
        / 'comparison_examples'
        / f'{run_dir.name}_{_safe_slug(manifest_row_id)}'
    )
    staged_images_dir = bundle_root / 'images'
    cohort_id = str(input_row.get('cohort_id') or '').strip() or 'comparison_case'
    staged_relative_path = f'{cohort_id}/{image_path.name}'
    staged_image_path = staged_images_dir / cohort_id / image_path.name
    staged_image_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, staged_image_path)
    evidence_dir = bundle_root / 'evidence'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    staged_manual_mask = evidence_dir / manual_mask_path.name
    staged_prediction_mask = evidence_dir / prediction_mask_path.name
    staged_overlay = evidence_dir / overlay_path.name
    shutil.copy2(manual_mask_path, staged_manual_mask)
    shutil.copy2(prediction_mask_path, staged_prediction_mask)
    shutil.copy2(overlay_path, staged_overlay)
    proof_source_id = (
        f'medsam_manual_glomeruli_comparison:{run_dir.name}:{COMPARISON_PROOF_METHOD}'
    )
    return ComparisonExample(
        comparison_run_dir=run_dir,
        comparison_run_id=run_dir.name,
        manifest_row_id=manifest_row_id,
        cohort_id=cohort_id,
        lane_assignment=str(input_row.get('lane_assignment') or ''),
        source_sample_id=str(input_row.get('source_sample_id') or ''),
        staged_images_dir=staged_images_dir,
        staged_relative_path=staged_relative_path,
        source_image_path=staged_image_path,
        manual_mask_path=staged_manual_mask,
        prediction_mask_path=staged_prediction_mask,
        overlay_path=staged_overlay,
        prediction_method=COMPARISON_PROOF_METHOD,
        proof_source_id=proof_source_id,
        metrics_by_method=metrics_by_method,
    )


def _find_csv_row(path: Path, key: str, expected: str) -> dict[str, str] | None:
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get(key) or '').strip() == expected:
                return row
    return None


def _comparison_input_path(
    row: dict[str, str], *, field: str, label: str
) -> Path:
    """Resolve a comparison input from its runtime-relative source field."""
    raw_path = str(row.get(field) or '').strip()
    if not raw_path:
        raise BootstrapError(f'Comparison row is missing {field!r} for {label}')
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = get_active_runtime_root() / path
    return _existing_path(path, label)


def _existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise BootstrapError(f'Missing {label}: {path}')
    return path.resolve()


def _metrics_for_manifest_row(
    metrics_path: Path, manifest_row_id: str
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    with metrics_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get('manifest_row_id') or '').strip() != manifest_row_id:
                continue
            method = str(row.get('method') or '').strip()
            if not method:
                continue
            parsed: dict[str, Any] = {
                'candidate_artifact': str(row.get('candidate_artifact') or '')
            }
            for field in COMPARISON_METRIC_FIELDS:
                value = str(row.get(field) or '').strip()
                if not value:
                    continue
                parsed[field] = (
                    int(float(value)) if field.endswith('_count') else float(value)
                )
            metrics[method] = parsed
    return metrics


def _comparison_example_payload(
    comparison_example: ComparisonExample,
) -> dict[str, Any]:
    medsam_metrics = comparison_example.metrics_by_method[
        comparison_example.prediction_method
    ]
    current_methods = {
        method: metrics
        for method, metrics in comparison_example.metrics_by_method.items()
        if method.startswith('current_segmenter_')
    }
    best_current_dice = max(
        float(metrics.get('dice', 0.0)) for metrics in current_methods.values()
    )
    return {
        'comparison_run_dir': str(comparison_example.comparison_run_dir),
        'comparison_run_id': comparison_example.comparison_run_id,
        'manifest_row_id': comparison_example.manifest_row_id,
        'cohort_id': comparison_example.cohort_id,
        'lane_assignment': comparison_example.lane_assignment,
        'source_sample_id': comparison_example.source_sample_id,
        'prediction_method': comparison_example.prediction_method,
        'proof_source_id': comparison_example.proof_source_id,
        'staged_images_dir': str(comparison_example.staged_images_dir),
        'staged_relative_path': comparison_example.staged_relative_path,
        'source_image_path': str(comparison_example.source_image_path),
        'manual_mask_path': str(comparison_example.manual_mask_path),
        'prediction_mask_path': str(comparison_example.prediction_mask_path),
        'overlay_path': str(comparison_example.overlay_path),
        'metrics_by_method': comparison_example.metrics_by_method,
        'medsam_beats_all_current_segmenters': bool(
            float(medsam_metrics.get('dice', 0.0)) > best_current_dice
        ),
        'dice_margin_vs_best_current': float(medsam_metrics.get('dice', 0.0))
        - float(best_current_dice),
    }


def _task_manifest_summary(task_manifest_path: Path) -> dict[str, Any]:
    tasks = json.loads(Path(task_manifest_path).read_text(encoding='utf-8'))
    if not isinstance(tasks, list):
        raise ValueError(f'Expected task manifest list: {task_manifest_path}')
    preload_status_counts: dict[str, int] = {}
    preload_prediction_count = 0
    preload_region_count = 0
    for task in tasks:
        if not isinstance(task, dict):
            continue
        status = str(
            (task.get('data') or {}).get('preload_prediction_status') or 'unknown'
        )
        preload_status_counts[status] = preload_status_counts.get(status, 0) + 1
        lineage = (task.get('data') or {}).get('preload_region_lineage')
        if isinstance(lineage, dict) and lineage:
            preload_prediction_count += 1
            preload_region_count += len(lineage)
    return {
        'task_count': len(tasks),
        'preload_prediction_count': preload_prediction_count,
        'preload_region_count': preload_region_count,
        'preload_status_counts': preload_status_counts,
    }


def _annotation_export_summary(annotation_export: Path) -> dict[str, Any]:
    export_path = Path(annotation_export).expanduser().resolve()
    payload = json.loads(export_path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'Expected Label Studio export list: {export_path}')
    summary = _summarize_glomerulus_export(payload)
    return {
        'annotation_export': str(export_path),
        'scoring_unit': 'glomerulus_instance',
        'rows': int(summary['rows']),
        'proposal_kind_counts': summary['proposal_kind_counts'],
        'region_edit_state_counts': summary['region_edit_state_counts'],
        'mask_release_counts': summary['mask_release_counts'],
    }


def _summarize_glomerulus_export(payload: list[dict[str, Any]]) -> dict[str, Any]:
    proposal_kind_counts: Counter[str] = Counter()
    region_edit_state_counts: Counter[str] = Counter()
    mask_release_counts: Counter[str] = Counter()
    row_count = 0

    for task in payload:
        if not isinstance(task, dict):
            continue
        task_data = task.get('data') or {}
        task_lineage = task_data.get('preload_region_lineage')
        if not isinstance(task_lineage, dict):
            task_lineage = {}
        prediction_region_ids = _prediction_region_ids(task)
        annotations = task.get('annotations') or []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            result = annotation.get('result') or []
            regions: dict[str, dict[str, Any]] = {}
            graded_region_ids: set[str] = set()
            for item in result:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get('id') or '').strip()
                item_type = str(item.get('type') or '').strip()
                if item_type == 'brushlabels' and item_id:
                    regions[item_id] = item
                    continue
                if item_type == 'choices':
                    parent_region_id = str(
                        item.get('parentID') or item.get('parent_id') or item_id
                    ).strip()
                    if _extract_grade_choice(item) is not None:
                        graded_region_ids.add(parent_region_id)

            for region_id, region in regions.items():
                labels = {
                    str(label).strip()
                    for label in (region.get('value') or {}).get('brushlabels', [])
                    if str(label).strip()
                }
                if 'complete_glomerulus' not in labels:
                    continue
                if region_id not in graded_region_ids:
                    continue
                row_count += 1
                lineage = task_lineage.get(region_id)
                if isinstance(lineage, dict):
                    proposal_kind = str(lineage.get('origin') or 'human_manual')
                    mask_release_id = str(lineage.get('mask_release_id') or '')
                else:
                    if region_id in prediction_region_ids:
                        proposal_kind = 'auto_preload'
                    elif annotation.get('parent_prediction'):
                        proposal_kind = 'box_assisted_manual'
                    else:
                        proposal_kind = 'human_manual'
                    mask_release_id = str(task_data.get('mask_release_id') or '')
                proposal_kind_counts[proposal_kind] += 1
                region_edit_state = (
                    'unedited_auto'
                    if proposal_kind == 'auto_preload'
                    else (
                        'refined_from_auto'
                        if proposal_kind == 'box_assisted_manual'
                        else 'human_manual'
                    )
                )
                region_edit_state_counts[region_edit_state] += 1
                mask_release_counts[mask_release_id] += 1

    return {
        'rows': row_count,
        'proposal_kind_counts': dict(proposal_kind_counts),
        'region_edit_state_counts': dict(region_edit_state_counts),
        'mask_release_counts': dict(mask_release_counts),
    }


def _prediction_region_ids(task: dict[str, Any]) -> set[str]:
    region_ids: set[str] = set()
    predictions = task.get('predictions') or []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        for item in prediction.get('result') or []:
            if not isinstance(item, dict):
                continue
            region_id = str(item.get('id') or '').strip()
            if region_id:
                region_ids.add(region_id)
    return region_ids


def _extract_grade_choice(result: dict[str, Any]) -> float | None:
    choices = (result.get('value') or {}).get('choices', [])
    if not isinstance(choices, list) or not choices:
        return None
    try:
        return float(str(choices[0]).strip())
    except ValueError:
        return None


def _operator_next_steps(
    *, dry_run: bool, project_url: str, live_export_only: bool
) -> list[str]:
    if live_export_only:
        return [
            f'Open {project_url or "the printed Label Studio project URL"} to continue the canonical project if more edits are needed.',
            'Use the refreshed acceptance artifact and live export JSON as the current production-proof evidence bundle.',
        ]
    if dry_run:
        return [
            'Start the MedSAM companion if hybrid companion status is required.',
            'Rerun without --dry-run to create or refresh the Label Studio project.',
            'Open the printed project URL, draw/edit brush regions, assign one grade per complete glomerulus, then export JSON.',
            'Rerun demo-hybrid with --annotation-export <export.json> to attach lineage evidence.',
        ]
    return [
        f'Open {project_url or "the printed Label Studio project URL"}.',
        'Draw or refine brush regions, assign one grade per complete glomerulus, and submit tasks.',
        'Export Label Studio JSON and rerun demo-hybrid with --annotation-export <export.json>.',
    ]


def _safe_slug(value: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '-', value.strip().lower())
    return '-'.join(part for part in slug.split('-') if part) or 'hybrid-demo'
