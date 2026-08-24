"""Local Label Studio bootstrap for glomerulus-instance grading."""

from __future__ import annotations

import copy
import csv
import http.client
import json
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml
from PIL import Image

from eq.labelstudio.brush_rle import mask2rle
from eq.utils.paths import (
    get_active_runtime_root,
    get_label_studio_medsam_hybrid_config_path,
    get_repo_root,
    get_runtime_generated_masks_glomeruli_manifest_path,
    get_runtime_medsam_finetuned_release_path,
)

SUPPORTED_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
DEFAULT_PROJECT_TITLE = 'EQ Glomerulus Grading'
DEFAULT_CONTAINER_NAME = 'eq-labelstudio'
# Pin Label Studio to an audited API contract version.
DEFAULT_DOCKER_IMAGE = 'heartexlabs/label-studio:1.23.0'
DEFAULT_USERNAME = 'eq-admin@example.local'
DEFAULT_PASSWORD = 'eq-labelstudio'
DEFAULT_API_TOKEN = 'eq-local-token'
DEFAULT_PORT = 8080
DEFAULT_TIMEOUT_SECONDS = 60
DOCKER_DESKTOP_APP_PATH = Path('/Applications/Docker.app')
CONTAINER_LOCAL_FILES_ROOT = '/label-studio/media'
DEFAULT_HYBRID_SELECTION_MODE = 'latest_valid'
DEFAULT_COMPANION_BASE_URL = 'http://localhost:8098'
DEFAULT_COMPANION_HEALTH_PATH = '/healthz'
DEFAULT_HYBRID_MODE = 'auto'
MIN_PRELOAD_COMPONENT_AREA_PX = 1000


class BootstrapError(RuntimeError):
    """Raised when local Label Studio bootstrap cannot proceed."""


@dataclass(frozen=True)
class HybridBootstrapSettings:
    enabled: bool
    preload_enabled: bool
    selection_mode: str
    mask_release_id: str
    companion_base_url: str
    companion_health_path: str
    require_box_assisted_medsam: bool
    offline_manual_only_allowed: bool
    fail_on_missing_preload: bool


@dataclass(frozen=True)
class BootstrapPlan:
    images_dir: Path
    runtime_root: Path
    data_dir: Path
    media_dir: Path
    media_mount_dir: Path
    local_files_dataset_dir: str
    imports_dir: Path
    bootstrap_dir: Path
    task_manifest_path: Path
    label_config_path: Path
    project_title: str
    url: str
    project_url: str
    container_name: str
    docker_image: str
    username: str
    password: str
    api_token: str
    timeout_seconds: int
    image_paths: tuple[Path, ...]
    task_count: int
    hybrid_mode: str
    selected_mask_release_id: str
    companion_base_url: str
    companion_health_path: str
    companion_required: bool
    offline_manual_only_allowed: bool
    fail_on_missing_preload: bool


@dataclass(frozen=True)
class BootstrapResult:
    plan: BootstrapPlan
    task_manifest_path: Path
    project_url: str
    message: str
    companion_status: str


DockerRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def discover_image_files(images_dir: Path) -> list[Path]:
    """Return supported image files under `images_dir` in deterministic order."""
    images_dir = Path(images_dir).expanduser()
    if not images_dir.exists() or not images_dir.is_dir():
        raise BootstrapError(f'Image directory does not exist: {images_dir}')
    return [
        path
        for path in sorted(images_dir.rglob('*'))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]


def build_labelstudio_tasks(
    images_dir: Path,
    image_paths: Iterable[Path],
    *,
    local_files_dataset_dir: str,
    prediction_by_relative_path: dict[str, dict[str, Any]] | None = None,
    mask_release_id: str = '',
) -> list[dict[str, Any]]:
    """Build Label Studio local-file tasks for image paths."""
    images_dir = Path(images_dir).expanduser().resolve()
    tasks: list[dict[str, Any]] = []
    for image_path in image_paths:
        image_path = Path(image_path).expanduser().resolve()
        relative = image_path.relative_to(images_dir).as_posix()
        served_relative = f'{local_files_dataset_dir}/{relative}'
        parts = Path(relative).parts
        subject_hint = parts[0] if len(parts) > 1 else image_path.stem
        task: dict[str, Any] = {
            'data': {
                'image': f'/data/local-files/?d={served_relative}',
                'source_relative_path': relative,
                'source_filename': image_path.name,
                'subject_hint': subject_hint,
                'mask_release_id': mask_release_id,
            }
        }
        if prediction_by_relative_path and relative in prediction_by_relative_path:
            prediction = prediction_by_relative_path[relative]
            task['data']['preload_prediction_status'] = 'available'
            task['data']['preload_region_lineage'] = _build_preload_region_lineage(
                prediction
            )
        else:
            task['data']['preload_prediction_status'] = 'missing'
        tasks.append(task)
    return tasks


def plan_bootstrap(
    *,
    images_dir: Path,
    runtime_root: Path | None = None,
    project_name: str = DEFAULT_PROJECT_TITLE,
    port: int = DEFAULT_PORT,
    container_name: str = DEFAULT_CONTAINER_NAME,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    api_token: str = DEFAULT_API_TOKEN,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    hybrid_settings: HybridBootstrapSettings | None = None,
    hybrid_mode: str = DEFAULT_HYBRID_MODE,
) -> BootstrapPlan:
    """Create a complete bootstrap plan without side effects."""
    images_dir = Path(images_dir).expanduser().resolve()
    image_paths = tuple(discover_image_files(images_dir))
    root = (
        Path(runtime_root).expanduser().resolve()
        if runtime_root is not None
        else get_active_runtime_root() / 'labelstudio'
    )
    url = f'http://localhost:{int(port)}'
    safe_project = _safe_slug(project_name)
    imports_dir = root / 'imports'
    settings = hybrid_settings or HybridBootstrapSettings(
        enabled=False,
        preload_enabled=False,
        selection_mode=DEFAULT_HYBRID_SELECTION_MODE,
        mask_release_id='',
        companion_base_url='',
        companion_health_path=DEFAULT_COMPANION_HEALTH_PATH,
        require_box_assisted_medsam=False,
        offline_manual_only_allowed=False,
        fail_on_missing_preload=False,
    )
    return BootstrapPlan(
        images_dir=images_dir,
        runtime_root=root,
        data_dir=root / 'data',
        media_dir=images_dir,
        media_mount_dir=images_dir.parent,
        local_files_dataset_dir=images_dir.name,
        imports_dir=imports_dir,
        bootstrap_dir=root / 'bootstrap',
        task_manifest_path=imports_dir / f'{safe_project}_tasks.json',
        label_config_path=get_repo_root()
        / 'configs'
        / 'label_studio_glomerulus_grading.xml',
        project_title=project_name,
        url=url,
        project_url='',
        container_name=container_name,
        docker_image=docker_image,
        username=username,
        password=password,
        api_token=api_token,
        timeout_seconds=int(timeout_seconds),
        image_paths=image_paths,
        task_count=len(image_paths),
        hybrid_mode=hybrid_mode,
        selected_mask_release_id=settings.mask_release_id,
        companion_base_url=settings.companion_base_url,
        companion_health_path=settings.companion_health_path,
        companion_required=settings.require_box_assisted_medsam,
        offline_manual_only_allowed=settings.offline_manual_only_allowed,
        fail_on_missing_preload=settings.fail_on_missing_preload,
    )


def run_bootstrap(
    *,
    images_dir: Path,
    runtime_root: Path | None = None,
    project_name: str = DEFAULT_PROJECT_TITLE,
    port: int = DEFAULT_PORT,
    container_name: str = DEFAULT_CONTAINER_NAME,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
    api_token: str = DEFAULT_API_TOKEN,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    dry_run: bool = False,
    config_path: Path | None = None,
    hybrid_mode: str = DEFAULT_HYBRID_MODE,
    prediction_by_relative_path_override: dict[str, dict[str, Any]] | None = None,
    mask_release_id_override: str = '',
    replace_existing_project: bool = False,
    docker_runner: DockerRunner | None = None,
    api_client: 'LabelStudioApiClient | Any | None' = None,
) -> BootstrapResult:
    """Prepare a local Label Studio project and import image tasks."""
    hybrid_settings = _resolve_hybrid_settings(
        config_path=config_path, hybrid_mode=hybrid_mode, runtime_root=runtime_root
    )
    plan = plan_bootstrap(
        images_dir=images_dir,
        runtime_root=runtime_root,
        project_name=project_name,
        port=port,
        container_name=container_name,
        docker_image=docker_image,
        username=username,
        password=password,
        api_token=api_token,
        timeout_seconds=timeout_seconds,
        hybrid_settings=hybrid_settings,
        hybrid_mode=hybrid_mode,
    )
    prediction_by_relative = (
        prediction_by_relative_path_override
        or _build_prediction_payloads(plan, hybrid_settings=hybrid_settings)
    )
    effective_mask_release_id = (
        str(mask_release_id_override).strip() or plan.selected_mask_release_id
    )
    if (
        not prediction_by_relative_path_override
        and plan.fail_on_missing_preload
        and plan.task_count
        and (len(prediction_by_relative) < plan.task_count)
    ):
        raise BootstrapError(
            'Hybrid config requires preload proposals for all tasks, '
            f'but only {len(prediction_by_relative)}/{plan.task_count} were resolved '
            f'from mask release {plan.selected_mask_release_id}.'
        )
    tasks = build_labelstudio_tasks(
        plan.images_dir,
        plan.image_paths,
        local_files_dataset_dir=plan.local_files_dataset_dir,
        prediction_by_relative_path=prediction_by_relative,
        mask_release_id=effective_mask_release_id,
    )
    _write_task_manifest(plan, tasks)
    companion_status = _companion_status_summary(plan, hybrid_settings=hybrid_settings)
    if dry_run:
        return BootstrapResult(
            plan=plan,
            task_manifest_path=plan.task_manifest_path,
            project_url='',
            message=(
                f'Dry run: would start {plan.container_name}, create project '
                f'{plan.project_title!r}, and import {plan.task_count} images. '
                f'Preload proposals: {len(prediction_by_relative)}.'
            ),
            companion_status=companion_status,
        )

    runner = docker_runner or _run_command
    ensure_docker_available(runner, timeout_seconds=plan.timeout_seconds)
    _enforce_companion_health(plan, hybrid_settings=hybrid_settings)
    start_labelstudio_container(plan, runner)
    client = api_client or LabelStudioApiClient(plan.url, plan.api_token)
    project_id = client.bootstrap_project(
        project_title=plan.project_title,
        label_config=plan.label_config_path.read_text(encoding='utf-8'),
        tasks=tasks,
        preload_annotation_by_relative_path=prediction_by_relative,
        timeout_seconds=plan.timeout_seconds,
        replace_existing_project=replace_existing_project,
    )
    project_url = f'{plan.url}/projects/{project_id}/data'
    return BootstrapResult(
        plan=plan,
        task_manifest_path=plan.task_manifest_path,
        project_url=project_url,
        message=(
            f'Label Studio project ready: {project_url} '
            f'({plan.task_count} image tasks imported, '
            f'{len(prediction_by_relative)} with preload proposals)'
        ),
        companion_status=companion_status,
    )


def _resolve_hybrid_settings(
    *, config_path: Path | None, hybrid_mode: str, runtime_root: Path | None
) -> HybridBootstrapSettings:
    if hybrid_mode not in {'auto', 'enabled', 'disabled'}:
        raise BootstrapError(
            f'Unsupported hybrid mode {hybrid_mode!r}; expected auto|enabled|disabled'
        )
    cfg_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else get_label_studio_medsam_hybrid_config_path()
    )
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        parsed = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
        if not isinstance(parsed, dict):
            raise BootstrapError(
                f'Invalid hybrid config at {cfg_path}: expected top-level mapping'
            )
        raw = parsed
    elif hybrid_mode == 'enabled':
        raise BootstrapError(
            f'Hybrid mode enabled but config file was not found: {cfg_path}'
        )

    hybrid_raw = raw.get('hybrid') if isinstance(raw.get('hybrid'), dict) else {}
    companion_raw = (
        raw.get('companion') if isinstance(raw.get('companion'), dict) else {}
    )
    enabled_default = bool(hybrid_raw.get('enabled', True))
    enabled = (
        False
        if hybrid_mode == 'disabled'
        else True
        if hybrid_mode == 'enabled'
        else enabled_default
    )
    preload_enabled = (
        bool(hybrid_raw.get('preload_enabled', False)) if enabled else False
    )
    pinned_release = str(hybrid_raw.get('mask_release_id') or '').strip()
    if enabled and not preload_enabled and pinned_release:
        raise BootstrapError(
            'hybrid.mask_release_id requires hybrid.preload_enabled: true'
        )
    selection_mode = str(
        hybrid_raw.get('selection_mode', DEFAULT_HYBRID_SELECTION_MODE)
    ).strip()
    if selection_mode not in {'latest_valid'}:
        raise BootstrapError(
            'Invalid hybrid.selection_mode; supported values: latest_valid'
        )

    selected_release = (
        _select_mask_release_id(
            pinned_id=pinned_release,
            selection_mode=selection_mode,
            runtime_root=runtime_root,
        )
        if preload_enabled
        else ''
    )
    companion_base = str(
        companion_raw.get('base_url', DEFAULT_COMPANION_BASE_URL)
    ).strip()
    companion_health = str(
        companion_raw.get('health_path', DEFAULT_COMPANION_HEALTH_PATH)
    ).strip()
    if enabled and not companion_health.startswith('/'):
        raise BootstrapError(
            'Invalid companion.health_path: expected URL path beginning with /'
        )
    return HybridBootstrapSettings(
        enabled=enabled,
        preload_enabled=preload_enabled,
        selection_mode=selection_mode,
        mask_release_id=selected_release,
        companion_base_url=companion_base if enabled else '',
        companion_health_path=companion_health,
        require_box_assisted_medsam=bool(
            hybrid_raw.get('require_box_assisted_medsam', True)
        )
        if enabled
        else False,
        offline_manual_only_allowed=bool(
            hybrid_raw.get('offline_manual_only_allowed', False)
        )
        if enabled
        else False,
        fail_on_missing_preload=bool(hybrid_raw.get('fail_on_missing_preload', False))
        if enabled
        else False,
    )


def _select_mask_release_id(
    *, pinned_id: str, selection_mode: str, runtime_root: Path | None
) -> str:
    manifest_path = get_runtime_generated_masks_glomeruli_manifest_path(runtime_root)
    if not manifest_path.exists():
        if pinned_id:
            raise BootstrapError(
                f'Pinned mask_release_id {pinned_id!r} not found because manifest is missing: {manifest_path}'
            )
        return ''
    release_rows = _load_release_rows(manifest_path)
    if pinned_id:
        if pinned_id not in release_rows:
            candidates = ', '.join(sorted(release_rows)[:8])
            raise BootstrapError(
                f'Pinned mask_release_id {pinned_id!r} not found in {manifest_path}. '
                f'Available candidates: {candidates or "<none>"}'
            )
        return pinned_id
    if selection_mode == 'latest_valid':
        ranked = sorted(
            release_rows.items(),
            key=lambda item: (
                item[1]['generated_count'],
                item[1]['latest_mtime'],
                item[0],
            ),
            reverse=True,
        )
        return ranked[0][0] if ranked else ''
    raise BootstrapError(f'Unsupported selection_mode {selection_mode!r}')


def _load_release_rows(manifest_path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    with manifest_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            release_id = str(row.get('mask_release_id') or '').strip()
            if not release_id:
                continue
            group = grouped.setdefault(
                release_id, {'rows': [], 'generated_count': 0, 'latest_mtime': 0.0}
            )
            group['rows'].append(row)
            status = str(row.get('generation_status') or '').strip().lower()
            mask_path = Path(str(row.get('generated_mask_path') or '')).expanduser()
            if status == 'generated' and mask_path.exists():
                group['generated_count'] += 1
                group['latest_mtime'] = max(
                    group['latest_mtime'], mask_path.stat().st_mtime
                )
    return grouped


def _build_prediction_payloads(
    plan: BootstrapPlan, *, hybrid_settings: HybridBootstrapSettings
) -> dict[str, dict[str, Any]]:
    if not hybrid_settings.enabled or not plan.selected_mask_release_id:
        return {}
    runtime_root = (
        plan.runtime_root.parent
        if plan.runtime_root.name == 'labelstudio'
        else plan.runtime_root
    )
    release_dir = get_runtime_medsam_finetuned_release_path(
        plan.selected_mask_release_id, runtime_root
    )
    release_manifest_path = release_dir / 'manifest.csv'
    if not release_manifest_path.exists():
        return {}

    by_name: dict[str, Path] = {}
    for image_path in plan.image_paths:
        rel = image_path.relative_to(plan.images_dir).as_posix()
        by_name[image_path.name.lower()] = Path(rel)

    prediction_by_relative: dict[str, dict[str, Any]] = {}
    with release_manifest_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            source_image = str(row.get('source_image_path') or '')
            key = Path(source_image).name.lower()
            if key not in by_name:
                continue
            mask_path = Path(str(row.get('generated_mask_path') or '')).expanduser()
            if not mask_path.exists():
                continue
            rel = by_name[key].as_posix()
            prediction_by_relative[rel] = _prediction_from_mask(
                prediction_id=f'auto_{idx:05d}',
                mask_path=mask_path,
                mask_release_id=plan.selected_mask_release_id,
            )
    return prediction_by_relative


def _prediction_from_mask(
    *, prediction_id: str, mask_path: Path, mask_release_id: str
) -> dict[str, Any]:
    mask = Image.open(mask_path).convert('L')
    mask_np = (np.array(mask) > 0).astype(np.uint8)
    height, width = mask_np.shape
    component_masks = _connected_component_masks(mask_np)
    if not component_masks:
        component_masks = [mask_np]
    result: list[dict[str, Any]] = []
    for component_idx, component_mask in enumerate(component_masks, start=1):
        rle = mask2rle((component_mask * 255).astype(np.uint8))
        result.append(
            {
                'id': f'{prediction_id}_{component_idx:03d}',
                'from_name': 'glomerulus_roi',
                'to_name': 'image',
                'type': 'brushlabels',
                'original_width': int(width),
                'original_height': int(height),
                'image_rotation': 0,
                'origin': 'prediction',
                'value': {
                    'format': 'rle',
                    'rle': rle,
                    'brushlabels': ['complete_glomerulus'],
                },
            }
        )
    return {
        'model_version': f'medsam-release:{mask_release_id}',
        'score': 1.0,
        'result': result,
    }


def _build_preload_region_lineage(
    prediction: dict[str, Any],
) -> dict[str, dict[str, str]]:
    result = prediction.get('result')
    if not isinstance(result, list):
        return {}
    model_version = str(prediction.get('model_version') or '')
    mask_release_id = _mask_release_from_model_version(model_version)
    mask_source = _mask_source_from_model_version(model_version)
    lineage: dict[str, dict[str, str]] = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        region_id = str(item.get('id') or '').strip()
        if not region_id:
            continue
        lineage[region_id] = {
            'origin': 'auto_preload',
            'mask_release_id': mask_release_id,
            'mask_source': mask_source,
            'source_prediction_id': region_id,
            'preload_geometry_signature': _preload_geometry_signature(item),
        }
    return lineage


def _preload_geometry_signature(item: dict[str, Any]) -> str:
    value = item.get('value') or {}
    return json.dumps(
        {
            'format': value.get('format'),
            'rle': value.get('rle'),
            'points': value.get('points'),
            'brushlabels': value.get('brushlabels'),
            'polygonlabels': value.get('polygonlabels'),
            'rectanglelabels': value.get('rectanglelabels'),
        },
        sort_keys=True,
    )


def _connected_component_masks(mask_np: np.ndarray) -> list[np.ndarray]:
    """Split a binary mask into disconnected per-instance binary masks."""
    if mask_np.ndim != 2:
        raise BootstrapError('Expected 2D mask for Label Studio preload conversion')
    active = mask_np > 0
    if not np.any(active):
        return []
    height, width = active.shape
    visited = np.zeros_like(active, dtype=bool)
    components: list[np.ndarray] = []
    for y in range(height):
        for x in range(width):
            if not active[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            component = np.zeros_like(mask_np, dtype=np.uint8)
            while queue:
                cy, cx = queue.popleft()
                component[cy, cx] = 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if not active[ny, nx] or visited[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            components.append(component)
    filtered: list[np.ndarray] = []
    for component in components:
        if int(component.sum()) < MIN_PRELOAD_COMPONENT_AREA_PX:
            continue
        filtered.append(component)
    return filtered


def _mask_source_from_model_version(model_version: str) -> str:
    if str(model_version).startswith('medsam-release:'):
        return 'medsam_finetuned_glomeruli'
    return ''


def _mask_release_from_model_version(model_version: str) -> str:
    text = str(model_version).strip()
    prefix = 'medsam-release:'
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return ''


def _binary_mask_bbox(mask_np: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _companion_status_summary(
    plan: BootstrapPlan, *, hybrid_settings: HybridBootstrapSettings
) -> str:
    if not hybrid_settings.enabled:
        return 'hybrid-disabled'
    if not plan.companion_required:
        return 'hybrid-enabled companion-optional'
    if plan.offline_manual_only_allowed:
        return 'hybrid-enabled companion-required-with-admin-bypass'
    return 'hybrid-enabled companion-required'


def _enforce_companion_health(
    plan: BootstrapPlan, *, hybrid_settings: HybridBootstrapSettings
) -> None:
    if not hybrid_settings.enabled or not plan.companion_required:
        return
    if not plan.companion_base_url:
        raise BootstrapError('Hybrid mode requires companion.base_url in config')
    health_url = f'{plan.companion_base_url.rstrip("/")}{plan.companion_health_path}'
    try:
        request = urllib.request.Request(health_url, method='GET')
        with urllib.request.urlopen(request, timeout=5) as response:
            if int(response.status) >= 400:
                raise BootstrapError(
                    f'Companion health check failed: {health_url} -> {response.status}'
                )
    except Exception as exc:
        if plan.offline_manual_only_allowed:
            return
        raise BootstrapError(
            'Hybrid companion health check failed and offline_manual_only_allowed is false. '
            f'URL: {health_url}. Error: {exc}'
        ) from exc


def ensure_docker_available(
    runner: DockerRunner | None = None,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Fail early if Docker is missing, and start Docker Desktop on macOS when possible."""
    if shutil.which('docker') is None:
        raise BootstrapError(_docker_install_message())
    runner = runner or _run_command
    if runner(['docker', 'info']).returncode == 0:
        return
    if _can_start_docker_desktop():
        start = runner(['open', '-a', 'Docker'])
        if start.returncode != 0:
            raise BootstrapError(
                'Docker Desktop is installed but could not be started. Open Docker Desktop manually and rerun the command.'
            )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if runner(['docker', 'info']).returncode == 0:
                return
            time.sleep(2)
        raise BootstrapError(
            'Docker Desktop was started but the Docker daemon did not become ready. '
            'Open Docker Desktop and wait for it to finish starting, then rerun the command.'
        )
    raise BootstrapError(
        'Docker is installed but the daemon is not running. Start Docker and rerun the command.'
    )


def start_labelstudio_container(
    plan: BootstrapPlan, runner: DockerRunner | None = None
) -> None:
    """Start or reuse the local Label Studio Docker container."""
    runner = runner or _run_command
    inspect = runner(['docker', 'inspect', plan.container_name])
    if inspect.returncode == 0:
        if not _container_matches_plan(plan, inspect.stdout):
            remove = runner(['docker', 'rm', '-f', plan.container_name])
            if remove.returncode != 0:
                raise BootstrapError(
                    f'Failed to replace container {plan.container_name}'
                )
            command = docker_run_command(plan)
            result = runner(command)
            if result.returncode != 0:
                raise BootstrapError(
                    f'Failed to start Label Studio Docker container: {result.stderr}'
                )
            return
        start = runner(['docker', 'start', plan.container_name])
        if start.returncode != 0:
            raise BootstrapError(f'Failed to start container {plan.container_name}')
        return

    plan.data_dir.mkdir(parents=True, exist_ok=True)
    command = docker_run_command(plan)
    result = runner(command)
    if result.returncode != 0:
        raise BootstrapError(
            f'Failed to start Label Studio Docker container: {result.stderr}'
        )


def docker_run_command(plan: BootstrapPlan) -> list[str]:
    """Return the Docker command used to start Label Studio."""
    return [
        'docker',
        'run',
        '-d',
        '--name',
        plan.container_name,
        '-p',
        f'{_port_from_url(plan.url)}:8080',
        '-v',
        f'{plan.data_dir}:/label-studio/data',
        '-v',
        f'{plan.media_mount_dir}:{CONTAINER_LOCAL_FILES_ROOT}:ro',
        '-e',
        'LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true',
        '-e',
        f'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={CONTAINER_LOCAL_FILES_ROOT}',
        '-e',
        f'LABEL_STUDIO_USERNAME={plan.username}',
        '-e',
        f'LABEL_STUDIO_PASSWORD={plan.password}',
        '-e',
        f'LABEL_STUDIO_USER_TOKEN={plan.api_token}',
        '-e',
        'LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN=true',
        plan.docker_image,
    ]


def _container_matches_plan(plan: BootstrapPlan, inspect_stdout: str) -> bool:
    try:
        details = json.loads(inspect_stdout)
    except json.JSONDecodeError:
        return False
    container = details[0] if isinstance(details, list) and details else {}
    mounts = container.get('Mounts', []) if isinstance(container, dict) else []
    mount_map = {
        str(mount.get('Destination')): str(mount.get('Source'))
        for mount in mounts
        if isinstance(mount, dict)
    }
    media_source = mount_map.get(CONTAINER_LOCAL_FILES_ROOT)
    data_source = mount_map.get('/label-studio/data')
    expected_media_source = str(plan.media_mount_dir)
    expected_data_source = str(plan.data_dir)
    return (
        media_source == expected_media_source
        and data_source == expected_data_source
        and f'{CONTAINER_LOCAL_FILES_ROOT}/images' not in mount_map
    )


def _can_start_docker_desktop() -> bool:
    return platform.system() == 'Darwin' and DOCKER_DESKTOP_APP_PATH.exists()


def _docker_install_message() -> str:
    if platform.system() == 'Darwin':
        return (
            'Docker Desktop is not installed. Install it with '
            '`brew install --cask docker`, open Docker Desktop once, then rerun this command.'
        )
    return 'Docker executable not found. Install Docker or use --dry-run.'


class LabelStudioApiClient:
    """Small stdlib HTTP client for the Label Studio bootstrap API calls."""

    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token

    def bootstrap_project(
        self,
        *,
        project_title: str,
        label_config: str,
        tasks: list[dict[str, Any]],
        preload_annotation_by_relative_path: dict[str, dict[str, Any]] | None = None,
        timeout_seconds: int,
        replace_existing_project: bool = False,
    ) -> int:
        self.wait_until_ready(timeout_seconds)
        project_id = self._find_project(project_title)
        if project_id is not None and replace_existing_project:
            self._request_json('DELETE', f'/api/projects/{project_id}/')
            project_id = None
        if project_id is None:
            response = self._request_json(
                'POST',
                '/api/projects/',
                {'title': project_title, 'label_config': label_config},
            )
            project_id = int(response['id'])
        else:
            self._request_json(
                'PATCH', f'/api/projects/{project_id}/', {'label_config': label_config}
            )
        self._ensure_local_file_storages(project_id, tasks)
        self._request_json('POST', f'/api/projects/{project_id}/import', tasks)
        self._materialize_preload_annotations(
            project_id,
            imported_tasks=tasks,
            preload_annotation_by_relative_path=preload_annotation_by_relative_path
            or {},
        )
        return project_id

    def find_project_id(self, project_title: str) -> int | None:
        self.wait_until_ready(DEFAULT_TIMEOUT_SECONDS)
        return self._find_project(project_title)

    def export_project(self, project_id: int) -> list[dict[str, Any]]:
        response = self._request_json(
            'GET', f'/api/projects/{int(project_id)}/export?exportType=JSON'
        )
        if not isinstance(response, list):
            raise BootstrapError(
                f'Expected JSON export list for project {project_id}, got {type(response).__name__}'
            )
        return response

    def wait_until_ready(self, timeout_seconds: int) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                self._request_json('GET', '/api/projects/')
                return
            except (
                BootstrapError,
                http.client.RemoteDisconnected,
                urllib.error.URLError,
                ConnectionResetError,
            ):
                time.sleep(1)
        raise BootstrapError(
            f'Label Studio did not become reachable at {self.base_url} within '
            f'{timeout_seconds}s. Inspect the Docker container logs.'
        )

    def _find_project(self, project_title: str) -> int | None:
        response = self._request_json('GET', '/api/projects/')
        projects = _list_response_items(response)
        for project in projects:
            if project.get('title') == project_title:
                return int(project['id'])
        return None

    def _ensure_local_file_storages(
        self, project_id: int, tasks: list[dict[str, Any]]
    ) -> None:
        response = self._request_json(
            'GET', f'/api/storages/localfiles/?project={project_id}'
        )
        storages = _list_response_items(response)
        existing_paths = {storage.get('path') for storage in storages}
        for storage_path in _local_file_storage_paths(tasks):
            if storage_path in existing_paths:
                continue
            self._request_json(
                'POST',
                '/api/storages/localfiles/',
                {
                    'project': project_id,
                    'path': storage_path,
                    'title': f'EQ Image Media {Path(storage_path).name}',
                    'use_blob_urls': True,
                    'recursive_scan': True,
                },
            )

    def _request_json(self, method: str, path: str, payload: Any | None = None) -> Any:
        data = None
        headers = {'Authorization': f'Token {self.api_token}'}
        if payload is not None:
            data = json.dumps(payload).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request(
            f'{self.base_url}{path}', data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            raise BootstrapError(
                f'Label Studio API {method} {path} failed: {exc.code} {body}'
            ) from exc
        return json.loads(raw) if raw else {}

    def _materialize_preload_annotations(
        self,
        project_id: int,
        *,
        imported_tasks: list[dict[str, Any]],
        preload_annotation_by_relative_path: dict[str, dict[str, Any]],
    ) -> None:
        """Create editable annotations from preload masks without importing predictions."""
        annotation_results_by_relative: dict[str, list[dict[str, Any]]] = {}
        for relative, prediction in preload_annotation_by_relative_path.items():
            if not isinstance(relative, str) or not isinstance(prediction, dict):
                continue
            result = prediction.get('result')
            if isinstance(result, list) and result:
                annotation_results_by_relative[relative] = _editable_annotation_results(
                    result
                )
        if not annotation_results_by_relative:
            return

        materialized: list[dict[str, Any]] = []
        for task in self._iter_project_tasks(project_id):
            if not isinstance(task, dict):
                continue
            relative = task.get('data', {}).get('source_relative_path')
            if (
                not isinstance(relative, str)
                or relative not in annotation_results_by_relative
            ):
                continue
            total_annotations = int(task.get('total_annotations') or 0)
            annotations = (
                task.get('annotations')
                if isinstance(task.get('annotations'), list)
                else []
            )
            if total_annotations > 0 or annotations:
                # Respect existing manual work; do not auto-create duplicates.
                materialized.append(
                    {
                        'task_id': task.get('id'),
                        'relative_image_path': relative,
                        'action': 'skipped_existing_annotations',
                        'total_annotations': total_annotations,
                    }
                )
                continue
            self._request_json(
                'POST',
                f'/api/tasks/{int(task.get("id"))}/annotations/',
                {'result': annotation_results_by_relative[relative]},
            )
            materialized.append(
                {
                    'task_id': task.get('id'),
                    'relative_image_path': relative,
                    'action': 'created_annotation',
                    'region_count': len(annotation_results_by_relative[relative]),
                }
            )

    def _iter_project_tasks(self, project_id: int) -> Iterable[dict[str, Any]]:
        path = f'/api/tasks/?project={project_id}'
        while path:
            response = self._request_json('GET', path)
            if isinstance(response, list):
                for item in response:
                    if isinstance(item, dict):
                        yield item
                return
            if not isinstance(response, dict):
                return
            results = response.get('results')
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        yield item
            tasks = response.get('tasks')
            if isinstance(tasks, list):
                for item in tasks:
                    if isinstance(item, dict):
                        yield item
            next_path = response.get('next')
            if not next_path:
                return
            if isinstance(next_path, str) and next_path.startswith(self.base_url):
                path = next_path[len(self.base_url) :]
            else:
                path = str(next_path)


def _editable_annotation_results(results: list[Any]) -> list[dict[str, Any]]:
    editable_results: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        result = copy.deepcopy(item)
        if result.get('type') == 'brushlabels':
            _validate_editable_brush_result(result)
            result['image_rotation'] = int(result.get('image_rotation') or 0)
            result['origin'] = 'manual'
        editable_results.append(result)
    return editable_results


def _validate_editable_brush_result(result: dict[str, Any]) -> None:
    region_id = str(result.get('id') or '<missing-id>')
    if result.get('original_width') is None or result.get('original_height') is None:
        raise BootstrapError(
            f'Preload brush region {region_id} cannot be materialized as an '
            'editable annotation without original_width and original_height.'
        )
    value = result.get('value')
    if not isinstance(value, dict):
        raise BootstrapError(
            f'Preload brush region {region_id} cannot be materialized as an '
            'editable annotation without a value mapping.'
        )
    if value.get('format') != 'rle' or not isinstance(value.get('rle'), list):
        raise BootstrapError(
            f'Preload brush region {region_id} cannot be materialized as an '
            'editable annotation without value.format=rle and list value.rle.'
        )
    if not isinstance(value.get('brushlabels'), list) or not value.get('brushlabels'):
        raise BootstrapError(
            f'Preload brush region {region_id} cannot be materialized as an '
            'editable annotation without value.brushlabels.'
        )


def _list_response_items(response: Any) -> list[Any]:
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        results = response.get('results')
        return results if isinstance(results, list) else []
    return []


def _local_file_storage_paths(tasks: list[dict[str, Any]]) -> list[str]:
    dataset_roots: set[str] = set()
    flat_files: list[str] = []
    for task in tasks:
        relative = task.get('data', {}).get('source_relative_path')
        image_ref = task.get('data', {}).get('image')
        if not isinstance(relative, str):
            continue
        parts = Path(relative).parts
        if len(parts) < 2:
            flat_files.append(relative)
        if isinstance(image_ref, str) and '?d=' in image_ref:
            served_relative = image_ref.split('?d=', 1)[1]
            served_parts = Path(served_relative).parts
            if served_parts:
                dataset_roots.add(f'{CONTAINER_LOCAL_FILES_ROOT}/{served_parts[0]}')
    if flat_files:
        examples = ', '.join(flat_files[:3])
        raise BootstrapError(
            'Label Studio local-file serving requires images to be inside at least one subfolder '
            f'under --images. Move flat image files into a subject/batch folder first. Examples: {examples}'
        )
    return sorted(dataset_roots)


def _write_task_manifest(plan: BootstrapPlan, tasks: list[dict[str, Any]]) -> None:
    plan.imports_dir.mkdir(parents=True, exist_ok=True)
    plan.bootstrap_dir.mkdir(parents=True, exist_ok=True)
    plan.task_manifest_path.write_text(json.dumps(tasks, indent=2), encoding='utf-8')


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def _safe_slug(value: str) -> str:
    slug = ''.join(char.lower() if char.isalnum() else '-' for char in value).strip('-')
    return '-'.join(part for part in slug.split('-') if part) or 'labelstudio-project'


def _port_from_url(url: str) -> int:
    return int(url.rsplit(':', 1)[-1])
