import json
from http.client import RemoteDisconnected
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eq.labelstudio.bootstrap import (
    CONTAINER_LOCAL_FILES_ROOT,
    BootstrapError,
    LabelStudioApiClient,
    _load_release_rows,
    _prediction_from_mask,
    build_labelstudio_tasks,
    discover_image_files,
    docker_run_command,
    ensure_docker_available,
    plan_bootstrap,
    run_bootstrap,
    start_labelstudio_container,
)
from eq.labelstudio.demo import run_hybrid_demo


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'image')
    return path


def _write_real_png(path: Path, value: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8), mode='RGB').save(path)
    return path


def test_discovers_supported_images_recursively_and_sorts(tmp_path: Path):
    images = tmp_path / 'images'
    tif = _write_image(images / 'animal_2' / 'b.tif')
    jpg = _write_image(images / 'animal_1' / 'a.JPG')
    _write_image(images / 'notes.txt')

    discovered = discover_image_files(images)

    assert discovered == [jpg, tif]


def test_task_payload_preserves_relative_path_and_subject_hint(tmp_path: Path):
    images = tmp_path / 'images'
    nested = _write_image(images / 'animal_1' / 'kidney_a' / 'image_001.tif')
    flat = _write_image(images / 'flat_image.png')

    tasks = build_labelstudio_tasks(
        images, [nested, flat], local_files_dataset_dir=images.name
    )

    assert tasks[0]['data']['image'] == (
        '/data/local-files/?d=images/animal_1/kidney_a/image_001.tif'
    )
    assert tasks[0]['data']['source_relative_path'] == (
        'animal_1/kidney_a/image_001.tif'
    )
    assert tasks[0]['data']['source_filename'] == 'image_001.tif'
    assert tasks[0]['data']['subject_hint'] == 'animal_1'
    assert tasks[1]['data']['subject_hint'] == 'flat_image'


def test_task_payload_stores_preload_lineage_in_task_data(tmp_path: Path):
    images = tmp_path / 'images'
    nested = _write_real_png(images / 'animal_1' / 'image_001.png')
    mask_path = tmp_path / 'mask.png'
    array = np.zeros((32, 32), dtype=np.uint8)
    array[4:20, 6:22] = 255
    Image.fromarray(array, mode='L').save(mask_path)
    prediction = _prediction_from_mask(
        prediction_id='auto_123', mask_path=mask_path, mask_release_id='release_x'
    )

    tasks = build_labelstudio_tasks(
        images,
        [nested],
        local_files_dataset_dir=images.name,
        prediction_by_relative_path={'animal_1/image_001.png': prediction},
        mask_release_id='release_x',
    )

    assert 'predictions' not in tasks[0]
    lineage = tasks[0]['data']['preload_region_lineage']
    assert lineage['auto_123_001']['origin'] == 'auto_preload'
    assert lineage['auto_123_001']['mask_release_id'] == 'release_x'
    assert lineage['auto_123_001']['mask_source'] == 'medsam_finetuned_glomeruli'
    assert lineage['auto_123_001']['source_prediction_id'] == 'auto_123_001'
    assert lineage['auto_123_001']['preload_geometry_signature']


def test_prediction_from_mask_splits_disconnected_regions(tmp_path: Path):
    mask_path = tmp_path / 'mask.png'
    array = np.zeros((120, 120), dtype=np.uint8)
    array[5:40, 5:40] = 255
    array[70:105, 70:105] = 255
    Image.fromarray(array, mode='L').save(mask_path)

    prediction = _prediction_from_mask(
        prediction_id='auto_123', mask_path=mask_path, mask_release_id='release_x'
    )

    assert prediction['model_version'] == 'medsam-release:release_x'
    assert len(prediction['result']) == 2
    assert prediction['result'][0]['id'] == 'auto_123_001'
    assert prediction['result'][1]['id'] == 'auto_123_002'
    assert prediction['result'][0]['image_rotation'] == 0
    assert prediction['result'][0]['origin'] == 'prediction'


def test_prediction_from_mask_drops_tiny_components(tmp_path: Path):
    mask_path = tmp_path / 'mask_tiny.png'
    array = np.zeros((80, 80), dtype=np.uint8)
    array[10:40, 10:40] = 255
    array[60:63, 60:63] = 255
    Image.fromarray(array, mode='L').save(mask_path)

    prediction = _prediction_from_mask(
        prediction_id='auto_200', mask_path=mask_path, mask_release_id='release_y'
    )

    assert len(prediction['result']) == 1
    assert prediction['result'][0]['id'] == 'auto_200_001'


def test_plan_bootstrap_uses_runtime_labelstudio_root(tmp_path: Path, monkeypatch):
    images = tmp_path / 'images'
    _write_image(images / 'a.png')
    runtime = tmp_path / 'runtime'
    monkeypatch.setenv('EQ_RUNTIME_ROOT', str(runtime))

    plan = plan_bootstrap(images_dir=images)

    assert plan.runtime_root == runtime / 'labelstudio'
    assert plan.data_dir == runtime / 'labelstudio' / 'data'
    assert plan.media_dir == images
    assert plan.media_mount_dir == tmp_path
    assert plan.local_files_dataset_dir == 'images'
    assert plan.imports_dir == runtime / 'labelstudio' / 'imports'
    assert plan.bootstrap_dir == runtime / 'labelstudio' / 'bootstrap'
    assert plan.project_title == 'EQ Glomerulus Grading'
    assert plan.url == 'http://localhost:8080'
    assert plan.task_count == 1


def test_docker_run_mounts_images_as_local_files_document_root(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'a.png')
    plan = plan_bootstrap(images_dir=images, runtime_root=tmp_path / 'runtime')

    command = docker_run_command(plan)

    assert f'{plan.media_mount_dir}:{CONTAINER_LOCAL_FILES_ROOT}:ro' in command
    assert (
        'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=' + CONTAINER_LOCAL_FILES_ROOT
        in command
    )


def test_start_container_reuses_existing_container_with_media_mount(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'a.png')
    plan = plan_bootstrap(images_dir=images, runtime_root=tmp_path / 'runtime')
    calls = []

    def fake_runner(command):
        calls.append(command)
        if command == ['docker', 'inspect', plan.container_name]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    '[{"Mounts":['
                    f'{{"Destination":"/label-studio/media","Source":"{plan.media_mount_dir}"}},'
                    f'{{"Destination":"/label-studio/data","Source":"{plan.data_dir}"}}'
                    ']}]'
                ),
            )
        return SimpleNamespace(returncode=0, stdout='')

    start_labelstudio_container(plan, fake_runner)

    assert ['docker', 'start', plan.container_name] in calls
    assert ['docker', 'rm', '-f', plan.container_name] not in calls


def test_start_container_recreates_failed_nested_media_mount(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'a.png')
    plan = plan_bootstrap(images_dir=images, runtime_root=tmp_path / 'runtime')
    calls = []

    def fake_runner(command):
        calls.append(command)
        if command == ['docker', 'inspect', plan.container_name]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    '[{"Mounts":['
                    f'{{"Destination":"/label-studio/media","Source":"{plan.media_mount_dir}"}},'
                    f'{{"Destination":"/label-studio/data","Source":"{plan.data_dir}"}},'
                    '{"Destination":"/label-studio/media/images"}'
                    ']}]'
                ),
            )
        return SimpleNamespace(returncode=0, stdout='')

    start_labelstudio_container(plan, fake_runner)

    assert ['docker', 'rm', '-f', plan.container_name] in calls
    assert docker_run_command(plan) in calls


def test_start_container_recreates_when_mount_source_changes(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'animal_1' / 'a.png')
    plan = plan_bootstrap(images_dir=images, runtime_root=tmp_path / 'runtime')
    calls = []

    def fake_runner(command):
        calls.append(command)
        if command == ['docker', 'inspect', plan.container_name]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    '[{"Mounts":['
                    '{"Destination":"/label-studio/media","Source":"/private/tmp"},'
                    f'{{"Destination":"/label-studio/data","Source":"{plan.data_dir}"}}'
                    ']}]'
                ),
            )
        return SimpleNamespace(returncode=0, stdout='')

    start_labelstudio_container(plan, fake_runner)

    assert ['docker', 'rm', '-f', plan.container_name] in calls
    assert docker_run_command(plan) in calls


def test_dry_run_writes_manifest_without_docker_or_api(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'animal_1' / 'a.png')
    runtime = tmp_path / 'runtime'

    result = run_bootstrap(
        images_dir=images,
        runtime_root=runtime,
        dry_run=True,
        docker_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('dry-run must not call docker')
        ),
        api_client=SimpleNamespace(
            bootstrap_project=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError('dry-run must not call api')
            )
        ),
    )

    assert result.plan.task_count == 1
    assert result.project_url == ''
    assert result.task_manifest_path.exists()
    assert 'Dry run' in result.message
    assert result.companion_status == 'hybrid-enabled companion-required'


def test_missing_image_directory_fails_before_side_effects(tmp_path: Path):
    with pytest.raises(BootstrapError, match='Image directory does not exist'):
        plan_bootstrap(images_dir=tmp_path / 'missing')


def test_docker_preflight_starts_installed_docker_desktop_on_macos(monkeypatch):
    calls = []
    attempts = {'docker_info': 0}

    def fake_runner(command):
        calls.append(command)
        if command == ['docker', 'info']:
            attempts['docker_info'] += 1
            return SimpleNamespace(returncode=1 if attempts['docker_info'] == 1 else 0)
        if command == ['open', '-a', 'Docker']:
            return SimpleNamespace(returncode=0)
        raise AssertionError(f'unexpected command: {command}')

    monkeypatch.setattr(
        'eq.labelstudio.bootstrap.shutil.which', lambda name: '/usr/local/bin/docker'
    )
    monkeypatch.setattr('eq.labelstudio.bootstrap.platform.system', lambda: 'Darwin')
    monkeypatch.setattr(
        'eq.labelstudio.bootstrap.DOCKER_DESKTOP_APP_PATH',
        Path('/Applications/Docker.app'),
    )
    monkeypatch.setattr(
        Path, 'exists', lambda self: str(self) == '/Applications/Docker.app'
    )
    monkeypatch.setattr('eq.labelstudio.bootstrap.time.sleep', lambda _seconds: None)

    ensure_docker_available(fake_runner, timeout_seconds=2)

    assert ['open', '-a', 'Docker'] in calls
    assert attempts['docker_info'] == 2


def test_docker_preflight_reports_install_command_when_docker_missing_on_macos(
    monkeypatch,
):
    monkeypatch.setattr('eq.labelstudio.bootstrap.shutil.which', lambda name: None)
    monkeypatch.setattr('eq.labelstudio.bootstrap.platform.system', lambda: 'Darwin')

    with pytest.raises(BootstrapError, match='brew install --cask docker'):
        ensure_docker_available(lambda command: SimpleNamespace(returncode=1))


def test_labelstudio_readiness_retries_remote_disconnect(monkeypatch):
    client = LabelStudioApiClient('http://localhost:8089', 'token')
    calls = {'count': 0}

    def flaky_request(method, path, payload=None):
        calls['count'] += 1
        if calls['count'] == 1:
            raise RemoteDisconnected('Remote end closed connection without response')
        return []

    monkeypatch.setattr(client, '_request_json', flaky_request)
    monkeypatch.setattr('eq.labelstudio.bootstrap.time.sleep', lambda _seconds: None)

    client.wait_until_ready(timeout_seconds=2)

    assert calls['count'] == 2


def test_labelstudio_readiness_retries_connection_reset(monkeypatch):
    client = LabelStudioApiClient('http://localhost:8089', 'token')
    calls = {'count': 0}

    def flaky_request(method, path, payload=None):
        calls['count'] += 1
        if calls['count'] == 1:
            raise ConnectionResetError('connection reset by peer')
        return []

    monkeypatch.setattr(client, '_request_json', flaky_request)
    monkeypatch.setattr('eq.labelstudio.bootstrap.time.sleep', lambda _seconds: None)

    client.wait_until_ready(timeout_seconds=2)
    assert calls['count'] == 2


def test_labelstudio_bootstrap_creates_local_file_storages_before_import(monkeypatch):
    client = LabelStudioApiClient('http://localhost:8089', 'token')
    calls = []

    def fake_request(method, path, payload=None):
        calls.append((method, path, payload))
        if method == 'GET' and path == '/api/projects/':
            return []
        if method == 'POST' and path == '/api/projects/':
            return {'id': 7}
        if method == 'GET' and path == '/api/storages/localfiles/?project=7':
            return []
        if method == 'POST' and path == '/api/storages/localfiles/':
            return {'id': 3}
        if method == 'POST' and path == '/api/projects/7/import':
            return {'task_count': 1}
        if method == 'GET' and path == '/api/tasks/?project=7':
            return {
                'tasks': [
                    {
                        'id': 11,
                        'total_annotations': 0,
                        'data': {'source_relative_path': 'animal_1/a.png'},
                    }
                ]
            }
        if method == 'POST' and path == '/api/tasks/11/annotations/':
            return {'id': 99}
        raise AssertionError(f'unexpected request: {method} {path} {payload}')

    monkeypatch.setattr(client, '_request_json', fake_request)

    project_id = client.bootstrap_project(
        project_title='Demo',
        label_config='<View />',
        tasks=[
            {
                'data': {
                    'image': '/data/local-files/?d=images/animal_1/a.png',
                    'source_relative_path': 'animal_1/a.png',
                }
            }
        ],
        preload_annotation_by_relative_path={
            'animal_1/a.png': {
                'result': [
                    {
                        'id': 'auto_001',
                        'from_name': 'glomerulus_roi',
                        'to_name': 'image',
                        'type': 'brushlabels',
                        'original_width': 32,
                        'original_height': 32,
                        'value': {
                            'format': 'rle',
                            'rle': [0, 1],
                            'brushlabels': ['complete_glomerulus'],
                        },
                    }
                ]
            }
        },
        timeout_seconds=2,
    )

    assert project_id == 7
    assert calls.index(
        (
            'POST',
            '/api/storages/localfiles/',
            {
                'project': 7,
                'path': f'{CONTAINER_LOCAL_FILES_ROOT}/images',
                'title': 'EQ Image Media images',
                'use_blob_urls': True,
                'recursive_scan': True,
            },
        )
    ) < calls.index(
        (
            'POST',
            '/api/projects/7/import',
            [
                {
                    'data': {
                        'image': '/data/local-files/?d=images/animal_1/a.png',
                        'source_relative_path': 'animal_1/a.png',
                    }
                }
            ],
        )
    )
    assert (
        'POST',
        '/api/tasks/11/annotations/',
        {
            'result': [
                {
                    'id': 'auto_001',
                    'from_name': 'glomerulus_roi',
                    'to_name': 'image',
                    'type': 'brushlabels',
                    'original_width': 32,
                    'original_height': 32,
                    'image_rotation': 0,
                    'origin': 'manual',
                    'value': {
                        'format': 'rle',
                        'rle': [0, 1],
                        'brushlabels': ['complete_glomerulus'],
                    },
                }
            ]
        },
    ) in calls
    assert not any(path.startswith('/api/predictions/') for _, path, _ in calls)


def test_labelstudio_bootstrap_replaces_existing_project_when_requested(monkeypatch):
    client = LabelStudioApiClient('http://localhost:8089', 'token')
    calls = []

    def fake_request(method, path, payload=None):
        calls.append((method, path, payload))
        if method == 'GET' and path == '/api/projects/':
            return [{'id': 7, 'title': 'Demo'}]
        if method == 'DELETE' and path == '/api/projects/7/':
            return {}
        if method == 'POST' and path == '/api/projects/':
            return {'id': 8}
        if method == 'GET' and path == '/api/storages/localfiles/?project=8':
            return []
        if method == 'POST' and path == '/api/storages/localfiles/':
            return {'id': 3}
        if method == 'POST' and path == '/api/projects/8/import':
            return {'task_count': 1}
        if method == 'GET' and path == '/api/tasks/?project=8':
            return {'tasks': []}
        raise AssertionError(f'unexpected request: {method} {path} {payload}')

    monkeypatch.setattr(client, '_request_json', fake_request)

    project_id = client.bootstrap_project(
        project_title='Demo',
        label_config='<View />',
        tasks=[
            {
                'data': {
                    'image': '/data/local-files/?d=images/animal_1/a.png',
                    'source_relative_path': 'animal_1/a.png',
                }
            }
        ],
        timeout_seconds=2,
        replace_existing_project=True,
    )

    assert project_id == 8
    assert ('DELETE', '/api/projects/7/', None) in calls
    assert (
        'POST',
        '/api/projects/',
        {'title': 'Demo', 'label_config': '<View />'},
    ) in calls


def test_materialized_preload_skips_existing_annotations(monkeypatch):
    client = LabelStudioApiClient('http://localhost:8089', 'token')
    calls = []

    def fake_request(method, path, payload=None):
        calls.append((method, path, payload))
        if method == 'GET' and path == '/api/tasks/?project=7':
            return {
                'tasks': [
                    {
                        'id': 11,
                        'total_annotations': 1,
                        'annotations': [{'id': 91}],
                        'data': {'source_relative_path': 'animal_1/a.png'},
                    }
                ]
            }
        raise AssertionError(f'unexpected request: {method} {path} {payload}')

    monkeypatch.setattr(client, '_request_json', fake_request)

    client._materialize_preload_annotations(
        7,
        imported_tasks=[{'data': {'source_relative_path': 'animal_1/a.png'}}],
        preload_annotation_by_relative_path={
            'animal_1/a.png': {'result': [{'id': 'auto_001'}]}
        },
    )

    assert not any(path == '/api/tasks/11/annotations/' for _, path, _ in calls)


def test_labelstudio_iter_project_tasks_supports_tasks_key(monkeypatch):
    client = LabelStudioApiClient('http://localhost:8089', 'token')

    def fake_request(method, path, payload=None):
        assert method == 'GET'
        assert path == '/api/tasks/?project=7'
        return {'total': 1, 'tasks': [{'id': 17}]}

    monkeypatch.setattr(client, '_request_json', fake_request)
    tasks = list(client._iter_project_tasks(7))
    assert tasks == [{'id': 17}]


def test_labelstudio_cli_command_forwards_start_options(
    monkeypatch, tmp_path: Path, capsys
):
    import eq.__main__ as main

    captured = {}

    def fake_run_bootstrap(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            message='Dry run: ready',
            project_url='',
            task_manifest_path=tmp_path / 'tasks.json',
            plan=SimpleNamespace(url='http://localhost:9090'),
            companion_status='hybrid-disabled',
        )

    monkeypatch.setattr(main, '_load_labelstudio_bootstrap', lambda: fake_run_bootstrap)

    args = SimpleNamespace(
        image_dir=None,
        images=str(tmp_path / 'images'),
        project_name='My Project',
        runtime_root=str(tmp_path / 'runtime'),
        port=9090,
        container_name='custom-ls',
        docker_image='labelstudio:test',
        username='admin@example.test',
        password='password',
        api_token='token123',
        timeout_seconds=5,
        dry_run=True,
        config=str(tmp_path / 'cfg.yaml'),
        hybrid_mode='auto',
    )

    main.labelstudio_start_command(args)

    output = capsys.readouterr().out
    assert captured['images_dir'] == tmp_path / 'images'
    assert captured['project_name'] == 'My Project'
    assert captured['runtime_root'] == tmp_path / 'runtime'
    assert captured['port'] == 9090
    assert captured['container_name'] == 'custom-ls'
    assert captured['docker_image'] == 'labelstudio:test'
    assert captured['username'] == 'admin@example.test'
    assert captured['password'] == 'password'
    assert captured['api_token'] == 'token123'
    assert captured['timeout_seconds'] == 5
    assert captured['dry_run'] is True
    assert captured['config_path'] == tmp_path / 'cfg.yaml'
    assert captured['hybrid_mode'] == 'auto'
    assert 'Dry run: ready' in output
    assert 'Login email: admin@example.test' in output
    assert 'Login password: password' in output


def test_labelstudio_cli_accepts_positional_image_dir(monkeypatch, tmp_path: Path):
    import eq.__main__ as main

    captured = {}

    def fake_run_bootstrap(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            message='ready',
            project_url='',
            task_manifest_path=tmp_path / 'tasks.json',
            plan=SimpleNamespace(url='http://localhost:9090'),
            companion_status='hybrid-disabled',
        )

    monkeypatch.setattr(main, '_load_labelstudio_bootstrap', lambda: fake_run_bootstrap)
    args = SimpleNamespace(
        image_dir=str(tmp_path / 'images'),
        images=None,
        project_name='My Project',
        runtime_root=str(tmp_path / 'runtime'),
        port=9090,
        container_name='custom-ls',
        docker_image='labelstudio:test',
        username='admin@example.test',
        password='password',
        api_token='token123',
        timeout_seconds=5,
        dry_run=True,
        config=None,
        hybrid_mode='disabled',
    )

    main.labelstudio_start_command(args)
    assert captured['images_dir'] == tmp_path / 'images'


def test_labelstudio_demo_cli_forwards_live_export_only(
    monkeypatch, tmp_path: Path, capsys
):
    import eq.__main__ as main

    captured = {}

    def fake_run_hybrid_demo(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            bootstrap=SimpleNamespace(
                message='Dry run: export refresh',
                project_url='',
                task_manifest_path=tmp_path / 'tasks.json',
                companion_status='hybrid-enabled companion-required',
                plan=SimpleNamespace(url='http://localhost:8080'),
            ),
            acceptance_path=tmp_path / 'acceptance.json',
            acceptance={'project_url': 'http://localhost:8080/projects/9/data'},
        )

    monkeypatch.setattr(
        main,
        '_load_labelstudio_demo',
        lambda: (fake_run_hybrid_demo, 'EQ Hybrid Production Proof'),
    )
    args = SimpleNamespace(
        image_dir=None,
        project_name='EQ Hybrid Production Proof',
        runtime_root=str(tmp_path / 'runtime'),
        port=8080,
        container_name='eq-labelstudio',
        docker_image='heartexlabs/label-studio:1.23.0',
        username='eq-admin@example.local',
        password='eq-labelstudio',
        api_token='eq-local-token',
        timeout_seconds=60,
        dry_run=True,
        config=None,
        hybrid_mode='auto',
        acceptance_dir=str(tmp_path / 'acceptance'),
        annotation_export=None,
        comparison_run_dir=str(tmp_path / 'comparison'),
        comparison_manifest_row_id='row_a',
        live_export_only=True,
    )

    main.labelstudio_demo_hybrid_command(args)

    output = capsys.readouterr().out
    assert captured['live_export_only'] is True
    assert 'Project URL: http://localhost:8080/projects/9/data' in output


def test_hybrid_dry_run_includes_preload_predictions(tmp_path: Path):
    images = tmp_path / 'images'
    img = _write_image(images / 'animal_1' / 'a.png')
    runtime = tmp_path / 'runtime'
    mask_dir = (
        runtime
        / 'derived_data'
        / 'generated_masks'
        / 'glomeruli'
        / 'medsam_finetuned'
        / 'release_a'
        / 'masks'
    )
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / 'mask_a.png'
    Image.new('L', (2, 2), color=255).save(mask_path)
    source_image = runtime / 'raw_data' / 'cohorts' / 'demo' / 'images' / img.name
    _write_image(source_image)
    manifest_dir = runtime / 'derived_data' / 'generated_masks' / 'glomeruli'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    relative_mask_path = mask_path.relative_to(runtime).as_posix()
    relative_release_manifest_path = (
        (mask_dir.parent / 'manifest.csv').relative_to(runtime).as_posix()
    )
    checkpoint_path = runtime / 'models' / 'medsam' / 'checkpoint.pth'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b'checkpoint')
    relative_checkpoint_path = checkpoint_path.relative_to(runtime).as_posix()
    provenance_path = mask_dir.parent / 'provenance.json'
    relative_provenance_path = provenance_path.relative_to(runtime).as_posix()
    provenance_path.write_text('{}', encoding='utf-8')
    manifest_fields = 'schema_version,path_encoding,generated_mask_id,mask_release_id,mask_source,adoption_tier,cohort_id,lane_assignment,source_sample_id,source_image_path,reference_mask_path,generated_mask_path,release_manifest_path,checkpoint_id,checkpoint_path,proposal_source,proposal_threshold,run_id,generation_status,provenance_path'
    (manifest_dir / 'manifest.csv').write_text(
        '\n'.join(
            [
                manifest_fields,
                f'eq.medsam.generated-mask-release/1,runtime-relative-posix,row1,release_a,medsam_finetuned_glomeruli,tier,cohort,lane,sample,{source_image.relative_to(runtime).as_posix()},,{relative_mask_path},{relative_release_manifest_path},ckpt,{relative_checkpoint_path},auto,0.2,run,generated,{relative_provenance_path}',
            ]
        ),
        encoding='utf-8',
    )
    (mask_dir.parent / 'manifest.csv').write_text(
        '\n'.join(
            [
                manifest_fields,
                f'eq.medsam.generated-mask-release/1,runtime-relative-posix,row1,release_a,medsam_finetuned_glomeruli,tier,cohort,lane,sample,{source_image.relative_to(runtime).as_posix()},,{relative_mask_path},{relative_release_manifest_path},ckpt,{relative_checkpoint_path},auto,0.2,run,generated,{relative_provenance_path}',
            ]
        ),
        encoding='utf-8',
    )
    provenance_path.write_text(
        json.dumps(
            {
                'schema_version': 'eq.medsam.generated-mask-release/1',
                'path_encoding': 'runtime-relative-posix',
                'checkpoint_path': relative_checkpoint_path,
                'manifest_path': relative_release_manifest_path,
            }
        ),
        encoding='utf-8',
    )
    cfg = tmp_path / 'hybrid.yaml'
    cfg.write_text(
        'hybrid:\n  enabled: true\n  preload_enabled: true\n  selection_mode: latest_valid\n  require_box_assisted_medsam: false\ncompanion:\n  base_url: "http://localhost:8098"\n  health_path: "/healthz"\n',
        encoding='utf-8',
    )

    result = run_bootstrap(
        images_dir=images,
        runtime_root=runtime,
        dry_run=True,
        config_path=cfg,
        hybrid_mode='enabled',
        docker_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('dry-run must not call docker')
        ),
        api_client=SimpleNamespace(
            bootstrap_project=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError('dry-run must not call api')
            )
        ),
    )
    tasks = json.loads(result.task_manifest_path.read_text(encoding='utf-8'))
    assert tasks[0]['data']['preload_prediction_status'] == 'available'
    assert 'predictions' not in tasks[0]
    assert len(tasks[0]['data']['preload_region_lineage']) == 1


def test_hybrid_preload_resolves_portable_release_paths_from_runtime_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    images = tmp_path / 'images'
    img = _write_real_png(images / 'animal_1' / 'a.png')
    runtime = tmp_path / 'runtime'
    release_dir = (
        runtime
        / 'derived_data'
        / 'generated_masks'
        / 'glomeruli'
        / 'medsam_finetuned'
        / 'release_a'
    )
    mask_path = release_dir / 'masks' / 'mask_a.png'
    _write_real_png(mask_path, value=255)
    source_image = runtime / 'raw_data' / 'cohorts' / 'demo' / 'images' / img.name
    _write_real_png(source_image)
    registry_path = (
        runtime / 'derived_data' / 'generated_masks' / 'glomeruli' / 'manifest.csv'
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    relative_mask = 'derived_data/generated_masks/glomeruli/medsam_finetuned/release_a/masks/mask_a.png'
    relative_manifest = (
        'derived_data/generated_masks/glomeruli/medsam_finetuned/release_a/manifest.csv'
    )
    checkpoint_path = runtime / 'models' / 'medsam' / 'checkpoint.pth'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b'checkpoint')
    relative_checkpoint = checkpoint_path.relative_to(runtime).as_posix()
    provenance_path = release_dir / 'provenance.json'
    relative_provenance = provenance_path.relative_to(runtime).as_posix()
    provenance_path.write_text('{}', encoding='utf-8')
    fields = 'schema_version,path_encoding,generated_mask_id,mask_release_id,mask_source,adoption_tier,cohort_id,lane_assignment,source_sample_id,source_image_path,reference_mask_path,generated_mask_path,release_manifest_path,checkpoint_id,checkpoint_path,proposal_source,proposal_threshold,run_id,generation_status,provenance_path'
    registry_path.write_text(
        '\n'.join(
            [
                fields,
                f'eq.medsam.generated-mask-release/1,runtime-relative-posix,row1,release_a,medsam_finetuned_glomeruli,tier,cohort,lane,sample,{source_image.relative_to(runtime).as_posix()},,{relative_mask},{relative_manifest},ckpt,{relative_checkpoint},auto,0.2,run,generated,{relative_provenance}',
            ]
        ),
        encoding='utf-8',
    )
    (release_dir / 'manifest.csv').write_text(
        '\n'.join(
            [
                fields,
                f'eq.medsam.generated-mask-release/1,runtime-relative-posix,row1,release_a,medsam_finetuned_glomeruli,tier,cohort,lane,sample,{source_image.relative_to(runtime).as_posix()},,{relative_mask},{relative_manifest},ckpt,{relative_checkpoint},auto,0.2,run,generated,{relative_provenance}',
            ]
        ),
        encoding='utf-8',
    )
    provenance_path.write_text(
        json.dumps(
            {
                'schema_version': 'eq.medsam.generated-mask-release/1',
                'path_encoding': 'runtime-relative-posix',
                'checkpoint_path': relative_checkpoint,
                'manifest_path': relative_manifest,
            }
        ),
        encoding='utf-8',
    )
    cfg = tmp_path / 'hybrid.yaml'
    cfg.write_text(
        'hybrid:\n  enabled: true\n  preload_enabled: true\n  selection_mode: latest_valid\n  require_box_assisted_medsam: false\n',
        encoding='utf-8',
    )
    other_cwd = tmp_path / 'another-process-cwd'
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    result = run_bootstrap(
        images_dir=images,
        runtime_root=runtime,
        dry_run=True,
        config_path=cfg,
        hybrid_mode='enabled',
    )

    tasks = json.loads(result.task_manifest_path.read_text(encoding='utf-8'))
    assert tasks[0]['data']['preload_prediction_status'] == 'available'


def test_hybrid_preload_rejects_stale_absolute_release_manifest_paths(tmp_path: Path):
    images = tmp_path / 'images'
    _write_real_png(images / 'animal_1' / 'a.png')
    runtime = tmp_path / 'runtime'
    registry_path = (
        runtime / 'derived_data' / 'generated_masks' / 'glomeruli' / 'manifest.csv'
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        '\n'.join(
            [
                'schema_version,path_encoding,generated_mask_id,mask_release_id,generated_mask_path,release_manifest_path,generation_status',
                'eq.medsam.generated-mask-release/1,runtime-relative-posix,row1,release_a,/stale/mask.png,/stale/manifest.csv,generated',
            ]
        ),
        encoding='utf-8',
    )
    cfg = tmp_path / 'hybrid.yaml'
    cfg.write_text(
        'hybrid:\n  enabled: true\n  preload_enabled: true\n  mask_release_id: "release_a"\n',
        encoding='utf-8',
    )

    with pytest.raises(BootstrapError, match='absolute'):
        run_bootstrap(
            images_dir=images,
            runtime_root=runtime,
            dry_run=True,
            config_path=cfg,
            hybrid_mode='enabled',
        )


def test_release_rows_validate_paths_before_release_id_filter(tmp_path: Path):
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text(
        '\n'.join(
            [
                'schema_version,path_encoding,generated_mask_id,mask_release_id,generated_mask_path,generation_status',
                'eq.medsam.generated-mask-release/1,runtime-relative-posix,row1,,/stale/mask.png,generated',
            ]
        ),
        encoding='utf-8',
    )

    with pytest.raises(BootstrapError, match='absolute'):
        _load_release_rows(manifest, runtime_root=tmp_path / 'runtime')


def test_hybrid_pinned_release_requires_preload_enabled(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'animal_1' / 'a.png')
    cfg = tmp_path / 'hybrid.yaml'
    cfg.write_text(
        'hybrid:\n  enabled: true\n  preload_enabled: false\n  mask_release_id: "release_a"\n',
        encoding='utf-8',
    )

    with pytest.raises(BootstrapError, match='preload_enabled: true'):
        run_bootstrap(
            images_dir=images,
            runtime_root=tmp_path / 'runtime',
            dry_run=True,
            config_path=cfg,
            hybrid_mode='enabled',
        )


def test_hybrid_demo_dry_run_writes_acceptance_artifact(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'animal_1' / 'a.png')

    result = run_hybrid_demo(
        images_dir=images,
        runtime_root=tmp_path / 'runtime',
        project_name='Hybrid Demo',
        dry_run=True,
        acceptance_dir=tmp_path / 'acceptance',
    )

    assert result.acceptance_path.exists()
    assert result.acceptance['dry_run'] is True
    assert result.acceptance['task_count'] == 1
    assert result.acceptance['preload_prediction_count'] == 0
    assert result.acceptance['preload_region_count'] == 0
    assert result.acceptance['preload_status_counts'] == {'missing': 1}
    assert result.acceptance['companion_status'] == 'hybrid-enabled companion-required'
    assert result.acceptance['selected_mask_release_id'] == ''


def test_hybrid_demo_summarizes_annotation_export(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'animal_1' / 'a.png')
    fixture = (
        Path(__file__).resolve().parents[1]
        / 'fixtures'
        / 'labelstudio_glomerulus_instances'
        / 'hybrid_box_assisted_export.json'
    )

    result = run_hybrid_demo(
        images_dir=images,
        runtime_root=tmp_path / 'runtime',
        project_name='Hybrid Demo',
        dry_run=True,
        acceptance_dir=tmp_path / 'acceptance',
        annotation_export=fixture,
    )

    summary = result.acceptance['annotation_export_summary']
    assert summary['scoring_unit'] == 'glomerulus_instance'
    assert summary['rows'] == 1
    assert summary['proposal_kind_counts'] == {'box_assisted_manual': 1}


def test_hybrid_demo_summarizes_multi_roi_annotation_export(tmp_path: Path):
    images = tmp_path / 'images'
    _write_image(images / 'animal_1' / 'a.png')
    fixture = (
        Path(__file__).resolve().parents[1]
        / 'fixtures'
        / 'labelstudio_glomerulus_instances'
        / 'hybrid_multi_roi_live_export.json'
    )

    result = run_hybrid_demo(
        images_dir=images,
        runtime_root=tmp_path / 'runtime',
        project_name='Hybrid Demo',
        dry_run=True,
        acceptance_dir=tmp_path / 'acceptance',
        annotation_export=fixture,
    )

    summary = result.acceptance['annotation_export_summary']
    assert summary['scoring_unit'] == 'glomerulus_instance'
    assert summary['rows'] == 3
    assert summary['proposal_kind_counts'] == {'auto_preload': 3}
    assert summary['region_edit_state_counts'] == {'unedited_auto': 3}


def test_hybrid_demo_live_export_only_refreshes_acceptance_without_replacing_project(
    monkeypatch, tmp_path: Path
):
    from eq.labelstudio import demo as demo_mod

    images = tmp_path / 'images'
    _write_image(images / 'animal_1' / 'a.png')
    runtime = tmp_path / 'runtime'
    export_fixture = (
        Path(__file__).resolve().parents[1]
        / 'fixtures'
        / 'labelstudio_glomerulus_instances'
        / 'hybrid_multi_roi_live_export.json'
    )

    def fake_run_bootstrap(**kwargs):
        assert kwargs['dry_run'] is True
        assert kwargs['replace_existing_project'] is False
        manifest = runtime / 'imports' / 'eq-hybrid-production-proof_tasks.json'
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('[]', encoding='utf-8')
        return SimpleNamespace(
            plan=SimpleNamespace(
                runtime_root=runtime,
                project_title='EQ Hybrid Production Proof',
                url='http://localhost:8080',
                username='eq-admin@example.local',
                task_count=0,
                hybrid_mode='auto',
                selected_mask_release_id='',
                companion_base_url='http://localhost:8098',
                companion_health_path='/healthz',
                companion_required=True,
            ),
            task_manifest_path=manifest,
            project_url='',
            companion_status='hybrid-enabled companion-required',
        )

    class FakeClient:
        def __init__(self, base_url, api_token):
            assert base_url == 'http://localhost:8080'
            assert api_token == 'eq-local-token'

        def find_project_id(self, project_title):
            assert project_title == 'EQ Hybrid Production Proof'
            return 9

        def export_project(self, project_id):
            assert project_id == 9
            return json.loads(export_fixture.read_text(encoding='utf-8'))

    monkeypatch.setattr(demo_mod, 'run_bootstrap', fake_run_bootstrap)
    monkeypatch.setattr(demo_mod, 'LabelStudioApiClient', FakeClient)

    result = run_hybrid_demo(
        images_dir=images,
        runtime_root=runtime,
        project_name='EQ Hybrid Production Proof',
        dry_run=True,
        live_export_only=True,
        acceptance_dir=tmp_path / 'acceptance',
    )

    assert result.acceptance['project_url'] == 'http://localhost:8080/projects/9/data'
    summary = result.acceptance['annotation_export_summary']
    assert summary['rows'] == 3
    assert summary['proposal_kind_counts'] == {'auto_preload': 3}
    assert Path(summary['annotation_export']).exists()


def test_hybrid_demo_comparison_example_writes_real_data_proof(tmp_path: Path):
    image_path = _write_real_png(tmp_path / 'source' / 'row_a.png')
    manual_mask = tmp_path / 'source' / 'row_a_mask.png'
    prediction_mask = (
        tmp_path / 'comparison' / 'medsam_masks' / 'row_a_medsam_oracle_union.png'
    )
    overlay = tmp_path / 'comparison' / 'overlays' / 'row_a_overlay.png'
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    prediction_mask.parent.mkdir(parents=True, exist_ok=True)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode='L').save(manual_mask)
    Image.fromarray(mask, mode='L').save(prediction_mask)
    Image.fromarray(np.full((32, 32, 3), 64, dtype=np.uint8), mode='RGB').save(overlay)
    inputs_csv = tmp_path / 'comparison' / 'inputs.csv'
    inputs_csv.parent.mkdir(parents=True, exist_ok=True)
    inputs_csv.write_text(
        'manifest_row_id,cohort_id,lane_assignment,source_sample_id,image_path,mask_path,image_path_resolved,mask_path_resolved\n'
        f'row_a,cohort_x,manual_mask_core,sample_1,{image_path},{manual_mask},{image_path},{manual_mask}\n',
        encoding='utf-8',
    )
    metrics_csv = tmp_path / 'comparison' / 'metrics.csv'
    metrics_csv.write_text(
        'method,candidate_artifact,manifest_row_id,dice,jaccard,precision,recall,pixel_accuracy,area_ratio,manual_component_count,prediction_component_count\n'
        'medsam_oracle,medsam,row_a,0.95,0.90,0.93,0.97,0.99,1.02,1,1\n'
        'current_segmenter_transfer,transfer,row_a,0.71,0.56,0.70,0.72,0.96,0.98,1,4\n'
        'current_segmenter_scratch,scratch,row_a,0.69,0.53,0.68,0.70,0.95,0.94,1,3\n',
        encoding='utf-8',
    )

    result = run_hybrid_demo(
        images_dir=None,
        runtime_root=tmp_path / 'runtime',
        project_name='Real Proof',
        dry_run=True,
        acceptance_dir=tmp_path / 'acceptance',
        comparison_run_dir=tmp_path / 'comparison',
        comparison_manifest_row_id='row_a',
    )

    assert result.acceptance['task_count'] == 1
    assert result.acceptance['preload_prediction_count'] == 1
    assert result.acceptance['preload_region_count'] == 1
    comparison = result.acceptance['comparison_example']
    assert comparison['manifest_row_id'] == 'row_a'
    assert comparison['cohort_id'] == 'cohort_x'
    assert comparison['prediction_method'] == 'medsam_oracle'
    assert comparison['medsam_beats_all_current_segmenters'] is True
    assert comparison['dice_margin_vs_best_current'] == pytest.approx(0.24)
    assert Path(comparison['source_image_path']).exists()
    tasks = json.loads(result.bootstrap.task_manifest_path.read_text(encoding='utf-8'))
    assert tasks[0]['data']['preload_prediction_status'] == 'available'
    assert tasks[0]['data']['mask_release_id'].startswith(
        'medsam_manual_glomeruli_comparison:comparison:medsam_oracle'
    )


def test_hybrid_demo_comparison_uses_runtime_relative_inputs(
    tmp_path: Path, monkeypatch
):
    runtime_root = tmp_path / 'runtime'
    monkeypatch.setenv('EQ_RUNTIME_ROOT', str(runtime_root))
    image_rel = Path('raw_data/cohorts/cohort_x/images/row_a.png')
    mask_rel = Path('raw_data/cohorts/cohort_x/masks/row_a_mask.png')
    image_path = _write_real_png(runtime_root / image_rel)
    manual_mask = runtime_root / mask_rel
    prediction_mask = (
        tmp_path / 'comparison' / 'medsam_masks' / 'row_a_medsam_oracle_union.png'
    )
    overlay = tmp_path / 'comparison' / 'overlays' / 'row_a_overlay.png'
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    manual_mask.parent.mkdir(parents=True, exist_ok=True)
    prediction_mask.parent.mkdir(parents=True, exist_ok=True)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode='L').save(manual_mask)
    Image.fromarray(mask, mode='L').save(prediction_mask)
    Image.fromarray(np.full((32, 32, 3), 64, dtype=np.uint8), mode='RGB').save(overlay)
    inputs_csv = tmp_path / 'comparison' / 'inputs.csv'
    inputs_csv.parent.mkdir(parents=True, exist_ok=True)
    inputs_csv.write_text(
        'manifest_row_id,cohort_id,lane_assignment,source_sample_id,image_path,mask_path,image_path_resolved,mask_path_resolved\n'
        f'row_a,cohort_x,manual_mask_core,sample_1,{image_rel},{mask_rel},/Users/ncamarda/ProjectsRuntime/stale/image.jpg,/Users/ncamarda/ProjectsRuntime/stale/mask.png\n',
        encoding='utf-8',
    )
    (tmp_path / 'comparison' / 'metrics.csv').write_text(
        'method,candidate_artifact,manifest_row_id,dice,jaccard,precision,recall,pixel_accuracy,area_ratio,manual_component_count,prediction_component_count\n'
        'medsam_oracle,medsam,row_a,0.95,0.90,0.93,0.97,0.99,1.02,1,1\n'
        'current_segmenter_transfer,transfer,row_a,0.71,0.56,0.70,0.72,0.96,0.98,1,4\n'
        'current_segmenter_scratch,scratch,row_a,0.69,0.53,0.68,0.70,0.95,0.94,1,3\n',
        encoding='utf-8',
    )

    result = run_hybrid_demo(
        images_dir=None,
        runtime_root=tmp_path / 'labelstudio',
        project_name='Relative Proof',
        dry_run=True,
        acceptance_dir=tmp_path / 'acceptance',
        comparison_run_dir=tmp_path / 'comparison',
        comparison_manifest_row_id='row_a',
    )

    comparison = result.acceptance['comparison_example']
    assert Path(comparison['source_image_path']).exists()
    assert Path(comparison['manual_mask_path']).exists()
    assert comparison['source_image_path'].endswith('/images/cohort_x/row_a.png')


def test_hybrid_demo_comparison_example_requires_metrics(tmp_path: Path):
    image_path = _write_real_png(tmp_path / 'source' / 'row_a.png')
    manual_mask = tmp_path / 'source' / 'row_a_mask.png'
    prediction_mask = (
        tmp_path / 'comparison' / 'medsam_masks' / 'row_a_medsam_oracle_union.png'
    )
    overlay = tmp_path / 'comparison' / 'overlays' / 'row_a_overlay.png'
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    prediction_mask.parent.mkdir(parents=True, exist_ok=True)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode='L').save(manual_mask)
    Image.fromarray(mask, mode='L').save(prediction_mask)
    Image.fromarray(np.full((32, 32, 3), 64, dtype=np.uint8), mode='RGB').save(overlay)
    inputs_csv = tmp_path / 'comparison' / 'inputs.csv'
    inputs_csv.parent.mkdir(parents=True, exist_ok=True)
    inputs_csv.write_text(
        'manifest_row_id,cohort_id,lane_assignment,source_sample_id,image_path,mask_path,image_path_resolved,mask_path_resolved\n'
        f'row_a,cohort_x,manual_mask_core,sample_1,{image_path},{manual_mask},{image_path},{manual_mask}\n',
        encoding='utf-8',
    )
    metrics_csv = tmp_path / 'comparison' / 'metrics.csv'
    metrics_csv.write_text(
        'method,candidate_artifact,manifest_row_id,dice,jaccard,precision,recall,pixel_accuracy,area_ratio,manual_component_count,prediction_component_count\n'
        'medsam_oracle,medsam,row_a,0.95,0.90,0.93,0.97,0.99,1.02,1,1\n',
        encoding='utf-8',
    )

    with pytest.raises(BootstrapError, match='must include current segmenter rows'):
        run_hybrid_demo(
            images_dir=None,
            runtime_root=tmp_path / 'runtime',
            project_name='Real Proof',
            dry_run=True,
            acceptance_dir=tmp_path / 'acceptance',
            comparison_run_dir=tmp_path / 'comparison',
            comparison_manifest_row_id='row_a',
        )


def test_hybrid_demo_multi_roi_comparison_counts_regions(tmp_path: Path):
    image_path = _write_real_png(tmp_path / 'source' / 'row_a.png')
    manual_mask = tmp_path / 'source' / 'row_a_mask.png'
    prediction_mask = (
        tmp_path / 'comparison' / 'medsam_masks' / 'row_a_medsam_oracle_union.png'
    )
    overlay = tmp_path / 'comparison' / 'overlays' / 'row_a_overlay.png'
    mask = np.zeros((160, 160), dtype=np.uint8)
    mask[5:45, 5:45] = 255
    mask[55:95, 55:95] = 255
    mask[105:145, 20:65] = 255
    prediction_mask.parent.mkdir(parents=True, exist_ok=True)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode='L').save(manual_mask)
    Image.fromarray(mask, mode='L').save(prediction_mask)
    Image.fromarray(np.full((160, 160, 3), 64, dtype=np.uint8), mode='RGB').save(
        overlay
    )
    inputs_csv = tmp_path / 'comparison' / 'inputs.csv'
    inputs_csv.parent.mkdir(parents=True, exist_ok=True)
    inputs_csv.write_text(
        'manifest_row_id,cohort_id,lane_assignment,source_sample_id,image_path,mask_path,image_path_resolved,mask_path_resolved\n'
        f'row_a,cohort_x,manual_mask_core,sample_1,{image_path},{manual_mask},{image_path},{manual_mask}\n',
        encoding='utf-8',
    )
    metrics_csv = tmp_path / 'comparison' / 'metrics.csv'
    metrics_csv.write_text(
        'method,candidate_artifact,manifest_row_id,dice,jaccard,precision,recall,pixel_accuracy,area_ratio,manual_component_count,prediction_component_count\n'
        'medsam_oracle,medsam,row_a,0.95,0.90,0.93,0.97,0.99,1.02,3,3\n'
        'current_segmenter_transfer,transfer,row_a,0.71,0.56,0.70,0.72,0.96,0.98,3,4\n'
        'current_segmenter_scratch,scratch,row_a,0.69,0.53,0.68,0.70,0.95,0.94,3,3\n',
        encoding='utf-8',
    )

    result = run_hybrid_demo(
        images_dir=None,
        runtime_root=tmp_path / 'runtime',
        project_name='Multi ROI Proof',
        dry_run=True,
        acceptance_dir=tmp_path / 'acceptance',
        comparison_run_dir=tmp_path / 'comparison',
        comparison_manifest_row_id='row_a',
    )

    assert result.acceptance['preload_prediction_count'] == 1
    assert result.acceptance['preload_region_count'] == 3
    tasks = json.loads(result.bootstrap.task_manifest_path.read_text(encoding='utf-8'))
    assert 'predictions' not in tasks[0]
    assert len(tasks[0]['data']['preload_region_lineage']) == 3
