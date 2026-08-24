# Label Studio Glomerulus-Instance Grading

This document describes the Stage 1 Label Studio grading contract for complete glomeruli. It is a data-collection and validation contract, not a model-assisted second-review workflow.

## Default Development Startup

Use this two-terminal path for hybrid MedSAM + Label Studio development. Label Studio runs in Docker; the MedSAM companion runs on the Mac host so MPS/Metal works.

Set `EQ_SOURCE_ROOT` to the source checkout and `EQ_RUNTIME_ROOT` to the active
runtime tree before running the commands below.

Terminal 1:

```bash
cd "$EQ_SOURCE_ROOT"
conda activate eq-mac
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m eq.labelstudio.medsam_companion \
  --checkpoint "$EQ_RUNTIME_ROOT/output/segmentation_evaluation/medsam_glomeruli_fine_tuning/deploy_conservative_mps_glomeruli/finetuned_evaluation/medsam_glomeruli_best_sam_state_dict.pth" \
  --device mps \
  --port 8098
```

Terminal 2:

```bash
cd "$EQ_SOURCE_ROOT"
conda activate eq-mac
eq labelstudio demo-hybrid \
  --comparison-run-dir "$EQ_RUNTIME_ROOT/output/segmentation_evaluation/medsam_manual_glomeruli_comparison/pilot_medsam_manual_glomeruli_comparison" \
  --comparison-manifest-row-id lauren_preeclampsia__000009 \
  --project-name "EQ Hybrid Production Proof"
```

Open the `Project URL` printed by `eq labelstudio demo-hybrid`. Reuse the same `--project-name` each time; the command replaces the existing project with that title instead of accumulating more demo projects. Do not hardcode the numeric project id. The command also prints an acceptance artifact path containing the task manifest, preload status, companion policy, project URL, exact `manifest_row_id`, and the real MedSAM-versus-ground-truth proof fields used for the demo.

For arbitrary image folders that are not tied to a proof row, pass an image directory directly:

```bash
eq labelstudio demo-hybrid /path/to/images --project-name "Kidney Glomerulus Grading"
```

The command recursively imports `.jpg`, `.jpeg`, `.png`, `.tif`, and `.tiff` files, starts a local Docker Label Studio instance, creates or reuses the named project, applies `configs/label_studio_glomerulus_grading.xml`, imports the image tasks, and prints the Label Studio URL plus the project URL.

Hybrid preload and companion controls are YAML-first:

- Default config: `configs/label_studio_medsam_hybrid.yaml`
- Optional override: `--config /path/to/label_studio_medsam_hybrid.yaml`
- Admin override: `--hybrid-mode auto|enabled|disabled`
- Default posture: `hybrid.preload_enabled: false`, so the demo starts clean and uses box-assist/manual brush rather than noisy full-image preload masks.

For companion launch and HTTP contract details, see `docs/LABEL_STUDIO_MEDSAM_COMPANION.md`.

On macOS, if Docker Desktop is installed but not running, the command tries to start it. If Docker Desktop is missing, install it first:

```bash
brew install --cask docker
open -a Docker
```

The default local Docker login is:

```text
Email: eq-admin@example.local
Password: eq-labelstudio
```

These credentials belong to the `eq` Docker-backed Label Studio instance. A separate conda `label-studio` environment uses a different data directory and cannot reset this Docker instance's password.

To inspect the plan without starting Docker or calling Label Studio:

```bash
eq labelstudio demo-hybrid /path/to/images --dry-run
```

To inspect the real-data proof path without starting Docker:

```bash
eq labelstudio demo-hybrid \
  --dry-run \
  --comparison-run-dir "$EQ_RUNTIME_ROOT/output/segmentation_evaluation/medsam_manual_glomeruli_comparison/pilot_medsam_manual_glomeruli_comparison" \
  --comparison-manifest-row-id lauren_preeclampsia__000009
```

After completing and exporting annotations, validate the export and attach lineage evidence:

```bash
eq labelstudio demo-hybrid /path/to/images --dry-run --annotation-export /path/to/export.json
```

To refresh the canonical production-proof bundle from the current live project without replacing it:

```bash
eq labelstudio demo-hybrid \
  --dry-run \
  --live-export-only \
  --comparison-run-dir "$EQ_RUNTIME_ROOT/output/segmentation_evaluation/medsam_manual_glomeruli_comparison/pilot_medsam_manual_glomeruli_comparison" \
  --comparison-manifest-row-id vegfri_dox__000528 \
  --project-name "EQ Hybrid Production Proof"
```

This fetches the current export for the named live project, rewrites the canonical acceptance artifact, and keeps the existing project intact.

Useful options:

```bash
eq labelstudio start \
  /path/to/images \
  --project-name "Kidney Glomerulus Grading" \
  --port 8080 \
  --runtime-root "$EQ_RUNTIME_ROOT/labelstudio"
```

Legacy `--images /path/to/images` remains supported for automation compatibility.

The command is local/admin bootstrap only. With explicit preload enabled, it materializes existing mask-release proposals as editable brush annotations with task-level lineage, and requires the companion for box-assist health when hybrid mode is enabled; it does not train segmentation, run quantification, run model second review, or adjudicate grades.

## Purpose

The lab workflow should grade complete glomeruli, not whole images. Each human grade must be linked to the exact glomerulus region it describes so `eq` can later support second review, adjudication, grader variability analysis, and animal-level rollups.

Historical image-level scores remain useful as legacy baseline data. They must not be treated as per-glomerulus ground truth because the assignment between each grade and each glomerulus was not preserved.

## Collaborator Workflow

Collaborators use Label Studio only:

1. Open the assigned image in Label Studio.
2. Mark each complete glomerulus with the `complete_glomerulus` region label.
3. Select that region and assign exactly one `endotheliosis_grade`.
4. Mark cut-off or partial glomeruli with `cutoff_partial_glomerulus`.
5. Do not grade cut-off or partial glomeruli.
6. Submit the task.

Primary grading is human-first and model-blind. The primary Label Studio config intentionally does not expose model grade, confidence, disagreement, or decision-state fields.

## Admin And Developer Boundary

Use `eq labelstudio demo-hybrid --comparison-run-dir ... --comparison-manifest-row-id ...` as the primary demonstrable setup path because it writes acceptance evidence tied to a real evaluated MedSAM row. Use `eq labelstudio demo-hybrid /path/to/images` when you want the same wrapper on an arbitrary image folder. Use `eq labelstudio start /path/to/images` when you only need the reusable bootstrap primitive. Use `configs/label_studio_glomerulus_grading.xml` only when inspecting or manually debugging the Label Studio config. Use `configs/label_studio_medsam_hybrid.yaml` for mask-release selection, companion URL/health policy, and fail-closed preload behavior.

By default, bootstrap runtime artifacts live under:

```text
<active-runtime-root>/labelstudio/
```

That runtime tree contains Label Studio data, generated import manifests, and bootstrap metadata. Keep those generated artifacts out of Git.

The initial ingestion authority is a Label Studio JSON export. The repo-owned parser lives in `src/eq/labelstudio/glomerulus_grading.py` and validates that:

- every complete glomerulus grade is linked to one region ID
- every complete glomerulus has exactly one grade from each grader pass
- cut-off or partial glomeruli have no grade
- named grader provenance is present from Label Studio annotation identity
- legacy image-level average scores are rejected for this contract

The parser emits glomerulus-level records and rollup-ready records in memory. Persist generated CSVs under configured runtime or output roots when adding command wrappers; do not commit raw Label Studio exports or generated grading outputs.

## Record Identity

The atomic unit is:

```text
image_id + glomerulus_instance_id
```

The stable source record ID used by the Stage 1 parser is:

```text
image_id::glomerulus_instance_id::annotation_id::grader_user_id
```

Rollups must preserve this source record ID so image, kidney, or animal averages can be traced back to included complete glomeruli.

## Not Yet Implemented

The following are intentionally deferred to later OpenSpec changes:

- model second-review inference
- second-review or adjudication Label Studio project creation
- production hosted Label Studio ML backend deployment
- collaborator-visible model suggestions during primary grading
