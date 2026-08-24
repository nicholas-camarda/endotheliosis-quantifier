# Endotheliosis Quantifier

Endotheliosis Quantifier (`eq`) is a research pipeline for glomeruli segmentation and image-level endotheliosis review triage in kidney histology.

The current usable endpoint is a binary review-triage workflow:

- `no_low`: score `0` or `0.5`
- `moderate_severe`: score `1.5`, `2`, or `3`
- `borderline_review`: score `1.0`, routed for review

The model is useful for prioritizing human review of no/low versus moderate/severe cases. It is not a clinical diagnostic device, an autonomous grader, or external validation evidence.


## Quick Start

Install the package in the environment that matches the machine.

Linux/CUDA:

```bash
mamba env create -f environment.yml
conda activate eq
pip install -e .[dev]
```

macOS Apple Silicon/MPS:

```bash
mamba env create -f environment-macos.yml
conda activate eq-mac
pip install -e .[dev]
```

Check that the CLI imports:

```bash
python -m eq --help
eq capabilities
eq mode --show
```

Raw cohort inputs belong under `${EQ_RUNTIME_ROOT}/raw_data/cohorts`; generated
data, models, logs, and outputs remain under the configured workspace runtime
rather than the source checkout.

## Start A Label Studio Grading Project

Use this when collaborators need to grade complete glomeruli from a directory of images. Docker Desktop is required because Label Studio runs as a separate local web app, not inside the `eq-mac` environment.

macOS setup:

```bash
brew install --cask docker
open -a Docker
```

Wait for Docker Desktop to finish starting. The current hybrid demo path uses box-assist/manual-first setup by default.

Set these variables to the local source and runtime locations before running the
commands below:

```bash
export EQ_SOURCE_ROOT=/path/to/endotheliosis-quantifier/source
export EQ_RUNTIME_ROOT=/path/to/endotheliosis-quantifier/runtime
```

Terminal 1 starts the MedSAM companion on the Mac host so MPS/Metal is available:

```bash
cd "$EQ_SOURCE_ROOT"
conda activate eq-mac
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m eq.labelstudio.medsam_companion \
  --checkpoint "$EQ_RUNTIME_ROOT/output/segmentation_evaluation/medsam_glomeruli_fine_tuning/deploy_conservative_mps_glomeruli/finetuned_evaluation/medsam_glomeruli_best_sam_state_dict.pth" \
  --device mps \
  --port 8098
```

Terminal 2 creates or refreshes the Label Studio project:

```bash
cd "$EQ_SOURCE_ROOT"
conda activate eq-mac
eq labelstudio demo-hybrid \
  --comparison-run-dir "$EQ_RUNTIME_ROOT/output/segmentation_evaluation/medsam_manual_glomeruli_comparison/pilot_medsam_manual_glomeruli_comparison" \
  --comparison-manifest-row-id lauren_preeclampsia__000009 \
  --project-name "EQ Hybrid Production Proof"
```

The command stages the real source image for that comparison row, preloads the exact MedSAM mask that was scored against the manual mask, creates a local Label Studio project with `configs/label_studio_glomerulus_grading.xml`, applies hybrid bootstrap settings from `configs/label_studio_medsam_hybrid.yaml` (or `--config` override), and writes `<runtime-root>/labelstudio/acceptance/<project>_hybrid_demo_acceptance.json`. The default config keeps full-image preload masks off and requires the companion for box-assisted MedSAM. It prints the Label Studio URL, project URL, acceptance artifact path, and hybrid companion status.

Reuse the same `--project-name` each time. `demo-hybrid` replaces the existing project with that title instead of accumulating more demo projects. Open the `Project URL` printed by the command; do not hardcode the numeric project id.

Default local login:

```text
Email: eq-admin@example.local
Password: eq-labelstudio
```

Preview without starting Docker or importing tasks:

```bash
eq labelstudio demo-hybrid \
  --dry-run \
  --comparison-run-dir "$EQ_RUNTIME_ROOT/output/segmentation_evaluation/medsam_manual_glomeruli_comparison/pilot_medsam_manual_glomeruli_comparison" \
  --comparison-manifest-row-id lauren_preeclampsia__000009
```

After exporting completed Label Studio annotations, attach lineage evidence to the acceptance artifact:

```bash
eq labelstudio demo-hybrid /path/to/images --dry-run --annotation-export /path/to/export.json
```

To refresh the canonical proof bundle from the current live project without replacing it:

```bash
eq labelstudio demo-hybrid \
  --dry-run \
  --live-export-only \
  --comparison-run-dir "$EQ_RUNTIME_ROOT/output/segmentation_evaluation/medsam_manual_glomeruli_comparison/pilot_medsam_manual_glomeruli_comparison" \
  --comparison-manifest-row-id vegfri_dox__000528 \
  --project-name "EQ Hybrid Production Proof"
```

That command fetches the current Label Studio export for the named project, rewrites the canonical acceptance artifact, and leaves the existing project in place.

Legacy automation remains valid:

```bash
eq labelstudio start --images /path/to/images
```

For details, see [docs/LABEL_STUDIO_GLOMERULUS_GRADING.md](docs/LABEL_STUDIO_GLOMERULUS_GRADING.md).
For MedSAM box-assist companion launch/contract, see [docs/LABEL_STUDIO_MEDSAM_COMPANION.md](docs/LABEL_STUDIO_MEDSAM_COMPANION.md).

### Current Hybrid Status

The current operator-safe posture is box-assist/manual-first. Full-image auto-preload is available only when `hybrid.preload_enabled: true` is set in YAML and should be used only for a release that passes target-image instance-quality checks. The active deploy release `deploy_conservative_mps_glomeruli` is useful evaluation evidence, but its full-image masks are not the default operator demo because they can produce noisy regions on some demo images.

Local login:

```text
Email: eq-admin@example.local
Password: eq-labelstudio
```

Suggested annotation mode while iterating:

1. Keep `Compare All` off.
2. Work only in `eq-admin_local`.
3. Correct masks with brush/eraser, delete obvious false positives, and relabel cutoffs as `cutoff_partial_glomerulus`.
4. Grade complete glomeruli per-region.

Export and run quant contract-first check:

```bash
conda activate eq-mac
eq run-config --config configs/endotheliosis_quantification.yaml
```

If you need to refresh the canonical demo project after code changes, rerun:

```bash
eq labelstudio demo-hybrid /tmp/eq_hybrid_demo/images --project-name "EQ Hybrid Production Proof"
```

## MedSAM glomerulus fine-tuning (domain adaptation)

YAML-first workflow: trains/fine-tunes MedSAM glomeruli masks on admitted manual-mask cohort rows, evaluates on fixed splits, optionally **packages** prediction masks into `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` when fine-tuned evaluation completes.

Operator handoff (paths, logs, gates, mask releases):

- [docs/MEDSAM_GLOMERULI_FINETUNING_HANDOFF.md](docs/MEDSAM_GLOMERULI_FINETUNING_HANDOFF.md)

Quick dry-run:

```bash
eq run-config --config configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml --dry-run
```

## Run The Current Quantification Workflow

The main entrypoint is YAML-first:

```bash
eq run-config --config configs/endotheliosis_quantification.yaml
eq run-config --config configs/label_free_roi_embedding_atlas.yaml
```

The first command builds the scored ROI, embedding, burden, comparator, learned-ROI, and review artifact tree. The second command builds the label-free ROI embedding atlas and the binary no/low versus moderate/severe triage handoff.

The committed configs expect the runtime root to contain the required data, masks, labels, and model artifacts. Use `EQ_RUNTIME_ROOT` or edit the YAML when running against a different runtime tree.

Authoritative grading loop for reruns:

1. Finalize Label Studio export (image-level legacy or per-glomerulus hybrid export).
2. Rebuild scored examples through contract-first quantification (`eq run-config --config ...`).
3. Review `${EQ_RUNTIME_ROOT}/output/quantification_results/<run>/scored_examples/lineage_summary.json` to confirm which grading snapshot and scoring unit were consumed.
4. Compare new burden/quant outputs against prior run roots before promoting decisions.

## Review The Result

Open the atlas output in this order:

1. `${EQ_RUNTIME_ROOT}/output/quantification_results/<run>/burden_model/embedding_atlas/INDEX.md`
2. `${EQ_RUNTIME_ROOT}/output/quantification_results/<run>/burden_model/embedding_atlas/summary/atlas_verdict.json`
3. `${EQ_RUNTIME_ROOT}/output/quantification_results/<run>/burden_model/embedding_atlas/evidence/embedding_atlas_review.html`
4. `${EQ_RUNTIME_ROOT}/output/quantification_results/<run>/burden_model/embedding_atlas/binary_review_triage/INDEX.md`
5. `${EQ_RUNTIME_ROOT}/output/quantification_results/<run>/burden_model/embedding_atlas/binary_review_triage/evidence/binary_triage_review.html`

The binary review HTML is a bounded QA sample, not a request to review every prediction row. Review the cards shown, export the CSV beside the HTML file, and inspect the full prediction table only if the sample reveals a systematic failure pattern.

## Current Result

Current selected triage model:

```text
roi_qc_binary_logistic
```

Current grouped-development measurements and cohort support remain runtime
evidence, not README claims. Review the generated atlas summary and the
reproducibility handoff before interpreting or reporting performance.

## Current Segmentation Artifacts

Hybrid Label Studio uses the following canonical runtime artifacts:

| Role | Model / artifact | Where it lives |
| --- | --- | --- |
| Live box-assist inference | Segment Anything **ViT-B** (`vit_b`) weights loaded from the deploy-run evaluation checkpoint | `${EQ_RUNTIME_ROOT}/output/segmentation_evaluation/medsam_glomeruli_fine_tuning/deploy_conservative_mps_glomeruli/finetuned_evaluation/medsam_glomeruli_best_sam_state_dict.pth` |
| Full-image preload masks | Fine-tuned MedSAM mask release `deploy_conservative_mps_glomeruli` (`mask_source=medsam_finetuned_glomeruli`) | `${EQ_RUNTIME_ROOT}/derived_data/generated_masks/glomeruli/medsam_finetuned/deploy_conservative_mps_glomeruli/` (see `manifest.csv`) |

The default operator posture remains box-assist/manual-first. Full-image preload
requires a quality-gated release. Legacy transfer/scratch pickles are evaluation
artifacts unless a separate promotion audit certifies them.

For the full checkpoint and release policy, see [docs/REPRODUCIBILITY_HANDOFF_2026-04-30.md](docs/REPRODUCIBILITY_HANDOFF_2026-04-30.md).

## Other Maintained Workflows

All maintained workflows use the same `eq run-config` entrypoint.


| Task                                                   | Config                                                              |
| ------------------------------------------------------ | ------------------------------------------------------------------- |
| Endotheliosis quantification                           | `configs/endotheliosis_quantification.yaml`                         |
| Glomeruli candidate comparison                         | `configs/glomeruli_candidate_comparison.yaml`                       |
| Glomeruli fine-tuning                                  | `configs/glomeruli_finetuning_config.yaml`                          |
| Glomeruli transport audit                              | `configs/glomeruli_transport_audit.yaml`                            |
| High-resolution concordance                            | `configs/highres_glomeruli_concordance.yaml`                        |
| Label-free atlas and binary triage                     | `configs/label_free_roi_embedding_atlas.yaml`                       |
| MedSAM glomeruli fine-tuning (conservative MPS deploy) | `configs/medsam_glomeruli_fine_tuning_deploy_conservative_mps.yaml` |
| MedSAM glomeruli fine-tuning (pilot)                   | `configs/medsam_glomeruli_fine_tuning.yaml`                         |
| Mitochondria pretraining                               | `configs/mito_pretraining_config.yaml`                              |


Use `--dry-run` before long-running training or audit jobs:

```bash
eq run-config --config configs/glomeruli_candidate_comparison.yaml --dry-run
```

## Repository Layout

```text
configs/      Runnable workflow YAMLs
docs/         Supporting guides, methods notes, and handoff docs
openspec/     Spec-driven change history and current contracts
src/eq/       Python package and CLI implementation
tests/        Unit and integration tests
assets/       Public-safe images for README/docs
```

Large data, trained models, logs, generated reports, notebooks, and review exports stay out of Git. Runtime paths are configured through `EQ_RUNTIME_ROOT`, `analysis_registry.yaml`, and `src/eq/utils/paths.py`.

## Documentation Map

- [docs/BINARY_REVIEW_TRIAGE_GUIDE.md](docs/BINARY_REVIEW_TRIAGE_GUIDE.md): how to review the binary triage HTML and interpret its dropdowns.
- [docs/REPRODUCIBILITY_HANDOFF_2026-04-30.md](docs/REPRODUCIBILITY_HANDOFF_2026-04-30.md): source checkpoint, metrics, release policy, and resume plan.
- [docs/ONBOARDING_GUIDE.md](docs/ONBOARDING_GUIDE.md): longer walkthrough for collaborators and future-you.
- [docs/OUTPUT_STRUCTURE.md](docs/OUTPUT_STRUCTURE.md): runtime directory layout and artifact locations.
- [docs/SEGMENTATION_ENGINEERING_GUIDE.md](docs/SEGMENTATION_ENGINEERING_GUIDE.md): segmentation engineering details.
- [docs/TECHNICAL_LAB_NOTEBOOK.md](docs/TECHNICAL_LAB_NOTEBOOK.md): detailed lab notes and current internal evidence.
- [docs/HISTORICAL_NOTES.md](docs/HISTORICAL_NOTES.md): archived implementation history.

## Development Checks

Run before committing:

```bash
ruff check .
python -m pytest -q
openspec validate --specs --strict
```
