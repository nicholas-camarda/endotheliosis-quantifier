## 1. Spec and contract foundation

- [x] 1.1 Audit Label Studio image version + JSON export fields for predictions vs annotations (document findings in `implementation-notes.md`)
- [x] 1.2 Audit available generated-mask releases in `derived_data/generated_masks/glomeruli/manifest.csv` and document latest-valid selection criteria
- [x] 1.3 Finalize `configs/label_studio_medsam_hybrid.yaml` schema (release selection mode + optional pinned mask release, companion URLs, enforcement flags) and add example file to Git
- [x] 1.4 Add path helpers in `src/eq/utils/paths.py` for YAML + generated-mask release resolution (reuse existing helpers when present)

## 2. CLI ergonomics

- [x] 2.1 Extend `src/eq/__main__.py` `labelstudio start` parser with positional `<image-dir>` while keeping `--images`
- [x] 2.2 Thread parsed config + hybrid flags into `run_bootstrap`

## 3. Bootstrap preload + health

- [x] 3.1 Implement mask release manifest reader + preload proposal builder in `src/eq/labelstudio/bootstrap.py` or dedicated helper module justified in notes
- [x] 3.2 Wire Local Files + Label Studio API import of hybrid tasks with task-level lineage and API-created editable brush annotations
- [x] 3.3 Add companion health probe with fail-closed default and admin-only bypass flag surfaced via YAML
- [x] 3.4 Update README + `docs/LABEL_STUDIO_GLOMERULUS_GRADING.md` with single-line collaborator command and pointer to YAML knobs

## 4. Export ingestion + validation

- [x] 4.1 Extend `src/eq/labelstudio/glomerulus_grading.py` to emit hybrid lineage columns and enforce validator rules
- [x] 4.2 Add fixtures under `tests/fixtures/labelstudio_glomerulus_instances/` covering auto preload, manual refine, box-assisted, contradictory lineage failure
- [x] 4.3 Expand unit tests for parser + validator edge cases

## 5. Companion service (thin wrapper)

- [x] 5.1 Define HTTP contract + README section for companion (separate small service or script) with MedSAM invocation details
- [x] 5.2 Provide docker-compose fragment or documented launch steps keeping secrets out of Git

## 6. Validation gates

- [x] 6.1 Run `openspec validate label-studio-medsam-hybrid-grading --strict`
- [x] 6.2 Run `python3 scripts/check_openspec_explicitness.py label-studio-medsam-hybrid-grading`
- [x] 6.3 Run focused pytest `tests/unit/test_labelstudio_*` and `ruff check` on touched modules

## 7. Product-readiness amendment

- [x] 7.1 Make the default hybrid config box-assist/manual-first by adding `hybrid.preload_enabled: false`; require explicit YAML opt-in before full-image preload masks are shown to operators.
- [x] 7.2 Remove committed agent-debug instrumentation and hardcoded `.cursor` log writes from Label Studio/MedSAM production modules.
- [x] 7.3 Update README and Label Studio grading docs so the current demo path describes what actually works and does not advertise noisy preload masks as the default product.
- [x] 7.4 Add and pass a focused test proving a pinned `mask_release_id` is rejected when preload is disabled, while explicit preload-enabled configs still materialize release proposals as editable annotations.
- [x] 7.5 Produce a clean demo artifact or command transcript: companion health, `eq labelstudio start <image-dir> --dry-run` or live project creation, and exported per-glomerulus lineage ingestion when available.

## 8. Finished-product demo wrapper

- [x] 8.1 Add `eq labelstudio demo-hybrid <image-dir>` as the canonical showable workflow command; it must reuse `run_bootstrap` rather than duplicate Docker, Label Studio, or config logic.
- [x] 8.2 Write an acceptance artifact JSON containing command mode, task manifest, task count, preload count, companion status, Label Studio URL/project URL when live, selected mask release, config path, and operator next steps.
- [x] 8.3 Support `--annotation-export <export.json>` so a completed Label Studio export can be parsed through `load_glomerulus_grading_records` and summarized by rows, scoring unit, proposal-kind counts, edit-state counts, and mask-release counts.
- [x] 8.4 Add unit tests for dry-run acceptance artifact generation, preload-off default status, and optional export lineage summary.
- [x] 8.5 Update README and `docs/LABEL_STUDIO_GLOMERULUS_GRADING.md` to make `eq labelstudio demo-hybrid <image-dir>` the demonstrable product path.
- [x] 8.6 Run OpenSpec validation, explicitness check, focused Label Studio/demo tests, ruff on touched modules, and one demo-hybrid dry run on the runtime demo image directory.

## 9. Real-data proof path

- [x] 9.1 Extend `eq labelstudio demo-hybrid` so it can stage one admitted manual-mask comparison row from `medsam_manual_glomeruli_comparison` using `--comparison-run-dir` plus `--comparison-manifest-row-id`, without introducing a second bootstrap implementation.
- [x] 9.2 Reuse the existing MedSAM/manual comparison artifact contract (`inputs.csv`, `metrics.csv`, `medsam_masks/`, `overlays/`) to build a single-image demo case and preload the same MedSAM prediction that was scored against ground truth.
- [x] 9.3 Add acceptance-artifact proof fields naming the exact row, source image, manual mask, MedSAM mask, overlay, and same-row comparator metrics, including a boolean summary of whether MedSAM beat the current segmenters on that example.
- [x] 9.4 Add focused unit tests for real-data comparison staging, proof-field generation, and invalid comparison-row failure handling.
- [x] 9.5 Run one real-data `demo-hybrid --dry-run --comparison-run-dir ... --comparison-manifest-row-id ...` example and confirm the acceptance artifact references the actual runtime evidence.
- [x] 9.6 Prove a fresh live multi-ROI comparison row renders as separate MedSAM brush regions in Label Studio and exports back to three authoritative glomerulus-instance rows with preserved `auto_preload` lineage.
- [x] 9.7 Add repo regression coverage from the fresh multi-ROI export fixture so disconnected-component grading and `parentID`-linked score parsing remain locked.
