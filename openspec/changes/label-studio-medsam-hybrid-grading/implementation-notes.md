# label-studio-medsam-hybrid-grading implementation notes

## Task 1.1 audit: Label Studio version and export contract

Date: 2026-05-01  
Host: macOS (`eq-mac` development context)

### Audited image/version

- Local image tag previously used by bootstrap: `heartexlabs/label-studio:latest`
- Image metadata inspection:
  - `org.opencontainers.image.version = 1.23.0`
  - `org.opencontainers.image.revision = 2a9bfbcbf0a844b999de97e601d16050a893f5fb`
- Pin decision for this change:
  - `DEFAULT_DOCKER_IMAGE = heartexlabs/label-studio:1.23.0`
  - CLI default `--docker-image` updated to `heartexlabs/label-studio:1.23.0`

### API behavior audited (prediction vs annotation)

Using a temporary local container (`localhost:8099`) with token auth:

1. Create project with `Image + BrushLabels + Choices` label config.
2. Import task payload containing `data` plus `predictions`:
   - prediction `result` item uses:
     - `type = brushlabels`
     - `from_name = glomerulus_roi`
     - `to_name = image`
     - `value.format = rle`
     - `value.rle = [...]`
     - `value.brushlabels = [...]`
3. Verified import response includes `prediction_count = 1`.
4. Verified task retrieval includes top-level `predictions` list.
5. Posted a matching annotation for the imported prediction and exported project JSON.
6. Verified export task includes both:
   - `predictions` (preloaded model proposals)
   - `annotations` (human/submitted records)
7. Verified exported annotation rows include lineage hooks:
   - `parent_prediction`
   - `parent_annotation`
   - `result` entries with `brushlabels` geometry and `choices` grade link by shared `id`.

### Contract decision from audit

- Hybrid preload is encoded as task-level lineage plus editable brush annotations created through the Label Studio annotation API. It is not imported through task `predictions`, because prediction-linked task state does not behave like native editable annotations during region selection.
- Grader output remains authoritative in `annotations`.
- Export parsers rely on `annotations` as authoritative for scored rows while preserving auto-preload lineage from `data.preload_region_lineage`.

### Follow-on implications for tasks 3.x/4.x

- Task builder should generate Label Studio-compatible `brushlabels` + RLE annotation payloads and store preload lineage in task data.
- Export validation should require coherent lineage when a row claims preload/box-assist provenance.
- Parser logic should preserve `parent_prediction`, annotation ids, and region ids for training/audit traceability.

## Task 1.2 audit: generated-mask registry and latest-valid selection

Source audited: `derived_data/generated_masks/glomeruli/manifest.csv`

Observed contract fields:

- `mask_release_id`
- `generated_mask_path`
- `generation_status`
- `mask_source`
- `release_manifest_path`
- `provenance_path`

Current local rows are dominated by release `deploy_conservative_mps_glomeruli` with `generation_status=generated` and `mask_source=medsam_finetuned_glomeruli`.

Selection criteria implemented for `selection_mode: latest_valid`:

1. Group central manifest rows by `mask_release_id`.
2. Within each release, count rows where:
   - `generation_status == generated`, and
   - `generated_mask_path` exists on disk.
3. Compute release recency by max filesystem mtime of existing generated masks.
4. Select release by descending tuple:
   - `generated_count`
   - `latest_mtime`
   - `mask_release_id` (tie-break)
5. If YAML pins `mask_release_id`, fail closed when absent from registry.

Rationale:

- Works with current manifest schema without assuming an extra `created_at` column.
- Fails closed for invalid pinned releases.
- Still allows cold-start import when no release is selected or preload does not cover all images.

## Product-momentum checkpoint (mid-implementation)

Question: *Did this slice move the product measurably closer to final operator value?*  
Answer: **Yes**.

Evidence:

- Hybrid bootstrap now supports positional CLI image-dir, YAML-first config, release selection, editable preload annotation payloads, and companion health gating.
- Parser now emits hybrid lineage fields and rejects contradictory lineage payloads.
- Quant contract path now accepts per-glomerulus instance exports and writes lineage summaries.

## Real-data E2E evidence (MPS host)

Runtime setup used:

- Demo image subset copied from runtime manifest rows into `/tmp/eq_hybrid_demo/images/animal_1/`.
- Companion checkpoint:
  - `/Users/ncamarda/Workspaces/endotheliosis-quantifier/runtime/output/segmentation_evaluation/medsam_glomeruli_fine_tuning/deploy_conservative_mps_glomeruli/finetuned_evaluation/medsam_glomeruli_best_sam_state_dict.pth`

Commands run and outcomes:

1. Start companion on MPS:
   - `PYTORCH_ENABLE_MPS_FALLBACK=1 ... python -m eq.labelstudio.medsam_companion --checkpoint ... --device mps --port 8098`
   - Health check passed (`GET /healthz` -> `200`).
2. Real box assist inference:
   - `POST /v1/box_infer` on copied runtime image returned non-empty mask (`foreground_pixels=34956`, non-empty RLE).
3. Bootstrap Label Studio hybrid project:
   - `python -m eq labelstudio start /tmp/eq_hybrid_demo/images`
   - Imported 2 tasks with 2 preload proposals (runtime-default root path).
4. Create multi-ROI annotation + export:
   - Used Label Studio API to submit one preloaded ROI + one box-assisted ROI, each with a grade.
   - Export written to `/tmp/eq_hybrid_demo/project2_export.json`.
5. Quant contract ingestion with per-glom scoring:
   - `run_contract_first_quantification(..., annotation_source=/tmp/eq_hybrid_demo/project2_export.json, stop_after='contract')`
   - Outputs written under `/tmp/eq_hybrid_demo/quant_run/`.
   - `scored_examples.csv` has 2 glomerulus-instance rows from one image.
   - `lineage_summary.json` reports `scoring_unit=glomerulus_instance`, `rows=2`, `unique_instances=2`.

Issue discovered + fixed during E2E:

- Label Studio readiness polling failed on transient `ConnectionResetError`.
- Fix: `LabelStudioApiClient.wait_until_ready` now retries `ConnectionResetError` alongside existing transient HTTP startup failures.

## Current production-path proof state

The current hybrid workflow is no longer half-finished at the Label Studio contract layer. The repo now has direct evidence for the production-like path that mattered most:

- a fresh real-data comparison row can be staged into Label Studio with MedSAM preload,
- multiple disconnected ROIs in one image render as separate editable brush regions,
- each ROI can receive its own grade,
- the exported Label Studio JSON parses back into one authoritative glomerulus-instance row per ROI,
- and the parsed rows preserve `auto_preload`, `unedited_auto`, and the shared `mask_release_id`.

Concrete live proof completed on 2026-05-23:

- comparison row: `vegfri_dox__000528`
- live project URL during validation: `http://localhost:8080/projects/8/data`
- rendered ROIs: `3`
- exported scored rows: `3`
- parsed grades: `0.5`, `2.0`, `1.0`
- parsed lineage:
  - `proposal_kind = auto_preload`
  - `region_edit_state = unedited_auto`
  - `mask_release_id = medsam_manual_glomeruli_comparison:pilot_medsam_manual_glomeruli_comparison:medsam_oracle`

Implementation decisions tightened during this proof:

- Preload lineage for materialized editable annotations is stored in task-level `data.preload_region_lineage` keyed by region id instead of writing unsupported custom keys onto Label Studio result objects.
- Export parsing now links per-region grades through `parentID` when present, and the linkage validation no longer depends on result ordering.
- Disconnected preload components remain separate regions end to end; they are not collapsed into one union-scored export row.

Remaining product risks are now primarily about model quality and operator policy, not the Label Studio data contract:

1. Some release masks may still be scientifically noisy even though the UI/export path is correct.
2. Additional real rows should still be spot-checked before broader collaborator rollout.
3. Optional future policy gates such as border-touch rejection or stricter component filtering remain product-tuning work, not blockers for the verified contract.
