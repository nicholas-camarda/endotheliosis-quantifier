## Context

`medsam-glomeruli-fine-tuning` will emit reusable generated-mask releases under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` with registry rows in `derived_data/generated_masks/glomeruli/manifest.csv`. Collaborators already use `eq labelstudio start` for Docker-backed Label Studio and per-instance grading XML. The hybrid workflow must connect those releases into Label Studio tasks as **regions** grader edits, supplemented by interactive **box-assisted** MedSAM, while keeping human-first scoring semantics and tightening export lineage so models are trained from honest segmentation provenance rather than naive auto-masks.

## Goals / Non-Goals

**Goals:**

- Consume a named MedSAM finetuned **mask_release_id** binding (via YAML—not a proliferation of collaborator CLI flags).
- Provide **preload** proposals when available without blocking labeling when preload is incomplete (cold start **B**).
- Treat box-assisted MedSAM as a required companion for hybrid usability; fail closed startup when hybrid mode mandates it vs allowing silent degradation across multi-glomerulus images.
- Extend export validation with explicit region lineage enums and latest authoritative records with reversible provenance anchors.
- Keep collaborator ergonomics anchored on `**eq labelstudio start <image-dir>`**.
- Extend rather than bypass existing owners (`src/eq/labelstudio/bootstrap.py`, `src/eq/labelstudio/glomerulus_grading.py`, `src/eq/utils/paths.py`).

**Non-Goals:**

- Performing MedSAM training or fine-tuning inside this change.
- Embedding MedSAM CUDA training code into `src/eq`; wrap external checkpoints + inference/box endpoints per upstream contracts.
- Exposing Stage 2 model grade suggestions directly into Stage 1 primary grading UX (preserve model-blind contract already spec’d).
- Committing heavyweight runtime artifacts (`derived_data/`**, checkpoints) into Git.
- Inferring per-glomerulus grades or masks from legacy **image-level aggregate scores** (mathematically ill-posed without new annotation or vision models; out of scope for this change).

## Decisions

### Decision: Canonical collaborator CLI minimizes flags

**Choice:** Teach `eq labelstudio start <image-dir>` as the advertised path while retaining backward-compatible `--images` invocation already shipped for automation.

**Alternative:** Eliminate positional support and keep `--images` only → rejected due to collaborator preference.

### Decision: Finished-product demo command owns acceptance evidence

**Choice:** Add `eq labelstudio demo-hybrid` as the showable workflow wrapper. It calls the same bootstrap implementation as `eq labelstudio start`, can stage a real MedSAM/manual comparison row from runtime evaluation outputs, writes `hybrid_demo_acceptance.json`, and records task manifest path, URL/project URL when live, companion policy, preload count/status, optional parsed export lineage summary from `--annotation-export`, and comparison-proof fields when a runtime-backed example is requested.

**Alternative:** Keep README command snippets only → rejected because the user needs a finished product artifact that can be shown without reconstructing chat history.

### Decision: Real-data proof reuses the manual-mask pilot contract

**Choice:** The canonical proof example comes from `output/segmentation_evaluation/medsam_manual_glomeruli_comparison/<run_id>/`, reusing `inputs.csv`, `metrics.csv`, `medsam_masks/`, and `overlays/` instead of inventing a synthetic image or a second evaluation path. `demo-hybrid` stages one selected `manifest_row_id` into a single-image demo directory and preloads the corresponding MedSAM mask into Label Studio.

**Alternative:** Invent a synthetic image or use an unrelated generated-mask release → rejected because it does not prove MedSAM behavior against known ground truth.

### Decision: Multi-ROI proof stays per-component end to end

**Choice:** When a preload or proof mask contains multiple disconnected glomerulus components in one image, bootstrap splits them into separate `brushlabels` regions with stable ids, task-level provenance keeps one lineage entry per region id, and export ingestion emits one scored glomerulus-instance row per region rather than collapsing them into one union row.

**Alternative:** Treat the union mask as one scored ROI → rejected because it breaks per-glomerulus grading semantics and hides the exact production edge case that previously mattered.

### Decision: YAML carries integration bindings

**Choice:** Provide `configs/label_studio_medsam_hybrid.yaml` documenting release-selection policy (`selection_mode: latest_valid`), explicit `preload_enabled`, optional pinned `mask_release_id`, ML companion base URL, auth token reference (env-backed), Docker/LS defaults, hybrid enablement.

**Alternative:** Environment variables exclusively → simpler shell but brittle for onboarding; YAML remains authoritative with optional overrides.

### Decision: Native editable MedSAM annotation ingestion

**Choice:** Offline ingestion maps release manifest masks to task-level lineage metadata and creates editable Label Studio brush annotations through the annotation API when `preload_enabled: true`; it does not import those masks through Label Studio `predictions`. Interactive box assist uses the companion HTTP contract `GET /healthz` and `POST /v1/box_infer` with minimal payload (`image_path`, `box_xyxy`).

**Alternative:** Only synchronous companion generation → brittle for laptops; preload gives instant UX when release exists.

**Operator intent:** **Box-assist** is the high-value path for **images or glomeruli not yet covered** by a mask release (cold coverage). **Preload** accelerates **known** images and MUST remain compatible with **repeat review** of the same tasks (QA, grade updates, boundary fixes)—the workflow is not limited to one-and-done preload consumption.

**Default product posture:** full-image preload is disabled in the default YAML because current release masks can be noisy as operator-facing instance proposals. Preload is an explicit admin opt-in (`preload_enabled: true`) after a target release passes instance-quality checks.

### Decision: Operational posture for companion unavailability

**Choice:** Startup health ping when hybrid mode flag true; refusal to advertise ready collaborator session if ping fails unless admin explicitly enables `offline_manual_only_allowed` bypass (non-default guard).

**Alternative:** Soft degrade silently → contradicts collaborator clarity for multi-glomerulus labeling.

### Decision: Versioned audit fields without full duplication

**Choice:** Persist `annotation_id`, LS `completed_at`, `mask_release_id`, `proposal_kind`, `region_edit_state`, optional superseded references to prior annotation drafts when LS exposes them cheaply—otherwise deterministic latest-completed semantics only.

### Decision: Parsing ownership

Extend `glomerulus_grading.py` parsers (or refactor into shared labelstudio contract helpers) versus duplicating ingestion script—reuse existing ingestion surface.

### Decision: Audit registry and select latest valid release before bootstrap

**Choice:** Before task construction, bootstrap audits `derived_data/generated_masks/glomeruli/manifest.csv`, filters release rows by validity status required for Label Studio consumption, and selects the latest candidate using explicit sort precedence (for example: `is_current`/release status then `created_at` descending). A pinned `mask_release_id` in YAML or admin override is allowed but not required.

**Alternative:** Require a hard-coded release id in every run. Rejected because release names are fluid during fine-tuning iteration and this burdens collaborators.

## Risks / Trade-offs

- **[Risk]** Label Studio version drift changes prediction API → mitigation: pin container tag or document audited API + contract tests referencing fixture exports.
- **[Risk]** Over-specifying ML HTTP schema before companion exists → mitigation: stabilize minimal request/response in spec with explicit extension points flagged `[defer_ok]` for batching/gpu fields.
- **[Risk]** Preload polygons misaligned vs displayed image → mitigation: deterministic coordinate normalization tests + QA overlay snapshot hook.
- **[Risk]** Admin bypass flags inadvertently used in production datasets → mitigation: fail-closed default; bypass requires YAML opt-in loudly logged.
- **[Risk]** A cherry-picked real-data example can mislead if it is not traceable → mitigation: acceptance artifacts must name the exact `manifest_row_id`, source paths, method, and same-row current-segmenter comparator metrics.

## Migration Plan

1. Land YAML config + parsers + bootstrap wiring after MedSAM releases exist (`depends_on` sequencing with fine-tuning change).
2. Add pre-bootstrap registry audit step and deterministic latest-valid selection logging.
3. Introduce staging profile for collaborators; document rotation between releases referencing registry manifest.
4. Rollback by pinning a prior `mask_release_id`, setting YAML to manual-only mode, or disabling hybrid block.

## Explicit Decisions

- Collaborator advertised command: `**eq labelstudio start <image-dir>`**.
- Demonstration/handoff command: `eq labelstudio demo-hybrid <image-dir>`.
- Real-data proof command extension: `eq labelstudio demo-hybrid --comparison-run-dir <run-dir> --comparison-manifest-row-id <manifest_row_id>`.
- Acceptance artifact: `<runtime-root>/labelstudio/acceptance/<safe-project-name>_hybrid_demo_acceptance.json` unless overridden with `--acceptance-dir`.
- Default integration settings file path: `**configs/label_studio_medsam_hybrid.yaml`**.
- Default startup policy: hybrid companion required, full-image preload disabled unless `hybrid.preload_enabled: true`.
- Cold start policy (**B**) remains: zero preload masks must NOT block importing tasks.
- Box-assist default posture: hybrid mode refuses open without healthy companion unless `offline_manual_only_allowed: true`.
- **Canonical region primitive for human refinement** is **brush** (`brushlabels` in the grading XML): graders edit MedSAM-derived geometry with the brush; box-assist is an **input modality to MedSAM**, not a replacement for brush-finalized masks.
- Real-data proof artifact fields SHALL include: `comparison_run_dir`, `comparison_run_id`, `manifest_row_id`, `cohort_id`, `lane_assignment`, `source_image_path`, `manual_mask_path`, `prediction_mask_path`, `overlay_path`, per-method same-row metrics, and a boolean summary of whether MedSAM outperformed the current segmenters on that row.
- Multi-ROI proof contract: disconnected components in one staged image remain separate preloaded regions, can each receive a grade, and must parse back into one authoritative row per region with preserved `auto_preload` lineage.

## Open Questions

- [defer_ok] Compose packaging versus standalone script to launch the companion beside the Label Studio container — operational convenience only.
