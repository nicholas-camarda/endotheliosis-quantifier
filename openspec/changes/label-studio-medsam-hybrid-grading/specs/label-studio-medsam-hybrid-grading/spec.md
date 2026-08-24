## ADDED Requirements

### Requirement: Hybrid mode SHALL preload MedSAM proposal regions without blocking cold-start labeling

When hybrid grading is enabled and `hybrid.preload_enabled: true`, the system SHALL import Label Studio tasks and materialize MedSAM-derived proposal regions from the configured generated-mask release as editable brush annotations whenever matching mask artifacts exist, and MUST NOT refuse task import solely because some images lack preload proposal masks.

#### Scenario: Default demo starts without full-image preload

- **WHEN** the default `configs/label_studio_medsam_hybrid.yaml` is used
- **THEN** hybrid bootstrap MUST require the box-assist companion according to policy
- **AND** MUST NOT attach full-image preload proposal regions unless `hybrid.preload_enabled: true` is explicitly configured

#### Scenario: Warm start with preload coverage

- **WHEN** `configs/label_studio_medsam_hybrid.yaml` sets `hybrid.preload_enabled: true` and binds a valid `mask_release_id` whose release manifest resolves masks or polygons for a subset of images under `<image-dir>`
- **THEN** each matching task MUST include preload proposal geometries consumable by the labeling UI prior to submission
- **AND** lineage metadata tying proposals to `{mask_release_id, mask_source}` MUST be persisted for deterministic export ingestion

#### Scenario: Cold start with missing preload masks

- **WHEN** an image exists under `<image-dir>` but the mask release lacks a preload artifact or metadata entry for it
- **THEN** bootstrap MUST still create a Label Studio task for that image unless an explicit YAML policy rejects missing rows
- **AND** lineage MUST record absent preload artifacts so downstream parsers can distinguish inferred vs hydrated proposals

### Requirement: Bootstrap SHALL audit available releases and default to the latest valid release

Hybrid bootstrap MUST inspect the generated-mask registry before task construction, determine the latest valid release using explicit ordering rules, and use that release when no pinned `mask_release_id` is provided.

#### Scenario: No release pinned in YAML

- **WHEN** `configs/label_studio_medsam_hybrid.yaml` omits a pinned `mask_release_id`
- **THEN** bootstrap MUST audit `derived_data/generated_masks/glomeruli/manifest.csv`
- **AND** select the latest valid release by documented precedence
- **AND** emit the selected `mask_release_id` in startup logs and provenance metadata

#### Scenario: Pinned release missing or invalid

- **WHEN** YAML pins `mask_release_id` that is absent or invalid in the registry
- **THEN** bootstrap MUST fail closed with an actionable error listing audit results and candidate valid releases

### Requirement: Box-assisted MedSAM MUST be reachable when hybrid mode requires interactive proposals

When hybrid grading is configured with companion enforcement (default-on), startup MUST probe the companion health endpoint referenced in YAML and MUST refuse to finalize bootstrap if the probe fails unless `offline_manual_only_allowed: true` is explicitly enabled for admin-only contingency.

#### Scenario: Companion healthy

- **WHEN** YAML sets `require_box_assisted_medsam: true` or equivalent default-on hybrid flag AND the companion returns success on the documented health probe
- **THEN** bootstrap completes and emits operator messaging that hybrid assist is operational

#### Scenario: Companion unavailable with strict hybrid policy

- **WHEN** strict hybrid companion enforcement is enabled AND the health probe fails
- **THEN** bootstrap terminates with actionable diagnostics before collaborators open the labeling queue

### Requirement: Collaborative CLI surface MUST remain minimally flagged

Collaborator documentation MUST cite `eq labelstudio start <image-dir>` without requiring ancillary flags; advanced knobs MUST reside in YAML, optional `--config`, or explicit environment overrides surfaced only for administrators.

#### Scenario: Collaborator runbook path

- **WHEN** a collaborator executes `eq labelstudio start /path/to/images` with default repository config resolution
- **THEN** YAML supplies MedSAM preload binding, companion URL, LS/Docker defaults, and authentication references without prompting for additional positional parameters

### Requirement: Exported hybrid lineage MUST capture proposal derivation and edits

Hybrid exports MUST annotate each authoritative record with enumerated lineage fields distinguishing auto preload vs box-assisted origin, classify human edit fidelity, preserve `mask_release_id`, tie to Label Studio `region_id`/internal instance mapping, and keep references suitable for rollup or training ingestion audits.

#### Scenario: Auto preload refined by annotator brush edits

- **WHEN** a grader submits a finalized complete-glomerulus region originally materialized from preload proposal masks and edits segmentation boundaries prior to assigning a discrete grade
- **THEN** the emitted record MUST include proposal kind auto, edit state reflecting human refinement, authoritative grade, preserved mask release lineage, stable `glomerulus_instance_id`

#### Scenario: Box-assisted region graded after interactive assist

- **WHEN** a grader submits a finalized complete-glomerulus region synthesized through box-assisted MedSAM after preload absence
- **THEN** lineage MUST denote box-assisted derivation with companion provenance timestamps and authoritative grade linkage identical to preload-derived rows

### Requirement: Hybrid workflow SHALL treat brush refinement as the canonical graded mask primitive

MedSAM proposals imported from releases and masks returned from box-assisted inference MUST be presented and edited as **brush-compatible** region geometry in the labeling configuration (for example `brushlabels` with RLE or equivalent), so graders refine boundaries with the brush tool rather than committing final science masks as axis-aligned rectangles alone.

#### Scenario: Grader refines preload proposal

- **WHEN** a task includes preload MedSAM geometry and the grader adjusts segmentation before assigning a grade
- **THEN** the labeling configuration MUST allow brush-based edits on that geometry without forcing a rectangle-only workflow

#### Scenario: Cold image relies on box-assist

- **WHEN** an image has no usable preload for one or more glomeruli and the grader uses box-assisted MedSAM to propose initial geometry
- **THEN** the resulting region MUST remain brush-editable to the same standard as preload-derived regions

#### Scenario: Repeat review of preloaded tasks

- **WHEN** operators re-open tasks that already had preload or prior annotations (QA, grade change, boundary correction)
- **THEN** hybrid bootstrap and Label Studio project behavior MUST continue to support labeling on those tasks without requiring net-new images only

### Requirement: Hybrid mode MUST support both cold coverage and repeat review

The integration SHALL NOT assume workflows are limited to first-pass consumption of preload masks: box-assisted and manual-brush paths remain available for **unseen** coverage, while tasks with existing preload or prior work remain valid for **repeat** grading sessions.

#### Scenario: New image without release coverage

- **WHEN** an image under `<image-dir>` has no matching preload artifact in the bound mask release
- **THEN** the task MUST still be importable and graders MUST be able to obtain complete-glomerulus regions via box-assisted MedSAM (when companion policy allows), manual brush, or both

#### Scenario: Previously reviewed image returns to the queue

- **WHEN** an image already associated with a mask release or prior export re-enters the labeling queue
- **THEN** operators MUST be able to run another authoritative grading pass without a special “new-only” restriction imposed by hybrid bootstrap logic

### Requirement: Demo command SHALL write product-readiness acceptance evidence

The system SHALL expose `eq labelstudio demo-hybrid <image-dir>` as the canonical demonstrable hybrid workflow command, reusing the same bootstrap implementation as `eq labelstudio start` and writing an acceptance artifact that proves what was prepared.

#### Scenario: Dry-run demo creates an acceptance artifact

- **WHEN** an operator runs `eq labelstudio demo-hybrid <image-dir> --dry-run`
- **THEN** the command MUST write a JSON acceptance artifact under the Label Studio runtime acceptance directory or the administrator-provided `--acceptance-dir`
- **AND** the artifact MUST include task count, task manifest path, selected mask release, preload prediction count, companion status, Label Studio URL, dry-run status, config path, and operator next steps
- **AND** the command MUST NOT start Docker or call the Label Studio API in dry-run mode

#### Scenario: Live demo records project URL

- **WHEN** an operator runs `eq labelstudio demo-hybrid <image-dir>` and bootstrap succeeds
- **THEN** the acceptance artifact MUST include the live project URL returned by Label Studio
- **AND** the command output MUST print the acceptance artifact path alongside the project URL

#### Scenario: Completed export is attached to acceptance evidence

- **WHEN** an operator provides `--annotation-export <export.json>`
- **THEN** the command MUST parse the export through the glomerulus-instance grading validator
- **AND** the acceptance artifact MUST summarize row count, scoring unit, proposal-kind counts, region edit-state counts, and mask-release counts
- **AND** contradictory or invalid exports MUST fail closed before the artifact is advertised as ready

### Requirement: Demo command SHALL support a real-data MedSAM proof case

The system SHALL support a canonical demo mode that stages one admitted manual-mask comparison row from the runtime `medsam_manual_glomeruli_comparison` outputs, imports the real source image into Label Studio, preloads the exact MedSAM mask scored in that comparison, and records proof fields tying the demo to ground truth.

#### Scenario: Runtime-backed comparison row is staged into the demo

- **WHEN** an operator runs `eq labelstudio demo-hybrid --comparison-run-dir <run-dir> --comparison-manifest-row-id <manifest_row_id>`
- **THEN** the command MUST resolve the row from `<run-dir>/inputs.csv`
- **AND** MUST verify that the corresponding manual mask, MedSAM prediction mask, and overlay exist under the same run directory
- **AND** MUST stage exactly that image into a single-image demo import directory without requiring the operator to hand-assemble files

#### Scenario: Acceptance artifact proves what the demo shows

- **WHEN** the runtime-backed comparison demo succeeds
- **THEN** the acceptance artifact MUST include `comparison_run_dir`, `comparison_run_id`, `manifest_row_id`, `cohort_id`, `lane_assignment`, `source_image_path`, `manual_mask_path`, `prediction_mask_path`, `overlay_path`, and the same-row MedSAM/current-segmenter metrics used for interpretation
- **AND** the artifact MUST state whether the shown MedSAM example outperformed the current segmenters on that exact row

#### Scenario: Multi-ROI proof preserves one scored record per preloaded region

- **WHEN** the staged comparison row contains multiple disconnected MedSAM/manual glomerulus components in one image
- **THEN** the imported task MUST preserve those components as separate brush regions with stable region ids
- **AND** the completed export MUST parse into one authoritative glomerulus-instance record per scored region
- **AND** the parsed records MUST preserve the shared `mask_release_id`, per-region grades, and `auto_preload` lineage without collapsing multiple ROIs into one union-scored row

#### Scenario: Invalid comparison row fails closed

- **WHEN** the requested `manifest_row_id` is absent from `inputs.csv`, lacks a MedSAM mask, or lacks same-row metrics
- **THEN** `eq labelstudio demo-hybrid` MUST fail with an actionable error before it advertises a ready demo
