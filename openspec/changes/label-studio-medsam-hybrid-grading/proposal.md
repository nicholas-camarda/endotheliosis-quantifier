## Why

Fine-tuned MedSAM generated-mask releases from `medsam-glomeruli-fine-tuning` will be scientifically ready for labeling only if graders can reliably turn proposals into finalized glomerulus regions and attach endotheliosis scores per region inside Label Studio, with minimal CLI friction.

Current implementation evidence shows the integration plumbing exists, but the product is not finished unless the default demo path avoids noisy full-image preload masks, stages a real admitted manual-mask example from runtime data, and gives operators explicit proof that the shown MedSAM region matches ground truth on that example.

## What Changes

- Add a dedicated integration capability for **hybrid** grading: preload MedSAM proposal regions from a configured mask release while allowing human edit/delete/add—including **box-assisted** MedSAM for missing proposals—then export one authoritative row per scored complete-glomerulus region with full lineage (mask release, proposal kind, edit state, versioning fields).
- Preserve disconnected multi-glomerulus components in one image as separate scored ROIs end to end so production images with multiple glomeruli do not collapse back into one union-scored row.
- Make the default operator demo **box-assist/manual-first**: hybrid mode may require the companion while keeping full-image auto-preload disabled unless a YAML policy explicitly enables it for a quality-gated release.
- Add a single demo/handoff command, `eq labelstudio demo-hybrid`, that can either bootstrap a provided `<image-dir>` or stage a real comparison case from the MedSAM manual-mask pilot, then write an acceptance bundle with the Label Studio URL or dry-run plan, task manifest, companion policy, preload status, and explicit MedSAM-versus-ground-truth proof fields.
- Simplify collaborator-facing invocation: `**eq labelstudio start <image-dir>`** as the advertised entrypoint; optional YAML config (`configs/` path TBD by design) carries MedSAM mask release binding, ML backend endpoint, project defaults, and policy flags—not a growing list of CLI flags for standard users.
- Extend local bootstrap gates: optional preload-from-release validation; startup health-check for ML box-assist when required for hybrid mode; preserve existing Docker-backed Label Studio model.
- Extend glomerulus grading export contract parsers/validators with region lineage enums, latest-vs-history provenance hooks, and fail-closed rules aligned to per-region Choices.
- Remove implementation scaffolding and committed investigation hooks from production modules before treating the workflow as demonstrable.

## Capabilities

### New Capabilities

- `label-studio-medsam-hybrid-grading`: Label Studio UX and data contracts for preload + edit + box-assisted MedSAM regions, authoritative per-region grading, ML backend coupling, versioning/provenance semantics, and config-minimized CLI surfaces.
- `label-studio-medsam-real-example-proof`: A runtime-backed proof bundle contract that stages one admitted manual-mask comparison row into a showable demo and records the exact source image, manual mask, MedSAM prediction, overlay, and metrics used to justify the example.

### Modified Capabilities

- `label-studio-local-bootstrap`: Collaborator CLI shape (positional `<image-dir>`), YAML-backed parameters, optional MedSAM preload from `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`, startup health gates for hybrid mode including ML backend.
- `label-studio-glomerulus-grading`: Exported record shape and validation rules to include mask release lineage, proposal kind (`auto` vs `box_assisted`), human edit classification, versioning fields for latest authoritative grade/mask references, consistent `region_id` / `glomerulus_instance_id`.
- `medsam-glomeruli-fine-tuning`: Generated-mask releases intended for Label Studio hybrid review MUST expose import-ready manifest fields and stable release identifiers consumable by bootstrap preload construction.

## Impact

- Affected code (expected owners): `src/eq/__main__.py`, `src/eq/labelstudio/bootstrap.py`, `src/eq/labelstudio/demo.py`, `src/eq/labelstudio/glomerulus_grading.py`, `src/eq/evaluation/run_medsam_manual_glomeruli_comparison_workflow.py` (read-only contract reuse only), configs under `configs/`, new or extended unit tests mirroring fixtures.
- Affected integrations: Label Studio labeling config XML (region types, preannotations wiring), ML backend companion service boundary (implement or wrap—not vendored inside `eq` unless a later architectural change says otherwise).
- Affected artifact contracts: consumes `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` and registry `derived_data/generated_masks/glomeruli/manifest.csv` as documented by `medsam-glomeruli-fine-tuning`.
- Dependency ordering: assumes `medsam-glomeruli-fine-tuning` applies first for any mask release artifact contract this change binds to.

## Explicit Decisions

- Change name (OpenSpec folder): `label-studio-medsam-hybrid-grading`.
- New capability id: `label-studio-medsam-hybrid-grading`.
- Collaborator CLI primary form: `**eq labelstudio start <image-dir>**` where `<image-dir>` is a filesystem path argument (POSIX friendly); YAML config resolves MedSAM preload and companion service settings unless explicitly overridden via **limited** `--config`/`EQ_`* for admins only.
- Demonstration CLI primary form: `eq labelstudio demo-hybrid <image-dir>` with `--dry-run` for non-Docker proof and `--annotation-export <export.json>` for post-labeling lineage validation.
- Real-data proof CLI extension: `eq labelstudio demo-hybrid --comparison-run-dir <run-dir> --comparison-manifest-row-id <manifest_row_id>` stages exactly one runtime-backed MedSAM/manual comparison row into the demo and attaches proof fields to the acceptance artifact.
- Cold-import policy (**B**): images may ship with zero preload masks; graders recover via box-assisted MedSAM or manual contours; bootstrap does not refuse the project solely for absent preload proposal masks.
- Default demo policy: `configs/label_studio_medsam_hybrid.yaml` sets `hybrid.preload_enabled: false`; full-image preload requires an explicit administrator YAML change and quality-gated release.
- Real-data proof default: the canonical showable example SHALL come from `output/segmentation_evaluation/medsam_manual_glomeruli_comparison/<run_id>/` and MUST name the exact `manifest_row_id` used; synthetic images are not sufficient acceptance evidence.
- Box-assist is a **production dependency** whenever hybrid grading is enabled: health-check before usable labeling; labeling cannot silently pretend multi-glomerulus work remains easy if companion is unreachable (fail closed for that labeling session/profile).
- Versioning stance: authoritative exports expose **latest** committed annotation state with explicit lineage fields permitting audit; finer-grained LC history deferred unless LS export makes it trivial.
- Release-selection stance: implementation MUST audit the generated-mask registry first and choose the latest eligible MedSAM release by explicit ordering rules (status + timestamp), not by assuming a fixed release name.
- Required region source states for downstream ingestion: `auto_medsam`, `human_edited_medsam`, `box_assisted_medsam`, `excluded_partial_glomerulus`, and `rejected_medsam_candidate`.
- **Brush-first UX:** Final glomerulus regions remain **brush-editable** in Label Studio; box-assist supplies MedSAM prompts for new or missing coverage, not rectangle-only graded ROIs.
- **Production edge-case bar:** a real comparison row with multiple disconnected glomeruli in one image must render as separate MedSAM ROIs, accept one grade per ROI, and parse back into one authoritative export row per ROI with preserved preload lineage.
- **Use-case balance:** **Box-assist** is essential for **novel** images/glomeruli without preload; **preload + repeat labeling** of the same tasks remains a supported path (re-review, QA, updated grades).
- **No legacy score decomposition:** This change does **not** introduce models or heuristics that infer per-glomerulus scores from historical image-level aggregate grades (see `endotheliosis-grading-quant-learning-loop` for cohort-era flexibility).

## Open Questions

- [defer_ok] Whether administrators ever need a second canned YAML profile beside `configs/label_studio_medsam_hybrid.yaml`; v1 ships a single authoritative example.
- [defer_ok] Whether a later packaged release starts the MedSAM companion automatically; v1 keeps host-side companion launch explicit for MPS correctness.

## logging-contract

This change extends `eq labelstudio start`, `eq labelstudio demo-hybrid`, and companion operators only. Durable execution logging remains the existing `eq` CLI log stream; no new repository log root or subprocess tee contract is introduced. The MedSAM companion MAY emit its own local logs under operator control and MUST NOT write into Git-tracked paths.

## docs-impact

Update `README.md` and `docs/LABEL_STUDIO_GLOMERULUS_GRADING.md` with the real-data `eq labelstudio demo-hybrid --comparison-run-dir ... --comparison-manifest-row-id ...` path as the canonical proof workflow, keep `eq labelstudio start <image-dir>` documented as the reusable bootstrap primitive, and document hybrid YAML plus companion health expectations. Add git-tracked `configs/label_studio_medsam_hybrid.yaml` (template or fully documented example) as part of the implementation tasks.
