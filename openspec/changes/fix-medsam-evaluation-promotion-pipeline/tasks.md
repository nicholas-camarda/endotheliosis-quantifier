# Fix MedSAM evaluation and promotion provenance

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan task-by-task with a fresh implementer and task review for each task.

**Goal:** Make MedSAM glomerular evaluation portable, auditable, and fail-closed while preserving the current scientific promotion thresholds and technical gates.

**Architecture:** Extend the existing workflow and central path contract; do not create a second promotion pipeline. Use explicit artifact-backed metric evidence and one runtime-relative provenance resolver shared by release packaging and Label Studio consumption. Keep dataset and box diagnostics as structured run artifacts driven by the locked evaluation rows.

**Tech Stack:** Python 3.10+, pandas, NumPy, PIL, pytest, Ruff, YAML/OpenSpec.

**Spec:** `openspec/changes/fix-medsam-evaluation-promotion-pipeline/proposal.md` and `design.md`.

## Global Constraints

- Preserve Dice `0.90`, Jaccard `0.82`, maximum oracle Dice gap `0.05`, and maximum prompt failures `0`.
- Never use oracle metrics as the deployable automatic-pipeline promotion gate.
- Missing or incompatible evidence must be unavailable or an explicit error, never numeric zero by default.
- Do not add fallback roots, username substitution, silent remapping, or temporary compatibility logic.
- Keep runtime data and model artifacts outside Git.
- Do not merge or push automatically.

### Task 1: Fail-closed metric evidence and correct oracle gap

**Files:**
- Modify: `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`
- Test: `tests/test_medsam_glomeruli_fine_tuning_workflow.py`

**Interfaces:**
- Add a canonical metric-evidence loader used by `_build_finetuned_comparison_summary`.
- Preserve existing test-facing helpers where possible, but extend returned summaries with `status`, `reason`, artifact identifiers, and sample counts.

- [x] Write a regression test with a real temporary oracle CSV (42 rows, mean Dice `0.921631295838860`) and fine-tuned CSV (42 rows, mean Dice `0.815511056209316`) while `external_baselines_summary={"status":"skipped_config"}`. Assert the report gap is `0.106120239629544` and the two artifact paths differ.
- [x] Write a regression test with missing oracle evidence and assert the comparison is blocked with an unavailable reason; assert no numeric zero gap is emitted.
- [x] Run the focused tests and observe the pre-fix failure caused by the current `0.0` fallback.
- [x] Implement explicit CSV validation, selected-artifact loading, sample-count recording, SHA-256 recording, and gap computation only when both metric records are available.
- [x] Add structured `failed_gates`/`reason_codes` for missing evidence while preserving the existing adoption thresholds and automatic-path semantics.
- [x] Rerun the focused tests and the existing workflow test module.

### Task 2: Portable generated-mask release provenance

**Files:**
- Create or modify: `src/eq/utils/artifact_provenance.py` (only if the existing path owner cannot absorb the contract)
- Modify: `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`
- Modify: `src/eq/labelstudio/bootstrap.py`
- Test: `tests/test_medsam_glomeruli_fine_tuning_workflow.py`
- Test: `tests/unit/test_labelstudio_bootstrap.py`

**Interfaces:**
- Define one `runtime-relative-posix` serializer/resolver with explicit `runtime_root` input and visible stale-path errors.
- Release manifests and provenance JSON use `schema_version=eq.medsam.generated-mask-release/1` and relative path fields.

- [x] Write tests that serialize the same logical release under two temporary runtime roots and assert identical relative fields.
- [x] Write tests that reject absolute paths, `..` traversal, paths outside runtime root, missing generated artifacts, and stale existing absolute manifests.
- [x] Write a Label Studio test that resolves relative release paths against the supplied runtime root from a different process CWD.
- [x] Run these tests before implementation and observe failure against the current absolute-path writer/consumer.
- [x] Implement the single serializer/resolver and migrate release manifest, provenance JSON, registry `release_manifest_path`, and Label Studio loading to it.
- [x] Regenerate only disposable temporary fixtures for verification; do not rewrite historical runtime artifacts in Git.

### Task 3: Reproducible dataset/split audit

**Files:**
- Create or modify: `src/eq/evaluation/medsam_glomeruli_audit.py`
- Modify: `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`
- Test: `tests/test_medsam_glomeruli_audit.py`
- Test: `tests/test_medsam_glomeruli_fine_tuning_workflow.py`

**Interfaces:**
- Add `audit_medsam_evaluation_cohort(rows, runtime_root)` returning JSON-serializable counts and issues.
- Add a workflow writer that emits `dataset_split_audit.json` and `dataset_split_audit.csv` from the locked split rows.

- [x] Write fixture tests for train/validation/test counts, unique source groups, duplicate image/mask hashes, positive/background counts, mask area quantiles, cohort/lane composition, and cross-split leakage.
- [x] Write a fail-closed test for a duplicated source group crossing splits.
- [x] Implement the audit using the strongest available grouping key (`split_group_id`, subject/specimen/source identifiers) and explicit `independence_status`.
- [x] Record that positive-only evidence is a distribution limitation; do not invent whole-image negatives. Reuse the existing validated negative-crop contract if a negative manifest is explicitly supplied.
- [x] Attach audit summary/hash to run provenance and make invalid split/leakage a promotion blocker.

### Task 4: Automatic-box diagnostics and coordinate invariants

**Files:**
- Create or modify: `src/eq/evaluation/medsam_glomeruli_audit.py`
- Modify: `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`
- Test: `tests/test_medsam_glomeruli_audit.py`

**Interfaces:**
- Add `box_diagnostics_for_row(gt_mask, automatic_box, image_shape, oracle_metrics, automatic_metrics)` returning box IoU, GT coverage, area ratio, center displacement, proposal status, and paired segmentation metrics.
- Add invariant helpers for original-image/resized-encoder coordinate round trips on square and non-square images.

- [x] Write tests for perfect, partial, clipped, invalid, and missing boxes, including a non-square `2448x2048` image.
- [x] Write an invariant test that converts a box to resized coordinates and back without changing integer pixel bounds beyond one pixel.
- [x] Implement the metrics and write one row per evaluation sample to `box_diagnostics.csv`.
- [x] Add a summary that relates proposal quality strata to automatic Dice/Jaccard; do not alter thresholds based on held-out test rows.
- [x] Use the resulting evidence to classify the current failure as prompt geometry, segmentation, or insufficient data; do not retrain without that classification.

### Task 5: Promotion/report provenance and documentation contract

**Files:**
- Modify: `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`
- Modify: `openspec/specs/medsam-glomeruli-fine-tuning/spec.md`
- Test: `tests/test_medsam_glomeruli_fine_tuning_workflow.py`

- [x] Add report fields for checkpoint ID/hash, split manifest IDs/hashes, dataset audit ID, box-diagnostic artifact, preprocessing/prompt config, automatic/oracle metric artifacts, evaluation run ID/timestamp, thresholds, failed gates, and decision.
- [x] Add a test proving a sub-threshold automatic result cannot be promoted even when oracle metrics pass.
- [x] Add a test proving stale provenance or split leakage blocks promotion.
- [x] Update the active OpenSpec contract with exact schema and failure behavior, then run strict validation.

### Task 6: Full verification and independent review

- [ ] Run focused red-green tests, then the full `PYTHONPATH=src /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest -q` suite.
- [ ] Run `ruff check src tests`, `git diff --check`, and `openspec validate --all --strict`.
- [ ] Run deterministic evaluation on the corrected protocol and inspect summary, metric CSVs, promotion report, generated-mask manifest, dataset audit, and box diagnostics.
- [ ] Run available live MPS companion and ephemeral Label Studio/Docker tests without making them routine CI dependencies.
- [ ] Dispatch whole-branch code review, address material findings, rerun verification, and leave the isolated worktree clean without merge or push.
