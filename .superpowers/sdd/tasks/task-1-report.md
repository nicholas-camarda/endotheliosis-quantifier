# Task 1 Report: Fail-closed metric evidence and correct oracle gap

## Files changed

- `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`
  - Added a canonical CSV metric-evidence loader that validates the selected artifact, required Dice/Jaccard columns, finite numeric values, SHA-256, and sample count.
  - Updated fine-tuned comparison construction to use the selected oracle CSV even when the external-baseline execution status is `skipped_config`.
  - Added fail-closed comparison statuses, reasons, reason codes, and failed-gate records. Missing evidence does not flow into the adoption classifier and cannot emit an artificial `0.0` oracle gap.
- `tests/test_medsam_glomeruli_fine_tuning_workflow.py`
  - Added the 42-row selected-oracle regression (oracle mean Dice `0.921631295838860`; fine-tuned mean Dice `0.815511056209316`; asserted gap `0.106120239629544`, distinct artifacts, hashes, and sample counts).
  - Added a missing-oracle-evidence regression that asserts a blocked result and no numeric oracle gap.
  - Updated the existing workflow-level expectation to require the new fail-closed evidence block when external baselines are disabled without selected artifacts.

## Root cause

`_build_finetuned_comparison_summary` read only scalar means from the executed external-baseline summary. When that execution was skipped, it substituted missing metrics with `0.0`; this both discarded an explicitly selected oracle artifact and could produce an invalid oracle gap/adoption decision.

## Test commands and results

1. Red test run:

   ```bash
   PYTHONPATH=src conda run -n eq-mac python -m pytest -q tests/test_medsam_glomeruli_fine_tuning_workflow.py -k 'selected_oracle_metrics_csv or oracle_metric_evidence_is_unavailable'
   ```

   Result before the fix: 2 failed as expected. The selected oracle CSV produced `oracle_dice_gap == 0.0`, and missing oracle evidence produced `improved_candidate_not_oracle`.

2. Focused green test run: same command; 2 passed, 21 deselected (4 third-party PyTorch deprecation warnings).

3. Full requested MedSAM workflow module:

   ```bash
   PYTHONPATH=src conda run -n eq-mac python -m pytest -q tests/test_medsam_glomeruli_fine_tuning_workflow.py
   ```

   Result: 23 passed (4 third-party PyTorch deprecation warnings).

4. Static/diff checks:

   ```bash
   PYTHONPATH=src conda run -n eq-mac ruff check src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py tests/test_medsam_glomeruli_fine_tuning_workflow.py
   git diff --check
   ```

   Result: both passed.

## Concerns

- No blocking concerns. The deploy configuration can keep `external_baselines: false`; promotion remains blocked unless the comparison receives valid selected evidence for all required lanes.
- The test environment emits four PyTorch `torch.jit.script` deprecation warnings from dependencies; they are unrelated to this change.

## Fix round 1

Review found three additional fail-closed gaps in commit `26fe938`:

- Generated-mask release packaging and registry updates checked fine-tuned inference completion but did not require an eligible comparison. Packaging now requires `finetuned_comparison.status == "completed"`; blocked or unavailable evidence records `skipped_comparison_not_eligible` and writes neither release nor registry.
- Metric CSVs now require `manifest_row_id` and `method` provenance. Each lane's caller-supplied `run_id`, `fixed_split`, and complete fixed-row identity must match the fine-tuned evaluation context and the CSV row identities. Missing context or mismatched run/split/rows fails closed.
- Automatic and oracle evidence now require distinct paths and SHA-256 identities and lane-correct methods (`medsam_automatic` versus `medsam_oracle`). A shared path/hash or wrong method adds a structured blocking reason.
- The selected oracle-minus-fine-tuned Dice gap remains `0.106120239629544`; unavailable or provenance-invalid evidence emits no numeric zero gap.

### Fix-round tests

1. Focused RED run:

   ```bash
   PYTHONPATH=src conda run -n eq-mac python -m pytest -q tests/test_medsam_glomeruli_fine_tuning_workflow.py -k 'mismatched_fixed_split_metric_evidence or shared_oracle_automatic_artifact_and_wrong_method or workflow_summary_marks_finetuned_evaluation_completed_when_real_inference_runs'
   ```

   Result against `26fe938`: 3 failed as expected. The comparison omitted provenance/collision/method reasons, and packaging completed despite blocked evidence.

2. Focused GREEN run including the original selected-oracle and unavailable-evidence regressions:

   ```bash
   PYTHONPATH=src conda run -n eq-mac python -m pytest -q tests/test_medsam_glomeruli_fine_tuning_workflow.py -k 'selected_oracle_metrics_csv or oracle_metric_evidence_is_unavailable or mismatched_fixed_split_metric_evidence or shared_oracle_automatic_artifact_and_wrong_method or workflow_summary_marks_finetuned_evaluation_completed_when_real_inference_runs'
   ```

   Result: 5 passed, 20 deselected, 4 third-party PyTorch deprecation warnings.

3. Full requested MedSAM workflow module:

   ```bash
   PYTHONPATH=src conda run -n eq-mac python -m pytest -q tests/test_medsam_glomeruli_fine_tuning_workflow.py
   ```

   Result: 25 passed, 4 third-party PyTorch deprecation warnings.

4. Static/diff checks:

   ```bash
   PYTHONPATH=src conda run -n eq-mac ruff check src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py tests/test_medsam_glomeruli_fine_tuning_workflow.py
   git diff --check
   ```

   Result: both passed.

### Fix-round concerns

- No blocking concern. Four dependency-originated `torch.jit.script` deprecation warnings remain unchanged.

## Fix round 2

Review found two remaining fail-closed gaps in commit `97dd46a`:

- Generated-mask packaging treated every comparison with `status: completed` as reusable. Packaging now additionally requires an eligible adoption tier: `oracle_level_preferred` or `improved_candidate_not_oracle`. A completed comparison with `adoption_tier: blocked` writes neither a release nor the central registry. The reliable non-preferred candidate path remains packageable.
- Metric evidence provenance reduced manifest identities to a set before aggregation, allowing duplicate rows to alter means silently. The canonical loader now normalizes method and manifest-row identities and rejects every duplicate `(method, manifest_row_id)` combination before mean or per-method aggregation. This rejects repeated single-method manifest rows while preserving the expected one-row-per-method structure of multi-method baselines.
- Existing provenance, method, path/hash-distinctness, finite-value, and fixed-split checks remain fail closed. The selected oracle-minus-fine-tuned Dice gap remains `0.106120239629544`.

### Fix-round 2 tests

1. Focused RED run:

   ```bash
   PYTHONPATH=src conda run -n eq-mac python -m pytest -q tests/test_medsam_glomeruli_fine_tuning_workflow.py -k 'metric_evidence_rejects_duplicate_evaluation_rows or workflow_summary_does_not_package_ineligible_completed_inference'
   ```

   Result against `97dd46a`: 3 failed and 1 passed as expected. Both duplicate-evidence fixtures were accepted and averaged, and the completed/blocked comparison packaged a release and updated the registry.

2. Focused GREEN run, including the preserved non-preferred release path:

   ```bash
   PYTHONPATH=src conda run -n eq-mac python -m pytest -q tests/test_medsam_glomeruli_fine_tuning_workflow.py -k 'metric_evidence_rejects_duplicate_evaluation_rows or workflow_summary_packages_only_eligible_completed_inference or adoption_gate_distinguishes_oracle_level_candidate_and_blocked or generated_mask_release_and_registry_include_required_fields'
   ```

   Result: 7 passed, 22 deselected, 4 third-party PyTorch deprecation warnings.

3. Full requested MedSAM workflow module:

   ```bash
   PYTHONPATH=src conda run -n eq-mac python -m pytest -q tests/test_medsam_glomeruli_fine_tuning_workflow.py
   ```

   Result: 29 passed, 4 third-party PyTorch deprecation warnings.

4. Static and diff checks:

   ```bash
   PYTHONPATH=src conda run -n eq-mac ruff check src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py tests/test_medsam_glomeruli_fine_tuning_workflow.py
   git diff --check
   ```

   Result: both passed.

### Fix-round 2 concerns

- No blocking concern. Four dependency-originated `torch.jit.script` deprecation warnings remain unchanged.
- The pre-existing untracked `openspec/changes/fix-medsam-evaluation-promotion-pipeline/` directory was not read, modified, staged, or committed.
