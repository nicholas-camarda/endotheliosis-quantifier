## Why

The MedSAM glomerular evaluation workflow has two evidence-integrity defects: a skipped or missing oracle artifact is converted into a numeric zero, and generated-mask release manifests serialize machine-specific absolute paths inherited from an older runtime root. The workflow therefore produces a misleading oracle gap and cannot be moved or independently audited safely.

## What Changes

- Make automatic, oracle, and candidate metrics explicit artifact-backed records with sample counts, artifact identifiers, and unavailable states; missing evidence fails closed instead of becoming zero.
- Select oracle metrics from the explicitly selected oracle metrics artifact and reject stale, incompatible, or mismatched evaluation evidence.
- Serialize generated-mask release and registry paths as runtime-relative POSIX paths under a versioned manifest contract, with an explicit runtime-root resolver and visible stale-manifest rejection.
- Add deterministic dataset/split audit and automatic-box diagnostics that report cohort, source-group, positivity, leakage, proposal geometry, and segmentation relationships without changing promotion thresholds.
- Preserve the existing Docker/Label Studio and MPS companion gates as optional capability checks, separate from portable deterministic tests.

## Capabilities

### New Capabilities

- `medsam-evaluation-provenance`: fail-closed metric and portable release provenance contracts for MedSAM evaluation and generated-mask packaging.
- `medsam-dataset-box-audit`: reproducible cohort/split/positivity and automatic-box quality diagnostics for promotion evidence.

## Impact

- `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`
- `src/eq/utils/paths.py` or a single shared artifact-provenance helper
- `src/eq/labelstudio/bootstrap.py` release-manifest resolution
- focused MedSAM workflow tests and new audit tests
- current OpenSpec contract and runtime-generated audit artifacts only

## Explicit Decisions

- Existing Dice/Jaccard thresholds remain unchanged: Dice `0.90`, Jaccard `0.82`, oracle Dice gap `0.05`, and zero prompt failures.
- Oracle metrics are diagnostic only and never satisfy the automatic/deployable promotion gate.
- A missing metric is represented as unavailable with a reason and cannot participate in adoption or promotion as a passing value.
- Release path fields use `schema_version=eq.medsam.generated-mask-release/1` and `path_encoding=runtime-relative-posix`.
- Stale absolute manifests are rejected; they are not repaired by username replacement, path guessing, or fallback roots.
- This change does not retrain or promote the current checkpoint unless corrected automatic evidence independently satisfies the existing gates.
