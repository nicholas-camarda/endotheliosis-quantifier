## Context

The fine-tuning workflow evaluates fixed validation/test rows, writes CSV metrics, builds a comparison summary, and optionally packages generated masks. The current comparison summary reads oracle means only from `external_baselines_summary`, while the deploy configuration can intentionally skip that computation. The workflow also converts missing file/metric values to `0.0`. A historical release artifact still contains absolute paths under the retired `ProjectsRuntime` tree even though the active runtime root is configurable.

## Design

### Metric evidence contract

Add one canonical loader for metric CSV evidence. It validates the file, required metric columns, finite numeric values, and expected row count when supplied. It returns a JSON-serializable record containing `status`, `reason`, `artifact_path` as a runtime-relative identifier when possible, `artifact_sha256`, `sample_count`, `mean_dice`, and `mean_jaccard`. A valid metric value of zero remains distinct from an unavailable metric.

The comparison builder receives explicit evidence records for automatic, oracle, current-segmenter, trivial, and fine-tuned pathways. Oracle evidence is selected from the explicitly named oracle CSV/output record, not inferred from a skipped summary. If evidence is unavailable or from a different evaluation run/split, the comparison is blocked and records a structured reason code. `oracle_dice_gap` is computed only when both automatic and oracle Dice are available.

### Portable release provenance

Introduce a shared runtime-relative path serializer/resolver. Serialization requires an explicit runtime root and rejects paths outside it, absolute manifest values, traversal, and missing generated artifacts where status is `generated`. The manifest and provenance JSON include the schema/version contract and relative artifact identifiers. Consumers resolve fields against the caller-supplied runtime root, independent of process CWD, and fail closed when the path is stale or escapes the root. The central registry stores the same relative fields and relative release-manifest identifier.

### Dataset and proposal audit

Add deterministic audit helpers that operate on the same fixed split rows used by evaluation. They report counts by split/cohort/lane, unique image and strongest available biological/source grouping, duplicate content hashes, positive/background mask counts and foreground-size summaries, and cross-split group/image overlap. For each row with a ground-truth mask and automatic box, compute box IoU, GT-mask coverage, proposal-to-GT area ratio, center displacement, proposal failure, and automatic/oracle Dice/Jaccard. The output is a CSV plus JSON summary under the evaluation run root. No threshold or test-set tuning is introduced.

### Promotion gate

The existing automatic Dice/Jaccard thresholds and provenance requirements remain in force. The report gains explicit `failed_gates` and `reason_codes`. Any unavailable metric, stale artifact, invalid split, missing artifact hash, leakage, or missing required cohort evidence blocks promotion. The existing MPS and live Label Studio checks stay capability-gated and are not made routine CI dependencies.

## Failure Boundaries

- Missing or incompatible metric CSVs are unavailable, never numeric zeros.
- Existing absolute release manifests fail validation as stale and are not silently migrated.
- An automatic-box failure is reported separately from a segmentation failure; neither is hidden by an aggregate metric.
- Dataset audit findings do not manufacture independent sample size from correlated crops.
- Retraining is not part of the first fix unless corrected validation evidence identifies it as the remaining limiting component.

## Verification

- Red-green regression tests reproduce the zero-gap and stale-path defects before implementation.
- Focused workflow, provenance, Label Studio resolver, dataset-audit, and box-coordinate tests pass.
- Full pytest, Ruff, `git diff --check`, strict OpenSpec validation, and available MPS/Label Studio capability checks are rerun.
- Newly generated summary/metric/provenance/audit artifacts are inspected for path portability, distinct automatic/oracle evidence, correct gap math, accurate N/cohort counts, and fail-closed promotion.
