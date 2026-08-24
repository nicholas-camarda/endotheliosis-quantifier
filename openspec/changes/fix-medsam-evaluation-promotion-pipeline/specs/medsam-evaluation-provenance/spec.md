## ADDED Requirements

### Requirement: Metric evidence is artifact-backed and fail-closed

The MedSAM fine-tuning comparison SHALL represent automatic, oracle, and fine-tuned metrics as artifact-backed evidence records with status, artifact identity, sample count, and metric values. Missing, stale, incompatible, or invalid evidence SHALL be unavailable or an explicit error and SHALL NOT be converted to numeric zero.

#### Scenario: Oracle evidence is selected from an explicit artifact
- **WHEN** the external-baseline summary is skipped but an explicitly selected oracle metrics CSV exists for the locked evaluation rows
- **THEN** the comparison SHALL load that CSV, record its artifact identity and sample count, and compute the oracle-minus-automatic Dice gap from distinct artifacts

#### Scenario: Oracle evidence is unavailable
- **WHEN** no valid oracle metrics artifact is available for the locked evaluation
- **THEN** the comparison SHALL record an unavailable reason, omit the numeric oracle gap, and block adoption/promotion evidence

#### Scenario: Automatic and oracle artifacts collide
- **WHEN** automatic and oracle metric records resolve to the same artifact identity
- **THEN** the comparison SHALL block the evidence and report an artifact-collision reason

### Requirement: Generated-mask provenance is portable

Generated-mask release manifests, central registries, and adjacent provenance JSON SHALL use `schema_version=eq.medsam.generated-mask-release/1` and `path_encoding=runtime-relative-posix` for runtime-owned artifacts. Resolution SHALL require an explicit runtime root and SHALL reject absolute, traversal, outside-root, missing, or stale paths.

#### Scenario: A release moves between runtime roots
- **WHEN** the same logical release is generated under two different runtime roots
- **THEN** the serialized relative path fields SHALL be identical and resolve correctly against either supplied runtime root

#### Scenario: A stale absolute release manifest is loaded
- **WHEN** a release manifest contains an absolute path from a retired or unavailable runtime root
- **THEN** loading SHALL fail visibly as stale provenance and SHALL NOT rewrite the path by username or guessed prefix

#### Scenario: Generated artifact is missing
- **WHEN** a manifest row has `generation_status=generated` but its resolved source image, generated mask, checkpoint, or provenance artifact is missing
- **THEN** release validation SHALL fail closed with the missing artifact and row identity

#### Scenario: Optional reference mask is declared
- **WHEN** a generated-mask row declares a reference mask path
- **THEN** release validation SHALL resolve it beneath the explicit runtime root and fail closed if it is stale, unsafe, or missing
- **AND** a row without an available reference mask SHALL remain valid with an empty reference-mask field
