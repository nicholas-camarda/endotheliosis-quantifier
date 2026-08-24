## ADDED Requirements

### Requirement: Evaluation cohort independence is auditable

The MedSAM evaluation workflow SHALL emit a deterministic dataset/split audit for the locked train, validation, and test rows, including counts, cohorts, lanes, unique source images, strongest available biological/source grouping, duplicate content hashes, positive/background mask composition, mask-area summaries, and cross-split overlap findings.

#### Scenario: A biological/source group crosses splits
- **WHEN** the strongest available independent grouping identifier occurs in more than one split
- **THEN** the audit SHALL record the group and mark the evidence not promotion eligible

#### Scenario: Evaluation contains only positive masks
- **WHEN** all admitted training/evaluation masks have foreground pixels
- **THEN** the audit SHALL report positive-only composition and its distribution-mismatch limitation without manufacturing background examples

### Requirement: Automatic-box quality is separate from segmentation quality

For each evaluation row with a valid ground-truth mask and automatic proposal, the workflow SHALL report proposal status, box IoU, ground-truth mask coverage, proposal-to-ground-truth area ratio, center displacement, coordinate-frame metadata, and paired automatic/oracle segmentation metrics.

#### Scenario: Non-square coordinate round trip
- **WHEN** a source-image box is mapped to resized encoder coordinates and back for a non-square image
- **THEN** the round trip SHALL preserve the box within the declared one-pixel integer tolerance

#### Scenario: Proposal failure is observed
- **WHEN** no valid automatic box is produced for an evaluation row
- **THEN** the audit SHALL record a proposal failure separately from segmentation metric failure
