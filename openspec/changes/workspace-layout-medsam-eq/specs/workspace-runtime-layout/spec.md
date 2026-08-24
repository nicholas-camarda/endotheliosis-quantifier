## ADDED Requirements

### Requirement: Canonical EQ runtime root

EQ SHALL resolve its default active runtime root to `/Users/ncamarda/Workspaces/endotheliosis-quantifier/runtime` when `EQ_RUNTIME_ROOT` is unset.

#### Scenario: Default runtime resolution

- **WHEN** a workflow runs without `EQ_RUNTIME_ROOT`
- **THEN** raw data, derived data, models, logs, and outputs SHALL resolve below the workspace runtime root
- **AND** no operational default SHALL resolve below a retained legacy runtime root

#### Scenario: Explicit runtime override

- **WHEN** `EQ_RUNTIME_ROOT` is set
- **THEN** the existing override SHALL remain authoritative

### Requirement: Canonical MedSAM dependency roots

EQ SHALL use `/Users/ncamarda/Workspaces/medsam/source` as the default MedSAM repository and `/Users/ncamarda/Workspaces/medsam/runtime/work_dir/MedSAM/medsam_vit_b.pth` as the default base checkpoint.

#### Scenario: MedSAM workflow preflight

- **WHEN** a MedSAM workflow uses its defaults
- **THEN** code SHALL load from the workspace source
- **AND** the base checkpoint SHALL load from the workspace runtime
- **AND** a missing workspace dependency SHALL fail closed without consulting a legacy path

### Requirement: Verification isolation

Migration verification SHALL not write into canonical runtime data or retained legacy roots.

#### Scenario: Automated verification

- **WHEN** tests and smoke checks execute during workspace cutover
- **THEN** generated outputs and caches SHALL be directed to temporary roots
- **AND** protected source/runtime surfaces SHALL be reverified afterward
