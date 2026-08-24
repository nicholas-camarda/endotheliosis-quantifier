## 1. Copy and evidence

- [x] 1.1 Capture Git, dirty/untracked state, worktrees, submodules, symlinks, path consumers, processes, and target absence.
- [x] 1.2 Copy MedSAM source and payload mirrors plus EQ source/runtime without dereferencing symlinks.
- [x] 1.3 Run the stable checksum pass and capture original SHA-256 evidence.

## 2. Runtime contracts

- [x] 2.1 Add failing tests for workspace EQ runtime and MedSAM source/checkpoint defaults.
- [x] 2.2 Update centralized defaults, active configs, scripts, docs, and live OpenSpec contracts.
- [x] 2.3 Add an active-contract regression test that rejects retired local roots.

## 3. Verification

- [x] 3.1 Complete source/target SHA-256 comparison for the EQ runtime.
- [x] 3.2 Run complete isolated test, lint, format-check, type-check, CLI, MedSAM, Label Studio, bootstrap, and quantification checks.
- [x] 3.3 Verify protected originals and canonical runtime surfaces after tests.
- [x] 3.4 Run `openspec validate workspace-layout-medsam-eq --strict`.

## 4. Cutover

- [x] 4.1 Write workspace manifests and controller verification evidence.
- [x] 4.2 Mark the new paths canonical only after all required verification succeeds.
