## Why

The local source and runtime contracts must identify the verified workspace copies directly. MedSAM code, its base checkpoint, and EQ generated state have separate canonical roots and must not depend on legacy project paths.

## What Changes

- Set EQ's default runtime root to `/Users/ncamarda/Workspaces/endotheliosis-quantifier/runtime` while preserving `EQ_RUNTIME_ROOT` as the explicit override.
- Set the MedSAM repository default to `/Users/ncamarda/Workspaces/medsam/source` and the base-checkpoint default to `/Users/ncamarda/Workspaces/medsam/runtime/work_dir/MedSAM/medsam_vit_b.pth`.
- Align active code, configs, scripts, tests, documentation, and OpenSpec contracts to those roots.
- Retain the legacy source/runtime trees and MedSAM source-local payload mirrors as read-only rollback evidence; add no compatibility symlinks or fallback paths.

## Capabilities

### New Capabilities

- `workspace-runtime-layout`: Canonical local source/runtime boundaries for EQ and its external MedSAM dependency.

## Impact

- `src/eq/utils/paths.py` and MedSAM workflow defaults.
- Active YAML configs, operator scripts, README/docs, and path-contract tests.
- No scientific methods, result artifacts, cloud paths, or external-service behavior changes.

## Explicit Decisions

- EQ workspace slug: `endotheliosis-quantifier`.
- MedSAM workspace slug: `medsam`.
- MedSAM checkpoint ownership: workspace runtime, not the source checkout.
- Legacy roots remain present but are not operational defaults.
