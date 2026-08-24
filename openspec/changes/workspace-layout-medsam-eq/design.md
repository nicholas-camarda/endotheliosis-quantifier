## Context

EQ resolves generated artifacts through `EQ_RUNTIME_ROOT` and `src/eq/utils/paths.py`. MedSAM evaluation and fine-tuning workflows separately require an upstream repository and base checkpoint. The copied workspace preserves all original Git state and runtime artifacts while making these ownership boundaries explicit.

## Explicit Decisions

1. `get_active_runtime_root()` returns `/Users/ncamarda/Workspaces/endotheliosis-quantifier/runtime` when `EQ_RUNTIME_ROOT` is unset.
2. Active configs and scripts use the same EQ runtime root; environment overrides continue to resolve first.
3. MedSAM code defaults to `/Users/ncamarda/Workspaces/medsam/source`.
4. The base checkpoint defaults to `/Users/ncamarda/Workspaces/medsam/runtime/work_dir/MedSAM/medsam_vit_b.pth`.
5. Tests execute with workspace source first on `PYTHONPATH` and temporary runtime/output roots so verification cannot write into canonical or legacy runtime data.

## Failure Boundaries

- A missing MedSAM repo or checkpoint fails the existing preflight; there is no legacy-path fallback.
- A caller can select another runtime only through the existing explicit environment/config interfaces.
- Historical archived OpenSpec evidence retains the paths that were true when it was generated.

## Verification

- Path-contract tests prove new defaults and reject retired roots in active text surfaces.
- Full suite, lint, format-check, type check, CLI/help, MedSAM import/checkpoint load, and isolated Label Studio/MedSAM workflow tests run from the workspace copy.
- SHA-256 manifests and post-test protected-surface checks prove retained originals and copied runtimes were not changed by verification.
