# Label Studio MedSAM Companion (MPS host)

This companion exposes real MedSAM box-prompt inference over HTTP for hybrid Label Studio grading.

## Runtime contract

- `GET /healthz`
  - Returns `200` with `{status, device, checkpoint, model_type}` when model is loaded.
- `POST /v1/box_infer`
  - Request JSON:
    - `image_path`: absolute path to source image on host filesystem
    - `box_xyxy`: `[x0, y0, x1, y1]` in original pixel coordinates
  - Response JSON:
    - `width`, `height`
    - `rle` (binary run-length payload for `brushlabels`)
    - `foreground_pixels`
    - `model_type`, `device`, `image_path`

## Launch on macOS (`eq-mac`, MPS)

Set `EQ_SOURCE_ROOT` to the source checkout and `EQ_RUNTIME_ROOT` to the active
runtime tree before launching the companion.

Default development launch:

```bash
cd "$EQ_SOURCE_ROOT"
conda activate eq-mac
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m eq.labelstudio.medsam_companion \
  --checkpoint "$EQ_RUNTIME_ROOT/output/segmentation_evaluation/medsam_glomeruli_fine_tuning/deploy_conservative_mps_glomeruli/finetuned_evaluation/medsam_glomeruli_best_sam_state_dict.pth" \
  --device mps \
  --port 8098
```

Then run `eq labelstudio start ...` in a second terminal and open the printed project URL. Configure `configs/label_studio_medsam_hybrid.yaml` to point to the same base URL and health path when changing host/port.

## Notes

- Keep the companion on the host when using MPS; Docker containers on macOS do not expose Metal/MPS acceleration for this workflow.
- The Label Studio Docker container continues to run separately via `eq labelstudio start`.
- Do not commit checkpoints or runtime logs to Git.
