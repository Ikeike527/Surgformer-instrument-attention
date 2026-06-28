# Runbook

## Phase Metric Aggregation

```bash
python3 scripts/eval_phase_python.py \
  --main-path outputs/test_eval_off/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

python3 scripts/eval_phase_python.py \
  --main-path outputs/test_eval_on/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

python3 scripts/eval_phase_python.py \
  --main-path outputs/test_eval_ablate_r1.0/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

python3 scripts/eval_phase_python.py \
  --main-path outputs/test_eval_ablate_r0.9/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

python3 scripts/eval_phase_python.py \
  --main-path outputs/test_eval_ablate_r0.8/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4
```

## Ablation-vs-Baseline Comparison

```bash
python3 scripts/compare_predictions.py \
  --baseline outputs/test_eval_on/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4 \
  --ablation outputs/test_eval_ablate_r1.0/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

python3 scripts/compare_predictions.py \
  --baseline outputs/test_eval_on/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4 \
  --ablation outputs/test_eval_ablate_r0.9/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

python3 scripts/compare_predictions.py \
  --baseline outputs/test_eval_on/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4 \
  --ablation outputs/test_eval_ablate_r0.8/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4
```

## Raw Result Sources

- Main log:
  - `outputs/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/log.txt`
- Converted predictions:
  - `outputs/test_eval_on/Cholec80/.../prediction/`
  - `outputs/test_eval_on/Cholec80/.../phase_annotations/`

## Notes

- `log.txt` provides frame-level `Top-1` / `Top-5`.
- `scripts/eval_phase_python.py` provides per-video and per-phase evaluation.
- `scripts/compare_predictions.py` provides flip rate and KL-based perturbation analysis.
