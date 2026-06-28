# Test Result Summary

## Scope

- Dataset: `Cholec80`
- Test split: `video49-80` (`32` videos)
- Frame count used in comparison runs: `76,757`
- Main metrics source:
  - frame-level `Top-1` / `Top-5`: each run's `log.txt`
  - per-video / per-phase metrics: `python3 scripts/eval_phase_python.py --main-path ...`
  - ablation flip/KL stats: `python3 scripts/compare_predictions.py --baseline ... --ablation ...`

## Main Test Result

Primary checkpoint run:

- `outputs/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4`

This run matches the replayed `test_eval_off` result.

| Metric | Value |
|---|---:|
| Frame Top-1 accuracy | 90.79 |
| Frame Top-5 accuracy | 99.67 |
| Mean Accuracy (per-video) | 92.83 |
| Mean Jaccard (per-phase) | 83.58 |
| Mean Precision (per-phase) | 92.30 |
| Mean Recall (per-phase) | 91.42 |
| Mean F1 (per-phase) | 91.65 |

### Phase-wise Result

| Phase | Jaccard | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Preparation | 81.41 | 96.93 | 84.98 | 91.93 |
| CalotTriangleDissection | 91.55 | 95.15 | 96.46 | 95.88 |
| ClippingCutting | 87.60 | 94.16 | 93.30 | 93.64 |
| GallbladderDissection | 88.84 | 92.36 | 96.89 | 93.81 |
| GallbladderPackaging | 82.73 | 87.78 | 97.18 | 92.99 |
| CleaningCoagulation | 69.58 | 87.76 | 78.61 | 80.98 |
| GallbladderRetraction | 83.32 | 91.93 | 92.53 | 92.34 |

## Mask ON/OFF Comparison

Compared runs:

- `outputs/test_eval_off/Cholec80/...`
- `outputs/test_eval_on/Cholec80/...`

| Condition | Top-1 | Top-5 | Mean Acc | Mean Jaccard | Mean Precision | Mean Recall | Mean F1 | Delta Top-1 vs OFF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mask OFF | 90.79 | 99.67 | 92.83 | 83.58 | 92.30 | 91.42 | 91.65 | +0.00 |
| Mask ON | 90.65 | 99.55 | 92.70 | 83.45 | 92.26 | 91.32 | 91.54 | -0.14 |

Interpretation:

- `mask ON` and `mask OFF` are nearly identical.
- Pure-black patch masking does not materially change test performance on this split.

## Off-field Input Ablation

Baseline for this comparison: `test_eval_on`

| Condition | Top-1 | Top-5 | Mean Acc | Mean Jaccard | Mean F1 | Flip Rate | Correct->Wrong / Wrong->Correct | Mean KL | Delta Top-1 vs ON |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Baseline (no ablation) | 90.65 | 99.55 | 92.70 | 83.45 | 91.54 | - | - | - | +0.00 |
| Ablate `r=1.0` | 90.22 | 99.47 | 92.32 | 82.86 | 91.33 | 1.90% | 798 / 470 | 0.01283 | -0.43 |
| Ablate `r=0.9` | 89.61 | 99.52 | 91.92 | 82.29 | 90.93 | 3.01% | 1403 / 601 | 0.02584 | -1.05 |
| Ablate `r=0.8` | 88.51 | 99.49 | 91.17 | 81.10 | 90.19 | 4.66% | 2393 / 746 | 0.04833 | -2.15 |

Interpretation:

- Removing more off-field / boundary-region pixels causes monotonic degradation.
- The effect exists, but it is modest at `r=1.0` and still limited even at `r=0.8`.
- The model shows weak causal dependence on off-field / peripheral regions.

## Key Points

- The main checkpoint reaches `90.79` frame-level `Top-1` and `92.83` mean per-video accuracy on `video49-80`.
- `CleaningCoagulation` is the weakest phase (`F1 = 80.98`), while `CalotTriangleDissection` is the strongest (`F1 = 95.88`).
- Pure-black spatial masking alone has negligible impact.
- Input ablation reveals a small but real dependence on peripheral / off-field content.

## Source Paths

- Main checkpoint log:
  - `outputs/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/log.txt`
- Replayed evaluation logs:
  - `outputs/test_eval_off/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/log.txt`
  - `outputs/test_eval_on/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/log.txt`
  - `outputs/test_eval_ablate_r1.0/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/log.txt`
  - `outputs/test_eval_ablate_r0.9/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/log.txt`
  - `outputs/test_eval_ablate_r0.8/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/log.txt`
