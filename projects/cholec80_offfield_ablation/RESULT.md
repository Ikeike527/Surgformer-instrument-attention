# Result

## Main Test Result

Target run:

- `outputs/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4`

| Metric | Value |
|---|---:|
| Frame Top-1 accuracy | 90.79 |
| Frame Top-5 accuracy | 99.67 |
| Mean Accuracy (per-video) | 92.83 |
| Mean Jaccard (per-phase) | 83.58 |
| Mean Precision (per-phase) | 92.30 |
| Mean Recall (per-phase) | 91.42 |
| Mean F1 (per-phase) | 91.65 |

## Phase-wise Result

| Phase | Jaccard | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Preparation | 81.41 | 96.93 | 84.98 | 91.93 |
| CalotTriangleDissection | 91.55 | 95.15 | 96.46 | 95.88 |
| ClippingCutting | 87.60 | 94.16 | 93.30 | 93.64 |
| GallbladderDissection | 88.84 | 92.36 | 96.89 | 93.81 |
| GallbladderPackaging | 82.73 | 87.78 | 97.18 | 92.99 |
| CleaningCoagulation | 69.58 | 87.76 | 78.61 | 80.98 |
| GallbladderRetraction | 83.32 | 91.93 | 92.53 | 92.34 |

## Mask ON/OFF

| Condition | Top-1 | Top-5 | Mean Acc | Mean Jaccard | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mask OFF | 90.79 | 99.67 | 92.83 | 83.58 | 92.30 | 91.42 | 91.65 |
| Mask ON | 90.65 | 99.55 | 92.70 | 83.45 | 92.26 | 91.32 | 91.54 |

Delta (`ON - OFF`):

- Top-1: `-0.14`
- Mean Accuracy: `-0.13`
- Mean Jaccard: `-0.13`
- Mean F1: `-0.11`

## Off-field Ablation

Baseline for this block: `test_eval_on`

| Condition | Top-1 | Top-5 | Mean Acc | Mean Jaccard | Mean F1 | Flip Rate | Correct->Wrong / Wrong->Correct | Mean KL |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Baseline | 90.65 | 99.55 | 92.70 | 83.45 | 91.54 | - | - | - |
| Ablate `r=1.0` | 90.22 | 99.47 | 92.32 | 82.86 | 91.33 | 1.90% | 798 / 470 | 0.01283 |
| Ablate `r=0.9` | 89.61 | 99.52 | 91.92 | 82.29 | 90.93 | 3.01% | 1403 / 601 | 0.02584 |
| Ablate `r=0.8` | 88.51 | 99.49 | 91.17 | 81.10 | 90.19 | 4.66% | 2393 / 746 | 0.04833 |

## Summary

- Pure-black patch masking alone is effectively neutral on the test split.
- Stronger off-field removal produces monotonic degradation.
- The model depends on peripheral/off-field information, but the dependence is limited rather than dominant.
