# Cholec80 Off-field Ablation

## Purpose

This project isolates the Cholec80 phase-classification experiment that tests:

- baseline phase-recognition performance
- pure-black spatial patch masking on/off
- causal dependence on off-field / peripheral image regions via input ablation

## Scope

- Dataset: `Cholec80`
- Split: `train=video01-40`, `val=video41-48`, `test=video49-80`
- Model: `surgformer_HTA`
- Main run name:
  - `surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4`

## Files

- [RESULT.md](./RESULT.md): consolidated test results
- [RUNBOOK.md](./RUNBOOK.md): commands used to reproduce evaluation summaries
- [../../Result.md](../../Result.md): root-level result summary created during analysis
- [../../PLAN.md](../../PLAN.md): detailed analysis log and interpretation

## Referenced Output Directories

- `outputs/Cholec80/...`
- `outputs/test_eval_off/...`
- `outputs/test_eval_on/...`
- `outputs/test_eval_ablate_r1.0/...`
- `outputs/test_eval_ablate_r0.9/...`
- `outputs/test_eval_ablate_r0.8/...`

## Conclusion

The main checkpoint reaches `Top-1 = 90.79` and `Mean Accuracy (per-video) = 92.83` on `video49-80`.
Pure-black patch masking has negligible effect, while off-field ablation causes small but monotonic degradation, indicating weak causal dependence on peripheral regions.
