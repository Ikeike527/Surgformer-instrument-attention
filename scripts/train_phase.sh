#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-12324}"
MODEL="${MODEL:-surgformer_HTA}"
DATA_SET="${DATA_SET:-AutoLaparo}"
DATA_PATH="${DATA_PATH:-/workspace/data/AutoLaparo}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$DATA_PATH}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs}"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"
DEVICE="${DEVICE:-cuda}"
DATA_FPS="${DATA_FPS:-1fps}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-5e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
NUM_FRAMES="${NUM_FRAMES:-16}"
SAMPLING_RATE="${SAMPLING_RATE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_PATH="${RESUME_PATH:-}"
FINETUNE="${FINETUNE:-}"        # 既存 ckpt から重みを読み fine-tune (epoch/optim は初期化)
UPDATE_FREQ="${UPDATE_FREQ:-1}" # 勾配累積。BATCH_SIZE を下げても実効バッチを保つ (OOM 対策)
SPLIT_TAG="${SPLIT_TAG:-}"
# 器具ソフトバイアス学習: INSTR_ATTN_BIAS=1 で有効化。
# 有効時はマスクとフレームの整合を保つため mixup/cutmix を無効化する
# (mixup は 2 サンプルを混合するが instr_mask は混合されないため)。
INSTR_ATTN_BIAS="${INSTR_ATTN_BIAS:-0}"
INSTR_LAMBDA="${INSTR_LAMBDA:-1.0}"
INSTR_BIAS_BLOCKS="${INSTR_BIAS_BLOCKS:-all}"
INSTR_MASK_DIRNAME="${INSTR_MASK_DIRNAME:-instrument_masks}"
if [[ "${INSTR_ATTN_BIAS}" == "1" ]]; then
  MIXUP="${MIXUP:-0.0}"
  CUTMIX="${CUTMIX:-0.0}"
else
  MIXUP="${MIXUP:-0.8}"
  CUTMIX="${CUTMIX:-1.0}"
fi
EXTRA_ARGS=()

if [[ "${INSTR_ATTN_BIAS}" == "1" ]]; then
  EXTRA_ARGS+=(--instr_attn_bias)
  EXTRA_ARGS+=(--instr_lambda "${INSTR_LAMBDA}")
  EXTRA_ARGS+=(--instr_bias_blocks "${INSTR_BIAS_BLOCKS}")
  EXTRA_ARGS+=(--instr_mask_dirname "${INSTR_MASK_DIRNAME}")
fi

if [[ -n "${PRETRAINED_PATH}" ]]; then
  EXTRA_ARGS+=(--pretrained_path "${PRETRAINED_PATH}")
fi
if [[ "${CUT_BLACK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--cut_black)
fi
if [[ "${DISABLE_SPATIAL_BLACK_MASK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_spatial_black_mask)
fi
if [[ -n "${BLACK_PIXEL_THRESHOLD:-}" ]]; then
  EXTRA_ARGS+=(--black_pixel_threshold "${BLACK_PIXEL_THRESHOLD}")
fi
if [[ "${DIST_EVAL:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--dist_eval)
fi
if [[ "${ENABLE_DEEPSPEED:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_deepspeed)
fi
if [[ "${AUTO_RESUME}" == "0" ]]; then
  EXTRA_ARGS+=(--no_auto_resume)
fi
if [[ -n "${RESUME_PATH}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_PATH}")
fi
if [[ -n "${FINETUNE}" ]]; then
  EXTRA_ARGS+=(--finetune "${FINETUNE}")
fi
if [[ -n "${SPLIT_TAG}" ]]; then
  EXTRA_ARGS+=(--split_tag "${SPLIT_TAG}")
fi

torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  --master_port "${MASTER_PORT}" \
  downstream_phase/run_phase_training.py \
  --batch_size "${BATCH_SIZE}" \
  --update_freq "${UPDATE_FREQ}" \
  --epochs "${EPOCHS}" \
  --save_ckpt_freq "${SAVE_CKPT_FREQ}" \
  --model "${MODEL}" \
  --mixup "${MIXUP}" \
  --cutmix "${CUTMIX}" \
  --smoothing 0.1 \
  --lr "${LR}" \
  --layer_decay 0.75 \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --data_path "${DATA_PATH}" \
  --eval_data_path "${EVAL_DATA_PATH}" \
  --nb_classes 7 \
  --data_strategy online \
  --output_mode key_frame \
  --num_frames "${NUM_FRAMES}" \
  --sampling_rate "${SAMPLING_RATE}" \
  --data_set "${DATA_SET}" \
  --data_fps "${DATA_FPS}" \
  --output_dir "${OUTPUT_ROOT}/${DATA_SET}" \
  --log_dir "${OUTPUT_ROOT}/${DATA_SET}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
