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
NUM_FRAMES="${NUM_FRAMES:-16}"
SAMPLING_RATE="${SAMPLING_RATE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_CKPT_FREQ="${SAVE_CKPT_FREQ:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_PATH="${RESUME_PATH:-}"
SPLIT_TAG="${SPLIT_TAG:-}"
EXTRA_ARGS=()

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
if [[ -n "${SPLIT_TAG}" ]]; then
  EXTRA_ARGS+=(--split_tag "${SPLIT_TAG}")
fi

torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  --master_port "${MASTER_PORT}" \
  downstream_phase/run_phase_training.py \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --save_ckpt_freq "${SAVE_CKPT_FREQ}" \
  --model "${MODEL}" \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --lr 5e-4 \
  --layer_decay 0.75 \
  --warmup_epochs 5 \
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
