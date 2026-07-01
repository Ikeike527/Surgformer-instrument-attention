#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-surgformer-repro}"
DATA_PATH_HOST="${DATA_PATH_HOST:-/home/ikeido/datasets/Cholec80/cholec80}"
RUN_NAME="${RUN_NAME:-surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/workspace/outputs/Cholec80/${RUN_NAME}/checkpoint-best.pth}"
OUT_DIR="${OUT_DIR:-/workspace/outputs/attn_vis}"
VIDEO_ID="${VIDEO_ID:-49}"
FRAME_ID="${FRAME_ID:-1000}"
CONTAINER_NAME="${CONTAINER_NAME:-surgformer-attn-vis}"
GPU_FLAG="${GPU_FLAG:---gpus all}"
# ホスト全体の OOM/フリーズを防ぐためコンテナのメモリを上限制限する。
# MEM_LIMIT を空にすると無制限(従来動作)。--memory-swap を同値にして swap 流入も抑止。
MEM_LIMIT="${MEM_LIMIT:-24g}"
BLACK_PIXEL_THRESHOLD="${BLACK_PIXEL_THRESHOLD:-}"
VIS_SCRIPT="${VIS_SCRIPT:-downstream_phase/visualize_attention.py}"

if [[ ! -d "${DATA_PATH_HOST}" ]]; then
  echo "DATA_PATH_HOST does not exist: ${DATA_PATH_HOST}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${VIDEO_ID}" ]]; then
  EXTRA_ARGS+=(--video_id "${VIDEO_ID}")
fi
if [[ -n "${FRAME_ID}" ]]; then
  EXTRA_ARGS+=(--frame_id "${FRAME_ID}")
fi
if [[ "${CUT_BLACK:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--cut_black)
fi
if [[ "${SAVE_NPZ:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--save_npz)
fi
if [[ "${HIDE_FRAME_W:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--hide_frame_w)
fi
if [[ "${NO_VISIBLE_MASK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no_visible_mask)
fi
if [[ "${DISABLE_SPATIAL_BLACK_MASK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_spatial_black_mask)
fi
if [[ -n "${BLACK_PIXEL_THRESHOLD}" ]]; then
  EXTRA_ARGS+=(--black_pixel_threshold "${BLACK_PIXEL_THRESHOLD}")
fi
if [[ "${PER_PHASE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--per_phase)
fi
if [[ -n "${PER_PHASE_MODE:-}" ]]; then
  EXTRA_ARGS+=(--per_phase_mode "${PER_PHASE_MODE}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ "${INSTR_ATTN_BIAS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--instr_attn_bias)
fi
if [[ -n "${INSTR_LAMBDA:-}" ]]; then
  EXTRA_ARGS+=(--instr_lambda "${INSTR_LAMBDA}")
fi
if [[ -n "${INSTR_BIAS_BLOCKS:-}" ]]; then
  EXTRA_ARGS+=(--instr_bias_blocks "${INSTR_BIAS_BLOCKS}")
fi
if [[ -n "${INSTR_MASK_DIRNAME:-}" ]]; then
  EXTRA_ARGS+=(--instr_mask_dirname "${INSTR_MASK_DIRNAME}")
fi

MEM_ARGS=()
if [[ -n "${MEM_LIMIT}" ]]; then
  MEM_ARGS+=(--memory "${MEM_LIMIT}" --memory-swap "${MEM_LIMIT}")
fi

docker run --rm ${GPU_FLAG} \
  --name "${CONTAINER_NAME}" \
  "${MEM_ARGS[@]}" \
  --ipc=host \
  -v "${REPO_ROOT}:/workspace" \
  -v "${DATA_PATH_HOST}:/workspace/data/Cholec80" \
  -w /workspace \
  "${IMAGE}" \
  python "${VIS_SCRIPT}" \
    --finetune "${CHECKPOINT_PATH}" \
    --data_path /workspace/data/Cholec80 \
    --out_dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"
