#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-surgformer-repro}"
DATA_PATH_HOST="${DATA_PATH_HOST:-/home/ikeido/datasets/Cholec80/cholec80}"
RUN_NAME="${RUN_NAME:-surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4}"
FINETUNE_PATH="${FINETUNE_PATH:-/workspace/outputs/Cholec80/${RUN_NAME}/checkpoint-best.pth}"
# on/off で結果が混ざらないよう、実行ごとに出力先を分ける
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/test_eval}"
CONTAINER_NAME="${CONTAINER_NAME:-surgformer-test-phase}"
GPU_FLAG="${GPU_FLAG:---gpus all}"
GPUS="${GPUS:-1}"
# ホスト全体の OOM/フリーズを防ぐためコンテナのメモリを上限制限する。
# MEM_LIMIT を空にすると無制限(従来動作)。--memory-swap を同値にして swap 流入も抑止。
MEM_LIMIT="${MEM_LIMIT:-24g}"

if [[ ! -d "${DATA_PATH_HOST}" ]]; then
  echo "DATA_PATH_HOST does not exist: ${DATA_PATH_HOST}" >&2
  exit 1
fi

MEM_ARGS=()
if [[ -n "${MEM_LIMIT}" ]]; then
  MEM_ARGS+=(--memory "${MEM_LIMIT}" --memory-swap "${MEM_LIMIT}")
fi

# test_phase.sh へ渡す env を引き継ぐ（DISABLE_SPATIAL_BLACK_MASK 等）
docker run --rm ${GPU_FLAG} \
  --name "${CONTAINER_NAME}" \
  "${MEM_ARGS[@]}" \
  --ipc=host \
  -v "${REPO_ROOT}:/workspace" \
  -v "${DATA_PATH_HOST}:/workspace/data/Cholec80" \
  -w /workspace \
  -e GPUS="${GPUS}" \
  -e FINETUNE_PATH="${FINETUNE_PATH}" \
  -e OUTPUT_ROOT="${OUTPUT_ROOT}" \
  -e DISABLE_SPATIAL_BLACK_MASK="${DISABLE_SPATIAL_BLACK_MASK:-0}" \
  -e BLACK_PIXEL_THRESHOLD="${BLACK_PIXEL_THRESHOLD:-}" \
  -e ABLATE_OFFFIELD="${ABLATE_OFFFIELD:-0}" \
  -e OFFFIELD_RADIUS_SCALE="${OFFFIELD_RADIUS_SCALE:-}" \
  -e OFFFIELD_INVERT="${OFFFIELD_INVERT:-0}" \
  -e INSTR_ATTN_BIAS="${INSTR_ATTN_BIAS:-0}" \
  -e INSTR_LAMBDA="${INSTR_LAMBDA:-}" \
  -e INSTR_BIAS_BLOCKS="${INSTR_BIAS_BLOCKS:-}" \
  -e INSTR_MASK_DIRNAME="${INSTR_MASK_DIRNAME:-}" \
  -e CUT_BLACK="${CUT_BLACK:-1}" \
  -e DATA_SET="${DATA_SET:-Cholec80}" \
  -e BATCH_SIZE="${BATCH_SIZE:-32}" \
  -e NUM_WORKERS="${NUM_WORKERS:-8}" \
  -e MASTER_PORT="${MASTER_PORT:-12326}" \
  "${IMAGE}" \
  bash scripts/test_phase.sh
