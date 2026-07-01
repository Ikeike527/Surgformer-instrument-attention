#!/usr/bin/env bash
# SAM3 器具マスク生成 (train+val split) → 器具ソフトバイアス込みで fine-tune 学習まで一括実行。
#
# マスク生成は sam3env(host) で 1 video ずつ、学習は Docker(surgformer-repro) で実施する。
# 既存マスクは generate_instrument_masks.py 側でスキップされる (--overwrite で上書き)。
#
# 使い方 (既定: video01-48 のマスク生成 → 既存 best ckpt から bias 込み fine-tune):
#   bash scripts/gen_masks_and_train.sh
# マスク生成をスキップして学習だけ:
#   SKIP_GEN=1 bash scripts/gen_masks_and_train.sh
# 生成だけ (学習しない):
#   SKIP_TRAIN=1 bash scripts/gen_masks_and_train.sh
# バッチや range の調整例:
#   START=1 END=40 BATCH_SIZE=2 UPDATE_FREQ=4 bash scripts/gen_masks_and_train.sh
set -euo pipefail

# ---- 共通パス ----
DATA_ROOT="${DATA_ROOT:-/home/ikeido/datasets/Cholec80/cholec80}"
SAM3_DIR="${SAM3_DIR:-/home/ikeido/test/sam3}"
SAM3_PY="${SAM3_PY:-${SAM3_DIR}/sam3env/bin/python}"
GEN_SCRIPT="${GEN_SCRIPT:-${SAM3_DIR}/generate_instrument_masks.py}"
TS_DIR="${TS_DIR:-/home/ikeido/test/Timesformer}"

# 学習に必要なマスクは train(video01-40) + val(video41-48)。test(49-80) は生成済み前提。
START="${START:-1}"
END="${END:-48}"
OVERWRITE="${OVERWRITE:-0}"      # 1 で既存マスクを再生成
SKIP_GEN="${SKIP_GEN:-0}"        # 1 でマスク生成をスキップ (学習のみ)
SKIP_TRAIN="${SKIP_TRAIN:-0}"    # 1 で学習をスキップ (生成のみ)

# ---- 学習 (Docker) 設定 ----
IMAGE="${IMAGE:-surgformer-repro}"
REPO_ROOT="${TS_DIR}"
DATA_PATH_HOST="${DATA_ROOT}"
CONTAINER_NAME="${CONTAINER_NAME:-surgformer-train-phase}"
GPU_FLAG="${GPU_FLAG:---gpus all}"
GPUS="${GPUS:-1}"

# --- メモリ不足(OOM)対策 ---
# GPU: BATCH_SIZE を小さく、UPDATE_FREQ(勾配累積)で実効バッチを補う。
#      実効バッチ = BATCH_SIZE * UPDATE_FREQ * GPUS。
# HOST RAM: NUM_WORKERS を絞り、コンテナ MEM_LIMIT で上限を掛ける。
BATCH_SIZE="${BATCH_SIZE:-4}"        # 保守的既定。OOM なら 2 へ、余裕あれば 8/16 へ
UPDATE_FREQ="${UPDATE_FREQ:-2}"      # 実効バッチ = 4*2 = 8
NUM_WORKERS="${NUM_WORKERS:-2}"      # host RAM 保護。増やすと DataLoader が RAM を食う
MEM_LIMIT="${MEM_LIMIT:-32g}"        # 空で無制限。swap 流入も同値で抑止

# --- 学習ハイパラ ---
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"                     # fine-tune なのでスクラッチ(5e-4)より低め
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
DATA_SET="${DATA_SET:-Cholec80}"
SPLIT_TAG="${SPLIT_TAG:-split_tr01-40_val41-48_test49-80}"

# --- 器具バイアス設定 ---
INSTR_LAMBDA="${INSTR_LAMBDA:-2.0}"      # 推論検証で効果のあった値
INSTR_BIAS_BLOCKS="${INSTR_BIAS_BLOCKS:-all}"
INSTR_MASK_DIRNAME="${INSTR_MASK_DIRNAME:-instrument_masks}"

# --- fine-tune 元 ckpt (既存 phase 学習済みの best) ---
BASE_RUN_NAME="${BASE_RUN_NAME:-surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4}"
FINETUNE="${FINETUNE:-/workspace/outputs/Cholec80/${BASE_RUN_NAME}/checkpoint-best.pth}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/phase_train_instr_l${INSTR_LAMBDA}}"

# ---- 1) マスク生成 (train+val split) ----
if [[ "${SKIP_GEN}" != "1" ]]; then
  if [[ ! -x "${SAM3_PY}" ]]; then
    echo "SAM3 python not found: ${SAM3_PY}" >&2
    exit 1
  fi
  gen_args=()
  [[ "${OVERWRITE}" == "1" ]] && gen_args+=(--overwrite)
  for n in $(seq "${START}" "${END}"); do
    vid="$(printf 'video%02d' "${n}")"
    if [[ ! -d "${DATA_ROOT}/frames/${vid}" && ! -d "${DATA_ROOT}/frames_cutmargin/${vid}" ]]; then
      echo "[skip] ${vid}: frames dir not found" >&2
      continue
    fi
    echo "=== SAM3 masks: ${vid} ==="
    "${SAM3_PY}" "${GEN_SCRIPT}" \
      --video_id "${vid}" --data_root "${DATA_ROOT}" "${gen_args[@]}"
  done
  echo "マスク生成完了: video${START}-${END}"
else
  echo "SKIP_GEN=1: マスク生成をスキップ"
fi

# ---- 2) 器具バイアス込み fine-tune (Docker) ----
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "SKIP_TRAIN=1: 学習をスキップして終了"
  exit 0
fi

if [[ ! -d "${DATA_PATH_HOST}" ]]; then
  echo "DATA_PATH_HOST does not exist: ${DATA_PATH_HOST}" >&2
  exit 1
fi

MEM_ARGS=()
if [[ -n "${MEM_LIMIT}" ]]; then
  MEM_ARGS+=(--memory "${MEM_LIMIT}" --memory-swap "${MEM_LIMIT}")
fi

echo "=== train (instr bias): lambda=${INSTR_LAMBDA} batch=${BATCH_SIZE}x${UPDATE_FREQ} epochs=${EPOCHS} lr=${LR} ==="
echo "    finetune from: ${FINETUNE}"
echo "    output:        ${OUTPUT_ROOT}/${DATA_SET}"

cd "${TS_DIR}"
docker run --rm ${GPU_FLAG} \
  --name "${CONTAINER_NAME}" \
  "${MEM_ARGS[@]}" \
  --ipc=host \
  -v "${REPO_ROOT}:/workspace" \
  -v "${DATA_PATH_HOST}:/workspace/data/Cholec80" \
  -w /workspace \
  -e GPUS="${GPUS}" \
  -e MODEL="${MODEL:-surgformer_HTA}" \
  -e DATA_SET="${DATA_SET}" \
  -e DATA_PATH="/workspace/data/Cholec80" \
  -e EVAL_DATA_PATH="/workspace/data/Cholec80" \
  -e SPLIT_TAG="${SPLIT_TAG}" \
  -e OUTPUT_ROOT="${OUTPUT_ROOT}" \
  -e BATCH_SIZE="${BATCH_SIZE}" \
  -e UPDATE_FREQ="${UPDATE_FREQ}" \
  -e NUM_WORKERS="${NUM_WORKERS}" \
  -e EPOCHS="${EPOCHS}" \
  -e LR="${LR}" \
  -e WARMUP_EPOCHS="${WARMUP_EPOCHS}" \
  -e CUT_BLACK="${CUT_BLACK:-1}" \
  -e FINETUNE="${FINETUNE}" \
  -e INSTR_ATTN_BIAS=1 \
  -e INSTR_LAMBDA="${INSTR_LAMBDA}" \
  -e INSTR_BIAS_BLOCKS="${INSTR_BIAS_BLOCKS}" \
  -e INSTR_MASK_DIRNAME="${INSTR_MASK_DIRNAME}" \
  -e MASTER_PORT="${MASTER_PORT:-12327}" \
  "${IMAGE}" \
  bash scripts/train_phase.sh

echo "完了: masks(video${START}-${END}) + instr-bias fine-tune (lambda=${INSTR_LAMBDA})"
echo "出力: ${OUTPUT_ROOT}/${DATA_SET}"
