#!/usr/bin/env bash
# 学習済み(fine-tune済み)モデルを test split(video49-80) で評価し、指標を集計する。
# 任意の checkpoint と 推論時λ を指定できる。マスク(video49-80)は生成済み前提。
#
# 使い方 (既定: phase_train_instr_l2.0 の best を λ=2 で評価):
#   bash scripts/eval_trained_instr.sh
# checkpoint と λ を指定:
#   CKPT=/workspace/outputs/.../checkpoint-6.pth EVAL_LAMBDA=2 bash scripts/eval_trained_instr.sh
#   EVAL_LAMBDA=0 bash scripts/eval_trained_instr.sh    # 学習済みモデルをバイアスOFFで評価
set -euo pipefail

TS_DIR="${TS_DIR:-/home/ikeido/test/Timesformer}"
IMAGE="${IMAGE:-surgformer-repro}"
DATA_ROOT="${DATA_ROOT:-/home/ikeido/datasets/Cholec80/cholec80}"
REPO_ROOT="${TS_DIR}"

START="${START:-49}"
END="${END:-80}"
MEM_LIMIT="${MEM_LIMIT:-24g}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-16}"

# 評価対象 checkpoint (コンテナ内パス /workspace/...)
TRAIN_RUN="${TRAIN_RUN:-surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0001_0.75_online_key_frame_frame16_Fixed_Stride_4}"
CKPT="${CKPT:-/workspace/outputs/phase_train_instr_l2.0/Cholec80/${TRAIN_RUN}/checkpoint-best.pth}"

# 推論時に適用するλ (学習時と揃えるのが基本。0 で bias OFF)
EVAL_LAMBDA="${EVAL_LAMBDA:-2}"
INSTR_BIAS_BLOCKS="${INSTR_BIAS_BLOCKS:-all}"

# 出力先 (評価ごとに分離)
TAG="${TAG:-trained_l2_eval_l${EVAL_LAMBDA}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/test_${TAG}}"
DATA_SET="${DATA_SET:-Cholec80}"
ds_dir="${OUTPUT_ROOT}/${DATA_SET}"

cd "${TS_DIR}"

echo "=== eval: ckpt=${CKPT}"
echo "         eval_lambda=${EVAL_LAMBDA}  out=${OUTPUT_ROOT} ==="
INSTR_ATTN_BIAS=1 INSTR_LAMBDA="${EVAL_LAMBDA}" INSTR_BIAS_BLOCKS="${INSTR_BIAS_BLOCKS}" \
  FINETUNE_PATH="${CKPT}" OUTPUT_ROOT="${OUTPUT_ROOT}" MEM_LIMIT="${MEM_LIMIT}" \
  bash scripts/docker_test_phase.sh

echo "=== aggregate ==="
docker run --rm \
  --memory "${MEM_LIMIT}" --memory-swap "${MEM_LIMIT}" --ipc=host \
  -v "${REPO_ROOT}:/workspace" \
  -v "${DATA_ROOT}:/workspace/data/Cholec80" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "mp=\$(dirname \"\$(ls ${ds_dir}/*/0.txt | head -1)\") && \
            echo \"main-path=\$mp\" && \
            python datasets/convert_results/convert_cholec80.py --main-path \"\$mp\" && \
            python scripts/eval_phase_python.py --main-path \"\$mp\" --start ${START} --end ${END} \
              | tee \"\$mp/metrics_${TAG}.txt\""

echo "完了: ${ds_dir}/*/metrics_${TAG}.txt"
