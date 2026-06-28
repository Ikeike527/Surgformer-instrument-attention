#!/usr/bin/env bash
# SAM3 器具マスク生成 (test split 全 video) → そのまま test_phase 評価まで一括実行。
#
# マスク生成は sam3env(host) で 1 video ずつ、評価は Docker(surgformer-repro) で実施する。
# 既存マスクは generate_instrument_masks.py 側でスキップされるため再実行は安全 (--overwrite で上書き)。
#
# 使い方 (既定: video49-80 を生成し、lambda=0 と 2 で評価):
#   bash scripts/gen_masks_and_test.sh
# 例: lambda を 1 本だけ、video 範囲を限定:
#   START=49 END=55 LAMBDAS="2" bash scripts/gen_masks_and_test.sh
# 例: マスク生成をスキップして評価だけ:
#   SKIP_GEN=1 LAMBDAS="0 2 3" bash scripts/gen_masks_and_test.sh
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/ikeido/datasets/Cholec80/cholec80}"
SAM3_DIR="${SAM3_DIR:-/home/ikeido/test/sam3}"
SAM3_PY="${SAM3_PY:-${SAM3_DIR}/sam3env/bin/python}"
GEN_SCRIPT="${GEN_SCRIPT:-${SAM3_DIR}/generate_instrument_masks.py}"
TS_DIR="${TS_DIR:-/home/ikeido/test/Timesformer}"

START="${START:-49}"          # test split = video49-80
END="${END:-80}"
LAMBDAS="${LAMBDAS:-0 2}"      # 空白区切り。マスクは一度生成し各 lambda で評価
OVERWRITE="${OVERWRITE:-0}"    # 1 で既存マスクを再生成
SKIP_GEN="${SKIP_GEN:-0}"      # 1 でマスク生成をスキップ (評価のみ)

# 評価時 Docker へ渡す既定 (host RAM/GPU 保護。docker_test_phase.sh の既定に準拠)
export NUM_WORKERS="${NUM_WORKERS:-2}"
export MEM_LIMIT="${MEM_LIMIT:-50g}"
export BATCH_SIZE="${BATCH_SIZE:-16}"

# ---- 1) マスク生成 (test split 全 video) ----
if [[ "${SKIP_GEN}" != "1" ]]; then
  gen_args=()
  [[ "${OVERWRITE}" == "1" ]] && gen_args+=(--overwrite)
  for n in $(seq "${START}" "${END}"); do
    vid="video${n}"
    if [[ ! -d "${DATA_ROOT}/frames/${vid}" ]]; then
      echo "[skip] ${vid}: frames dir not found" >&2
      continue
    fi
    echo "=== SAM3 masks: ${vid} ==="
    "${SAM3_PY}" "${GEN_SCRIPT}" \
      --video_id "${vid}" --data_root "${DATA_ROOT}" "${gen_args[@]}"
  done
fi

# ---- 2) test_phase 評価 + 集計 (instr bias, lambda ごとに出力先を分ける) ----
cd "${TS_DIR}"
IMAGE="${IMAGE:-surgformer-repro}"
REPO_ROOT="${TS_DIR}"
DATA_PATH_HOST="${DATA_ROOT}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/workspace/outputs/test_instr}"

for lam in ${LAMBDAS}; do
  out_root="${OUTPUT_ROOT_BASE}_l${lam}"          # コンテナ内パス (/workspace/...)
  ds_dir="${out_root}/${DATA_SET:-Cholec80}"

  echo "=== test_phase: instr_lambda=${lam} (OUTPUT_ROOT=${out_root}) ==="
  INSTR_ATTN_BIAS=1 INSTR_LAMBDA="${lam}" OUTPUT_ROOT="${out_root}" \
    bash scripts/docker_test_phase.sh

  echo "=== aggregate: instr_lambda=${lam} ==="
  # 0.txt は ${ds_dir}/<RUN_NAME>/0.txt と 1 階層深い。RUN_NAME 非依存に動的解決する。
  # 出力が root 所有のため集計も Docker 内で実行。結果は metrics_l${lam}.txt に保存。
  docker run --rm \
    --memory "${MEM_LIMIT}" --memory-swap "${MEM_LIMIT}" --ipc=host \
    -v "${REPO_ROOT}:/workspace" \
    -v "${DATA_PATH_HOST}:/workspace/data/Cholec80" \
    -w /workspace \
    "${IMAGE}" \
    bash -lc "mp=\$(dirname \"\$(ls ${ds_dir}/*/0.txt | head -1)\") && \
              echo \"main-path=\$mp\" && \
              python datasets/convert_results/convert_cholec80.py --main-path \"\$mp\" && \
              python scripts/eval_phase_python.py --main-path \"\$mp\" --start ${START} --end ${END} \
                | tee \"\$mp/metrics_l${lam}.txt\""
done

echo "完了: masks(${START}-${END}) + test+集計 lambda={${LAMBDAS}}"
echo "各 lambda の指標: ${OUTPUT_ROOT_BASE}_l<λ>/${DATA_SET:-Cholec80}/metrics_l<λ>.txt"
