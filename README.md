# Surgformer Reproduction

This workspace is a Docker-friendly local reproduction of [`isyangshu/Surgformer`](https://github.com/isyangshu/Surgformer). The upstream code has been normalized for local use by removing hard-coded paths, making preprocessing scripts configurable, and adding Docker-based setup and run instructions.

## What You Need

- A CUDA-capable GPU and NVIDIA Container Toolkit on the host.
- One of the surgical phase datasets used by the paper:
  - `Cholec80`
  - `AutoLaparo`
- Recommended pretrained initialization weights:
  - `TimeSformer_divST_8x32_224_K400.pyth`

The dataset is required for training or evaluation. The TimeSformer checkpoint is not required to make the code run, but it is effectively required if you want to reproduce the reported performance instead of training from scratch.

## Repository Layout

```text
.
├── data/
├── datasets/
├── downstream_phase/
├── model/
├── outputs/
├── pretrain_params/
├── scripts/
└── Dockerfile
```

- Put datasets under `data/`.
- Put pretrained weights under `pretrain_params/`.
- Training outputs are written to `outputs/`.

## Build The Docker Image

```bash
docker build -t surgformer-repro .
```

## Start A Container

```bash
docker run --gpus all --ipc=host --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  surgformer-repro
```

If your datasets live outside this repository, mount them explicitly:

```bash
docker run --gpus all --ipc=host --rm -it \
  -v "$(pwd)":/workspace \
  -v /absolute/path/to/datasets:/workspace/data \
  -w /workspace \
  surgformer-repro
```

`--ipc=host` is recommended for training and evaluation runs that use multi-worker `DataLoader`s.
Without it, Docker's default shared-memory limit can cause `DataLoader worker ... exited unexpectedly` errors.

## Expected Data Layout

### Cholec80

```text
data/Cholec80/
├── tool_annotations/
├── phase_annotations/
├── frames/
├── frames_cutmargin/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The native layout mounted from `/home/ikeido/datasets/Cholec80/cholec80` is supported directly.
This reproduction uses the split `train=video01-40`, `val=video41-48`, `test=video49-80`.
If `labels/train/*.pickle`, `labels/val/*.pickle`, and `labels/test/*.pickle` are missing, the loader will build sample metadata on the fly from `frames/`, `phase_annotations/`, and `tool_annotations/`.

### AutoLaparo

```text
data/AutoLaparo/
├── videos/
├── labels/
├── frames/
└── labels_pkl/
    ├── train/
    ├── val/
    └── test/
```

## Preprocess Datasets

### Cholec80

```bash
python3 datasets/data_preprosses/extract_frames_ch80.py \
  --root-dir data/Cholec80 \
  --output-fps 1

python3 datasets/data_preprosses/generate_labels_ch80.py \
  --root-dir data/Cholec80 \
  --fps-tag 1fps \
  --output-fps 1

python3 datasets/data_preprosses/frame_cutmargin.py \
  --source-dir data/Cholec80/frames \
  --save-dir data/Cholec80/frames_cutmargin
```

### AutoLaparo

```bash
python3 datasets/data_preprosses/extract_frames_autolaparo.py \
  --root-dir data/AutoLaparo \
  --output-fps 1

python3 datasets/data_preprosses/generate_labels_autolaparo.py \
  --root-dir data/AutoLaparo \
  --fps-tag 1fps
```

## Pretrained Weights

Download the recommended TimeSformer initialization and place it at:

```text
pretrain_params/TimeSformer_divST_8x32_224_K400.pyth
```

If you leave `PRETRAINED_PATH` empty, the model will initialize randomly.

## Training

`bash scripts/train_phase.sh` now saves a checkpoint every epoch by default and keeps auto-resume enabled.
If you restart with the same training arguments and the same mounted `outputs/`, the latest `checkpoint-*.pth` is picked up automatically.
If a run only has `checkpoint-best.pth`, that file is used as the fallback resume point.
For `Cholec80`, run directories are automatically tagged with `split_tr01-40_val41-48_test49-80`, so new models are stored separately from the earlier `val=test` runs.

### AutoLaparo Example

```bash
GPUS=1 \
DATA_SET=AutoLaparo \
DATA_PATH=/workspace/data/AutoLaparo \
EVAL_DATA_PATH=/workspace/data/AutoLaparo \
PRETRAINED_PATH=/workspace/pretrain_params/TimeSformer_divST_8x32_224_K400.pyth \
bash scripts/train_phase.sh
```

### Cholec80 Example

```bash
GPUS=1 \
DATA_SET=Cholec80 \
DATA_PATH=/workspace/data/Cholec80 \
EVAL_DATA_PATH=/workspace/data/Cholec80 \
PRETRAINED_PATH=/workspace/pretrain_params/TimeSformer_divST_8x32_224_K400.pyth \
CUT_BLACK=1 \
bash scripts/train_phase.sh
```

### AVT Baseline

```bash
GPUS=1 \
DATA_SET=Cholec80 \
DATA_PATH=/workspace/data/Cholec80 \
EVAL_DATA_PATH=/workspace/data/Cholec80 \
bash scripts/train_phase_avt.sh
```

Optional overrides:

```bash
SAVE_CKPT_FREQ=5 AUTO_RESUME=0 bash scripts/train_phase.sh
RUN_NAME=surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4
SAVE_CKPT_FREQ=1 RESUME_PATH=/workspace/outputs/Cholec80/$RUN_NAME/checkpoint-9.pth bash scripts/train_phase.sh
SPLIT_TAG=my_experiment_tag bash scripts/train_phase.sh
```

### Pause Or Resume Later

- `docker pause surgformer-train-ch80`
  Pauses the container in place. GPU memory stays allocated.
- `docker unpause surgformer-train-ch80`
  Continues the paused run.
- `docker stop surgformer-train-ch80`
  Stops the run and releases resources. Resume by starting a new container with the same mounted `outputs/` and the same training arguments.

Restart example with automatic resume:

```bash
docker run -d --name surgformer-train-ch80 --gpus all --ipc=host \
  -v /home/ikeido/test/Timesformer:/workspace \
  -v /home/ikeido/datasets/Cholec80/cholec80:/workspace/data/Cholec80 \
  -w /workspace \
  surgformer-repro \
  bash -lc 'GPUS=1 DATA_SET=Cholec80 DATA_PATH=/workspace/data/Cholec80 EVAL_DATA_PATH=/workspace/data/Cholec80 PRETRAINED_PATH=/workspace/pretrain_params/TimeSformer_divST_8x32_224_K400.pyth OUTPUT_ROOT=/workspace/outputs CUT_BLACK=0 BATCH_SIZE=4 NUM_WORKERS=8 bash scripts/train_phase.sh'
```

## Evaluation

```bash
RUN_NAME=surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

GPUS=1 \
DATA_SET=Cholec80 \
DATA_PATH=/workspace/data/Cholec80 \
EVAL_DATA_PATH=/workspace/data/Cholec80 \
FINETUNE_PATH=/workspace/outputs/Cholec80/$RUN_NAME/checkpoint-best.pth \
CUT_BLACK=1 \
bash scripts/test_phase.sh
```

Convert distributed rank outputs into per-video prediction files:

```bash
RUN_NAME=surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

python3 datasets/convert_results/convert_cholec80.py \
  --main-path /workspace/outputs/Cholec80/$RUN_NAME

RUN_NAME=surgformer_AutoLaparo_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4
python3 datasets/convert_results/convert_autolaparo.py \
  --main-path /workspace/outputs/AutoLaparo/$RUN_NAME
```

MATLAB evaluation scripts remain under `evaluation_matlab/`.

Visualize test predictions directly from raw rank outputs:

```bash
RUN_NAME=surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

python3 scripts/visualize_phase_predictions.py \
  --run-dir /workspace/outputs/Cholec80/$RUN_NAME \
  --videos video49 video50
```

Figures are written to `$RUN_NAME/figs/` by default.
If that directory is Docker-owned and not writable from the host, the script falls back to `./figs/$RUN_NAME/`.

Visualize attention maps for one example per phase from a test video:

```bash
RUN_NAME=surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

python3 downstream_phase/visualize_attention.py \
  --finetune /workspace/outputs/Cholec80/$RUN_NAME/checkpoint-best.pth \
  --data_path /workspace/data/Cholec80 \
  --video_id 49 \
  --per_phase \
  --out_dir /workspace/outputs/attn_vis
```

To collect failure examples for each ground-truth phase instead, add `--per_phase_mode incorrect`.
If one video does not contain failures for every phase, omit `--video_id` to search across the whole test split.
When running on the host instead of inside Docker, replace `/workspace/...` with repo-local paths such as `outputs/...` and `data/...`, and install dependencies with `python3 -m pip install -r requirements.txt`.
Long per-phase searches print `[progress] ...` every 25 candidates by default; change this with `--progress_every`.

Plot training curves from the saved `log.txt`:

```bash
RUN_NAME=surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4

python3 scripts/plot_training_curves.py \
  --run-dir /workspace/outputs/Cholec80/$RUN_NAME
```

This writes `training_curves.png` into the run directory when possible, otherwise it falls back to `./figs/$RUN_NAME/training_curves.png`.

## Monitor Training Progress

For Docker-based training, you can summarize the latest progress from container logs:

```bash
python3 scripts/check_training_progress.py --container surgformer-train-ch80
```

For a live view that refreshes every 10 seconds:

```bash
python3 scripts/check_training_progress.py --container surgformer-train-ch80 --watch
```

## Notes

- `deepspeed` is not installed by default in the Docker image because it tends to be environment-sensitive; the training entrypoint still supports `ENABLE_DEEPSPEED=1` if you extend the image yourself.
- The original upstream repository is preserved under `upstream_surgformer/` for reference.
- This repository keeps the original model variants:
  - `surgformer_base`
  - `surgformer_HTA`
  - `surgformer_HTA_KCA`
  - `AVT`


##　学習の中断
docker stop surgformer-train-ch80

##　学習の再開
cd /home/ikeido/test/Timesformer

docker run -d --name surgformer-train-ch80 --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v /home/ikeido/datasets/Cholec80/cholec80:/workspace/data/Cholec80 \
  -w /workspace \
  surgformer-repro \
  bash -lc 'GPUS=1 DATA_SET=Cholec80 DATA_PATH=/workspace/data/Cholec80 EVAL_DATA_PATH=/workspace/data/Cholec80 PRETRAINED_PATH=/workspace/pretrain_params/TimeSformer_divST_8x32_224_K400.pyth OUTPUT_ROOT=/workspace/outputs CUT_BLACK=0 BATCH_SIZE=4 NUM_WORKERS=8 bash scripts/train_phase.sh'


##　最新モデルでのテスト
/workspace/outputs/Cholec80/surgformer_HTA_Cholec80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-11.pth

##　最高性能モデルでのテスト
/workspace/outputs/Cholec80/surgformer_HTA_Cholec80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-best.pth

##　Dockerで行う場合
cd /home/ikeido/test/Timesformer

docker run --gpus all --ipc=host --rm -it \
  -v "$(pwd)":/workspace \
  -v /home/ikeido/datasets/Cholec80/cholec80:/workspace/data/Cholec80 \
  -w /workspace \
  surgformer-repro \
  bash -lc 'GPUS=1 DATA_SET=Cholec80 DATA_PATH=/workspace/data/Cholec80 EVAL_DATA_PATH=/workspace/data/Cholec80 FINETUNE_PATH=/workspace/outputs/Cholec80/surgformer_HTA_Cholec80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-11.pth CUT_BLACK=0 BATCH_SIZE=8 NUM_WORKERS=4 bash scripts/test_phase.sh'


##　次やる学習
新しい split での学習を始める

```bash
cd /home/ikeido/test/Timesformer

docker run -d --name surgformer-train-ch80-split --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v /home/ikeido/datasets/Cholec80/cholec80:/workspace/data/Cholec80 \
  -w /workspace \
  surgformer-repro \
  bash -lc 'GPUS=1 DATA_SET=Cholec80 DATA_PATH=/workspace/data/Cholec80 EVAL_DATA_PATH=/workspace/data/Cholec80 PRETRAINED_PATH=/workspace/pretrain_params/TimeSformer_divST_8x32_224_K400.pyth OUTPUT_ROOT=/workspace/outputs CUT_BLACK=0 BATCH_SIZE=4 NUM_WORKERS=8 bash scripts/train_phase.sh'
```

進捗確認

```bash
python3 scripts/check_training_progress.py --container surgformer-train-ch80-split --watch
```

起動ログ確認

```bash
docker logs -f surgformer-train-ch80-split
```

保存先：新しい split 用ディレクトリになります。

```text
outputs/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4
```

古い同名コンテナが残っている場合

```bash
docker rm -f surgformer-train-ch80-split
```
