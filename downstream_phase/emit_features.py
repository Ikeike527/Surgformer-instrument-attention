"""Surgformer フレーム別特徴+確率の emit (Docker内実行)。

videotree の手法(tree+Viterbi)にフェアに載せるため、分類ヘッド出力(logits→probs)に加え
ヘッド直前のCLS特徴(forward_features, 768次元)を feats として動画別npzに保存する。
データ/前処理/サンプリングは test_phase.sh の評価設定と完全一致させる
(online, key_frame, num_frames=16, sampling_rate=4, cut_black, 1fps)。

出力: {out_dir}/{video}.npz  (probs: (N,7), feats: (N,768))  ※frame_id昇順
videotree側は data/emissions/surgformer_feat/ に置いて run_decode --backend surgformer_feat。

実行例 (Docker内, repo=/workspace, データ=/workspace/data/Cholec80):
  python downstream_phase/emit_features.py \
    --finetune /workspace/outputs/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-best.pth \
    --data_path /workspace/data/Cholec80 \
    --out_dir /workspace/outputs/emissions_surgformer_feat
"""
import os, sys, argparse
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, "/workspace")
from timm.models import create_model
import model.surgformer_HTA  # noqa: F401  (registers surgformer_HTA)
import utils
from downstream_phase.datasets_phase import build_dataset


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", required=True)
    ap.add_argument("--data_path", default="/workspace/data/Cholec80")
    ap.add_argument("--out_dir", default="/workspace/outputs/emissions_surgformer_feat")
    ap.add_argument("--model", default="surgformer_HTA")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    # データセット/モデル構築に必要な固定値 (test_phase.sh と一致)
    ap.add_argument("--data_set", default="Cholec80")
    ap.add_argument("--data_fps", default="1fps")
    ap.add_argument("--nb_classes", type=int, default=7)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--sampling_rate", type=int, default=4)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--short_side_size", type=int, default=224)
    ap.add_argument("--data_strategy", default="online")
    ap.add_argument("--output_mode", default="key_frame")
    ap.add_argument("--cut_black", action="store_true", default=True)
    # モデルのdrop系 (hyerparamter.txt と一致)
    ap.add_argument("--fc_drop_rate", type=float, default=0.5)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--attn_drop_rate", type=float, default=0.0)
    ap.add_argument("--disable_spatial_black_mask", action="store_true")
    ap.add_argument("--black_pixel_threshold", type=float, default=15.0 / 255.0)
    ap.add_argument("--model_key", default="model|module")
    ap.add_argument("--model_prefix", default="")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = build_args()
    device = torch.device(args.device)

    dataset_test, _ = build_dataset(is_train=False, test_mode=True, fps=args.data_fps, args=args)
    loader = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test),
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, drop_last=False)

    net = create_model(args.model, pretrained=False, num_classes=args.nb_classes,
                       all_frames=args.num_frames, fc_drop_rate=args.fc_drop_rate,
                       drop_rate=args.drop, drop_path_rate=args.drop_path,
                       attn_drop_rate=args.attn_drop_rate,
                       spatial_black_mask=not args.disable_spatial_black_mask,
                       black_pixel_threshold=args.black_pixel_threshold,
                       drop_block_rate=None)
    ckpt = torch.load(args.finetune, map_location="cpu", weights_only=False)
    state = None
    for k in args.model_key.split("|"):
        if k in ckpt:
            state = ckpt[k]; break
    if state is None:
        state = ckpt
    # num_frames=16 はcheckpointと一致するためpos/time embedの補間は不要、そのまま読み込む
    utils.load_state_dict(net, state, prefix=args.model_prefix)
    net.to(device).eval()

    # video別に (frame_id, probs, feats) を蓄積
    acc = defaultdict(list)
    softmax = torch.nn.functional.softmax
    n_done = 0
    for videos, _target, ids, _flag in loader:
        videos = videos.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            feats = net.forward_features(videos)      # (B, 768) CLS
            logits = net.head(feats)                  # (B, 7) eval時fc_dropoutは無効
        probs = softmax(logits.float(), dim=-1).cpu().numpy()
        feats = feats.float().cpu().numpy()
        for i in range(len(ids)):
            _uid, video_id, frame_id = ids[i].strip().split("_")
            acc[video_id].append((int(frame_id), probs[i], feats[i]))
        n_done += len(ids)
        if n_done % (args.batch_size * 20) == 0:
            print(f"  processed {n_done} samples", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for v, rows in acc.items():
        rows.sort(key=lambda r: r[0])
        probs = np.stack([r[1] for r in rows]).astype(np.float32)
        feats = np.stack([r[2] for r in rows]).astype(np.float32)
        np.savez_compressed(os.path.join(args.out_dir, f"{v}.npz"), probs=probs, feats=feats)
        print(f"saved {args.out_dir}/{v}.npz  probs{probs.shape} feats{feats.shape}", flush=True)


if __name__ == "__main__":
    main()
