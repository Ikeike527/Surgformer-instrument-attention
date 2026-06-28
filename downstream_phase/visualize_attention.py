"""Surgformer 推論時のフレーム別アテンションマップ可視化 (Docker内実行)。

ある対象フレームの推論で入力された num_frames 枚それぞれについて、最終 Transformer
ブロックの Spatial Attention (CLS -> 各パッチ) をヘッド平均し、14x14 -> 入力解像度へ
補間したヒートマップを元フレームに重畳して保存する。

データ/前処理/サンプリングは test_phase.sh / emit_features.py の評価設定と一致させる
(online, key_frame, sampling_rate=4, cut_black, 1fps)。

出力:
  {out_dir}/{video}_{frame}/frame_00.png ... frame_{T-1}.png  (各フレームの重畳)
  {out_dir}/{video}_{frame}/grid.png                          (全フレーム一覧)
  {out_dir}/{video}_{frame}/attn.npz  (--save_npz 時, attn: (T,14,14))

実行例 (Docker内, repo=/workspace):
  python downstream_phase/visualize_attention.py \
    --finetune /workspace/outputs/.../checkpoint-best.pth \
    --data_path /workspace/data/Cholec80 \
    --num_frames 24 --sampling_rate 4 \
    --video_id 49 --frame_id 1000 \
    --out_dir /workspace/outputs/attn_vis

  python downstream_phase/visualize_attention.py \
    --finetune /workspace/outputs/.../checkpoint-best.pth \
    --data_path /workspace/data/Cholec80 \
    --video_id 49 --per_phase \
    --per_phase_mode incorrect \
    --out_dir /workspace/outputs/attn_vis_failures
"""
import os, sys, argparse, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace")
try:
    from timm.models import create_model
except ModuleNotFoundError as exc:
    if exc.name == "timm":
        raise ModuleNotFoundError(
            "timm is not installed. Run this script inside the surgformer Docker "
            "container, or install host dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    raise
import model.surgformer_HTA  # noqa: F401  (registers surgformer_HTA)
import utils
from downstream_phase.datasets_phase import build_dataset

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", required=True)
    ap.add_argument("--data_path", default="/workspace/data/Cholec80")
    ap.add_argument("--out_dir", default="/workspace/outputs/attn_vis")
    ap.add_argument("--model", default="surgformer_HTA")
    ap.add_argument("--device", default="cuda")
    # 対象フレームの指定 (省略時は先頭から --max_samples 件を処理)
    ap.add_argument("--video_id", default=None, help="例: 49 または video49")
    ap.add_argument("--frame_id", default=None, help="例: 1000 (id中の frame_id と一致)")
    ap.add_argument("--max_samples", type=int, default=1, help="video/frame未指定時の処理件数")
    ap.add_argument("--alpha", type=float, default=0.5, help="ヒートマップ重畳の不透明度")
    ap.add_argument("--save_npz", action="store_true", default=False)
    # Phase別一括モード: 指定videoの各phaseにつき1フレームを選んで可視化
    ap.add_argument("--per_phase", action="store_true", default=False,
                    help=("各phase(1..nb_classes)につき1フレームを可視化。"
                          "--video_id 指定時はその動画内のみ、未指定時はテスト全体から探索"))
    ap.add_argument("--per_phase_mode", choices=("correct", "incorrect", "any"),
                    default=None,
                    help=("per_phase時のサンプル選択条件。"
                          "correct=正解例, incorrect=失敗例, any=正誤を問わない"))
    ap.add_argument("--progress_every", type=int, default=25,
                    help=("per_phase探索中の進捗表示間隔。"
                          "対象videoの候補を何件見るごとに進捗を出すか。0で無効"))
    ap.add_argument("--no_require_correct", dest="require_correct", action="store_false",
                    default=True,
                    help=("互換用: per_phase時に予測正解を要求しない "
                          "(= --per_phase_mode any と同等)"))
    # データセット/モデル構築 (test_phase.sh と一致)
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
    ap.add_argument("--fc_drop_rate", type=float, default=0.5)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--attn_drop_rate", type=float, default=0.0)
    ap.add_argument("--disable_spatial_black_mask", action="store_true")
    ap.add_argument("--black_pixel_threshold", type=float, default=15.0 / 255.0)
    # 器具マスク (SAM3) による Spatial Attention 強調 (ソフトバイアス)
    ap.add_argument("--instr_attn_bias", action="store_true", default=False)
    ap.add_argument("--instr_lambda", type=float, default=0.0)
    ap.add_argument("--instr_bias_blocks", default="all",
                    help='"all" または整数N (最終N Block のみ適用)')
    ap.add_argument("--instr_mask_dirname", default="instrument_masks")
    ap.add_argument("--model_key", default="model|module")
    ap.add_argument("--model_prefix", default="")
    return ap.parse_args()


def denormalize(frame_chw):
    """(C,H,W) 正規化済みtensor -> (H,W,C) [0,1] numpy"""
    img = frame_chw.cpu().numpy().transpose(1, 2, 0)
    img = img * IMNET_STD + IMNET_MEAN
    return np.clip(img, 0.0, 1.0)


def resolve_per_phase_mode(args):
    if args.per_phase_mode is not None:
        return args.per_phase_mode
    return "correct" if args.require_correct else "any"


def should_keep_per_phase_sample(pred, gt, mode):
    correct = (pred == gt)
    if mode == "correct":
        return correct
    if mode == "incorrect":
        return not correct
    if mode == "any":
        return True
    raise ValueError(f"unknown per_phase mode: {mode}")


def parse_video_id_token(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("video"):
        text = text[5:]
    if text.isdigit():
        return int(text)
    return None


def video_id_matches(sample_video_id, requested_video_id):
    if requested_video_id is None:
        return True
    sample_num = parse_video_id_token(sample_video_id)
    requested_num = parse_video_id_token(requested_video_id)
    if sample_num is not None and requested_num is not None:
        return sample_num == requested_num
    return str(sample_video_id) == str(requested_video_id)


def frame_id_matches(sample_frame_id, requested_frame_id):
    if requested_frame_id is None:
        return True
    try:
        return int(sample_frame_id) == int(requested_frame_id)
    except (TypeError, ValueError):
        return str(sample_frame_id) == str(requested_frame_id)


def get_dataset_sample_meta(dataset, index):
    if hasattr(dataset, "dataset_samples"):
        sample = dataset.dataset_samples[index]
        return {
            "video_id": str(sample["video_id"]),
            "frame_id": str(sample["frame_id"]),
            "phase_gt": int(sample["phase_gt"]),
        }
    return None


def collect_candidate_indices(dataset, video_id=None, frame_id=None):
    if not hasattr(dataset, "dataset_samples"):
        return None

    indices = []
    for idx, sample in enumerate(dataset.dataset_samples):
        sample_video_id = str(sample["video_id"])
        sample_frame_id = str(sample["frame_id"])
        if not video_id_matches(sample_video_id, video_id):
            continue
        if not frame_id_matches(sample_frame_id, frame_id):
            continue
        indices.append(idx)
    return indices


def foreground_mask(img_hwc, pixel_threshold):
    """元画像(H,W,3 [0,1]) から黒背景以外の領域マスク(H,W)を返す。"""
    return (img_hwc.max(axis=-1) > pixel_threshold).astype(np.float32)


def overlay(img_hwc, heat_hw, alpha, visible_mask=None):
    """元画像(H,W,3 [0,1]) に jet ヒートマップ(H,W [0,1])を重畳した (H,W,3) を返す"""
    heat = np.asarray(plt.cm.jet(heat_hw)[..., :3], dtype=np.float32)  # (H,W,3)
    if visible_mask is None:
        return (1 - alpha) * img_hwc + alpha * heat
    mask = visible_mask[..., None].astype(np.float32)
    return img_hwc + alpha * (heat - img_hwc) * mask


def upsample(heat_hw, size):
    """14x14 -> size の bilinear 補間 (torch利用)"""
    t = torch.from_numpy(heat_hw)[None, None].float()
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                        align_corners=False)
    return t[0, 0].numpy()


def infer_sample(net, last_attn, last_temporal_attn, buffer, instr_mask=None):
    """1サンプルを推論し (pred, prob, frames_hwc_list, maps, frame_w) を返す。図保存はしない。"""
    device = next(net.parameters()).device
    softmax = torch.nn.functional.softmax
    videos = buffer.unsqueeze(0).to(device)  # (1, C, T, H, W)
    mask_arg = None
    if instr_mask is not None:
        mask_arg = instr_mask.unsqueeze(0).to(device)  # (1, T, S, S)
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        logits = net(videos, instr_mask=mask_arg)
    prob = softmax(logits.float(), dim=-1)[0].cpu().numpy()
    pred = int(prob.argmax())

    # attn_map: (B*T, num_heads, K, K), B=1 -> (T, num_heads, K, K)
    attn = last_attn.attn_map.float().cpu()  # K = 1 + P
    T = attn.shape[0]
    P = attn.shape[-1] - 1
    side = int(math.sqrt(P))
    # CLS(row0) -> 各パッチ(col 1..) をヘッド平均
    cls_to_patch = attn[:, :, 0, 1:].mean(dim=1)  # (T, P)
    maps = cls_to_patch.reshape(T, side, side).numpy()  # (T,side,side)

    # フレーム別重み: 最終ブロックの時間 attention attn_16 (BK,num_heads,T,T) から、
    # 各フレームが key としてどれだけ注目されたかを patch/head/query 方向で平均。
    # 各 query 行が softmax で和1のため、結果は T 上で和1の分布になる。
    t_attn = last_temporal_attn.attn_map.float().cpu()  # (BK, num_heads, T, T)
    frame_w = t_attn.mean(dim=(0, 1, 2)).numpy()  # (T,)

    frames = [denormalize(videos[0, :, t]) for t in range(T)]  # 各 (H,W,3)
    return pred, prob, frames, maps, frame_w


def save_figures(frames, maps, args, dir_name, title, prob=None, pred=None,
                 frame_w=None):
    """frames(各H,W,3) と maps(T,side,side) からフレーム別ヒートマップを保存。
    frame_w(T,) を与えると各フレーム下部に時間 attention 重みを併記する。"""
    T = len(frames)
    size = frames[0].shape[0]
    sample_dir = os.path.join(args.out_dir, dir_name)
    os.makedirs(sample_dir, exist_ok=True)

    wmax = float(frame_w.max()) if frame_w is not None else None
    ncol = min(8, T)
    nrow = math.ceil(T / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2.2))
    axes = np.array(axes).reshape(-1)
    for t in range(T):
        heat = maps[t]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat_up = upsample(heat, size)
        visible_mask = foreground_mask(frames[t], args.black_pixel_threshold)
        ov = np.clip(
            overlay(frames[t], heat_up, args.alpha, visible_mask=visible_mask),
            0,
            1,
        )
        plt.imsave(os.path.join(sample_dir, f"frame_{t:02d}.png"), ov)
        axes[t].imshow(ov)
        is_top = frame_w is not None and abs(frame_w[t] - wmax) < 1e-12
        label_lines = [f"t={t}{' target' if t == T - 1 else ''}"]
        if frame_w is not None:
            label_lines.append(f"w={frame_w[t]:.3f}")
        axes[t].text(
            0.03,
            0.97,
            "\n".join(label_lines),
            transform=axes[t].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=("red" if is_top else "black"),
            fontweight=("bold" if is_top else "normal"),
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": ("red" if is_top else "none"),
                "alpha": 0.8,
            },
        )
        axes[t].axis("off")
    for t in range(T, len(axes)):
        axes[t].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(sample_dir, "grid.png"), dpi=120)
    plt.close(fig)

    if args.save_npz:
        extra = {}
        if prob is not None:
            extra["prob"] = prob
        if pred is not None:
            extra["pred"] = pred
        if frame_w is not None:
            extra["frame_w"] = frame_w.astype(np.float32)
        np.savez_compressed(os.path.join(sample_dir, "attn.npz"),
                            attn=maps.astype(np.float32), **extra)
    print(f"saved {sample_dir}  T={T}", flush=True)


@torch.no_grad()
def main():
    args = build_args()
    device = torch.device(args.device)

    dataset_test, _ = build_dataset(is_train=False, test_mode=True,
                                    fps=args.data_fps, args=args)

    net = create_model(args.model, pretrained=False, num_classes=args.nb_classes,
                       all_frames=args.num_frames, fc_drop_rate=args.fc_drop_rate,
                       drop_rate=args.drop, drop_path_rate=args.drop_path,
                       attn_drop_rate=args.attn_drop_rate,
                       spatial_black_mask=not args.disable_spatial_black_mask,
                       black_pixel_threshold=args.black_pixel_threshold,
                       instr_attn_bias=args.instr_attn_bias,
                       instr_lambda=args.instr_lambda,
                       instr_bias_blocks=args.instr_bias_blocks,
                       drop_block_rate=None)
    ckpt = torch.load(args.finetune, map_location="cpu", weights_only=False)
    state = None
    for k in args.model_key.split("|"):
        if k in ckpt:
            state = ckpt[k]; break
    if state is None:
        state = ckpt
    utils.load_state_dict(net, state, prefix=args.model_prefix)
    net.to(device).eval()

    # 最終ブロックの Spatial / Temporal Attention の保存を有効化
    last_attn = net.blocks[-1].attn
    last_attn.save_attn = True
    last_temporal_attn = net.blocks[-1].temporal_attn
    last_temporal_attn.save_attn = True

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Phase別一括モード ----
    if args.per_phase:
        per_phase_mode = resolve_per_phase_mode(args)
        print(f"per_phase mode: {per_phase_mode}", flush=True)
        candidate_indices = collect_candidate_indices(dataset_test, video_id=args.video_id)
        scope_label = (f"video {args.video_id}"
                       if args.video_id is not None else "all test videos")
        if candidate_indices is not None:
            print(f"per_phase candidates in {scope_label}: {len(candidate_indices)}",
                  flush=True)
        else:
            candidate_indices = range(len(dataset_test))
        found = {}  # gt_phase -> frame_id
        scanned_video = 0
        correct_seen = 0
        incorrect_seen = 0
        total_candidates = len(candidate_indices)
        for pos, idx in enumerate(candidate_indices, start=1):
            meta = get_dataset_sample_meta(dataset_test, idx)
            if args.progress_every > 0 and ((pos - 1) % args.progress_every == 0):
                print(
                    f"[progress] scanned={scanned_video} "
                    f"candidate={pos}/{total_candidates} "
                    f"found={[p + 1 for p in sorted(found)]} "
                    f"correct_seen={correct_seen} incorrect_seen={incorrect_seen}",
                    flush=True,
                )
            if meta is not None and meta["phase_gt"] in found:
                continue
            scanned_video += 1
            sample = dataset_test[idx]
            buffer, target, sample_id, _flag = sample[:4]
            instr_mask = sample[4] if len(sample) > 4 else None
            _uid, video_id, frame_id = sample_id.strip().split("_")
            gt = int(target)
            if gt in found:  # このphaseは取得済み (推論もスキップ)
                continue
            pred, prob, frames, maps, frame_w = infer_sample(
                net, last_attn, last_temporal_attn, buffer, instr_mask=instr_mask)
            correct = (pred == gt)
            if correct:
                correct_seen += 1
            else:
                incorrect_seen += 1
            if not should_keep_per_phase_sample(pred, gt, per_phase_mode):
                continue
            dir_name = f"{video_id}_phase{gt + 1}_frame{frame_id}"
            if not correct:
                dir_name += "_ng"
            elif per_phase_mode != "correct":
                dir_name += "_ok"
            title = (f"{video_id} Phase{gt + 1} (GT) frame {frame_id}  "
                     f"pred=Phase{pred + 1} (p={prob[pred]:.2f}) "
                     f"{'OK' if correct else 'NG'}")
            save_figures(frames, maps, args, dir_name, title, prob=prob, pred=pred,
                         frame_w=frame_w)
            found[gt] = frame_id
            print(f"  -> Phase{gt + 1}: frame {frame_id} pred=Phase{pred + 1} "
                  f"{'OK' if correct else 'NG'}", flush=True)
            if len(found) == args.nb_classes:
                break
        if scanned_video == 0:
            print(f"{scope_label} のサンプルが見つかりませんでした", flush=True)
            return
        missing = [p + 1 for p in range(args.nb_classes) if p not in found]
        print(f"完了: 取得 Phase {sorted(p + 1 for p in found)} / "
              f"未取得 Phase {missing} / scanned={scanned_video} "
              f"(correct_seen={correct_seen}, incorrect_seen={incorrect_seen})",
              flush=True)
        return

    # ---- 通常モード (単一/先頭N件) ----
    candidate_indices = collect_candidate_indices(
        dataset_test, video_id=args.video_id, frame_id=args.frame_id)
    if candidate_indices is None:
        candidate_indices = range(len(dataset_test))
    elif args.video_id is not None or args.frame_id is not None:
        print(f"matched candidates: {len(candidate_indices)}", flush=True)
    done = 0
    for idx in candidate_indices:
        sample = dataset_test[idx]
        buffer, target, sample_id, _flag = sample[:4]
        instr_mask = sample[4] if len(sample) > 4 else None
        _uid, video_id, frame_id = sample_id.strip().split("_")
        if not video_id_matches(video_id, args.video_id):
            continue
        if not frame_id_matches(frame_id, args.frame_id):
            continue

        pred, prob, frames, maps, frame_w = infer_sample(
            net, last_attn, last_temporal_attn, buffer, instr_mask=instr_mask)
        title = (f"video {video_id} frame {frame_id}  pred=Phase{pred + 1} "
                 f"(p={prob[pred]:.2f})  GT=Phase{int(target) + 1}")
        save_figures(frames, maps, args, f"{video_id}_{frame_id}", title,
                     prob=prob, pred=pred, frame_w=frame_w)
        done += 1
        if args.video_id is None and args.frame_id is None and done >= args.max_samples:
            break

    if done == 0:
        print("対象サンプルが見つかりませんでした (video_id/frame_id を確認)", flush=True)


if __name__ == "__main__":
    main()
