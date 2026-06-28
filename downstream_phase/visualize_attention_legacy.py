"""Legacy Surgformer attention visualization.

This preserves the older behavior that:
- overlays the heatmap over the entire frame without masking black background
- disables the spatial black-patch attention mask in the model
- shows frame id as title and temporal weight as xlabel outside the image
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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

import model.surgformer_HTA  # noqa: F401
import utils
from downstream_phase.datasets_phase import build_dataset
from downstream_phase import visualize_attention as base


def overlay(img_hwc, heat_hw, alpha):
    """Overlay the heatmap over the whole frame."""
    heat = np.asarray(plt.cm.jet(heat_hw)[..., :3], dtype=np.float32)
    return (1 - alpha) * img_hwc + alpha * heat


def save_figures(frames, maps, args, dir_name, title, prob=None, pred=None, frame_w=None):
    """Save figures with the legacy layout and full-frame overlay."""
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
        heat_up = base.upsample(heat, size)
        ov = np.clip(overlay(frames[t], heat_up, args.alpha), 0, 1)
        plt.imsave(os.path.join(sample_dir, f"frame_{t:02d}.png"), ov)
        axes[t].imshow(ov)
        ttl = f"t={t} (target)" if t == T - 1 else f"t={t}"
        axes[t].set_title(ttl, fontsize=7)
        axes[t].axis("off")
        if frame_w is not None:
            is_top = abs(frame_w[t] - wmax) < 1e-12
            axes[t].set_xlabel(
                f"w={frame_w[t]:.3f}",
                fontsize=8,
                color=("red" if is_top else "black"),
                fontweight=("bold" if is_top else "normal"),
            )
            axes[t].axis("on")
            axes[t].set_xticks([])
            axes[t].set_yticks([])
            for sp in axes[t].spines.values():
                sp.set_visible(False)
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
        np.savez_compressed(
            os.path.join(sample_dir, "attn.npz"),
            attn=maps.astype(np.float32),
            **extra,
        )
    print(f"saved {sample_dir}  T={T}", flush=True)


@torch.no_grad()
def main():
    args = base.build_args()
    device = torch.device(args.device)

    dataset_test, _ = build_dataset(is_train=False, test_mode=True, fps=args.data_fps, args=args)

    net = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        spatial_black_mask=False,
        black_pixel_threshold=args.black_pixel_threshold,
        drop_block_rate=None,
    )
    ckpt = torch.load(args.finetune, map_location="cpu", weights_only=False)
    state = None
    for k in args.model_key.split("|"):
        if k in ckpt:
            state = ckpt[k]
            break
    if state is None:
        state = ckpt
    utils.load_state_dict(net, state, prefix=args.model_prefix)
    net.to(device).eval()

    last_attn = net.blocks[-1].attn
    last_attn.save_attn = True
    last_temporal_attn = net.blocks[-1].temporal_attn
    last_temporal_attn.save_attn = True

    os.makedirs(args.out_dir, exist_ok=True)

    if args.per_phase:
        per_phase_mode = base.resolve_per_phase_mode(args)
        print(f"per_phase mode: {per_phase_mode}", flush=True)
        candidate_indices = base.collect_candidate_indices(dataset_test, video_id=args.video_id)
        scope_label = f"video {args.video_id}" if args.video_id is not None else "all test videos"
        if candidate_indices is not None:
            print(f"per_phase candidates in {scope_label}: {len(candidate_indices)}", flush=True)
        else:
            candidate_indices = range(len(dataset_test))
        found = {}
        scanned_video = 0
        correct_seen = 0
        incorrect_seen = 0
        total_candidates = len(candidate_indices)
        for pos, idx in enumerate(candidate_indices, start=1):
            meta = base.get_dataset_sample_meta(dataset_test, idx)
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
            buffer, target, sample_id, _flag = dataset_test[idx]
            _uid, video_id, frame_id = sample_id.strip().split("_")
            gt = int(target)
            if gt in found:
                continue
            pred, prob, frames, maps, frame_w = base.infer_sample(
                net, last_attn, last_temporal_attn, buffer
            )
            correct = pred == gt
            if correct:
                correct_seen += 1
            else:
                incorrect_seen += 1
            if not base.should_keep_per_phase_sample(pred, gt, per_phase_mode):
                continue
            dir_name = f"{video_id}_phase{gt + 1}_frame{frame_id}"
            if not correct:
                dir_name += "_ng"
            elif per_phase_mode != "correct":
                dir_name += "_ok"
            title = (
                f"{video_id} Phase{gt + 1} (GT) frame {frame_id}  "
                f"pred=Phase{pred + 1} (p={prob[pred]:.2f}) "
                f"{'OK' if correct else 'NG'}"
            )
            save_figures(frames, maps, args, dir_name, title, prob=prob, pred=pred, frame_w=frame_w)
            found[gt] = frame_id
            print(
                f"  -> Phase{gt + 1}: frame {frame_id} pred=Phase{pred + 1} "
                f"{'OK' if correct else 'NG'}",
                flush=True,
            )
            if len(found) == args.nb_classes:
                break
        if scanned_video == 0:
            print(f"{scope_label} のサンプルが見つかりませんでした", flush=True)
            return
        missing = [p + 1 for p in range(args.nb_classes) if p not in found]
        print(
            f"完了: 取得 Phase {sorted(p + 1 for p in found)} / "
            f"未取得 Phase {missing} / scanned={scanned_video} "
            f"(correct_seen={correct_seen}, incorrect_seen={incorrect_seen})",
            flush=True,
        )
        return

    candidate_indices = base.collect_candidate_indices(
        dataset_test, video_id=args.video_id, frame_id=args.frame_id
    )
    if candidate_indices is None:
        candidate_indices = range(len(dataset_test))
    elif args.video_id is not None or args.frame_id is not None:
        print(f"matched candidates: {len(candidate_indices)}", flush=True)
    done = 0
    for idx in candidate_indices:
        buffer, target, sample_id, _flag = dataset_test[idx]
        _uid, video_id, frame_id = sample_id.strip().split("_")
        if not base.video_id_matches(video_id, args.video_id):
            continue
        if not base.frame_id_matches(frame_id, args.frame_id):
            continue

        pred, prob, frames, maps, frame_w = base.infer_sample(
            net, last_attn, last_temporal_attn, buffer
        )
        title = (
            f"video {video_id} frame {frame_id}  pred=Phase{pred + 1} "
            f"(p={prob[pred]:.2f})  GT=Phase{int(target) + 1}"
        )
        save_figures(frames, maps, args, f"{video_id}_{frame_id}", title, prob=prob, pred=pred, frame_w=frame_w)
        done += 1
        if args.video_id is None and args.frame_id is None and done >= args.max_samples:
            break

    if done == 0:
        print("対象サンプルが見つかりませんでした (video_id/frame_id を確認)", flush=True)


if __name__ == "__main__":
    main()
