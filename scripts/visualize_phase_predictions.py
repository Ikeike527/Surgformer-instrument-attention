#!/usr/bin/env python3
import argparse
import ast
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.special import softmax


CHOLEC80_CLASS_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

DEFAULT_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize phase predictions from Surgformer test outputs."
    )
    parser.add_argument("--run-dir", required=True, help="Path to a run output directory.")
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Video ids like video49 video50. Defaults to every video found.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save figures. Defaults to <run-dir>/figs.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame sampling rate used by the evaluation outputs.",
    )
    return parser.parse_args()


def normalize_video_id(video_id):
    video_id = str(video_id)
    if video_id.startswith("video"):
        return video_id
    return f"video{int(video_id):02d}"


def parse_raw_outputs(run_dir):
    txt_files = sorted(glob.glob(os.path.join(run_dir, "[0-9]*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No raw rank txt files found in {run_dir}")

    records = {}
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as handle:
            for line in handle.readlines()[1:]:
                line = line.strip()
                if not line:
                    continue
                left = line.find("[")
                right = line.rfind("]")
                if left < 0 or right < left:
                    continue
                prefix = line[:left].strip().split()
                suffix = line[right + 1 :].strip().split()
                if len(prefix) < 3 or not suffix:
                    continue
                video_id = prefix[1]
                frame_id = int(prefix[2])
                logits_text = line[left : right + 1]
                target_text = suffix[-1]
                logits = np.asarray(ast.literal_eval(logits_text), dtype=np.float32)
                target = int(target_text)
                probs = softmax(logits)
                prediction = int(np.argmax(probs))
                confidence = float(np.max(probs))
                records.setdefault(video_id, {})[frame_id] = {
                    "target": target,
                    "prediction": prediction,
                    "confidence": confidence,
                }
    return {
        video_id: dict(sorted(frame_map.items()))
        for video_id, frame_map in sorted(records.items(), key=lambda item: item[0])
    }


def ensure_writable_directory(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    probe_path = directory / ".write_probe"
    with open(probe_path, "w", encoding="utf-8") as handle:
        handle.write("ok\n")
    probe_path.unlink(missing_ok=True)
    return str(directory)


def resolve_output_dir(run_dir, requested_output_dir):
    if requested_output_dir:
        try:
            return ensure_writable_directory(requested_output_dir)
        except OSError as exc:
            raise PermissionError(
                f"Cannot write to --output-dir: {requested_output_dir}"
            ) from exc

    default_output_dir = os.path.join(run_dir, "figs")
    try:
        return ensure_writable_directory(default_output_dir)
    except OSError:
        fallback_output_dir = os.path.join(
            os.getcwd(),
            "figs",
            os.path.basename(os.path.normpath(run_dir)),
        )
        resolved = ensure_writable_directory(fallback_output_dir)
        print(
            f"default output dir is not writable, using fallback: {resolved}",
            flush=True,
        )
        return resolved


def plot_video_timeline(video_id, frames, class_names, fps, output_path):
    frame_ids = np.asarray(list(frames.keys()), dtype=np.int32)
    targets = np.asarray([item["target"] for item in frames.values()], dtype=np.int32)
    predictions = np.asarray(
        [item["prediction"] for item in frames.values()], dtype=np.int32
    )
    confidence = np.asarray(
        [item["confidence"] for item in frames.values()], dtype=np.float32
    )
    seconds = frame_ids / fps if fps > 0 else frame_ids
    accuracy = float((predictions == targets).mean()) if len(targets) else 0.0

    cmap = ListedColormap(DEFAULT_COLORS[: len(class_names)])
    legend_handles = [
        Patch(facecolor=DEFAULT_COLORS[idx], label=f"{idx}: {name}")
        for idx, name in enumerate(class_names)
    ]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(18, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 0.7, 2.0, 1.0]},
    )

    gt_band = targets[np.newaxis, :]
    pred_band = predictions[np.newaxis, :]

    axes[0].imshow(
        gt_band,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=len(class_names) - 1,
        extent=[seconds[0], seconds[-1], 0, 1],
    )
    axes[0].set_ylabel("GT")
    axes[0].set_yticks([])

    axes[1].imshow(
        pred_band,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=len(class_names) - 1,
        extent=[seconds[0], seconds[-1], 0, 1],
    )
    axes[1].set_ylabel("Pred")
    axes[1].set_yticks([])

    axes[2].step(seconds, targets, where="post", linewidth=2.0, color="#111111", label="GT")
    axes[2].step(
        seconds,
        predictions,
        where="post",
        linewidth=1.8,
        color="#D62728",
        alpha=0.9,
        label="Pred",
    )
    axes[2].set_ylim(-0.2, max(len(class_names) - 1, 1) + 0.2)
    axes[2].set_yticks(range(len(class_names)))
    axes[2].set_yticklabels(class_names)
    axes[2].set_ylabel("Phase")
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].legend(loc="upper right")

    axes[3].fill_between(
        seconds,
        0.0,
        confidence,
        step="post",
        alpha=0.18,
        color="#4E79A7",
        label="Confidence",
    )
    axes[3].plot(seconds, confidence, color="#4E79A7", linewidth=1.2, label="Confidence")
    axes[3].set_ylim(0.0, 1.02)
    axes[3].set_ylabel("Conf.")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(axis="x", alpha=0.25)
    axes[3].legend(loc="upper right")

    fig.suptitle(
        f"{video_id} | frames={len(frame_ids)} | top1={accuracy * 100:.2f}%",
        fontsize=14,
    )
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Classes",
    )
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    output_dir = resolve_output_dir(run_dir, args.output_dir)

    records = parse_raw_outputs(run_dir)
    if not records:
        raise RuntimeError(f"No prediction records found in {run_dir}")

    requested_videos = (
        [normalize_video_id(video_id) for video_id in args.videos]
        if args.videos
        else sorted(records.keys())
    )

    for video_id in requested_videos:
        if video_id not in records:
            print(f"skip {video_id}: no records found")
            continue
        output_path = os.path.join(output_dir, f"{video_id}.png")
        plot_video_timeline(
            video_id,
            records[video_id],
            CHOLEC80_CLASS_NAMES,
            args.fps,
            output_path,
        )
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
