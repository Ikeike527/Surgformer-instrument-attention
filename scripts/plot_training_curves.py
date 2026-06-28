#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training curves from a Surgformer run directory."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a run output directory containing log.txt.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <run-dir>/training_curves.png or a writable fallback.",
    )
    return parser.parse_args()


def ensure_writable_parent(path):
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    probe_path = parent / ".write_probe"
    with open(probe_path, "w", encoding="utf-8") as handle:
        handle.write("ok\n")
    probe_path.unlink(missing_ok=True)


def resolve_output_path(run_dir, requested_output):
    if requested_output:
        try:
            ensure_writable_parent(requested_output)
            return os.path.abspath(requested_output)
        except OSError as exc:
            raise PermissionError(
                f"Cannot write to --output: {requested_output}"
            ) from exc

    default_output = os.path.join(run_dir, "training_curves.png")
    try:
        ensure_writable_parent(default_output)
        return default_output
    except OSError:
        fallback_output = os.path.join(
            os.getcwd(),
            "figs",
            os.path.basename(os.path.normpath(run_dir)),
            "training_curves.png",
        )
        ensure_writable_parent(fallback_output)
        print(
            f"default output path is not writable, using fallback: {fallback_output}",
            flush=True,
        )
        return fallback_output


def load_history(log_path):
    history = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "epoch" not in record:
                continue
            history.append(record)
    if not history:
        raise RuntimeError(f"No epoch records found in {log_path}")
    return sorted(history, key=lambda record: int(record["epoch"]))


def values(history, key):
    xs = []
    ys = []
    for record in history:
        if key not in record:
            continue
        xs.append(int(record["epoch"]))
        ys.append(float(record[key]))
    return xs, ys


def plot_curves(history, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    train_epochs, train_loss = values(history, "train_loss")
    val_epochs, val_loss = values(history, "val_loss")
    if train_loss:
        axes[0].plot(train_epochs, train_loss, marker="o", linewidth=1.8, label="train_loss")
    if val_loss:
        axes[0].plot(val_epochs, val_loss, marker="o", linewidth=1.8, label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    val_acc1_epochs, val_acc1 = values(history, "val_acc1")
    val_acc5_epochs, val_acc5 = values(history, "val_acc5")
    if val_acc1:
        axes[1].plot(val_acc1_epochs, val_acc1, marker="o", linewidth=1.8, label="val_acc1")
    if val_acc5:
        axes[1].plot(val_acc5_epochs, val_acc5, marker="o", linewidth=1.8, label="val_acc5")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    lr_epochs, train_lr = values(history, "train_lr")
    min_lr_epochs, train_min_lr = values(history, "train_min_lr")
    if train_lr:
        axes[2].plot(lr_epochs, train_lr, marker="o", linewidth=1.8, label="train_lr")
    if train_min_lr:
        axes[2].plot(
            min_lr_epochs,
            train_min_lr,
            marker="o",
            linewidth=1.8,
            label="train_min_lr",
        )
    axes[2].set_ylabel("LR")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    best_epoch = max(history, key=lambda record: float(record.get("val_acc1", float("-inf"))))
    fig.suptitle(
        (
            f"{os.path.basename(os.path.normpath(os.path.dirname(output_path)))}"
            f" | best val_acc1={float(best_epoch.get('val_acc1', 0.0)):.2f}%"
            f" @ epoch {int(best_epoch['epoch'])}"
        ),
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    log_path = os.path.join(run_dir, "log.txt")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"log.txt not found: {log_path}")

    output_path = resolve_output_path(run_dir, args.output)
    history = load_history(log_path)
    plot_curves(history, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
