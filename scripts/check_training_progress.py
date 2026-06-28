#!/usr/bin/env python3
import argparse
import datetime as dt
import re
import subprocess
import sys
import time


START_RE = re.compile(r"Start training for (?P<epochs>\d+) epochs")
EPOCH_RE = re.compile(
    r"Epoch:\s+\[(?P<epoch>\d+)\]\s+\[\s*(?P<step>\d+)/(?P<total_steps>\d+)\]"
    r"\s+eta:\s+(?P<epoch_eta>[0-9:]+).*?"
    r"lr:\s+(?P<lr>[0-9.eE+-]+).*?"
    r"loss:\s+(?P<loss_current>[0-9.eE+-]+)\s+\((?P<loss_avg>[0-9.eE+-]+)\).*?"
    r"time:\s+(?P<iter_time>[0-9.]+)"
)
VAL_RE = re.compile(
    r"Val:\s+\[\s*(?P<step>\d+)/(?P<total_steps>\d+)\]"
    r"\s+eta:\s+(?P<eta>[0-9:]+).*?"
    r"loss:\s+(?P<loss_current>[0-9.eE+-]+)\s+\((?P<loss_avg>[0-9.eE+-]+)\).*?"
    r"acc1:\s+(?P<acc1_current>[0-9.eE+-]+)\s+\((?P<acc1_avg>[0-9.eE+-]+)\).*?"
    r"time:\s+(?P<iter_time>[0-9.]+)"
)
TEST_RE = re.compile(
    r"Test:\s+\[\s*(?P<step>\d+)/(?P<total_steps>\d+)\]"
    r"\s+eta:\s+(?P<eta>[0-9:]+).*?"
    r"loss:\s+(?P<loss_current>[0-9.eE+-]+)\s+\((?P<loss_avg>[0-9.eE+-]+)\).*?"
    r"time:\s+(?P<iter_time>[0-9.]+)"
)


def fetch_logs(container_name: str) -> str:
    result = subprocess.run(
        ["docker", "logs", container_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip() or f"docker logs failed for {container_name}")
    return result.stdout


def parse_status(log_text: str):
    total_epochs = None
    for line in log_text.splitlines():
        start_match = START_RE.search(line)
        if start_match:
            total_epochs = int(start_match.group("epochs"))

    last_train = None
    latest = None
    for line in log_text.splitlines():
        match = EPOCH_RE.search(line)
        if match:
            last_train = (match, line)
            latest = ("train", match, line)
            continue
        match = VAL_RE.search(line)
        if match:
            latest = ("val", match, line)
            continue
        match = TEST_RE.search(line)
        if match:
            latest = ("test", match, line)

    if latest is None:
        raise RuntimeError("No training progress line found in logs yet.")

    stage, latest_match, latest_line = latest
    epoch = int(last_train[0].group("epoch")) if last_train is not None else 0
    lr = float(last_train[0].group("lr")) if last_train is not None else None

    if stage == "val":
        step = int(latest_match.group("step"))
        total_steps = int(latest_match.group("total_steps"))
        eta = latest_match.group("eta")
        iter_time = float(latest_match.group("iter_time"))
        loss_current = float(latest_match.group("loss_current"))
        loss_avg = float(latest_match.group("loss_avg"))
        acc1_current = float(latest_match.group("acc1_current"))
        acc1_avg = float(latest_match.group("acc1_avg"))
    elif stage == "test":
        step = int(latest_match.group("step"))
        total_steps = int(latest_match.group("total_steps"))
        eta = latest_match.group("eta")
        iter_time = float(latest_match.group("iter_time"))
        loss_current = float(latest_match.group("loss_current"))
        loss_avg = float(latest_match.group("loss_avg"))
        acc1_current = None
        acc1_avg = None
    else:
        step = int(latest_match.group("step"))
        total_steps = int(latest_match.group("total_steps"))
        eta = latest_match.group("epoch_eta")
        iter_time = float(latest_match.group("iter_time"))
        loss_current = float(latest_match.group("loss_current"))
        loss_avg = float(latest_match.group("loss_avg"))
        acc1_current = None
        acc1_avg = None

    stage_progress = 100.0 * step / total_steps if total_steps else 0.0
    overall_progress = None
    overall_eta = None
    if total_epochs and last_train is not None:
        train_step = int(last_train[0].group("step"))
        train_total_steps = int(last_train[0].group("total_steps"))
        total_iterations = total_epochs * train_total_steps
        done_iterations = epoch * train_total_steps + train_step
        overall_progress = 100.0 * done_iterations / total_iterations if total_iterations else 0.0
        remaining_iterations = total_iterations - done_iterations
        overall_eta = str(dt.timedelta(seconds=int(max(remaining_iterations, 0) * iter_time)))

    return {
        "total_epochs": total_epochs,
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "total_steps": total_steps,
        "stage_progress": stage_progress,
        "overall_progress": overall_progress,
        "stage_eta": eta,
        "overall_eta": overall_eta,
        "lr": lr,
        "loss_current": loss_current,
        "loss_avg": loss_avg,
        "acc1_current": acc1_current,
        "acc1_avg": acc1_avg,
        "iter_time": iter_time,
        "latest_line": latest_line,
    }


def render(status, container_name: str):
    stage_label = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }[status["stage"]]
    lines = [
        f"container: {container_name}",
        f"epoch: {status['epoch'] + 1}/{status['total_epochs'] or '?'}",
        f"stage: {stage_label}",
        f"step: {status['step']}/{status['total_steps']} ({status['stage_progress']:.2f}% of current {stage_label})",
    ]
    if status["overall_progress"] is not None:
        lines.append(f"overall training: {status['overall_progress']:.2f}%")
    lines.append(f"{stage_label} eta: {status['stage_eta']}")
    if status["overall_eta"] is not None:
        lines.append(f"overall eta: {status['overall_eta']}")
    lines.append(f"iter time: {status['iter_time']:.4f}s")
    if status["lr"] is not None:
        lines.append(f"lr: {status['lr']:.8f}")
    lines.append(f"loss: current={status['loss_current']:.4f}, avg={status['loss_avg']:.4f}")
    if status["acc1_current"] is not None:
        lines.append(f"acc1: current={status['acc1_current']:.4f}, avg={status['acc1_avg']:.4f}")
    lines.append("latest log line:")
    lines.append(status["latest_line"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize Surgformer Docker training progress.")
    parser.add_argument("--container", default="surgformer-train-ch80")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    while True:
        try:
            logs = fetch_logs(args.container)
            status = parse_status(logs)
            output = render(status, args.container)
        except Exception as exc:
            output = f"error: {exc}"

        if args.watch:
            print("\033[2J\033[H", end="")
            print(output)
            sys.stdout.flush()
            time.sleep(args.interval)
        else:
            print(output)
            return


if __name__ == "__main__":
    main()
