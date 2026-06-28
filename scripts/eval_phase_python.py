#!/usr/bin/env python3
"""Cholec80 phase 評価の Python 移植版.

evaluation_matlab/Main.m と Evaluate.m を忠実に再現する (10 秒 relaxed
boundary, phase 4/5/6/7 の遷移許容). MATLAB/Octave が無い環境用.
prediction/ と phase_annotations/ は convert_cholec80.py が生成したものを使う.
"""
import argparse
import os

import numpy as np

PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


def read_phase_label(path):
    # "Frame\tPhase" ヘッダをスキップし label (0-6) を読む
    labels = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            labels.append(int(parts[1]))
    return np.array(labels, dtype=int)


def runs(mask):
    # 1D 連結成分 (bwconncomp 相当) を [start, end] (両端含む) で返す
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    return [(g[0], g[-1]) for g in groups]


def evaluate(gt, pred, fps=1):
    # gt, pred は 1..7
    oriT = 10 * fps
    diff = (pred - gt).astype(float)
    updated = diff.copy()

    for iphase in range(1, 8):
        for start, end in runs(gt == iphase):
            seg = updated[start : end + 1]
            t = min(oriT, len(seg))
            head = seg[:t]
            tail = seg[-t:]
            if iphase in (4, 5):
                head[head == -1] = 0
                tail[(tail == 1) | (tail == 2)] = 0
            elif iphase in (6, 7):
                head[(head == -1) | (head == -2)] = 0
                tail[(tail == 1) | (tail == 2)] = 0
            else:
                head[head == -1] = 0
                tail[tail == 1] = 0
            seg[:t] = head
            seg[-t:] = tail
            updated[start : end + 1] = seg

    jacc = np.full(7, np.nan)
    prec = np.full(7, np.nan)
    rec = np.full(7, np.nan)
    for i, iphase in enumerate(range(1, 8)):
        gtmask = gt == iphase
        if gtmask.sum() == 0:
            continue
        predmask = pred == iphase
        union = np.where(gtmask | predmask)[0]
        tp = np.sum(updated[union] == 0)
        jacc[i] = tp / len(union) * 100
        sum_pred = predmask.sum()
        sum_gt = gtmask.sum()
        prec[i] = tp * 100 / sum_pred if sum_pred > 0 else np.nan
        rec[i] = tp * 100 / sum_gt
    acc = np.sum(updated == 0) / len(gt) * 100
    with np.errstate(invalid="ignore"):
        f1 = 2 * prec * rec / (prec + rec)
    return jacc, prec, rec, acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-path", required=True)
    ap.add_argument("--start", type=int, default=49)
    ap.add_argument("--end", type=int, default=80)
    ap.add_argument("--fps", type=int, default=1)
    args = ap.parse_args()

    ann_dir = os.path.join(args.main_path, "phase_annotations")
    pred_dir = os.path.join(args.main_path, "prediction")
    vids = list(range(args.start, args.end + 1))
    n = len(vids)

    J = np.zeros((7, n))
    P = np.zeros((7, n))
    R = np.zeros((7, n))
    F = np.zeros((7, n))
    A = np.zeros(n)

    for col, k in enumerate(vids):
        vid = f"video-{k:02d}.txt"
        gt = read_phase_label(os.path.join(ann_dir, vid)) + 1
        pred = read_phase_label(os.path.join(pred_dir, vid)) + 1
        if len(gt) != len(pred):
            raise ValueError(f"{vid}: size mismatch gt={len(gt)} pred={len(pred)}")
        j, p, r, a, f = evaluate(gt, pred, args.fps)
        J[:, col], P[:, col], R[:, col], A[col], F[:, col] = j, p, r, a, f

    J[J > 100] = 100
    P[P > 100] = 100
    R[R > 100] = 100
    F[F > 100] = 100

    jacc_phase = np.nanmean(J, axis=1)
    prec_phase = np.nanmean(P, axis=1)
    rec_phase = np.nanmean(R, axis=1)
    f1_phase = np.nanmean(F, axis=1)

    print(f"test videos: video{args.start}-{args.end} ({n} 本), fps={args.fps}")
    print("=" * 64)
    print(f"{'Phase':>25}|{'Jacc':>7}|{'Prec':>7}|{'Rec':>7}|{'F1':>7}|")
    print("-" * 64)
    for i, name in enumerate(PHASES):
        print(
            f"{name:>25}|{jacc_phase[i]:7.2f}|{prec_phase[i]:7.2f}|"
            f"{rec_phase[i]:7.2f}|{f1_phase[i]:7.2f}|"
        )
    print("=" * 64)
    print(f"Mean Accuracy (per-video) : {np.mean(A):6.2f} +/- {np.std(A):5.2f}")
    print(f"Mean Jaccard  (per-phase) : {np.mean(jacc_phase):6.2f} +/- {np.std(jacc_phase):5.2f}")
    print(f"Mean Precision(per-phase) : {np.nanmean(prec_phase):6.2f} +/- {np.nanstd(prec_phase):5.2f}")
    print(f"Mean Recall   (per-phase) : {np.mean(rec_phase):6.2f} +/- {np.std(rec_phase):5.2f}")
    print(f"Mean F1       (per-phase) : {np.mean(f1_phase):6.2f} +/- {np.std(f1_phase):5.2f}")


if __name__ == "__main__":
    main()
