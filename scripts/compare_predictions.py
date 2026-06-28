#!/usr/bin/env python3
"""2 回の eval の rank txt (0.txt 形式) を突き合わせ、入力 ablation の因果効果を測る.

各行: "<unique_id> <video_id> <frame_id> [logit0, logit1, ...] <target>"
baseline と ablation を unique_id で揃え、以下を出力する:
  - flip 率 (argmax 予測が変わったフレーム割合)
  - baseline 正解→不正解 / 不正解→正解の内訳
  - 平均 |Δ max-prob|, 平均 KL(baseline || ablation)
  - 各々の frame-level accuracy と差
"""
import argparse
import glob
import os

import numpy as np


def softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def load(main_path):
    txts = sorted(glob.glob(os.path.join(main_path, "[0-9]*.txt")))
    if not txts:
        raise FileNotFoundError(f"No rank txt files in {main_path}")
    rec = {}
    for t in txts:
        for line in open(t).readlines()[1:]:
            if "[" not in line:
                continue
            name = line.split("[")[0].split()
            uid = name[0]
            logits = np.fromstring(line.split("[")[1].split("]")[0], sep=",")
            target = int(line.split("]")[1].split()[0])
            rec[uid] = (softmax(logits), target)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="ablation なし eval の 0.txt があるディレクトリ")
    ap.add_argument("--ablation", required=True, help="--ablate_offfield 付き eval の 0.txt があるディレクトリ")
    args = ap.parse_args()

    base = load(args.baseline)
    abla = load(args.ablation)
    keys = sorted(set(base) & set(abla))
    n = len(keys)
    if n == 0:
        raise ValueError("baseline と ablation で共通の unique_id が無い")
    only_b, only_a = len(base) - n, len(abla) - n

    flips = 0
    corr_to_wrong = wrong_to_corr = 0
    base_correct = abla_correct = 0
    dmaxprob = []
    kls = []
    for k in keys:
        pb, tb = base[k]
        pa, _ = abla[k]
        cb, ca = int(pb.argmax()), int(pa.argmax())
        base_correct += cb == tb
        abla_correct += ca == tb
        if cb != ca:
            flips += 1
            if cb == tb and ca != tb:
                corr_to_wrong += 1
            elif cb != tb and ca == tb:
                wrong_to_corr += 1
        dmaxprob.append(abs(pb.max() - pa.max()))
        kls.append(float(np.sum(pb * (np.log(pb + 1e-12) - np.log(pa + 1e-12)))))

    print(f"共通フレーム数: {n}  (baseline のみ {only_b}, ablation のみ {only_a})")
    print("=" * 56)
    print(f"flip 率 (予測クラス変化)   : {flips / n * 100:6.3f}%  ({flips}/{n})")
    print(f"  正解→不正解              : {corr_to_wrong}")
    print(f"  不正解→正解              : {wrong_to_corr}")
    print(f"平均 |Δ max-prob|          : {np.mean(dmaxprob):8.5f}")
    print(f"平均 KL(base || ablation)  : {np.mean(kls):8.5f}")
    print("-" * 56)
    print(f"frame accuracy  baseline   : {base_correct / n * 100:6.3f}%")
    print(f"frame accuracy  ablation   : {abla_correct / n * 100:6.3f}%")
    print(f"accuracy 差 (abl - base)   : {(abla_correct - base_correct) / n * 100:+.3f}pt")


if __name__ == "__main__":
    main()
