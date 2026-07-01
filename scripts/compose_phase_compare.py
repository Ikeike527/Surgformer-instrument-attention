"""per_phase で出力した成功(correct)/失敗(incorrect)の grid.png を Phase ごとに
縦に合成し、phaseN_compare.png を作る。成功=上(緑帯), 失敗=下(赤帯)。

使い方: python scripts/compose_phase_compare.py --dir outputs/attn_vis_cmp [--nb_classes 7]
"""
import os, glob, argparse
from PIL import Image, ImageDraw

BAND = 26  # ラベル帯の高さ(px)


def find_dirs(base, phase):
    ok = [d for d in glob.glob(os.path.join(base, f"*phase{phase}_frame*"))
          if os.path.isdir(d) and not d.endswith("_ng") and not d.endswith("_ok")]
    ng = glob.glob(os.path.join(base, f"*phase{phase}_frame*_ng"))
    return (ok[0] if ok else None), (ng[0] if ng else None)


def labeled(path, text, color):
    img = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (img.width, img.height + BAND), "white")
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, img.width, BAND], fill=color)
    d.text((6, 6), text, fill="white")
    canvas.paste(img, (0, BAND))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None, help="出力先(省略時は --dir)")
    ap.add_argument("--nb_classes", type=int, default=7)
    args = ap.parse_args()
    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    made = []
    for p in range(1, args.nb_classes + 1):
        ok_dir, ng_dir = find_dirs(args.dir, p)
        blocks = []
        if ok_dir:
            blocks.append(labeled(os.path.join(ok_dir, "grid.png"),
                                  f"Phase{p}  SUCCESS (OK)  [{os.path.basename(ok_dir)}]",
                                  (30, 130, 30)))
        if ng_dir:
            blocks.append(labeled(os.path.join(ng_dir, "grid.png"),
                                  f"Phase{p}  FAILURE (NG)  [{os.path.basename(ng_dir)}]",
                                  (170, 30, 30)))
        if not blocks:
            print(f"Phase{p}: no sample"); continue
        if len(blocks) == 1:
            tag = "OKonly" if ok_dir else "NGonly"
            print(f"Phase{p}: {tag} (片方のみ)")
        w = max(b.width for b in blocks)
        h = sum(b.height for b in blocks) + (len(blocks) - 1) * 4
        comp = Image.new("RGB", (w, h), "black")
        y = 0
        for b in blocks:
            comp.paste(b, (0, y)); y += b.height + 4
        out = os.path.join(out_dir, f"phase{p}_compare.png")
        comp.save(out)
        made.append(out)
        print(f"saved {out}")
    print(f"\n計 {len(made)} 枚")


if __name__ == "__main__":
    main()
