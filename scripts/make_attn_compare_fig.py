"""論文用: video49/frame1000 の λ=0 vs λ=3 注目ヒートマップ比較図を作成。
(a) SAM3器具マスク位置 / (b) λ=0 / (c) λ=3。個別PNG(subfigure用)と結合プレビューを出力。
素材: outputs/attn_vis_l{0,3}/video49_1000/frame_15.png (clean overlay), SAM3 mask。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = "/home/ikeido/test/Timesformer"
DATA = "/home/ikeido/datasets/Cholec80/cholec80"
FRAME_FILE = "video49_001001.png"   # frame_id=1000 の対象フレーム(=key frame)
OUT = os.path.join(REPO, "figs/attn_lambda_compare")
SIZE = 224
SHARE = {0: 0.098, 3: 0.481}        # npz から再確認済みの注目シェア
os.makedirs(OUT, exist_ok=True)


def resize(arr, mode):
    return np.asarray(Image.fromarray(arr).resize((SIZE, SIZE), mode))


# (a) 原フレーム + 器具マスク輪郭
frame = np.asarray(Image.open(f"{DATA}/frames/video49/{FRAME_FILE}").convert("RGB"))
frame = resize(frame, Image.BILINEAR).astype(np.float32) / 255.0
mask = np.asarray(Image.open(f"{DATA}/instrument_masks/video49/{FRAME_FILE}").convert("L"))
mask = (resize(mask, Image.NEAREST) > 127).astype(np.float32)
panel_a = frame.copy()
panel_a[mask > 0] = 0.5 * panel_a[mask > 0] + 0.5 * np.array([0.1, 1.0, 0.1])  # 緑で器具領域

# (b)(c) 既存の clean heatmap overlay
panel_b = np.asarray(Image.open(f"{REPO}/outputs/attn_vis_l0/video49_1000/frame_15.png").convert("RGB")) / 255.0
panel_c = np.asarray(Image.open(f"{REPO}/outputs/attn_vis_l3/video49_1000/frame_15.png").convert("RGB")) / 255.0

# 個別PNG (subfigure用, 余白なし)
for name, img in [("instrument_mask.png", panel_a),
                  ("lambda0.png", panel_b),
                  ("lambda3.png", panel_c)]:
    plt.imsave(os.path.join(OUT, name), np.clip(img, 0, 1))

# 結合プレビュー (ラベル付き)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
titles = [
    "(a) SAM3 instrument region",
    f"(b) $\\lambda=0$  (share={SHARE[0]:.3f})",
    f"(c) $\\lambda=3$  (share={SHARE[3]:.3f})",
]
for ax, img, t in zip(axes, [panel_a, panel_b, panel_c], titles):
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(t, fontsize=12)
    ax.axis("off")
fig.suptitle("video49 frame1000: CLS->patch attention shifts to instrument as $\\lambda$ increases",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_preview.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved:", *sorted(os.listdir(OUT)), sep="\n  ")
