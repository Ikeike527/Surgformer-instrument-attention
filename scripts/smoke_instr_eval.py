"""instr_attn_bias の評価経路を1バッチで高速検証 (Docker内実行)。
- test dataset が 5-tuple (mask 付) を返すか
- DataLoader collate で mask が (B,T,S,S) に束ねられるか
- engine と同じく batch[4] を model(videos, instr_mask=...) へ渡して forward が通るか
- lambda=0 と lambda>0 で出力が変わるか (実マスクのある video49 サンプルで)
"""
import os, sys
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader, default_collate

sys.path.insert(0, "/workspace")
from timm.models import create_model
import model.surgformer_HTA  # noqa: F401
import utils
from downstream_phase.datasets_phase import build_dataset

CKPT = os.environ.get(
    "FINETUNE_PATH",
    "/workspace/outputs/Cholec80/surgformer_HTA_Cholec80_split_tr01-40_val41-48_test49-80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4/checkpoint-best.pth",
)

def make_args(lam):
    return SimpleNamespace(
        data_set="Cholec80", data_path="/workspace/data/Cholec80",
        data_strategy="online", output_mode="key_frame", cut_black=True,
        num_frames=16, sampling_rate=4, input_size=224, short_side_size=224,
        nb_classes=7, data_fps="1fps",
        instr_attn_bias=True, instr_lambda=lam, instr_bias_blocks="all",
        instr_mask_dirname="instrument_masks", reprob=0.0,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds, _ = build_dataset(is_train=False, test_mode=True, fps="1fps", args=make_args(2.0))

# video49 の器具ありフレーム付近を探す
target = None
for i, s in enumerate(ds.dataset_samples):
    if str(s["video_id"]) in ("video49", "49") and 990 <= int(s["frame_id"]) <= 1010:
        target = i
        break
if target is None:
    target = 0
print("picked dataset index:", target)

sample = ds[target]
print("tuple len:", len(sample), "(expect 5)")
buffer, label, sid, flag, mask = sample
print("buffer:", tuple(buffer.shape), "| mask:", tuple(mask.shape),
      "| mask nonzero%%=%.2f" % (100 * (mask > 0).float().mean().item()), "| id:", sid)

# DataLoader collate (2 サンプル)
batch = default_collate([ds[target], ds[target]])
print("collated mask batch:", tuple(batch[4].shape))

# モデル
net = create_model("surgformer_HTA", pretrained=False, num_classes=7, all_frames=16,
                   fc_drop_rate=0.5, drop_rate=0.0, drop_path_rate=0.1,
                   attn_drop_rate=0.0, instr_attn_bias=True, instr_lambda=2.0,
                   instr_bias_blocks="all", drop_block_rate=None)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt.get("module", ckpt))
utils.load_state_dict(net, state, prefix="")
net.to(device).eval()

videos = batch[0].to(device)
m = batch[4].to(device)
with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
    out_l0 = net(videos, instr_mask=None)            # bias 無し
    net.instr_lambda = 2.0
    out_l2 = net(videos, instr_mask=m)               # bias 有り
print("output shape:", tuple(out_l2.shape))
print("pred l0:", out_l0.argmax(-1).tolist(), "| pred l2:", out_l2.argmax(-1).tolist())
print("max|out_l0 - out_l2| =", (out_l0 - out_l2).abs().max().item(),
      "(>0 expected if mask nonzero)")
print("SMOKE OK")
