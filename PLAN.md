# PLAN: 視野外（黒領域）アテンション依存の検証と対処

## 背景 / 仮説

surgformer の空間アテンション（`Spatial_Self_Attention` の `CLS->patch`）で、円形カメラ視野外の黒領域に重みが乗って見える。

- 仮説: モデルが視野外（黒背景・黒帯）の領域を分類に利用している可能性がある。
- もし実依存しているなら、推論時マスクだけでは train/test mismatch になり、fine-tuning のやり直しが必要。
- ただし「視野外に attention が乗る = 推論が不正」ではない。**実依存しているか**を数値・挙動で確かめてから判断する。

## 現状の実装（確認済み）

- 空間アテンション直前で token を `(b t) k c` に並べ替え、フレーム内 patch 間 attention を計算（`K x K`）。
- 黒パッチマスク: 入力動画を de-normalize（mean/std=0.5）して `0..1` に戻し、`max(R,G,B) > threshold` を前景画素とみなす。`16x16` patch 単位で `max_pool2d` し、**1画素でも前景があれば patch を保持**。純黒パッチのみ softmax 前に `-inf` で落とす。
  - 定義箇所: `model/surgformer_HTA.py:28` (`_compute_black_patch_mask`)
  - 適用箇所: `model/surgformer_HTA.py:109` (`Attention_Spatial.forward`)
  - threshold 既定: `15.0/255 ≈ 0.0588`
- 同じ変更を `surgformer_base` / `surgformer_HTA_KCA` にも展開済み。
- 入口フラグ: `--disable_spatial_black_mask` / `--black_pixel_threshold`（学習・推論・可視化・スクリプト env）。
- 可視化のにじみ対策（表示専用、推論に影響なし）: `visualize_attention.py` で視野外を塗らない表示マスク + `grid.png` に `t`/`w` をコマ上へ描画。
- 既知の限界:
  - temporal attention は spatial より先に走るため、黒パッチ token は temporal ではまだ使われている（spatial のみ排除）。
  - 「1画素でも前景なら保持」のため、円境界にかかる patch は残る（純黒のみ落ちる）。

## 確認の考え方（重要）

2つの仕組みを混同しない。
- **描画 foreground mask（`visualize_attention.py`）**: 見せ方だけ。推論に影響なし。＝視覚的に隠すだけ。
- **spatial black mask（`surgformer_HTA.py`）**: softmax 前に純黒パッチ key を `-inf`。raw attention 自体を変える。

そして本質的な区別:
- **attention 量 ≠ 依存**。attention の大小は説明の手掛かりで因果ではない。
- 「視野外に依存しているか」は attention の見た目ではなく **入力を変えたときの予測の挙動差**で確かめる。

## 検証フェーズ（再学習なし・最優先）

Docker 実行可。

### Step 1: マスク on/off で同一フレームの可視化比較 ✅ 完了（2026-06-24）

`scripts/docker_visualize_attention.sh` を mask on / off で実行し、`attn.npz`（raw CLS→patch, 描画前）を比較。

**結果（video49/frame1000）**:
- 視野外 attention 質量（近似円外 40/196 patch）: **off 37.5% → on 16.0%**。spatial mask は raw attention を実際に変えている（＝視覚だけでなく内部も変化）。
- 予測は on/off とも Phase3 で一致、Phase3 確率 0.885→0.870 と僅差。**この1本では視野外への実依存は弱い**示唆。
- 旧スクリプトで視野外が大きく見えた原因は (1) raw に ~37% 乗る実体 + (2) bilinear 補間の滲み、の重なりと確定。

**この Step の限界**:
- 1サンプル・1フレーム集合・`CLS→patch`・最終 block のみ。
- 視野外定義が近似円でモデルの黒判定と不一致（on で 0% にならず 16% 残る理由）。
- attention 量の比較であって依存の証明ではない。→ Step 2 / Step 3 で補う。

### Step 2: test 全体で精度比較（mask on/off）✅ 完了（2026-06-24）

**実装**: 不要。既存 `scripts/test_phase.sh` の env フラグで切替可。Docker は `scripts/docker_test_phase.sh` 経由。

```bash
# on/off で 0.txt が衝突しないよう OUTPUT_ROOT を分ける。eval は Docker、集計は Docker 内 numpy。
NUM_WORKERS=2 MEM_LIMIT=50g OUTPUT_ROOT=/workspace/outputs/test_eval_on bash scripts/docker_test_phase.sh
NUM_WORKERS=2 MEM_LIMIT=50g BATCH_SIZE=16 DISABLE_SPATIAL_BLACK_MASK=1 OUTPUT_ROOT=/workspace/outputs/test_eval_off bash scripts/docker_test_phase.sh
# 各 0.txt のあるディレクトリ（…/Cholec80/<RUN_NAME>/）に対して:
#   python datasets/convert_results/convert_cholec80.py --main-path <dir>
#   python scripts/eval_phase_python.py --main-path <dir> --start 49 --end 80
```

**実行メモ（つまずき）**:
- `docker_test_phase.sh` は `--memory` 制限が無く、既定 `NUM_WORKERS=8` で host RAM を圧迫し SIGKILL(-9)。→ スクリプトに `MEM_LIMIT`（既定 24g, `--memory`/`--memory-swap`）を追加し `NUM_WORKERS=2`。host RAM は実測 ~21GiB で安定（リークではなくベースライン）。`MEM_LIMIT=50g` で完走（host 62GiB）。
- OFF は batch32 で **CUDA OOM**（GPU 47.5GB に対し断片化込み 44.6GB）。→ `BATCH_SIZE=16` で回避。eval はフレーム毎独立推論なので batch は指標に影響せず ON と比較可能。
- 集計（convert/eval_phase）は出力が root 所有のため host から書けず、Docker 内で実行。

**結果（test set 全体, video49-80, 32本）**:

| 指標 | mask ON（既定・純黒落とす） | mask OFF | 差 (ON−OFF) |
|---|---|---|---|
| Top-1 accuracy (frame, raw) | 90.65 | 90.79 | −0.14 |
| Mean Accuracy (per-video) | 92.70 | 92.83 | −0.13 |
| Mean Jaccard (per-phase) | 83.45 | 83.58 | −0.13 |
| Mean Precision (per-phase) | 92.26 | 92.30 | −0.04 |
| Mean Recall (per-phase) | 91.32 | 91.42 | −0.10 |
| Mean F1 (per-phase) | 91.54 | 91.65 | −0.11 |

phase 別でも全 phase 差 1pt 未満（最大 ClippingCutting F1 −0.19）。

**判断**:
- off→on で全指標が誤差レベル（−0.04〜−0.14pt）で**ほぼ不変**（むしろ ON が極僅かに低い）。→ **純黒パッチには実依存していない**。この結果だけからは fine-tuning やり直し不要。
- **重要な限界**: 現行 mask は **純黒パッチのみ**落とす弱いマスク。「不変」は「純黒パッチに非依存」を意味するだけで「視野外全体に非依存」ではない。境界帯・円外の半端に明るい領域への依存は Step 2 では検出できない。→ **だから Step 3（入力 ablation）が必要**。

### Step 3: 入力 Ablation（視野外への因果依存を直接テスト）✅ 完了（2026-06-24）

attention ではなく **入力画素を実際に消して予測の変化**を見る。これが依存の直接証拠。

**実装（完了）**:
- `model/surgformer_HTA.py`: `_ablate_offfield()` を追加。画像中心の固定円（半径 = `offfield_radius_scale × min(H,W)/2`）の**円外画素を正規化後の真の黒 `(0-mean)/std`** に置換（patch_embed 直前）。`ablate_offfield` / `offfield_radius_scale` パラメータ。
- `run_phase_training.py`: `--ablate_offfield` / `--offfield_radius_scale`、create_model 2箇所に配線。
- `scripts/test_phase.sh` / `docker_test_phase.sh`: env `ABLATE_OFFFIELD` / `OFFFIELD_RADIUS_SCALE`。
- `scripts/compare_predictions.py`: 2 つの 0.txt から flip率 / 正↔誤内訳 / 平均|Δmax-prob| / 平均KL / accuracy差を算出。
- 正規化は ImageNet 値 [0.485,0.456,0.406]/[0.229,0.224,0.225]（dataset・モデル一致。PLAN 旧記述「0.5」は誤り）。

**実行**: baseline=`test_eval_on`（mask on, ablなし）を流用。半径 1.0/0.9/0.8 でスイープ（mask on 既定 + ablation, batch16, mem50g）。

**結果（test set 全体, video49-80, 32本）**:

| 条件 | 視野外% | flip率 | 正→誤 / 誤→正 | 平均KL | frame acc | per-video Acc |
|---|---|---|---|---|---|---|
| baseline (ablなし) | 0 | – | – | – | 90.65 | 92.70 |
| ablate r=1.0（四隅のみ） | ~21% | 1.90% | 798 / 470 | 0.013 | 90.22 (−0.43) | 92.32 |
| ablate r=0.9（+境界帯） | ~36% | 3.01% | 1403 / 601 | 0.026 | 89.61 (−1.05) | 91.92 |
| ablate r=0.8（円内まで） | ~50% | 4.66% | 2393 / 746 | 0.048 | 88.51 (−2.15) | 91.17 |

**判断**:
- **視野外/周辺に弱い因果依存あり**。削る面積を増やすほど flip率・KL・accuracy 低下が単調増加（用量反応）。flip は「正→誤」が「誤→正」を一貫して上回る（r=0.8 で約 3:1）。
- **Step 2 との整合**: Step 2（純黒 mask on/off）は ~0 だったが ablation r=1.0（四隅）で −0.43pt。現行 mask が「純黒のみ・1画素でも前景なら保持」と弱く四隅の**非純黒画素**を残していたため。→ PLAN が予測した「Step2 では検出できない境界帯・円外への依存」が実在と確認。
- **ただし依存は弱い**: 画面の約半分（r=0.8）を黒く潰しても per-video accuracy は −1.53pt に留まり、予測の 95%+ は不変。
- **交絡の注意**: r<1.0 の人工黒リングは学習時に無い OOD 入力で、低下の一部は分布シフトの可能性。r=1.0（四隅は元々ほぼ黒＝ほぼ in-distribution）でも −0.43pt 出ている点が比較的クリーンな依存の証拠。

**結論**: 視野外への因果依存は **存在するが弱い**。fine-tuning やり直しは必須ではないが、視野外を確実に落とす対処（円形マスク化）の価値は Step 2 単独より高い。再学習要否は「−1〜2pt をどう評価するか」次第。

## 補助検証（必要に応じて）

- `all queries -> patch` 可視化を追加し、`CLS->patch` だけでなく列方向の attention mass を確認（視野外 leakage の確認に適切）。
- 視野外 attention mass を数値化（視野外 patch への重み合計）。
- 黒パッチ判定を「前景画素率 < X%」または「円形マスク（中心+半径の幾何）」に変更する案を比較（現行の純黒判定より境界に効く）。

## 対処フェーズ（仮説が当たった場合のみ）

1. **マスク定義の見直し**: 純黒判定 → 円形マスク or 前景画素率閾値。視野外を確実に落とす。
2. **temporal 側のマスク**: 黒パッチ token を temporal attention でも排除するか、block 投入前に token ゼロ化。
3. **fine-tuning やり直し**: 同じ pretrained weight を初期値に、視野外マスクありで phase recognition を再学習（pretraining からのやり直しは通常不要）。
4. 再学習後、attention 可視化 + test metrics を再比較。

## 未確認 / 注意

- Docker 実行（`surgformer-repro` イメージ）の動作確認はまだ。
- host には `timm`/`decord` 無し → 検証は Docker 前提。
- `checkpoint-best.pth` の所在: `/workspace/outputs/Cholec80/<RUN_NAME>/checkpoint-best.pth`。

## 次アクション

- [x] Step 1 実行 → on/off 比較（raw 37.5%→16.0%、予測ほぼ不変）
- [x] **Step 2**: mask on/off で test 全体 eval → metrics ほぼ不変（Acc 92.70 vs 92.83 等、差 <0.15pt）→ 純黒パッチ非依存
- [x] **Step 3（入力 ablation）**: 円形 FOV 外を黒で潰す ablation を r=1.0/0.9/0.8 でスイープ → 単調用量反応（per-video Acc 92.70→92.32→91.92→91.17）。視野外に**弱い因果依存**を確認。
- [x] Step 2+3 の判断: 視野外への因果依存は**存在するが弱い**。再学習は必須でないが円形マスク化の価値は Step2 単独より高い。
- [ ] Step 2+3 の判断基準に従い「視野外への因果依存の有無」と「再学習要否」を確定
