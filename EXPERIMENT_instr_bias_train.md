# 実験メモ: 器具ソフトバイアス込み fine-tune（進行中）

最終更新: 2026-07-02 09:20 JST（学習継続中）

## 目的
SAM3 器具マスクによる Spatial Attention ソフトバイアスを、**推論時のみ**から
**学習時から適用**する方式へ拡張し、Phase 分類精度が向上するかを検証する。

## 設定
| 項目 | 値 |
|---|---|
| ベースモデル | surgformer_HTA (Cholec80, 16 frame, online/key_frame) |
| 初期化 | 既存 phase 学習済み `checkpoint-best.pth` から fine-tune |
| バイアス | `INSTR_ATTN_BIAS=1`, `INSTR_LAMBDA=2.0`, `INSTR_BIAS_BLOCKS=all` |
| マスク | SAM3 `metallic instrument`、video01-48 生成済み（train+val） |
| 前処理 | bias有効時は **幾何拡張オフ・決定的Resize**（マスク整合保証）、mixup/cutmix=0 |
| 最適化 | LR=1e-4, warmup=2ep, layer_decay=0.75, batch=4×update_freq2（実効8） |
| epochs | 20（既定） |
| GPU | RTX 6000 Ada (48GB) |
| split | train=video01-40 / val=video41-48 / test=video49-80 |
| 出力 | `outputs/phase_train_instr_l2.0/Cholec80/..._0.0001_..._Fixed_Stride_4/` |

## 比較基準（fine-tune 元、bias無し・50ep学習）
- **baseline best val_acc1 = 90.876（epoch 28/50）**

## 進捗（epoch 0–6 完了、epoch 7 進行中）
| epoch | train_loss | val_loss | val_acc1 | val_acc5 |
|---:|---:|---:|---:|---:|
| 0 | 0.4531 | 0.3744 | **90.432** | 99.520 |
| 1 | 0.4525 | 0.4327 | 88.632 | 98.899 |
| 2 | 0.4526 | 0.3945 | 90.353 | 99.081 |
| 3 | 0.4517 | 0.3849 | 90.418 | 99.347 |
| 4 | 0.4509 | 0.4016 | 90.325 | 99.170 |
| 5 | 0.4503 | 0.3956 | **90.535** ← 現best | 99.179 |
| 6 | 0.4495 | 0.4294 | 89.392 | 98.810 |

## 現時点の所見（暫定・7/20ep）
- **val_acc1 は約90.4でほぼ横ばい**。現best 90.535（ep5）は **baseline 90.876 を下回る**。
- train_loss もほぼ動かず（0.4531→0.4495）。fine-tune 元が既に収束済みのため頭打ち。
- 現状「学習時バイアスによる明確な向上は見えていない（むしろ僅かに下）」。
  ただし **まだ7/20ep で早期**・val のみ・test 未評価。best は今後上振れる可能性あり。

### 解釈上の注意（比較の非対称性）
1. bias 学習は **幾何拡張オフ**、baseline は **拡張あり**。前処理レジームが違うため純粋比較でない。
2. baseline best は 50ep/コサイン schedule の ep28。本実験は 20ep の短い schedule。
3. λ=2 固定・LR=1e-4 は 1 条件のみ。λ や適用 block を振っていない。

## 経過の見方（監視コマンド）
```bash
# エポック推移（val_acc1 / loss）
RUN=$(ls -d outputs/phase_train_instr_l2.0/Cholec80/*/ | head -1)
tail -f "$RUN/log.txt"

# 稼働状況・経過
docker ps --filter name=surgformer-train-phase --format '{{.Status}}'
ls -t "$RUN"/checkpoint-*.pth | head   # 最新epoch

# 現best のepoch確認
python3 -c "import json;print(max((json.loads(l) for l in open('$RUN/log.txt') if l.strip()),key=lambda d:d['val_acc1']))"
```
読みどころ:
- **val_acc1 が baseline 90.876 を安定して超えるか**が成否の一次判断。
- train_loss が下がるのに val_acc1 が伸びない → 過学習/バイアス過強（λを下げる）。
- val_acc1 が baseline 近傍で横ばい → バイアスが判別に寄与していない可能性。

## 停止・再開
- ckpt は毎エポック末保存。エポック区切りで停止推奨。
- 停止: `docker stop surgformer-train-phase`
- 再開: 同 env で `EPOCHS=20 bash scripts/gen_masks_and_train.sh`（`AUTO_RESUME` が最新ckptから継続、マスク生成はスキップ）
- ETA: 残り約13ep × 1.5h ≈ **+19.5h**（20ep 完走の場合）

## 次アクション候補
1. **20ep 完走 → test(video49-80) で評価**し baseline と正式比較。
2. 効果が出ない場合の振り直し:
   - λ スイープ（0.5 / 1 / 3）、`INSTR_BIAS_BLOCKS` を最終数 block 限定。
   - 前処理を揃える（baseline も幾何拡張オフで取り直す or bias側で拡張整合を実装）。
   - LR/epoch 調整（fine-tune なら 5–10ep で十分な可能性）。
