# PLAN: SAM3 器具マスクによる Spatial Attention 強調（ソフトバイアス方式）

## 実装状況（2026-06-26）— 推論時のみ・実装完了/検証済み
- モデル/データ/可視化/評価/スクリプトの配線をすべて実装。`λ=0`・mask無しは現行と
  数値完全一致（max|d|=0, モデル単体テスト）。
- 実マスク(video49)で end-to-end 検証済み。**video49 frame1000**:
  器具パッチ(19/196)への CLS→patch 注目シェアが **λ=0: 0.098 → λ=3: 0.481**（約5倍）、
  予測は Phase4 のまま安定 (p=0.870→0.834)。
- 実行例:
  ```
  INSTR_ATTN_BIAS=1 INSTR_LAMBDA=3 VIDEO_ID=49 FRAME_ID=1000 \
    OUT_DIR=/workspace/outputs/attn_vis_l3 bash scripts/docker_visualize_attention.sh
  ```
- 残: SAM3 マスクを video50〜80 へ拡大 → test 全体で λ スイープ精度評価。
- 実装ファイル: `model/surgformer_HTA.py`(`_compute_instr_patch_coverage` / `_make_instr_attn_bias` /
  `Attention_Spatial`・`Block`・`forward(_features)` 配線), `datasets/phase/Cholec80_phase.py`
  (`_resolve_mask_path` / `_load_instr_masks` / test 返り値5要素目),
  `downstream_phase/{visualize_attention,engine_for_phase,run_phase_training}.py`,
  `scripts/{test_phase,docker_test_phase,docker_visualize_attention}.sh`。
  フラグ: `--instr_attn_bias --instr_lambda <f> --instr_bias_blocks all|N --instr_mask_dirname`。

## 目的
SAM3 で抽出した手術器具のセグメンテーションを、Surgformer の分類時に **Spatial
Attention へ加算バイアスとして注入**し、器具領域のパッチへの注目を強める。
領域を除外する（ハードマスク）のではなく、重み付けで強調する **②ソフトバイアス**方式。

数式（空間 attention の softmax 直前）:

```
attn = (q · kᵀ) * scale
attn = attn + λ · m_key          # m_key: 各 key パッチの器具被覆率 (0..1)
attn = attn.softmax(dim=-1)
```

- `m_key` は「key 列（注目される側）」にのみ加える → 器具パッチが**より注目されやすく**なる。
- `λ` は強調の強さ（ハイパラ）。`λ=0` で現状と完全一致。
- 既存の黒背景ハードマスク（`masked_fill(-inf)`, surgformer_HTA.py:128-143）とは**共存**
  （黒パッチは除外のまま）。

---

## アーキテクチャ / データフロー

```
[SAM3 側]  各フレーム画像 → 器具マスク PNG  (Part A の仕様で出力)
                                   │
[Timesformer 側]                   ▼
  datasets_phase  ── フレームと同じサンプリング/Resize でマスクも読み込み
                  ── 224×224 へ Resize → 16px パッチで avg_pool → 14×14 被覆率
                  ── (B*T, 196) の attn_bias テンソルを buffer と一緒に返す
                                   │
  surgformer_HTA  forward_features ── attn_bias を各 Block へ伝播
                  Block.forward    ── Attention_Spatial へ渡す
                  Attention_Spatial.forward ── softmax 直前に λ·m_key を加算
```

---

## Part A: SAM3 側への依頼仕様
SAM3 へ渡す自己完結の依頼書は **[SAM3_mask_request.md](SAM3_mask_request.md)** に分離。
要点のみ:

- 入力 = 分類器が実際に読む画像 `{DATA}/frames_cutmargin/{video_id}/<filename>.png`
  （無い video のみ `frames/`）。**リサイズ厳禁・原寸のまま**処理。
- 出力 = **入力ファイル名を 1:1 でそのままコピー**し、ディレクトリ名のみ
  `instrument_masks` に差し替え: `{DATA}/instrument_masks/{video_id}/<同じfilename>.png`。
  - ファイル名は固定連番規則ではなく **実在 PNG 名がそのまま正**（分類器側は
    `img_path` の `frames_cutmargin`/`frames` を `instrument_masks` に文字列置換して探す）。
- SAM3 テキストプロンプトは **`metallic instrument`**（相性確認済み・確定）。
- 単一チャンネル PNG。器具=255/背景=0（または 0..255 尤度）。複数器具は 1 枚に OR 統合。
- 範囲: まず video49 → 整合 OK なら test split（video49〜80）。

---

## Part B: Timesformer 側の実装

### B-1. マスク読み込み（datasets/phase/Cholec80_phase.py）
フレーム読み込み (`_video_batch_loader` / `_video_batch_loader_for_key_frames`) と
**完全に同じ index・同じ順序**でマスク PNG を読み込む。

- マスクパス: `img_path` の `frames`(or `frames_cutmargin`) を `instrument_masks` に置換。
- マスクが存在しないフレームは **全 0（バイアスなし）** にフォールバック（後方互換）。
- フレームと同じ `data_resize`（224×224 Resize, bilinear）をマスクにも適用
  （マスクは nearest 推奨）→ `(T, 224, 224)` の float [0,1]。
- `__getitem__` の戻り値に `mask_buffer` を追加（既存 4-tuple → 5 要素へ拡張）。
  - 影響箇所: `infer_sample`（visualize_attention.py:209）, `engine_for_phase.py`,
    `emit_features.py` の unpack を更新。

### B-2. パッチ被覆率 → attn_bias（model/surgformer_HTA.py）
`(B, T, 224, 224)` のマスクを 16px パッチで avg_pool し `(B*T, 196)` の被覆率に変換。
`_compute_black_patch_mask`（surgformer_HTA.py:28）と同じ要領の関数を新設:

```python
def _compute_instr_bias(mask, patch_size):
    # mask: (B, T, H, W) in [0,1] → (B*T, P) 被覆率
    m = rearrange(mask, "b t h w -> (b t) 1 h w")
    pooled = F.avg_pool2d(m, kernel_size=patch_size, stride=patch_size)
    return pooled.flatten(1)            # (B*T, P)
```

### B-3. 配線（patch_mask と同じ経路）
1. `forward_features`（surgformer_HTA.py:559）で `attn_bias = _compute_instr_bias(...)`
   を作り、`blk(x, B, T, K, patch_mask=..., attn_bias=attn_bias)` で各 Block へ。
2. `Block.forward`（:309）に `attn_bias=None` 引数を追加し、
   `self.attn.forward(..., attn_bias=attn_bias)` へ素通し。
3. `Attention_Spatial.forward`（:109）で softmax 直前に加算:

```python
attn = (q @ k.transpose(-2, -1)) * self.scale
if attn_bias is not None:                       # (BT, P)
    cls_bias = attn.new_zeros(attn_bias.shape[0], 1)        # CLS 列は 0
    key_bias = torch.cat((cls_bias, attn_bias), dim=1)      # (BT, K)
    attn = attn + self.instr_lambda * key_bias[:, None, None, :]
# 既存の黒背景 masked_fill はこの後（-inf は加算より優先される）
if patch_mask is not None:
    ...
attn = attn.softmax(dim=-1)
```

### B-4. 設定フラグ（model 引数 + argparse）
- `instr_attn_bias`（bool, default False）: 機能の ON/OFF。
- `instr_lambda`（float, default 0.0 → 推奨初期値 1.0〜3.0 を探索）: 強調強度。
- `instr_bias_blocks`（"all" | "last_k"）: 全 Block か最終数 Block のみか（任意）。
  まずは **all** で実装し、効果を見て最終 N Block 限定を検討。

### B-5. 推論専用 vs 学習
- **推論時のみ強調**（学習済み ckpt をそのまま使用）: B-1〜B-4 のみで可。まずこれで効果検証。
- **バイアス込みで再学習/Fine-tune**: 上記がそのまま学習にも効く（`λ` 固定 or 学習可能
  パラメータ化）。推論専用で効果が出たら次段で検討。

---

## 整合性の注意点（崩れやすい順）
1. **解像度 1:1**: SAM3 マスクは入力 PNG と同解像度。Resize は Timesformer 側のみ。
2. **cut_black 一致**: モデルは `frames_cutmargin` を読む（存在時, Cholec80_phase.py の
   `_resolve_sample_path`）。マスクも同じ画像系列に。
3. **サンプリング一致**: マスクはフレームと同じ index 列（同じ `sampled_list`）で読む。
   独自に間引かないこと。
4. **pooling 種別**: マスクは avg_pool（被覆率）、黒背景マスクは max_pool（既存）と別物。
5. **CLS 列にバイアスを足さない**（CLS は 0 のまま）。

---

## 検証方法
1. **無効化テスト**: `instr_lambda=0` で出力が現行と完全一致（数値差 < 1e-5）を確認。
2. **可視化**: visualize_attention.py で `λ=0` と `λ>0` の attention マップを比較し、
   器具領域のヒートが強まることを目視確認（grid.png）。
3. **マスク整合**: 数サンプルで「入力フレーム・SAM3 マスク・14×14 被覆率」を並べて表示し、
   器具位置がパッチ上で一致するか確認。
4. **指標**: テストセットで `λ ∈ {0, 0.5, 1, 2, 3}` を振り、Phase 分類 Accuracy /
   Jaccard / F1 の変化を比較（scripts/test_phase.sh）。
5. **失敗例**: `--per_phase_mode incorrect` で誤分類例の attention 変化を観察。

---

## 実装ステップ（推奨順）
1. [SAM3] Part A の仕様で評価対象 video のマスクを `instrument_masks/` に出力。
2. [TS] B-1 マスク読み込み + 戻り値拡張（unpack 箇所も更新）。
3. [TS] B-2/B-3/B-4 attn_bias 配線とフラグ。`λ=0` で一致を確認（検証 1）。
4. [TS] 可視化で強調を確認（検証 2,3）。
5. [TS] `λ` スイープで指標評価（検証 4,5）。
6. 効果が出れば再学習 / 最終 Block 限定 / `λ` 学習可能化を検討。

---

## 影響ファイル一覧
- `datasets/phase/Cholec80_phase.py` — マスク読み込み・戻り値拡張（B-1）
- `model/surgformer_HTA.py` — `_compute_instr_bias`, 配線, バイアス加算（B-2〜B-4）
- `downstream_phase/datasets_phase.py` — 新 args の受け渡し
- `downstream_phase/engine_for_phase.py` / `emit_features.py` — unpack 更新
- `downstream_phase/visualize_attention.py` — unpack 更新 + 比較可視化（検証用）
- 学習/評価スクリプト（`scripts/*.sh`）— `--instr_attn_bias --instr_lambda ...` 追加
