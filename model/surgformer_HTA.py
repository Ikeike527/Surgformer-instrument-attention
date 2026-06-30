# 模型的base版本，重构了TimeSFormer模型，基本上与TimeSFormer模型结果一致；
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
import utils
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
import math

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 7,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


def _compute_black_patch_mask(video, patch_size, pixel_threshold, mean, std):
    """Return a (B*T, K) bool mask where True keeps non-black patches."""
    mean = video.new_tensor(mean).view(1, -1, 1, 1, 1)
    std = video.new_tensor(std).view(1, -1, 1, 1, 1)
    video = video * std + mean
    foreground = video.amax(dim=1, keepdim=True) > pixel_threshold
    foreground = rearrange(foreground.float(), "b c t h w -> (b t) c h w")
    pooled = F.max_pool2d(
        foreground,
        kernel_size=patch_size,
        stride=patch_size,
    )
    return pooled.flatten(1) > 0


def _compute_instr_patch_coverage(mask, patch_size):
    """Return a (B*T, P) float tensor of per-patch instrument coverage in [0,1].

    mask: (B, T, H, W) in [0,1] (1=instrument). avg_pool で 16x16 patch 単位の
    被覆率に縮約し、_compute_black_patch_mask と同じ row-major で flatten する。
    """
    mask = rearrange(mask.float(), "b t h w -> (b t) 1 h w")
    pooled = F.avg_pool2d(
        mask,
        kernel_size=patch_size,
        stride=patch_size,
    )
    return pooled.flatten(1)  # (B*T, P)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_Spatial(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # 可視化用: save_attn=True のときのみ softmax 後の空間 attention を保持する
        # (デフォルトFalseのため通常の学習/推論には影響しない)
        self.save_attn = False
        self.attn_map = None

    def forward(self, x, B, patch_mask=None, attn_bias=None):
        BT, K, C = x.shape
        T = BT // B
        qkv = self.qkv(x)
        # For Intra-Spatial: (BT, heads, K, C)
        # Atten: K*K, Values: K*C
        qkv = rearrange(
            qkv,
            "(b t) k (qkv num_heads c) -> qkv (b t) num_heads k c",
            t=T,
            qkv=3,
            num_heads=self.num_heads,
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 器具マスクによるソフトバイアス: softmax 前に key 列(注目される側)へ加算し、
        # 器具パッチへの注目を強める。CLS 列(col0)には足さない。
        if attn_bias is not None:
            cls_bias = attn_bias.new_zeros(attn_bias.shape[0], 1)  # (BT,1)
            key_bias = torch.cat((cls_bias, attn_bias), dim=1)  # (BT, K)
            attn = attn + key_bias[:, None, None, :].to(attn.dtype)
        token_mask = None
        if patch_mask is not None:
            cls_mask = torch.ones(
                (patch_mask.shape[0], 1),
                dtype=torch.bool,
                device=patch_mask.device,
            )
            token_mask = torch.cat((cls_mask, patch_mask), dim=1)
            attn = attn.masked_fill(
                ~token_mask[:, None, None, :],
                torch.finfo(attn.dtype).min,
            )
        attn = attn.softmax(dim=-1)
        if token_mask is not None:
            attn = attn * token_mask[:, None, :, None].to(attn.dtype)
        if self.save_attn:
            # (BT, num_heads, K, K)  K = 1(CLS) + パッチ数
            self.attn_map = attn.detach()
        attn = self.attn_drop(attn)

        x = attn @ v
        x = rearrange(
            x,
            "(b t) num_heads k c -> (b t) k (num_heads c)",
            b=B,
        )
        x = self.proj(x)
        return self.proj_drop(x)


class Attention_Temporal(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv_4 = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.qkv_8 = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.qkv_16 = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj_4 = nn.Linear(dim, dim)
            self.proj_8 = nn.Linear(dim, dim)
            self.proj_16 = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # 可視化用: save_attn=True のときのみ softmax 後の全スケール時間 attention
        # (attn_16, T×T) を保持する (デフォルトFalseのため学習/推論には影響しない)
        self.save_attn = False
        self.attn_map = None

    def forward(self, x, B):
        BK, T, C = x.shape
        t1 = T // 4
        t2 = T // 2
        x_4 = x[: , T-t1: , ]
        x_8 = x[: , t2: , ]
        x_16 = x
        K = BK // B
        
        qkv_4 = self.qkv_4(x_4)

        qkv_4 = rearrange(
            qkv_4,
            "(b k) t (qkv num_heads c) -> qkv (b k) num_heads t c",
            k=K,
            qkv=3,
            num_heads=self.num_heads,
        )
        q_4, k_4, v_4 = (qkv_4[0], qkv_4[1], qkv_4[2])

        qkv_8 = self.qkv_8(x_8)

        qkv_8 = rearrange(
            qkv_8,
            "(b k) t (qkv num_heads c) -> qkv (b k) num_heads t c",
            k=K,
            qkv=3,
            num_heads=self.num_heads,
        )
        q_8, k_8, v_8 = (qkv_8[0], qkv_8[1], qkv_8[2])

        qkv_16 = self.qkv_16(x_16)

        qkv_16 = rearrange(
            qkv_16,
            "(b k) t (qkv num_heads c) -> qkv (b k) num_heads t c",
            k=K,
            qkv=3,
            num_heads=self.num_heads,
        )
        q_16, k_16, v_16 = (qkv_16[0], qkv_16[1], qkv_16[2])
        
        attn_4 = (q_4 @ k_4.transpose(-2, -1)) * self.scale
        attn_4 = attn_4.softmax(dim=-1)
        attn_4 = self.attn_drop(attn_4)
        x_4 = attn_4 @ v_4
        x_4 = rearrange(x_4, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)

        attn_8 = (q_8 @ k_8.transpose(-2, -1)) * self.scale
        attn_8 = attn_8.softmax(dim=-1)
        attn_8 = self.attn_drop(attn_8)
        x_8 = attn_8 @ v_8
        x_8 = rearrange(x_8, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)

        attn_16 = (q_16 @ k_16.transpose(-2, -1)) * self.scale
        attn_16 = attn_16.softmax(dim=-1)
        if self.save_attn:
            # (BK, num_heads, T, T)  全フレームを q/k に持つ最終スケール
            self.attn_map = attn_16.detach()
        attn_16 = self.attn_drop(attn_16)
        x_16 = attn_16 @ v_16
        x_16 = rearrange(x_16, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)

        x_4 = self.proj_4(x_4)
        x_8[:, t1:, :] = 0.5 * x_8[:, t1:, :] + 0.5 * x_4
        x_8 = self.proj_8(x_8)
        x_16[:, t2: , :] = 0.5 * x_16[:, t2: , :] + 0.5 * x_8
        x_16 = self.proj_drop(self.proj_16(x_16))

        return x_16


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_Spatial(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        ## Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention_Temporal(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, B, T, K, patch_mask=None, attn_bias=None):
        # 如果alpha以及beta初始化为0，则xs、xt初始化为0, 在训练过程中降低了学习难度；
        # 仿照其余模型可以使用alpha.sigmoid()以及beta.sigmoid()；
        B, M, C = x.shape
        assert T * K + 1 == M

        # Temporal_Self_Attention
        xt = x[:, 1:, :]
        xt = rearrange(xt, "b (k t) c -> (b k) t c", t=T)

        res_temporal = self.drop_path(
            self.temporal_attn.forward(self.temporal_norm1(xt), B)
        )

        res_temporal = rearrange(
                res_temporal, "(b k) t c -> b (k t) c", b=B
            )  # 通过FC时需要将时空tokens合并，再通过残差连接连接输入特征
        xt = self.temporal_fc(res_temporal) + x[:, 1:, :]

        # Spatial_Self_Attention
        init_cls_token = x[:, 0, :].unsqueeze(1)  # B, 1, C
        cls_token = init_cls_token.repeat(1, T, 1)  # B, T, C
        cls_token = rearrange(cls_token, "b t c -> (b t) c", b=B, t=T).unsqueeze(1)
        xs = xt
        xs = rearrange(xs, "b (k t) c -> (b t) k c", t=T)

        xs = torch.cat((cls_token, xs), 1)  # BT, K+1, C
        res_spatial = self.drop_path(
            self.attn.forward(self.norm1(xs), B, patch_mask=patch_mask,
                              attn_bias=attn_bias)
        )

        ### Taking care of CLS token
        cls_token = res_spatial[:, 0, :]  # BT, C 表示了在每帧单独学习的class token
        cls_token = rearrange(cls_token, "(b t) c -> b t c", b=B, t=T)
        cls_token = torch.mean(cls_token, 1, True)  # 通过在全局帧上平均来建立时序关联（适用于视频分类任务）
        res_spatial = res_spatial[:, 1:, ]  # BT, xK, C
        res_spatial = rearrange(
            res_spatial, "(b t) k c -> b (k t) c", b=B)
        res = res_spatial
        x = xt
        ## Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 通过MLP学习时序对应的cls_token?

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=8,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(patch_size[0], patch_size[1]),
            stride=(patch_size[0], patch_size[1]),
        )
        # 直接使用3D卷积来映射时序帧到视频序列tokens，在过程中进行Temporal Sample
        # 对于逐帧计算的Tool以及Phase，怎么处理模型结构的变化？降低视频序列长度并且放弃时序采样

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2)
        x = rearrange(x, "(b t) c k -> b t k c", b=B)

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=7,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        fc_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        all_frames=16,
        spatial_black_mask=True,
        black_pixel_threshold=15.0 / 255.0,
        input_mean=(0.485, 0.456, 0.406),
        input_std=(0.229, 0.224, 0.225),
        ablate_offfield=False,
        offfield_radius_scale=1.0,
        offfield_invert=False,
        instr_attn_bias=False,
        instr_lambda=0.0,
        instr_bias_blocks="all",
    ):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.spatial_black_mask = spatial_black_mask
        self.black_pixel_threshold = float(black_pixel_threshold)
        # 器具マスクによる Spatial Attention 強調 (ソフトバイアス)
        self.instr_attn_bias = instr_attn_bias
        self.instr_lambda = float(instr_lambda)
        # "all" = 全 Block に適用 / 整数(または数値文字列) = 最終 N Block のみ
        self.instr_bias_blocks = instr_bias_blocks
        self.input_mean = tuple(float(v) for v in input_mean)
        self.input_std = tuple(float(v) for v in input_std)
        # Step3 入力 ablation: 円形 FOV 外の画素を真の黒に潰して因果依存をテストする
        self.ablate_offfield = ablate_offfield
        self.offfield_radius_scale = float(offfield_radius_scale)
        # 逆実験: FOV 内パッチを除外し視野外パッチのみで推論する (視野外の情報量を測る)
        self.offfield_invert = offfield_invert
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
        )
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches // all_frames + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.time_embed = nn.Parameter(torch.zeros(1, all_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc_dropout = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if "Block" in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "time_embed"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _make_spatial_patch_mask(self, x):
        if not self.spatial_black_mask:
            return None
        return _compute_black_patch_mask(
            x,
            patch_size=self.patch_embed.patch_size,
            pixel_threshold=self.black_pixel_threshold,
            mean=self.input_mean,
            std=self.input_std,
        )

    def _make_instr_attn_bias(self, instr_mask):
        """器具マスク (B,T,H,W in [0,1]) から (B*T, P) の加算バイアスを作る。
        instr_lambda を掛けて返す。無効時/マスク無し時は None。"""
        if not self.instr_attn_bias or instr_mask is None or self.instr_lambda == 0.0:
            return None
        coverage = _compute_instr_patch_coverage(
            instr_mask, patch_size=self.patch_embed.patch_size
        )
        return self.instr_lambda * coverage

    def _instr_bias_block_ids(self):
        """attn_bias を適用する Block index 集合を返す。"""
        if self.instr_bias_blocks == "all":
            return set(range(self.depth))
        n = int(self.instr_bias_blocks)
        return set(range(max(self.depth - n, 0), self.depth))

    def _ablate_offfield(self, x):
        """円形 FOV 外の画素を正規化後の真の黒 (0-mean)/std に置換して返す。
        視野外への因果依存テスト用 (Step3)。x: (B, C, T, H, W)。"""
        if not self.ablate_offfield:
            return x
        B, C, T, H, W = x.shape
        device = x.device
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        radius = self.offfield_radius_scale * (min(H, W) / 2.0)
        yy = torch.arange(H, device=device).view(H, 1).float()
        xx = torch.arange(W, device=device).view(1, W).float()
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        offfield = (dist2 > radius**2).view(1, 1, 1, H, W)  # True = 視野外
        fill = x.new_tensor(
            [(0.0 - m) / s for m, s in zip(self.input_mean, self.input_std)]
        ).view(1, C, 1, 1, 1)
        return torch.where(offfield, fill, x)

    def _make_offfield_patch_mask(self, x):
        """円形 FOV 外(視野外)のパッチのみ True(keep)、FOV 内パッチは False(除外)の
        (B*T, K) bool マスクを返す。視野外のみで推論する逆実験用。
        黒判定ではなく純粋な幾何(円)で判定するため、黒縁・オーバーレイ文字を問わず
        視野外パッチを全て attention に通し、FOV 内は -inf 除外する。"""
        B, C, T, H, W = x.shape
        device = x.device
        p = self.patch_embed.patch_size
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0
        radius = self.offfield_radius_scale * (min(H, W) / 2.0)
        yy = torch.arange(H, device=device).view(H, 1).float()
        xx = torch.arange(W, device=device).view(1, W).float()
        offfield = ((yy - cy) ** 2 + (xx - cx) ** 2 > radius**2).float()  # (H,W)
        pooled = F.avg_pool2d(offfield.view(1, 1, H, W), kernel_size=p, stride=p)
        keep = pooled.flatten(1) >= 0.5  # (1, K) パッチの過半が視野外なら keep
        return keep.expand(B * T, -1)

    def forward_features(self, x, instr_mask=None):
        # B, C, T, H, W
        if self.ablate_offfield and self.offfield_invert:
            # 視野外のみで推論: 画素は潰さず、FOV 内パッチを attention から除外する
            spatial_patch_mask = self._make_offfield_patch_mask(x)
        else:
            x = self._ablate_offfield(x)
            spatial_patch_mask = self._make_spatial_patch_mask(x)
        attn_bias = self._make_instr_attn_bias(instr_mask)
        bias_block_ids = self._instr_bias_block_ids() if attn_bias is not None else set()
        x = self.patch_embed(x)
        # B, T, K, C
        B, T, K, C = x.size()
        W = int(math.sqrt(K))

        # 添加Spatial Position Embedding
        x = rearrange(x, "b t k c -> (b t) k c")
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # BT, 1, C
        x = torch.cat((cls_tokens, x), dim=1)  # BT, HW+1, C  ---> 2*8, 196+1, 768
        x = x + self.pos_embed  # BT, HW, C  ---> 2*8, 196, 768
        x = self.pos_drop(x)

        # 添加Temporal Position Embedding
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:, 1:]  # 过滤掉cls_tokens
        x = rearrange(x, "(b t) k c -> (b k) t c", b=B)
        x = x + self.time_embed  # BK, T, C  ---> 2*196, 8, 768
        x = self.time_drop(x)

        # 添加Cls token
        x = rearrange(x, "(b k) t c -> b (k t) c", b=B)  # Spatial-Temporal tokens
        x = torch.cat((cls_tokens, x), dim=1)  # 时空tokens对应的class token的添加；

        for i, blk in enumerate(self.blocks):
            blk_bias = attn_bias if i in bias_block_ids else None
            x = blk(x, B, T, K, patch_mask=spatial_patch_mask, attn_bias=blk_bias)

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x, instr_mask=None):
        x = self.forward_features(x, instr_mask=instr_mask)
        x = self.head(self.fc_dropout(x))
        return x


@register_model
def surgformer_HTA(pretrained=False, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()

    if pretrained:
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()
        if "model_state" in checkpoint.keys():
            checkpoint = checkpoint["model_state"]
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # strip `model.` prefix
                name = k[6:] if k.startswith("model") else k
                new_state_dict[name] = v
            checkpoint = new_state_dict

            add_list = []
            for k in state_dict.keys():
                if "blocks" in k and "qkv_4" in k:
                    k_init = k.replace("qkv_4", "qkv")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "qkv_8" in k:
                    k_init = k.replace("qkv_8", "qkv")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "qkv_16" in k:
                    k_init = k.replace("qkv_16", "qkv")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_4" in k:
                    k_init = k.replace("proj_4", "proj")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_8" in k:
                    k_init = k.replace("proj_8", "proj")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_16" in k:
                    k_init = k.replace("proj_16", "proj")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
            print(f"Adding keys from pretrained checkpoint:", ", ".join(add_list))
            remove_list = []
            for k in state_dict.keys():
                if (
                    ("head" in k or "patch_embed" in k)
                    and k in checkpoint
                    and k in state_dict
                    and checkpoint[k].shape != state_dict[k].shape
                ):
                    remove_list.append(k)
                    del checkpoint[k]
            print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))

            # if 'time_embed' in checkpoint and state_dict['time_embed'].size(1) != checkpoint['time_embed'].size(1):
            #     print('Resize the Time Embedding, from %s to %s' % (str(checkpoint['time_embed'].size(1)), str(state_dict['time_embed'].size(1))))
            #     time_embed = checkpoint['time_embed'].transpose(1, 2)
            #     new_time_embed = F.interpolate(time_embed, size=(state_dict['time_embed'].size(1)), mode='nearest')
            #     checkpoint['time_embed'] = new_time_embed.transpose(1, 2)
            utils.load_state_dict(model, checkpoint)

        elif "model" in checkpoint.keys():
            checkpoint = checkpoint["model"]

            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # strip `model.` prefix
                name = k[8:] if k.startswith("encoder") else k
                new_state_dict[name] = v
            checkpoint = new_state_dict

            add_list = []
            for k in state_dict.keys():
                if "blocks" in k and "qkv_4" in k and "temporal_attn" in k:
                    k_init = k.replace("qkv_4", "qkv")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "qkv_8" in k and "temporal_attn" in k:
                    k_init = k.replace("qkv_8", "qkv")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "qkv_16" in k and "temporal_attn" in k:
                    k_init = k.replace("qkv_16", "qkv")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_4" in k and "temporal_attn" in k:
                    k_init = k.replace("proj_4", "proj")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_8" in k and "temporal_attn" in k:
                    k_init = k.replace("proj_8", "proj")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "proj_16" in k and "temporal_attn" in k:
                    k_init = k.replace("proj_16", "proj")
                    k_init = k_init.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "temporal_norm1" in k:
                    k_init = k.replace("temporal_norm1", "norm1")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)

            print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

            remove_list = []
            for k in state_dict.keys():
                if (
                    ("head" in k or "patch_embed" in k)
                    and k in checkpoint
                    and k in state_dict
                    and checkpoint[k].shape != state_dict[k].shape
                ):
                    remove_list.append(k)
                    del checkpoint[k]
            
            print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
            utils.load_state_dict(model, checkpoint)

        else:
            add_list = []
            for k in state_dict.keys():
                if "blocks" in k and "temporal_attn" in k:
                    k_init = k.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "temporal_norm1" in k:
                    k_init = k.replace("temporal_norm1", "norm1")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)

            print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

            remove_list = []
            for k in state_dict.keys():
                if (
                    ("head" in k or "patch_embed" in k)
                    and k in checkpoint
                    and k in state_dict
                    and checkpoint[k].shape != state_dict[k].shape
                ):
                    remove_list.append(k)
                    del checkpoint[k]
            print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
            utils.load_state_dict(model, checkpoint)

    return model
