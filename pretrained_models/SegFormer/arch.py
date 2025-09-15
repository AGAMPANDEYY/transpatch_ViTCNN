# -*- coding: utf-8 -*-
# Minimal SegFormer (MiT backbone + SegFormer head) for local-weight loading
# Matches mmseg naming so NVLabs checkpoints load with strict=False (or True if perfect).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# small utils
# ----------------------
def _to_2tuple(x):
    return (x, x) if not isinstance(x, (list, tuple)) else x

class DropPath(nn.Module):
    """Stochastic depth per-sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * rnd

# ----------------------
# MLP with DWConv (as in SegFormer)
# ----------------------
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ----------------------
# Attention with spatial reduction (SR)
# ----------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        if self.sr_ratio > 1:
            x_ = x.permute(0,2,1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0,2,1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ----------------------
# Transformer Block
# ----------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

# ----------------------
# OverlapPatchEmbed
# ----------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, stride):
        super().__init__()
        patch = _to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch, stride=stride,
                              padding=(patch[0]//2, patch[1]//2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)  # B, HW, C
        x = self.norm(x)
        return x, H, W

# ----------------------
# MixVisionTransformer backbone (MiT)
# ----------------------
class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=(32,64,160,256), depths=(2,2,2,2),
                 num_heads=(1,2,5,8), sr_ratios=(8,4,2,1), mlp_ratio=4.0, drop_rate=0.0,
                 drop_path_rate=0.1):
        super().__init__()
        self.depths = depths
        # stem / stages
        self.patch_embed1 = OverlapPatchEmbed(in_chans,       embed_dims[0], patch_size=7, stride=4)
        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0],  embed_dims[1], patch_size=3, stride=2)
        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1],  embed_dims[2], patch_size=3, stride=2)
        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2],  embed_dims[3], patch_size=3, stride=2)

        # stochastic depth decay
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        cur = 0
        self.block1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratio=mlp_ratio, drop=drop_rate,
                  attn_drop=drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratio=mlp_ratio, drop=drop_rate,
                  attn_drop=drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratio=mlp_ratio, drop=drop_rate,
                  attn_drop=drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratio=mlp_ratio, drop=drop_rate,
                  attn_drop=drop_rate, drop_path=dpr[cur+i], sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        B = x.size(0)

        x1, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.transpose(1,2).reshape(B, -1, H1, W1)   # 1/4

        x2, H2, W2 = self.patch_embed2(x1)
        for blk in self.block2:
            x2 = blk(x2, H2, W2)
        x2 = self.norm2(x2)
        x2 = x2.transpose(1,2).reshape(B, -1, H2, W2)   # 1/8

        x3, H3, W3 = self.patch_embed3(x2)
        for blk in self.block3:
            x3 = blk(x3, H3, W3)
        x3 = self.norm3(x3)
        x3 = x3.transpose(1,2).reshape(B, -1, H3, W3)   # 1/16

        x4, H4, W4 = self.patch_embed4(x3)
        for blk in self.block4:
            x4 = blk(x4, H4, W4)
        x4 = self.norm4(x4)
        x4 = x4.transpose(1,2).reshape(B, -1, H4, W4)   # 1/32

        return x1, x2, x3, x4

# ----------------------
# SegFormer Head (mmseg-like)
# ----------------------
class MLP_Linear(nn.Module):
    """flatten -> Linear -> embed_dim; mmseg names this 'MLP' but uses nn.Linear."""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):   # x: [B,C,H,W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)     # [B, HW, C]
        x = self.proj(x)                    # [B, HW, E]
        x = x.transpose(1,2).reshape(B, -1, H, W)  # [B, E, H, W]
        return x

class SegFormerHead(nn.Module):
    """
    Heads from SegFormer: 4 MLPs -> upsample to 1/4 -> concat -> fuse -> conv_seg
    Module names mimic mmseg:
      - linear_c1.proj, linear_c2.proj, ...
      - linear_fuse (Conv2d)
      - conv_seg (1x1 classifier)
    """
    def __init__(self, in_channels=(32,64,160,256), channels=128, num_classes=19):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        embed = channels

        self.linear_c1 = MLP_Linear(c1, embed)  # Direct MLP_Linear, no nn.Sequential
        self.linear_c2 = MLP_Linear(c2, embed)
        self.linear_c3 = MLP_Linear(c3, embed)
        self.linear_c4 = MLP_Linear(c4, embed)

        self.linear_fuse = nn.Conv2d(embed*4, embed, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embed)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(embed, num_classes, kernel_size=1)  # Rename conv_seg to linear_pred

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        n, _, H4, W4 = c1.shape

        _c1 = self.linear_c1(c1)
        _c2 = F.interpolate(self.linear_c2(c2), size=(H4, W4), mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=(H4, W4), mode='bilinear', align_corners=False)
        _c4 = F.interpolate(self.linear_c4(c4), size=(H4, W4), mode='bilinear', align_corners=False)

        x = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        x = self.linear_fuse(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear_pred(x)  # Use linear_pred instead of conv_seg
        return x

# ----------------------
# Full model (names: backbone.*, decode_head.*)
# ----------------------
class SegFormer_B(nn.Module):
    def __init__(self, variant='b0', num_classes=19, decoder_channels=128):
        super().__init__()
        if variant == 'b0':
            embed_dims  = (32, 64, 160, 256)
            depths      = (2, 2, 2, 2)
            num_heads   = (1, 2, 5, 8)
            sr_ratios   = (8, 4, 2, 1)
        elif variant == 'b1':
            embed_dims  = (64, 128, 320, 512)
            depths      = (2, 2, 2, 2)
            num_heads   = (1, 2, 5, 8)
            sr_ratios   = (8, 4, 2, 1)
        elif variant == 'b2':
            embed_dims  = (64, 128, 320, 512)
            depths      = (3, 4, 6, 3)
            num_heads   = (1, 2, 5, 8)
            sr_ratios   = (8, 4, 2, 1)
        elif variant == 'b3':
            embed_dims  = (64, 128, 320, 512)
            depths      = (3, 4, 18, 3)
            num_heads   = (1, 2, 5, 8)
            sr_ratios   = (8, 4, 2, 1)
        elif variant == 'b4':
            embed_dims  = (64, 128, 320, 512)
            depths      = (3, 8, 27, 3)
            num_heads   = (1, 2, 5, 8)
            sr_ratios   = (8, 4, 2, 1)
        elif variant == 'b5':
            embed_dims  = (64, 128, 320, 512)
            depths      = (3, 6, 40, 3)
            num_heads   = (1, 2, 5, 8)
            sr_ratios   = (8, 4, 2, 1)
        else:
            raise ValueError(f"Unknown SegFormer variant: {variant}")

        self.backbone = MixVisionTransformer(
            in_chans=3, embed_dims=embed_dims, depths=depths,
            num_heads=num_heads, sr_ratios=sr_ratios, mlp_ratio=4.0,
            drop_rate=0.0, drop_path_rate=0.1
        )
        self.decode_head = SegFormerHead(in_channels=embed_dims, channels=decoder_channels, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits_1_4 = self.decode_head(feats)     # B, num_classes, H/4, W/4
        return logits_1_4
