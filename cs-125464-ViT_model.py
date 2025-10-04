"""
Vision Transformer (ViT) encoder used to extract latent features for fog-visibility tasks.

Purpose:
    - Map grid-like or sequential predictors to token sequences.
    - Apply stacked Transformer encoder blocks.
    - Return a latent embedding for downstream classification (e.g., LightGBM).

Inputs:
    - Tensors prepared by the data pipeline and model hyperparameters.

Outputs:
    - A latent feature vector; classification head is external to this module.
"""
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange
import math

class Residual(nn.Module):
    """
    Apply the wrapped function and add the input (residual connection).

    Inputs:
        - A tensor and any keyword arguments for the wrapped function.

    Outputs:
        - A tensor with residual addition applied.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    """
    Layer normalization followed by the wrapped function (pre-norm style).

    Inputs:
        - A tensor to be normalized and passed to the wrapped function.

    Outputs:
        - The function output with pre-layer normalization applied.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activations and dropout.

    Inputs:
        - A token sequence or hidden representation.

    Outputs:
        - A transformed representation with the same sequence length.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention with optional output projection.

    Inputs:
        - A token sequence (batch-first) and an optional attention mask.

    Outputs:
        - The attention-refined token sequence of the same length.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5   # -0.5次方

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    """
    Stacked Transformer encoder blocks (pre-norm + residual + MSA + MLP).

    Inputs:
        - A token sequence and an optional mask.

    Outputs:
        - The transformed token sequence for pooling or further processing.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    """
    Vision Transformer encoder that returns a latent embedding.

    Inputs:
        - Tensors prepared by the data pipeline; hyperparameters include n_class,
          sampling_point, dim, depth, heads, mlp_dim, pooling mode, and dropouts.

    Outputs:
        - A latent feature vector; the classification head is intentionally
          kept external. (The commented line shows an alternative return with logits.)
    """
    def __init__(self, n_class, sampling_point, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        num_patches = 100

        self.to_patch_embedding = nn.Sequential(
             # nn.Conv2d(in_channels=35, out_channels=3, kernel_size=3, stride=1),
             Rearrange('b c h w  -> b h (c w)'),
             nn.Linear(sampling_point*35, sampling_point),
             nn.Linear(sampling_point, dim),
             nn.LayerNorm(dim)

         )
        self.pos_embedding_patch = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_patch = nn.Dropout(emb_dropout)
        self.transformer_patch = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_class)
           )

    def forward(self, img, mask=None):

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token_patch, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding_patch[:, :(n + 1)]
        x = self.dropout_patch(x)
        x = self.transformer_patch(x, mask)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x, self.mlp_head(x)
