"""
Vision Transformer (ViT) classifiers.

Two variants share the same `PatchTokenizer` front end and classification head:
`ViT` uses standard pre-LN transformer blocks, `ViT_SASP` uses the simplified
parallel SAS-P blocks of He & Hofmann (2024).

Shape notation in comments: B batch, C image channels, H/W image height/width,
N number of tokens, D embedding dim, h number of heads, d per-head q/k dim,
dv per-head value dim.
"""

from math import sqrt
from typing import Literal

import torch
from torch import nn


class PatchTokenizer(nn.Module):
    """
    Patch embeddings module in the style of ViT.

    Splits the image into non-overlapping patches with a strided convolution,
    flattens them to a token sequence, adds position embeddings, then applies
    LayerNorm and dropout. Optionally prepends a learnable CLS token, which
    receives no position embedding and bypasses norm and dropout.

    Args:
        image_size: (H, W) of the input images;
            preferably divisible by `patch_size` (remainder pixels are cut off).
        image_channels: Number of input channels C.
        patch_size: (pH, pW) patch size, also used as the convolution stride.
        embed_dim: Token embedding dimension D.
        position_embedding: "learned" for a learnable [N, D] embedding, or None
            for no position information.
        norm: Apply LayerNorm to the patch tokens.
        cls_token: Prepend a learnable CLS token to the sequence.
        dropout: Drop probability. Uses `Dropout1d` on [B, N, D], so it zeroes
            entire patch tokens rather than individual features.

    Weights (conv, position embedding, CLS token) are initialized with a
    truncated normal (std 0.02); the conv bias is initialized to zero.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        image_channels: int,
        patch_size: tuple[int, int],
        embed_dim: int,
        position_embedding: Literal["learned"] | None = "learned",
        norm: bool = True,
        cls_token: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.image_size = image_size
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.ntokens = (image_size[0] // patch_size[0]) * (
            image_size[1] // patch_size[1]
        )

        self.conv = nn.Conv2d(
            image_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()

        if position_embedding == "learned":
            self.position_embedding = nn.Parameter(torch.zeros(self.ntokens, embed_dim))
        else:
            self.position_embedding = None

        self.initial_cls_token = (
            nn.Parameter(torch.zeros(embed_dim)) if cls_token else None
        )
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

        if self.initial_cls_token is not None:
            nn.init.trunc_normal_(self.initial_cls_token, mean=0.0, std=0.02)
        if self.position_embedding is not None:
            nn.init.trunc_normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map images [B, C, H, W] to tokens [B, N, D] (or [B, N+1, D] with CLS)."""
        x = self.conv(x)  # [B,C,H,W] -> [B,D,H//pH,W//pW]
        x = x.flatten(-2, -1).permute(0, 2, 1)  # [B,N,D]
        if self.position_embedding is not None:
            x = x + self.position_embedding  # [B,N,D]
        x = self.dropout(self.norm(x))  # [B,N,D]
        if self.initial_cls_token is not None:
            # prepend a class token for each image
            x = torch.cat(
                (self.initial_cls_token.expand(x.shape[0], 1, -1), x), dim=1
            )  # [B,N,D] -> [B,N+1,D]
        return x.contiguous()


class QKVAttention(nn.Module):
    """
    Multi-head self-attention with bias-free query, key and value projections.

    Uses `scaled_dot_product_attention` without masking. The query weights are
    initialized to zero, so attention is uniform at initialization (each token
    receives the mean of the values); key and value weights use a truncated
    normal with std 1/sqrt(D).

    Args:
        embed_dim: Token embedding dimension D.
        num_heads: Number of attention heads h.
        head_dim: Per-head query/key dimension d; defaults to D // h. The value
            dimension is always dv = D // h, independent of this setting.
        dropout: Dropout probability on the attention weights (training only).

    The output projection is only present when `num_heads > 1`; with a single
    head the attended values (width D) are returned directly.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        head_dim: int | None = None,
        dropout: float = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.dropout = dropout

        self.query = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.key = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.value = nn.Linear(
            embed_dim, num_heads * (embed_dim // num_heads), bias=False
        )
        self.out = (
            nn.Linear(num_heads * (embed_dim // num_heads), embed_dim, bias=False)
            if num_heads > 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over tokens [B, N, D] -> [B, N, D]."""
        B, N, _ = x.shape
        h, d = self.num_heads, self.head_dim
        q, k, v = self.query(x), self.key(x), self.value(x)  # 2x[B,N,h*d],[B,N,h*dv]
        q = q.view(B, N, h, d).permute(0, 2, 1, 3).contiguous()  # [B,h,N,d]
        k = k.view(B, N, h, d).permute(0, 2, 1, 3).contiguous()  # [B,h,N,d]
        v = v.view(B, N, h, -1).permute(0, 2, 1, 3).contiguous()  # [B,h,N,dv]
        y = nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=(self.dropout if self.training else 0)
        )  # [B,h,N,dv]
        y = y.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B,N,h*dv]
        return self.out(y.contiguous())  # [B,N,D]


class TransformerLayer(nn.Module):
    """
    Standard pre-LN transformer block:

        x = x + Attention(LN(x))
        x = x + FF(LN(x))

    where FF is Linear -> SiLU -> Dropout -> Linear with hidden width
    `ff_fanout * embed_dim`.

    Args:
        embed_dim: Token embedding dimension D.
        head_dim: Per-head query/key dimension; see `QKVAttention`.
        num_heads: Number of attention heads.
        ff_fanout: Ratio of the feed-forward hidden width to D.
        ff_dropout: Dropout probability inside the feed-forward network.
        attention_dropout: Dropout probability on the attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        head_dim: int | None = None,
        num_heads: int = 1,
        ff_fanout: float = 2,
        ff_dropout: float = 0,
        attention_dropout: float = 0,
    ):
        super().__init__()

        self.pre_attn_ln = nn.LayerNorm(embed_dim)

        self.attention = QKVAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=attention_dropout,
        )

        self.pre_ff_ln = nn.LayerNorm(embed_dim)

        ff_hidden_dim = int(ff_fanout * embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.SiLU(),
            nn.Dropout(ff_dropout) if ff_dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.pre_attn_ln(x))
        x = x + self.ff(self.pre_ff_ln(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer classifier:
    PatchTokenizer -> `depth` x TransformerLayer -> linear classification head.

    Args:
        image_size: (H, W) of the input images.
        image_channels: Number of input channels C.
        patch_size: (pH, pW) patch size.
        num_classes: Number of output logits.
        embed_dim: Token embedding dimension D.
        depth: Number of transformer layers.
        head_dim: Per-head query/key dimension; defaults to D // num_heads.
        num_heads: Number of attention heads.
        pool: "cls" classifies from a prepended CLS token, "mean" from the
            average over all tokens.
        patch_dropout: Probability of dropping whole patch tokens.
        attention_dropout: Dropout probability on the attention weights.
        ff_dropout: Dropout probability inside the feed-forward networks.
        ff_fanout: Ratio of the feed-forward hidden width to D.
        position_embedding: "learned" or None; see `PatchTokenizer`.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        image_channels: int,
        patch_size: tuple[int, int],
        num_classes: int,
        embed_dim: int,
        depth: int,
        head_dim: int | None = None,
        num_heads: int = 1,
        pool: Literal["cls", "mean"] = "cls",
        patch_dropout: float = 0,
        attention_dropout: float = 0,
        ff_dropout: float = 0,
        ff_fanout: float = 2,
        position_embedding: Literal["learned"] | None = "learned",
    ):
        super().__init__()
        self.pool = pool

        # Get the Patch embedding
        self.tokenizer = PatchTokenizer(
            image_size=image_size,
            image_channels=image_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            cls_token=(pool == "cls"),
            position_embedding=position_embedding,
            dropout=patch_dropout,
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ff_fanout=ff_fanout,
                    ff_dropout=ff_dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        # Final fully connected linear layer that maps to the classes
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, img: torch.Tensor):
        """Map images [B, C, H, W] to class logits [B, num_classes]."""
        x = self.tokenizer(img)  # [B,N,D]

        for layer in self.layers:
            x = layer(x)  # [B,N,D]

        if self.pool == "cls":
            return self.head(x[:, 0])  # [B,num_classes]
        elif self.pool == "mean":
            return self.head(x.mean(dim=1))  # [B,num_classes]
        else:
            raise ValueError(f"Unknown pooling value: {self.pool}")


class ShapedAttention(nn.Module):
    """
    Shaped Attention Module

    Computes, per head,

        out = alpha * V + beta * softmax(Q K^T / sqrt(d)) V - gamma * C V

    with V = x, i.e. there are no value or output projections (as in SAS-P [2]).
    C is the uniform attention matrix, so C V is the token mean of V. Query
    weights are initialized to zero, so softmax(Q K^T) = C at initialization and
    the block starts as alpha * x. `alpha`, `beta` and `gamma` are learnable
    per-head scalars initialized to 1 - 1/sqrt(D), 1 and 1.

    References:
        [1] Noci et al. (2023). "The shaped transformer: Attention models in the
            infinite depth-and-width limit." NeurIPS 2023.
        [2] He & Hofmann. (2024). "Simplifying Transformer Blocks." ICLR 2024.

    Args:
        embed_dim: Token embedding dimension D; must be divisible by `num_heads`.
        num_heads: Number of attention heads h; each head has dimension D // h.
        dropout: Dropout probability on the attention weights (training only).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.query = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.key = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)

        self.alpha = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.beta = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.gamma = nn.Parameter(torch.ones(num_heads, 1, 1))

        nn.init.zeros_(self.query.weight)
        nn.init.trunc_normal_(self.key.weight, mean=0.0, std=1 / sqrt(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over tokens [B, N, D] -> [B, N, D]."""
        B, N, _ = x.shape
        h, d = self.num_heads, self.head_dim

        q, k, v = self.query(x), self.key(x), x  # 3x[B,N,D=h*d]
        q = q.view(B, N, h, d).permute(0, 2, 1, 3).contiguous()  # [B,h,N,d]
        k = k.view(B, N, h, d).permute(0, 2, 1, 3).contiguous()  # [B,h,N,d]
        v = v.view(B, N, h, -1).permute(0, 2, 1, 3).contiguous()  # [B,h,N,d]
        y = nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=(self.dropout if self.training else 0)
        )  # [B,h,N,d]
        c = v.mean(dim=-2, keepdim=True)  # [B,h,1,d], uniform attention = C V
        out = self.alpha * v + self.beta * y - self.gamma * c  # [B,h,N,d]
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B,N,D]
        return out.contiguous()  # [B,N,D]


class SASPLayer(nn.Module):
    """
    Parallel SAS-P transformer block of He & Hofmann (2024), without skip
    connection:

        x = LN(x)
        x = beta_sa * ShapedAttention(x) + beta_ff * FF(x)

    The identity path is carried by the `alpha * V` term of `ShapedAttention`.
    FF is Linear -> SiLU -> Dropout -> Linear with hidden width
    `ff_fanout * embed_dim`. `beta_sa` and `beta_ff` are learnable scalars
    initialized to 1 and 0.1.

    Args:
        embed_dim: Token embedding dimension D.
        num_heads: Number of attention heads.
        ff_fanout: Ratio of the feed-forward hidden width to D.
        ff_dropout: Dropout probability inside the feed-forward network.
        attention_dropout: Dropout probability on the attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        ff_fanout: float = 2,
        ff_dropout: float = 0,
        attention_dropout: float = 0,
    ):
        super().__init__()

        self.pre_ln = nn.LayerNorm(embed_dim)

        self.attention = ShapedAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=attention_dropout
        )

        ff_hidden_dim = int(ff_fanout * embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.SiLU(),
            nn.Dropout(ff_dropout) if ff_dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )

        self.beta_sa = nn.Parameter(torch.ones(1))
        self.beta_ff = nn.Parameter(torch.full((1,), fill_value=0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_ln(x)
        return self.beta_sa * self.attention(x) + self.beta_ff * self.ff(x)


class ViT_SASP(nn.Module):
    """
    Vision Transformer classifier with simplified SAS-P blocks:
    PatchTokenizer -> `depth` x SASPLayer -> linear classification head.

    References:
        He & Hofmann. (2024). "Simplifying Transformer Blocks." ICLR 2024.

    Args:
        image_size: (H, W) of the input images.
        image_channels: Number of input channels C.
        patch_size: (pH, pW) patch size.
        num_classes: Number of output logits.
        embed_dim: Token embedding dimension D; must be divisible by `num_heads`.
        depth: Number of SAS-P layers.
        num_heads: Number of attention heads; head dimension is D // num_heads.
        pool: "cls" classifies from a prepended CLS token, "mean" from the
            average over all tokens.
        patch_dropout: Probability of dropping whole patch tokens.
        attention_dropout: Dropout probability on the attention weights.
        ff_dropout: Dropout probability inside the feed-forward networks.
        ff_fanout: Ratio of the feed-forward hidden width to D.
        position_embedding: "learned" or None; see `PatchTokenizer`.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        image_channels: int,
        patch_size: tuple[int, int],
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int = 1,
        pool: Literal["cls", "mean"] = "cls",
        patch_dropout: float = 0,
        attention_dropout: float = 0,
        ff_dropout: float = 0,
        ff_fanout: float = 2,
        position_embedding: Literal["learned"] | None = "learned",
    ):
        super().__init__()
        self.pool = pool

        # Get the Patch embedding
        self.tokenizer = PatchTokenizer(
            image_size=image_size,
            image_channels=image_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            cls_token=(pool == "cls"),
            position_embedding=position_embedding,
            dropout=patch_dropout,
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                SASPLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_fanout=ff_fanout,
                    ff_dropout=ff_dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        # Final fully connected linear layer that maps to the classes
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, img: torch.Tensor):
        """Map images [B, C, H, W] to class logits [B, num_classes]."""
        x = self.tokenizer(img)  # [B,N,D]

        for layer in self.layers:
            x = layer(x)  # [B,N,D]

        if self.pool == "cls":
            return self.head(x[:, 0])  # [B,num_classes]
        elif self.pool == "mean":
            return self.head(x.mean(dim=1))  # [B,num_classes]
        else:
            raise ValueError(f"Unknown pooling value: {self.pool}")
