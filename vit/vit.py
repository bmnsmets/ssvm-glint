import torch.nn as nn
import torch
from typing import Literal
from math import sqrt


class PatchTokenizer(nn.Module):
    """Patch embeddings module in the style of ViT."""

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
            image_size[1] // patch_size[0]
        )

        self.conv = nn.Conv2d(
            image_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
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

        nn.init.trunc_normal_(self.conv.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.conv.bias, 0)
        if self.initial_cls_token is not None:
            nn.init.trunc_normal_(self.initial_cls_token, mean=0.0, std=0.02)
        if self.position_embedding is not None:
            nn.init.trunc_normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        nn.init.zeros_(self.query.weight)
        nn.init.trunc_normal_(self.key.weight, mean=0.0, std=1 / sqrt(self.embed_dim))
        nn.init.trunc_normal_(self.value.weight, mean=0.0, std=1 / sqrt(self.embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        self.pre_ln = nn.LayerNorm(embed_dim)

        self.attention = QKVAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=attention_dropout,
        )

        self.post_ln = nn.LayerNorm(embed_dim)

        ff_hidden_dim = int(ff_fanout * embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.SiLU(),
            nn.Dropout(ff_dropout) if ff_dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.pre_ln(x))
        x = x + self.ff(self.post_ln(x))
        return x


class ViT(nn.Module):
    """
    The ViT model.
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
        """
        Creates the model:
        Tokenizer -> ViT layers -> Classification head.
        """
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
        x = self.tokenizer(img)  # [B,N,D]

        for layer in self.layers:
            x = layer(x)  # [B,N,D]

        if self.pool == "cls":
            return self.head(x[:, 0])  # [B,num_classes]
        elif self.pool == "mean":
            return self.head(x.mean(dim=1))  # [B,num_classes]


class ShapedAttention(nn.Module):
    """
    Shaped Attention Module

    References:
        [1] Noci et al. (2023). "The shaped transformer: Attention models in the
            infinite depth-and-width limit." NeurIPS 2023.
        [2] He & Hofmann. (2024). "Simplifying Transformer Blocks." ICLR 2024.


    Args:

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

        self.alpha = nn.Parameter(
            torch.full((num_heads, 1, 1), fill_value=(1 - 1 / sqrt(embed_dim)))
        )
        self.beta = nn.Parameter(torch.ones(num_heads, 1, 1))

        nn.init.zeros_(self.query.weight)
        nn.init.trunc_normal_(self.key.weight, mean=0.0, std=1 / sqrt(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h, d = self.num_heads, self.head_dim

        q, k, v = self.query(x), self.key(x), x  # 3x[B,N,D=h*d]
        q = q.view(B, N, h, d).permute(0, 2, 1, 3).contiguous()  # [B,h,N,d]
        k = k.view(B, N, h, d).permute(0, 2, 1, 3).contiguous()  # [B,h,N,d]
        v = v.view(B, N, h, -1).permute(0, 2, 1, 3).contiguous()  # [B,h,N,d]
        y = nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=(self.dropout if self.training else 0)
        )  # [B,h,N,d]
        out = self.alpha * v + self.beta * y  # [B,h,N,d]
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B,N,D]
        return out.contiguous()  # [B,N,D]


class SASPLayer(nn.Module):

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
    ViT model with simplified SAS-P transformer blocks from:
    He, B., & Hofmann, T. (2023). Simplifying transformer blocks. arXiv preprint arXiv:2311.01906.
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
        """
        Creates the model:
        Tokenizer -> ViT layers -> Classification head.
        """
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
        x = self.tokenizer(img)  # [B,N,D]

        for layer in self.layers:
            x = layer(x)  # [B,N,D]

        if self.pool == "cls":
            return self.head(x[:, 0])  # [B,num_classes]
        elif self.pool == "mean":
            return self.head(x.mean(dim=1))  # [B,num_classes]
