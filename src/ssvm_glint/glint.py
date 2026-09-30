import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal
from ssvm_glint.tokens import AffPlus2Tokens


class GLInTClassifier(nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        semantic_embed_dim: int,
        invariant_embed_dim: int,
        depth: int,
        head_dim: int | None = None,
        num_heads: int = 1,
        pool: Literal["cls", "mean"] = "cls",
        token_dropout: float = 0,
        attention_dropout: float = 0,
        ff_dropout: float = 0,
        ff_fanout: float = 2,
    ):
        super().__init__()
        self.pool = pool

        # Embedding of the semantic part u [B,N,in_dim] -> [B,N,D]
        self.semantic_embedder = nn.Sequential(
            nn.Linear(in_dim, semantic_embed_dim),
            nn.LayerNorm(semantic_embed_dim),
            nn.Identity() if token_dropout == 0 else nn.Dropout1d(token_dropout),
        )

        # Generating invariants
        self.invariants = VariationalAffPlus2Invariants()

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                GLInTLayer(
                    embed_dim=semantic_embed_dim,
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
        self.head = nn.Linear(semantic_embed_dim, num_classes)

    def forward(self, tokens: AffPlus2Tokens) -> Tensor:
        """Map AffPlus2Tokens to class logits [B, num_classes]"""
        tokens = tokens.replace(u=self.semantic_embedder(tokens.u))  # u: [B,N,D]

        invariants = self.invariants(tokens)

        for layer in self.layers:
            tokens = layer(tokens)

        if self.pool == "cls":
            return self.head(tokens.u[:, 0])  # [B,num_classes]
        elif self.pool == "mean":
            return self.head(tokens.u.mean(dim=1))  # [B,num_classes]
        else:
            raise ValueError(f"Unknown pooling value: {self.pool}")


class VariationalAffPlus2Invariants(nn.Module):

    def __init__(self):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.full((5,), -1.0))

    def forward(self, tokens: AffPlus2Tokens) -> Tensor:
        affi = tokens.affine_invariant()
        t, (a, R, S) = tokens.t, tokens.polar()  # mathematical invariants
