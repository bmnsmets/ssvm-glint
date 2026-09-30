import torch
import torch.nn as nn
from typing import Literal


class GLInTClassifier(nn.Module):

    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        depth: int,
        head_dim: int | None = None,
        num_heads: int = 1,
        pool: Literal["cls", "mean"] = "cls",
        token_dropout: float = 0,
        attention_dropout: float = 0,
        ff_dropout: float = 0,
        ff_fanout: float = 2,
    ):
        pass
