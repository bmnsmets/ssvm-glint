import math
from math import exp, log, sqrt
from typing import Literal

import torch
from torch import Tensor, nn

from ssvm_glint.tokens import AffPlus2Tokens


class GLInTClassifier(nn.Module):
    """
    GLInT classifier on AffPlus2Tokens:

    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int = 1,
        token_dropout: float = 0,
        ff_fanout: float = 2,
        ff_dropout: float = 0,
        attention_dropout: float = 0,
        pool: Literal["cls", "mean"] = "cls",
        sigma_big: float = 0.0,
        sigma_loss_weight: float = 0.0,
        invariants_embedder_fanout: float = 2.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.pool = pool
        self.sigma_loss_weight = sigma_loss_weight

        # Embedding of the semantic part u [B,N,in_dim] -> [B,N,D]
        self.semantic_embedder = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Embedding of of pair-wise geometric invariants InvEmb(g_i,g_j)
        self.invariants = VariationalAffPlus2Invariants(sigma_big=sigma_big)
        iemb_hidden_dim = int(invariants_embedder_fanout * embed_dim)
        self.invariants_embedder = nn.Sequential(
            nn.Linear(self.invariants.num_invariants, iemb_hidden_dim),
            nn.SiLU(),
            nn.Linear(iemb_hidden_dim, iemb_hidden_dim),
            nn.SiLU(),
            nn.Linear(iemb_hidden_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
        )

        self.token_dropout = token_dropout

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                StaticGLInTLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_fanout=ff_fanout,
                    ff_dropout=ff_dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        # Semantic-only class token
        self.cls_token = nn.Parameter(torch.zeros(embed_dim)) if pool == "cls" else None
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02)

        # Final fully connected linear layer that maps to the classes
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: AffPlus2Tokens) -> Tensor:
        """Map AffPlus2Tokens to class logits [B, num_classes]"""
        # Token dropout: remove the same number of random tokens from every sample
        if self.training and self.token_dropout > 0:
            B, N = tokens.u.shape[:2]
            keep = torch.rand(B, N, device=tokens.u.device).argsort(dim=1)
            keep = keep[:, : N - int(self.token_dropout * N)]  # [B,N']
            b = torch.arange(B, device=keep.device)[:, None]  # [B,1]
            tokens = AffPlus2Tokens(
                g=tokens.g[b, keep], u=tokens.u[b, keep]
            )  # g: [B,N',2,3], u: [B,N',in_dim]

        tokens = tokens.replace(u=self.semantic_embedder(tokens.u))  # u: [B,N,D]

        inv = self.invariants(tokens)  # [B,N,N,14]
        inv_emb = self.invariants_embedder(inv)  # [B,N,N,D]
        inv_emb = inv_emb.mean(dim=-2)  # [B,N,D]

        if self.pool == "cls":
            assert self.cls_token is not None
            cls = self.cls_token.expand(tokens.u.shape[0], 1, -1)  # [B,1,D]
            for layer in self.layers:
                tokens, cls = layer(tokens, inv_emb, cls)
            return self.head(cls[:, 0])  # [B,num_classes]
        elif self.pool == "mean":
            for layer in self.layers:
                tokens = layer(tokens, inv_emb)
            return self.head(tokens.u.mean(dim=1))  # [B,num_classes]
        else:
            raise ValueError(f"Unknown pooling value: {self.pool}")

    def aux_loss(self) -> Tensor:
        """Regularization loss to add to the task loss during training."""
        return self.sigma_loss_weight * self.invariants.sigma_loss()

    def log_sigmas(self) -> dict[str, float]:
        """Current log noise level of each invariant feature group, keyed `log_sigma/<group>` for logging."""
        values = self.invariants.log_sigmas.tolist()
        return {
            f"log_sigma_{k}": v for k, v in zip(self.invariants.sigma_names, values)
        }


class VariationalAffPlus2Invariants(nn.Module):
    def __init__(self, sigma_big: float = 0.0, sigma_init: float = exp(-5.0)):
        super().__init__()
        self.log_sigmas = nn.Parameter(
            torch.full((5,), log(sigma_init) if sigma_big > 0 else -math.inf)
        )
        self.num_invariants = 14
        self.sigma_names = (
            "aff",
            "scale",
            "rot",
            "shear",
            "trans",
        )  # feature group of each sigma
        self.sigma_big = sigma_big

    def sigma_loss(self) -> Tensor:
        """Mean over the 5 feature groups of |sigma - sigma_big|."""
        return (self.log_sigmas.exp() - self.sigma_big).abs().mean()

    def forward(self, tokens: AffPlus2Tokens) -> Tensor:
        """Features [B,N,N,14] of tokens with g [B,N,2,3], computed in the dtype of g regardless of autocast."""
        N = tokens.ntokens
        with torch.autocast(tokens.g.device.type, enabled=False):
            affi = tokens.relative().flatten(-2, -1)  # [B,N,N,6]
            t, (a, R, S) = tokens.t, tokens.polar()
            a = a[..., None].log()  # [B,N,1]
            theta = torch.atan2(R[..., 1, 0], R[..., 0, 0])[..., None]  # [B,N,1]
            S00, S01, S11 = (
                S[..., 0, 0:1].log(),
                S[..., 0, 1:],
                S[..., 1, 1:].log(),
            )  # 3x[B,N,1]

            if self.sigma_big > 0 and self.training:
                sigmas = self.log_sigmas.exp()
                affi = affi + sigmas[0] * torch.randn_like(affi)  # [B,N,N,6]
                a = a + sigmas[1] * torch.randn_like(a)  # [B,N,1]
                theta = theta + torch.pi * sigmas[2] * torch.randn_like(
                    theta
                )  # [B,N,1]
                S00 = S00 + sigmas[3] * torch.randn_like(S00)  # [B,N,1]
                S01 = S01 + sigmas[3] * torch.randn_like(S01)  # [B,N,1]
                S11 = S11 + sigmas[3] * torch.randn_like(S11)  # [B,N,1]
                t = t + sigmas[4] * torch.randn_like(t)  # [B,N,2]

            per_token = (
                torch.cat((a, theta.cos(), theta.sin(), S00, S01, S11, t), dim=-1)
                .unsqueeze(-2)
                .expand(-1, -1, N, -1)
            )  # [B,N,8] -> [B,N,N,8]

            return torch.cat((affi, per_token), dim=-1)  # [B,N,N,14]


class StaticGLInTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        ff_fanout: float = 2,
        ff_dropout: float = 0,
        attention_dropout: float = 0,
    ):
        super().__init__()

        self.pre_attn_ln = nn.LayerNorm(embed_dim)

        self.attention = InvariantMixingAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )

        self.pre_ff_ln = nn.LayerNorm(embed_dim)

        ff_hidden_dim = int(ff_fanout * embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(ff_dropout) if ff_dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden_dim, embed_dim, bias=False),
        )

    def forward(
        self,
        tokens: AffPlus2Tokens,  # g: [B,N,2,3] u: [B,N,D]
        inv_emb: Tensor,  # [B,N,D]
        cls: Tensor | None = None,  # [B,1,D]
    ) -> AffPlus2Tokens | tuple[AffPlus2Tokens, Tensor]:

        u = tokens.u  # [B,N,D]

        if cls is None:
            u = u + self.attention(self.pre_attn_ln(u), inv_emb)
            u = u + self.ff(self.pre_ff_ln(u))
            return tokens.replace(u=u)
        else:
            y, y_cls = self.attention(self.pre_attn_ln(u), inv_emb, cls)
            u, cls = u + y, cls + y_cls  # [B,N,D], [B,1,D]
            u, cls = u + self.ff(self.pre_ff_ln(u)), cls + self.ff(self.pre_ff_ln(cls))
            return tokens.replace(u=u), cls


class InvariantMixingAttention(nn.Module):
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

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.zeros_(self.query.weight)
        nn.init.trunc_normal_(self.key.weight, mean=0.0, std=1 / sqrt(embed_dim))
        nn.init.trunc_normal_(self.value.weight, mean=0.0, std=1 / sqrt(embed_dim))

        self.alpha = nn.Parameter(torch.full((num_heads,), 0.0))  # [h]
        self.beta = nn.Parameter(torch.full((num_heads,), 1.0))  # [h]

    def forward(
        self, u: Tensor, inv_emb: Tensor, cls: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:

        B = u.shape[0]
        h, d = self.num_heads, self.head_dim

        alpha = self.alpha.repeat_interleave(d)
        beta = self.beta.repeat_interleave(d)

        x = alpha * inv_emb + beta * u  # [B,N,h*d]
        x = x if cls is None else torch.cat((cls, x), dim=1)  # [B,N+1,D]
        M = x.shape[1]  # N or N+1

        q, k, v = self.query(x), self.key(x), self.value(x)  # 3x[B,M,h*d]

        _heads = lambda z: z.view(B, M, h, d).permute(0, 2, 1, 3).contiguous()

        q, k, v = map(_heads, (q, k, v))  # 3x[B,h,N,d]
        y = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(self.dropout if self.training else 0),
        )  # [B,h,M,d]

        y = y.permute(0, 2, 1, 3).reshape(B, M, h * d).contiguous()  # [B,M,D]
        if cls is None:
            return y  # [B,N,D]
        return y[:, 1:], y[:, :1]  # [B,N,D], [B,1,D]
