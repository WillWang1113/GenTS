import math
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch import Tensor
import torch.nn.functional as F
from src.layers.norm import RMSNorm


from einops import repeat, rearrange

class SelfAttention(nn.Module):
    # fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        # self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def FeedForward(dim, mult=4):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.LayerNorm(dim),
        weight_norm(nn.Linear(dim, inner_dim, bias=False)),
        nn.GELU(),
        weight_norm(nn.LayerNorm(inner_dim)),
        nn.Linear(inner_dim, dim, bias=False),
    )


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


class DiffAttn(nn.Module):
    """
    Differential Attention module.

    This module computes attention weights based on the difference between two sets of queries and keys.

    Attributes:
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - W_q (nn.Linear): Linear layer for transforming queries.
    - W_k (nn.Linear): Linear layer for transforming keys.
    - W_v (nn.Linear): Linear layer for transforming values.
    """

    def __init__(self, d: int, embedding_dim: int):
        super(DiffAttn, self).__init__()
        self.d = d
        self.W_q = nn.Linear(embedding_dim, 2 * d)
        self.W_k = nn.Linear(embedding_dim, 2 * d)
        self.W_v = nn.Linear(embedding_dim, d)  # Changed to output d dimensions

    def forward(self, X: Tensor, λ: float) -> Tensor:
        """
        Forward pass of the Differential Attention module.

        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.

        Returns:
        - Tensor: Output tensor.
        """
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q1, Q2 = self.split(Q)
        K1, K2 = self.split(K)

        s = 1 / math.sqrt(self.d)

        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s

        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)

        result = (A1_softmax - λ * A2_softmax) @ V
        return result

    @staticmethod
    def split(X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Splits the input tensor into two halves along the last dimension.

        Args:
        - X (Tensor): Input tensor.

        Returns:
        - Tuple[Tensor, Tensor]: Two tensors, each containing half of the input dimensions.
        """
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention module.

    This module applies the Differential Attention mechanism multiple times in parallel.

    Attributes:
    - h (int): The number of attention heads.
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - λinit (float): The initial scaling factor for the difference.
    - diff_attn_heads (nn.ModuleList): List of Differential Attention modules.
    - W_o (nn.Linear): Linear layer for output transformation.
    - norm (nn.LayerNorm): Layer normalization module.
    """

    def __init__(self, h: int, d: int, embedding_dim: int, λinit: float = 0.05):
        super(MultiHeadDifferentialAttention, self).__init__()
        self.h = h
        self.d = d
        self.λinit = λinit
        self.embedding_dim = embedding_dim
        self.diff_attn_heads = nn.ModuleList(
            [DiffAttn(d, embedding_dim) for _ in range(h)]
        )
        self.W_o = nn.Linear(h * d, embedding_dim)  # Changed to h * d
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, X: Tensor, λ: float = 0.1) -> Tensor:
        """
        Forward pass of the Multi-Head Differential Attention module.

        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.

        Returns:
        - Tensor: Output tensor.
        """
        O_list = [head(X, λ) for head in self.diff_attn_heads]

        O_concat = torch.cat(O_list, dim=-1)

        # Apply the output transformation
        result = self.W_o(O_concat)

        # Apply LayerNorm
        result = self.norm(result)

        # Scale by λinit
        result = result * (1 - self.λinit)

        return result


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        cross_attend=False,
        # scale = 8,
    ):
        super().__init__()
        # self.scale = scale
        self.heads = heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = nn.LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = weight_norm(nn.Linear(dim, inner_dim, bias=False))
        self.to_kv = weight_norm(nn.Linear(dim, inner_dim * 2, bias=False))

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = weight_norm(nn.Linear(inner_dim, dim, bias=False))
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context=None, mask=None):
        if (exists(context) and not self.cross_attend) or (
            not exists(context) and self.cross_attend
        ):
            raise AssertionError(
                "Context and cross_attend must either both be present or both be absent."
            )

        # n = x.shape[-2]
        h = self.heads

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, "h 1 d -> b h 1 d", b=x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        out = F.scaled_dot_product_attention(
            q, k, v
        )  # scale by 1/√d is a default setting.

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        out = self.norm_out(out)
        return out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        # args,
        embed_dim: int,
        depth: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        # self.args = args
        self.embed_dim = embed_dim

        # arg num_heads set to half of Transformer's num_heads
        self.num_heads = num_heads

        # arg decoder_kv_attention_heads set to half of Transformer's num_kv_heads if use GQA
        # set to same as num_heads if use normal MHA
        self.num_kv_heads = num_heads  # args.decoder_kv_attention_heads if args.decoder_kv_attention_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = head_dim  # embed_dim // num_heads // 2
        self.scaling = self.head_dim**-0.5

        self.q_proj = weight_norm(
            nn.Linear(embed_dim, self.head_dim * num_heads * 2, bias=False)
        )
        self.k_proj = weight_norm(
            nn.Linear(
                embed_dim, self.head_dim * num_heads * 2 // self.n_rep, bias=False
            )
        )
        self.v_proj = weight_norm(
            nn.Linear(
                embed_dim, self.head_dim * num_heads * 2 // self.n_rep, bias=False
            )
        )
        self.out_proj = weight_norm(
            nn.Linear(self.head_dim * num_heads * 2, embed_dim, bias=False)
        )

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(
        self,
        x,
        # rel_pos,
        attn_mask=None,
    ):
        """
        x (Tensor)
        """
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(
            bsz, tgt_len, self.num_heads * 2 * self.head_dim
        )

        attn = self.out_proj(attn)
        return attn
