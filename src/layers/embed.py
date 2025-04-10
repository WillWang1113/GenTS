import collections.abc
import math

# from src.models.blocks import RevIN

from typing import Callable, Optional

import torch
import torch.nn as nn
from torchvision.ops import MLP
import numpy as np
from src.layers.mlp import CustomMLP

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=np.float32).reshape(1, -1)
    # grid_w = np.arange(grid_size, dtype=np.float32)
    # grid = np.meshgrid(grid_h)  # here w goes first
    # grid = np.stack(grid, axis=0)

    # grid = grid.reshape([1, grid_size])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed(nn.Module):
    # DiT
    def __init__(
        self,
        input_size,
        patch_size,
        in_channels,
        hidden_size,
        bias=True,
        stride=None,
        norm_layer: Optional[Callable] = None,
    ):
        super(PatchEmbed, self).__init__()
        # Patching
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride
        self.num_patches = input_size // patch_size
        self.proj = nn.Conv1d(
            in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.stride,
            bias=bias,
        )
        print(self.proj)
        self.norm = norm_layer(hidden_size) if norm_layer else nn.Identity()
        # self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        # self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        x = self.proj(x.permute(0, 2, 1))
        x = self.norm(x)
        return x.permute(0, 2, 1)


class TimestepEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# class ConditionEmbed(nn.Module):
#     def __init__(
#         self,
#         seq_channels,
#         seq_length,
#         hidden_size,
#         latent_dim,
#         cond_dropout_prob=0.5,
#         norm=True,
#         num_patches=1,
#     ) -> None:
#         super().__init__()

#         self.latent_dim = latent_dim
#         self.norm = norm
#         self.num_patches = num_patches
#         # self.embedder = nn.Linear(seq_length, hidden_size)
#         if norm:
#             self.rev = RevIN(seq_channels)
#         self.input_enc = MLP(
#             in_channels=seq_length,
#             hidden_channels=[hidden_size * 2, hidden_size],
#         )
#         self.pred_dec = torch.nn.Linear(hidden_size, latent_dim * num_patches)
#         self.dropout_prob = cond_dropout_prob
#         self.seq_channels = seq_channels

#     def token_drop(self, labels, force_drop_ids=None):
#         """
#         Drops labels to enable classifier-free guidance.
#         """
#         if force_drop_ids is None:
#             drop_ids = (
#                 torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
#             )
#         else:
#             drop_ids = force_drop_ids == 1
#         self.drop_ids = drop_ids.unsqueeze(1).unsqueeze(1)
#         labels = torch.where(self.drop_ids, torch.zeros_like(labels), labels)
#         return labels

#     def forward(self, observed_data, train, force_drop_ids=None, **kwargs):
#         use_dropout = self.dropout_prob > 0

#         if (train and use_dropout) or (force_drop_ids is not None):
#             observed_data = self.token_drop(observed_data, force_drop_ids)

#         if self.norm:
#             x_norm = self.rev(observed_data, "norm")
#         else:
#             x_norm = observed_data

#         latents = self.input_enc(x_norm.permute(0, 2, 1))
#         latents = latents.flatten(1)
#         latents = self.pred_dec(latents)

#         # if self.norm:
#         # latents = self.rev(latents, "denorm")

#         return latents
#         # mean_pred = self.mean_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)
#         # std_pred = self.std_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)

#         # return {"latents": latents, "mean_pred": mean_pred, "std_pred": std_pred}


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, cond_dim, hidden_size, dropout_prob):
        super().__init__()
        # use_cfg_embedding = dropout_prob > 0
        self.embedding_table = CustomMLP(
            cond_dim, hidden_features=hidden_size, out_features=hidden_size
        )
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids.unsqueeze(1), torch.zeros_like(labels), labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        # Learnable Embedding matrix to map time steps to embeddings
        self.embedding = nn.Embedding(
            num_embeddings=max_len, embedding_dim=d_model, max_norm=math.sqrt(d_model)
        )  # (max_len, d_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a positional encoding to the tensor x.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, max_len, d_emb) to which the positional encoding should be added

        Returns:
            torch.Tensor: Tensor with an additional positional encoding
        """
        position = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, max_len)
        pe = self.embedding(position)  # (1, max_len, d_emb)
        x = x + pe
        return x


class TimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_time: int, use_time_axis: bool = True):
        super().__init__()

        # Learnable Embedding matrix to map time steps to embeddings
        self.embedding = nn.Embedding(
            num_embeddings=max_time, embedding_dim=d_model, max_norm=math.sqrt(d_model)
        )  # (max_time, d_emb)
        self.use_time_axis = use_time_axis

    def forward(
        self, x: torch.Tensor, timesteps: torch.LongTensor, use_time_axis: bool = True
    ) -> torch.Tensor:
        """Adds a time encoding to the tensor x.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, max_len, d_emb) to which the time encoding should be added
            timesteps (torch.LongTensor): Tensor of shape (batch_size,) containing the current timestep for each sample in the batch

        Returns:
            torch.Tensor: Tensor with an additional time encoding
        """
        t_emb = self.embedding(timesteps)  # (batch_size, d_model)
        if use_time_axis:
            t_emb = t_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        assert isinstance(t_emb, torch.Tensor)
        return x + t_emb


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps.
    Courtesy of https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, d_model: int, scale: float = 30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.d_model = d_model
        self.W = nn.Parameter(
            torch.randn((d_model + 1) // 2) * scale, requires_grad=False
        )

        self.dense = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, use_time_axis: bool = True
    ) -> torch.Tensor:
        time_proj = timesteps[:, None] * self.W[None, :] * 2 * torch.pi
        embeddings = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)

        # Slice to get exactly d_model
        t_emb = embeddings[:, : self.d_model]  # (batch_size, d_model)

        if use_time_axis:
            t_emb = t_emb.unsqueeze(1)

        projected_emb: torch.Tensor = self.dense(t_emb)

        return x + projected_emb




class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)

        res = x - moving_mean
        return res, moving_mean