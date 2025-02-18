import torch
import torch.nn as nn
import numpy as np
import math
# from ..layers.MLP import Mlp
# from ..layers.SelfAttention_Family import Attention
from .embed import PatchEmbed, RevIN, ConditionEmbed, TimestepEmbed, CustomMLP
from torchvision.ops import MLP

# from src.models.blocks import RevIN

from functools import partial

from itertools import repeat
import collections.abc



    
class Attention(nn.Module):
    # fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm, **kwargs
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        # else:
        #     q = q * self.scale
        #     attn = q @ k.transpose(-2, -1)
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)
        #     x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)




#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = CustomMLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            **block_kwargs
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        seq_len=96,
        patch_size=16,
        seq_dim=1,
        latent_dim=1152,
        n_layers=4,
        num_heads=16,
        mlp_ratio=4.0,
        # cond_dropout_prob=0.5,
        # cond_seq_len=None,
        # cond_seq_dim=None,
        # norm=True,
        # freq_denoise=False,
        # learn_sigma=False,
        **kwargs,
    ):
        super().__init__()
        # self.freq_denoise = freq_denoise
        # self.learn_sigma = learn_sigma
        self.in_channels = seq_dim
        self.out_channels = seq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.seq_length = seq_len
        # if kwargs.get("freq", False):
        #     self.freq = kwargs.get("freq", False)
        #     in_seq_length = seq_len // 2 + 1
        #     in_channels = seq_dim * 2
        #     self.out_channels = in_channels
        # else:
        in_seq_length = seq_len
        in_channels = seq_dim
        self.in_channels = in_channels
        # print(in_seq_length)
        # print(patch_size)
        # print(seq_dim)
        # print(latent_dim)
        self.x_embedder = PatchEmbed(
            in_seq_length, patch_size, seq_dim, latent_dim, bias=True
        )
        self.t_embedder = TimestepEmbed(latent_dim)

        # self.cond = False
        # if (cond_seq_dim is not None) and (cond_seq_len is not None):
        #     self.cond = True
        #     self.cond_embed = ConditionEmbed(
        #         seq_channels=cond_seq_dim,
        #         seq_length=cond_seq_len,
        #         hidden_size=latent_dim * 2,
        #         latent_dim=latent_dim,
        #         norm=norm,
        #         cond_dropout_prob=cond_dropout_prob,
        #     )

        #     self.mean_dec = torch.nn.Linear(self.seq_length, 1)
        #     self.std_dec = torch.nn.Linear(self.seq_length, 1)
        #     self.pred_dec = torch.nn.Linear(latent_dim, self.seq_length * in_channels)

        # self.y_embedder = nn.Identity()
        # self.y_embedder = LabelEmbedder(cond_dim, d_model, cond_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, latent_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    latent_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=kwargs.get("dropout", 0),
                    proj_drop=kwargs.get("dropout", 0),
                    drop=kwargs.get("dropout", 0),
                )
                for _ in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(latent_dim, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # # Initialize label embedding table:
        # if self.cond:
        #     nn.init.normal_(self.cond_embed.input_enc[0].weight, std=0.02)
        #     nn.init.normal_(self.cond_embed.input_enc[-2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size * C)
        seq: (N, seq_len, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size
        num_patch = x.shape[1]
        # h = w = int(x.shape[1] ** 0.5)
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], num_patch, p, c))
        x = torch.einsum("nhpc->nchp", x)
        x = x.reshape(shape=(x.shape[0], c, num_patch * p))
        return x.permute(0, 2, 1)

    def forward(self, x, t, condition=None, train=True, force_drop_ids=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # input_freq = False
        # if torch.is_complex(x):
        #     input_freq = True
        #     fft_len = x.shape[1]
        #     x = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)
        # elif self.freq_denoise:
        #     x = torch.fft.rfft(x, dim=1, norm="ortho")
        #     fft_len = x.shape[1]
        #     x = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)

        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = (seq_len / patch_size)
        t = self.t_embedder(t)  # (N, D)
        c = t
        # TODO: condition
        # if condition is not None:
        #     # condition = self.cond_embed(condition, train, force_drop_ids)  # (N, D)
        #     # y_pred = self.pred_dec(condition).reshape(-1, self.seq_length, self.in_channels)
            
        #     # x_denorm = self.cond_embed.rev(y_pred, "denorm") if self.cond_embed.norm else y_pred
        #     # x_mean = self.mean_dec(x_denorm.permute(0, 2, 1)).permute(0, 2, 1)
        #     # x_std = self.std_dec(x_denorm.permute(0, 2, 1)).permute(0, 2, 1)
            
        #     c = t + self.cond_embed(condition, train, force_drop_ids)  # (N, D)
        # else:
            # x_mean, x_std = (
            #     torch.zeros_like(x)[:, [0], :],
            #     torch.ones_like(x)[:, [0], :],
            # )
            # y_pred = 0
            # c = t

        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size * out_channels)
        x = self.unpatchify(x)  # (N, seq_len, out_channels)

        # if condition is not None:
        #     x_denorm = self.cond_embed.rev(x, "denorm") if self.cond_embed.norm else x
        #     x_mean = self.mean_dec(x_denorm.permute(0, 2, 1)).permute(0, 2, 1)
        #     x_std = self.std_dec(x_denorm.permute(0, 2, 1)).permute(0, 2, 1)
        # else:
        #     x_mean, x_std = (
        #         torch.zeros_like(x)[:, [0], :],
        #         torch.ones_like(x)[:, [0], :],
        #     )

        # if self.freq_denoise or input_freq:
        #     x_re = x[:, :fft_len, :]
        #     x_im = torch.concat(
        #         [
        #             torch.zeros_like(x[:, [0], :]),
        #             x[:, fft_len:, :],
        #             torch.zeros_like(x[:, [0], :]),
        #         ],
        #         dim=1,
        #     )
        #     x = torch.stack([x_re, x_im], dim=-1)
        #     x = torch.view_as_complex(x)
        # if self.freq_denoise:
        #     x = torch.fft.irfft(x, dim=1, norm="ortho")

        # print('outshape')
        # print(x.shape)
        # print(x_mean.shape)
        # print(x_std.shape)
        # x = x + y_pred

        return x
        # return x, x_mean, x_std

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


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


# #################################################################################
# #                                   DiT Configs                                  #
# #################################################################################


# def DiT_XL_2(**kwargs):
#     return DiT(n_layers=28, d_model=1152, patch_size=2, num_heads=16, **kwargs)


# def DiT_XL_4(**kwargs):
#     return DiT(n_layers=28, d_model=1152, patch_size=4, num_heads=16, **kwargs)


# def DiT_XL_8(**kwargs):
#     return DiT(n_layers=28, d_model=1152, patch_size=8, num_heads=16, **kwargs)


# def DiT_L_2(**kwargs):
#     return DiT(n_layers=24, d_model=1024, patch_size=2, num_heads=16, **kwargs)


# def DiT_L_4(**kwargs):
#     return DiT(n_layers=24, d_model=1024, patch_size=4, num_heads=16, **kwargs)


# def DiT_L_8(**kwargs):
#     return DiT(n_layers=24, d_model=1024, patch_size=8, num_heads=16, **kwargs)


# def DiT_B_2(**kwargs):
#     return DiT(n_layers=12, d_model=768, patch_size=2, num_heads=12, **kwargs)


# def DiT_B_4(**kwargs):
#     return DiT(n_layers=12, d_model=768, patch_size=4, num_heads=12, **kwargs)


# def DiT_B_8(**kwargs):
#     return DiT(n_layers=12, d_model=768, patch_size=8, num_heads=12, **kwargs)


# def DiT_S_2(**kwargs):
#     return DiT(n_layers=12, d_model=384, patch_size=2, num_heads=6, **kwargs)


# def DiT_S_4(**kwargs):
#     return DiT(n_layers=12, d_model=384, patch_size=4, num_heads=6, **kwargs)


# def DiT_S_8(**kwargs):
#     return DiT(n_layers=12, d_model=384, patch_size=8, num_heads=6, **kwargs)

