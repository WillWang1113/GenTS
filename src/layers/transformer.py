from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch.nn.utils import weight_norm

from src.layers.attention import Attention, FeedForward, SelfAttention
from src.layers.conv import Upscale
from src.layers.embed import PatchEmbed, TimestepEmbed, get_1d_sincos_pos_embed
from src.layers.mlp import CustomMLP, FinalLayer, modulate

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
        self.attn = SelfAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = CustomMLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            **block_kwargs,
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


###############################################################################################
#                                 BidirectionalTransformer  Model                             #
###############################################################################################



class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for dep_i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                # MultiHeadDifferentialAttention(h=heads, d=dim_head, embedding_dim=dim),
                # MultiheadDiffAttn(dim, dep_i+1, heads, dim_head),
                FeedForward(dim = dim, mult = ff_mult),
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x
    


class BidirectionalTransformer(nn.Module):
    def __init__(self,
                 kind: str,
                 num_tokens: int,
                 codebook_sizes: dict,
                 embed_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 heads: int,
                 ff_mult: int,
                 use_rmsnorm: bool,
                 p_unconditional: float,
                 n_classes: int,
                 model_dropout:float=0.3,
                 emb_dropout:float=0.3,
                 **kwargs):
        """
        :param kind:
        :param num_tokens:
        :param codebook_sizes:
        :param embed_dim:
        :param hidden_dim:
        :param n_layers:
        :param heads:
        :param ff_mult:
        :param use_rmsnorm:
        :param p_unconditional:
        :param n_classes:
        :param num_tokens_l:
        :param kwargs:
        """
        super().__init__()
        kind = kind.lower()
        assert kind in ['lf', 'hf'], 'invalid `kind`.'
        self.kind = kind
        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim if kind == 'lf' else 2 * embed_dim
        # out_dim = embed_dim
        self.emb_dropout = emb_dropout
        self.mask_token_ind = {'lf':codebook_sizes, 'hf':codebook_sizes}

        # token embeddings
        self.tok_emb_l = nn.Embedding(codebook_sizes + 1, embed_dim)  # `+1` is for mask-token
        if kind == 'hf':
            self.tok_emb_h = nn.Embedding(codebook_sizes + 1, embed_dim)  # `+1` is for mask-token

        # transformer
        self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)  # `+1` is for no-condition
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = TransformerBlocks(dim=hidden_dim, depth=n_layers, dim_head=64, heads=heads, ff_mult=ff_mult)
        codebook_size = codebook_sizes if kind == 'lf' else codebook_sizes
        self.pred_head = nn.Sequential(*[
            weight_norm(nn.Linear(in_features=hidden_dim, out_features=hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            weight_norm(nn.Linear(in_features=hidden_dim, out_features=codebook_size)),
        ])
        # self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size + 1))

        if kind == 'hf':
            self.projector = Upscale(embed_dim, embed_dim, 2*embed_dim)

    def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
        cond_type = 'uncond' if isinstance(class_condition, type(None)) else 'class-cond'

        if cond_type == 'uncond':
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            cls_emb = self.class_condition_emb(class_uncondition)  # (b 1 dim)
            return cls_emb
        elif cond_type == 'class-cond':
            if self.training:
                ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional  # to enable classifier-free guidance
            else:
                ind = torch.ones_like(class_condition, dtype=torch.bool).to(device)
            class_condition = torch.where(ind, class_condition.long(), self.n_classes)  # (b 1)
            cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
            return cls_emb

    def _token_emb_dropout(self, s:torch.LongTensor, token_emb:torch.FloatTensor, freq_type:str, p:float):
        mask_ind = (s == self.mask_token_ind[freq_type])[:,:,None]  # (b n 1)
        token_emb_dropout = F.dropout(token_emb, p=p)  # (b n d); to make the prediction process more robust during sampling
        token_emb = torch.where(mask_ind, token_emb, token_emb_dropout)  # (b n d)
        return token_emb

    def forward_lf(self, s_M_l, class_condition: Union[None, torch.Tensor] = None):
        device = s_M_l.device

        token_embeddings = self.tok_emb_l(s_M_l)  # (b n dim)
        if self.training:
            token_embeddings = self._token_emb_dropout(s_M_l, token_embeddings, 'lf', p=self.emb_dropout)  # (b n d)

        cls_emb = self.class_embedding(class_condition, s_M_l.shape[0], device)  # (b 1 dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings  # (b, n, dim)
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+n, dim)
        embed = self.proj_in(embed)
        embed = self.blocks(embed)  # (b, 1+n, dim)
        logits = self.pred_head(embed)[:, 1:, :]  # (b, n, dim)

        # logits = torch.matmul(embed, self.tok_emb_l.weight.T) + self.bias  # (b, n, codebook_size+1)
        # logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, n, codebook_size)
        # return logits
        return logits

    def forward_hf(self, s_l, s_M_h, class_condition=None):
        """
        s_M_l (b n)
        s_M_h (b m); m > n
        """
        device = s_l.device

        token_embeddings_l = self.tok_emb_l(s_l)  # (b n dim)
        token_embeddings_h = self.tok_emb_h(s_M_h)  # (b m dim)

        if self.training:
            token_embeddings_l = self._token_emb_dropout(s_l, token_embeddings_l, 'lf', p=self.emb_dropout)
            token_embeddings_h = self._token_emb_dropout(s_M_h, token_embeddings_h, 'hf', p=self.emb_dropout)
        token_embeddings_l = self.projector(token_embeddings_l, upscale_size=token_embeddings_h.shape[1])  # (b m dim)
        token_embeddings = torch.cat((token_embeddings_l, token_embeddings_h), dim=-1)  # (b m 2*dim)

        cls_emb = self.class_embedding(class_condition, s_l.shape[0], device)  # (b 1 2*dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = token_embeddings + position_embeddings
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+m, 2*dim)
        embed = self.proj_in(embed)
        embed = self.blocks(embed)  # (b, 1+m, 2*dim)
        logits = self.pred_head(embed)[:, 1:, :]  # (b, m, dim)

        # logits = torch.matmul(embed, self.tok_emb_h.weight.T) + self.bias  # (b, m, codebook_size+1)
        # logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, m, codebook_size)
        # return logits
        return logits

    def forward(self, s_M_l, s_M_h=None, class_condition: Union[None, torch.Tensor] = None):
        """
        embed_ind: indices for embedding; (b n)
        class_condition: (b 1); if None, unconditional sampling is operated.
        """
        if self.kind == 'lf':
            logits = self.forward_lf(s_M_l, class_condition)
        elif self.kind == 'hf':
            logits = self.forward_hf(s_M_l, s_M_h, class_condition)
        else:
            raise ValueError
        return logits
    