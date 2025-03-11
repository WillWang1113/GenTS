from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch.nn.utils import weight_norm

from src.layers.attention import Attention, FeedForward, SelfAttention
from src.layers.conv import Upscale
from src.layers.embed import PatchEmbed, TimestepEmbed, get_1d_sincos_pos_embed
from src.layers.mlp import CustomMLP, FinalLayer, modulate
from src.utils.timefreq import zero_pad_high_freq, zero_pad_low_freq

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
    
    

# class MaskGIT(nn.Module):
#     """
#     ref: https://github.com/dome272/MaskGIT-pytorch/blob/cff485ad3a14b6ed5f3aa966e045ea2bc8c68ad8/transformer.py#L11
#     """

#     def __init__(self,
#                  dataset_name: str,
#                  in_channels:int,
#                  input_length: int,
#                  choice_temperatures: dict,
#                  T: dict,
#                  config: dict,
#                  n_classes: int,
#                  **kwargs):
#         super().__init__()
#         self.choice_temperature_l = choice_temperatures['lf']
#         self.choice_temperature_h = choice_temperatures['hf']
#         self.T = T
#         self.config = config
#         self.n_classes = n_classes
#         self.n_fft = config['VQ-VAE']['n_fft']
#         self.cfg_scale = config['MaskGIT']['cfg_scale']

#         self.mask_token_ids = {'lf': config['VQ-VAE']['codebook_sizes']['lf'], 'hf': config['VQ-VAE']['codebook_sizes']['hf']}
#         self.gamma = self.gamma_func("cosine")

#         # load the staeg1 model
#         self.stage1 = ExpStage1.load_from_checkpoint(os.path.join('saved_models', f'stage1-{dataset_name}.ckpt'), 
#                                                      in_channels=in_channels,
#                                                      input_length=input_length, 
#                                                      config=config,
#                                                      map_location='cpu')
#         freeze(self.stage1)
#         self.stage1.eval()

#         self.encoder_l = self.stage1.encoder_l
#         self.decoder_l = self.stage1.decoder_l
#         self.vq_model_l = self.stage1.vq_model_l
#         self.encoder_h = self.stage1.encoder_h
#         self.decoder_h = self.stage1.decoder_h
#         self.vq_model_h = self.stage1.vq_model_h

#         # token lengths
#         self.num_tokens_l = self.encoder_l.num_tokens.item()
#         self.num_tokens_h = self.encoder_h.num_tokens.item()

#         # latent space dim
#         self.H_prime_l, self.H_prime_h = self.encoder_l.H_prime.item(), self.encoder_h.H_prime.item()
#         self.W_prime_l, self.W_prime_h = self.encoder_l.W_prime.item(), self.encoder_h.W_prime.item()

#         # transformers / prior models
#         emb_dim = self.config['encoder']['hid_dim']
#         self.transformer_l = BidirectionalTransformer('lf',
#                                                       self.num_tokens_l,
#                                                       config['VQ-VAE']['codebook_sizes'],
#                                                       emb_dim,
#                                                       **config['MaskGIT']['prior_model_l'],
#                                                       n_classes=n_classes,
#                                                       )

#         self.transformer_h = BidirectionalTransformer('hf',
#                                                       self.num_tokens_h,
#                                                       config['VQ-VAE']['codebook_sizes'],
#                                                       emb_dim,
#                                                       **config['MaskGIT']['prior_model_h'],
#                                                       n_classes=n_classes,
#                                                       num_tokens_l=self.num_tokens_l,
#                                                       )

#     def load(self, model, dirname, fname):
#         """
#         model: instance
#         path_to_saved_model_fname: path to the ckpt file (i.e., trained model)
#         """
#         try:
#             model.load_state_dict(torch.load(dirname.joinpath(fname)))
#         except FileNotFoundError:
#             dirname = Path(tempfile.gettempdir())
#             model.load_state_dict(torch.load(dirname.joinpath(fname)))

#     @torch.no_grad()
#     def encode_to_z_q(self, x, encoder, vq_model, svq_temp:Union[float,None]=None):
#         """
#         encode x to zq

#         x: (b c l)
#         """
#         z = encoder(x)
#         zq, s, _, _ = quantize(z, vq_model, svq_temp=svq_temp)  # (b c h w), (b (h w) h), ...
#         return zq, s
    
#     def masked_prediction(self, transformer, class_condition, *s_in):
#         """
#         masked prediction with classifier-free guidance
#         """
#         if isinstance(class_condition, type(None)):
#             # unconditional 
#             logits_null = transformer(*s_in, class_condition=None)  # (b n k)
#             return logits_null
#         else:
#             # class-conditional
#             if self.cfg_scale == 1.0:
#                 logits = transformer(*s_in, class_condition=class_condition)  # (b n k)
#             else:
#                 # with CFG
#                 logits_null = transformer(*s_in, class_condition=None)
#                 logits = transformer(*s_in, class_condition=class_condition)  # (b n k)
#                 logits = logits_null + self.cfg_scale * (logits - logits_null)
#             return logits

#     def forward(self, x, y):
#         """
#         x: (B, C, L)
#         y: (B, 1)
#         straight from [https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py]
#         """
#         self.encoder_l.eval()
#         self.vq_model_l.eval()
#         self.encoder_h.eval()
#         self.vq_model_h.eval()
        
#         device = x.device
#         _, s_l = self.encode_to_z_q(x, self.encoder_l, self.vq_model_l)  # (b n)
#         _, s_h = self.encode_to_z_q(x, self.encoder_h, self.vq_model_h)  # (b m)

#         # mask tokens
#         s_l_M, mask_l = self._randomly_mask_tokens(s_l, self.mask_token_ids['lf'], device)  # (b n), (b n) where 0 for masking and 1 for un-masking
#         s_h_M, mask_h = self._randomly_mask_tokens(s_h, self.mask_token_ids['hf'], device)  # (b n), (b n) where 0 for masking and 1 for un-masking

#         # prediction
#         logits_l = self.masked_prediction(self.transformer_l, y, s_l_M)  # (b n k)
#         logits_h = self.masked_prediction(self.transformer_h, y, s_l, s_h_M)
        
#         # maksed prediction loss
#         logits_l_on_mask = logits_l[~mask_l]  # (bm k) where m < n
#         s_l_on_mask = s_l[~mask_l]  # (bm) where m < n
#         mask_pred_loss_l = F.cross_entropy(logits_l_on_mask.float(), s_l_on_mask.long())
        
#         logits_h_on_mask = logits_h[~mask_h]  # (bm k) where m < n
#         s_h_on_mask = s_h[~mask_h]  # (bm) where m < n
#         mask_pred_loss_h = F.cross_entropy(logits_h_on_mask.float(), s_h_on_mask.long())

#         mask_pred_loss = mask_pred_loss_l + mask_pred_loss_h
#         return mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h)

#     def _randomly_mask_tokens(self, s, mask_token_id, device):
#         """
#         s: token set
#         """
#         b, n = s.shape
        
#         # sample masking indices
#         ratio = torch.rand(0, 1, (b,))  # (b,)
#         n_unmasks = torch.floor(self.gamma(ratio) * n)  # (b,)
#         n_unmasks = torch.clip(n_unmasks, min=0, max=n-1).int()  # ensures that there's at least one masked token
#         rand = torch.rand((b, n), device=device)  # (b n)
#         mask = torch.zeros((b, n), dtype=torch.bool, device=device)  # (b n)

#         for i in range(b):
#             ind = rand[i].topk(n_unmasks[i], dim=-1).indices
#             mask[i].scatter_(dim=-1, index=ind, value=True)

#         # mask the token set
#         masked_indices = mask_token_id * torch.ones((b, n), device=device)  # (b n)
#         s_M = mask * s + (~mask) * masked_indices  # (b n); `~` reverses bool-typed data
#         return s_M.long(), mask
    
#     def gamma_func(self, mode="cosine"):
#         if mode == "linear":
#             return lambda r: 1 - r
#         elif mode == "cosine":
#             return lambda r: torch.cos(r * torch.pi / 2)
#         elif mode == "square":
#             return lambda r: 1 - r ** 2
#         elif mode == "cubic":
#             return lambda r: 1 - r ** 3
#         else:
#             raise NotImplementedError

#     def create_input_tokens_normal(self, num, num_tokens, mask_token_ids, device):
#         """
#         returns masked tokens
#         """
#         blank_tokens = torch.ones((num, num_tokens), device=device)
#         masked_tokens = mask_token_ids * blank_tokens
#         return masked_tokens.to(torch.int64)

#     def mask_by_random_topk(self, mask_len, probs, temperature=1.0, device='cpu'):
#         """
#         mask_len: (b 1)
#         probs: (b n); also for the confidence scores

#         This version keeps `mask_len` exactly.
#         """
#         def log(t, eps=1e-20):
#             return torch.log(t.clamp(min=eps))

#         def gumbel_noise(t):
#             """
#             Gumbel max trick: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
#             """
#             noise = torch.zeros_like(t).uniform_(0, 1)
#             return -log(-log(noise))

#         confidence = torch.log(probs + 1e-5) + temperature * gumbel_noise(probs).to(device)  # Gumbel max trick; 1e-5 for numerical stability; (b n)
#         mask_len_unique = int(mask_len.unique().item())
#         masking_ind = torch.topk(confidence, k=mask_len_unique, dim=-1, largest=False).indices  # (b k)
#         masking = torch.zeros_like(confidence).to(device)  # (b n)
#         for i in range(masking_ind.shape[0]):
#             masking[i, masking_ind[i].long()] = 1.
#         masking = masking.bool()
#         return masking



#     def first_pass(self,
#                    s_l: torch.Tensor,
#                    unknown_number_in_the_beginning_l,
#                    class_condition: Union[torch.Tensor, None],
#                    gamma,
#                    device):
#         for t in range(self.T['lf']):
#             logits_l = self.masked_prediction(self.transformer_l, class_condition, s_l)  # (b n k)

#             sampled_ids = torch.distributions.categorical.Categorical(logits=logits_l).sample()  # (b n)
#             unknown_map = (s_l == self.mask_token_ids['lf'])  # which tokens need to be sampled; (b n)
#             sampled_ids = torch.where(unknown_map, sampled_ids, s_l)  # keep the previously-sampled tokens; (b n)

#             # create masking according to `t`
#             ratio = 1. * (t + 1) / self.T['lf']  # just a percentage e.g. 1 / 12
#             mask_ratio = gamma(ratio)

#             probs = F.softmax(logits_l, dim=-1)  # convert logits into probs; (b n K)
#             selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b n)
#             _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
#             selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # assign inf probability to the previously-selected tokens; (b n)

#             mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning_l * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
#             mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

#             # Adds noise for randomness
#             masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature_l * (1. - ratio), device=device)

#             # Masks tokens with lower confidence.
#             s_l = torch.where(masking, self.mask_token_ids['lf'], sampled_ids)  # (b n)

#         # use ESS (Enhanced Sampling Scheme)
#         if self.config['MaskGIT']['ESS']['use']:
#             print(' ===== ESS: LF =====')
#             t_star, s_star = self.critical_reverse_sampling(s_l, unknown_number_in_the_beginning_l, class_condition, 'lf')
#             s_l = self.iterative_decoding_with_self_token_critic(t_star, s_star, 'lf', unknown_number_in_the_beginning_l, class_condition, device)

#         return s_l

#     def second_pass(self,
#                     s_l: torch.Tensor,
#                     s_h: torch.Tensor,
#                     unknown_number_in_the_beginning_h,
#                     class_condition: Union[torch.Tensor, None],
#                     gamma,
#                     device):
#         for t in range(self.T['hf']):
#             logits_h = self.masked_prediction(self.transformer_h, class_condition, s_l, s_h)  # (b m k)

#             sampled_ids = torch.distributions.categorical.Categorical(logits=logits_h).sample()  # (b m)
#             unknown_map = (s_h == self.mask_token_ids['hf'])  # which tokens need to be sampled; (b m)
#             sampled_ids = torch.where(unknown_map, sampled_ids, s_h)  # keep the previously-sampled tokens; (b m)

#             # create masking according to `t`
#             ratio = 1. * (t + 1) / self.T['hf']  # just a percentage e.g. 1 / 12
#             mask_ratio = gamma(ratio)

#             probs = F.softmax(logits_h, dim=-1)  # convert logits into probs; (b m K)
#             selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b m)
#             _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
#             selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # assign inf probability to the previously-selected tokens; (b m)

#             mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning_h * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
#             mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

#             # Adds noise for randomness
#             masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature_h * (1. - ratio), device=device)

#             # Masks tokens with lower confidence.
#             s_h = torch.where(masking, self.mask_token_ids['hf'], sampled_ids)  # (b n)
        
#         # use ESS (Enhanced Sampling Scheme)
#         if self.config['MaskGIT']['ESS']['use']:
#             print(' ===== ESS: HF =====')
#             t_star, s_star = self.critical_reverse_sampling(s_l, unknown_number_in_the_beginning_h, class_condition, 'hf', s_h=s_h)
#             s_h = self.iterative_decoding_with_self_token_critic(t_star, s_star, 'hf', unknown_number_in_the_beginning_h, class_condition, device, s_l=s_l)

#         return s_h

#     @torch.no_grad()
#     def iterative_decoding(self, num=1, mode="cosine", class_index=None, device='cpu'):
#         """
#         It performs the iterative decoding and samples token indices for LF and HF.
#         :param num: number of samples
#         :return: sampled token indices for LF and HF
#         """
#         s_l = self.create_input_tokens_normal(num, self.num_tokens_l, self.mask_token_ids['lf'], device)  # (b n)
#         s_h = self.create_input_tokens_normal(num, self.num_tokens_h, self.mask_token_ids['hf'], device)  # (b n)

#         unknown_number_in_the_beginning_l = torch.sum(s_l == self.mask_token_ids['lf'], dim=-1)  # (b,)
#         unknown_number_in_the_beginning_h = torch.sum(s_h == self.mask_token_ids['hf'], dim=-1)  # (b,)
#         gamma = self.gamma_func(mode)
#         class_condition = repeat(torch.Tensor([class_index]).int().to(device), 'i -> b i', b=num) if class_index != None else None  # (b 1)

#         s_l = self.first_pass(s_l, unknown_number_in_the_beginning_l, class_condition, gamma, device)
#         s_h = self.second_pass(s_l, s_h, unknown_number_in_the_beginning_h, class_condition, gamma, device)
#         return s_l, s_h

#     def decode_token_ind_to_timeseries(self, s: torch.Tensor, frequency: str, return_representations: bool = False):
#         """
#         It takes token embedding indices and decodes them to time series.
#         :param s: token embedding index
#         :param frequency:
#         :param return_representations:
#         :return:
#         """
#         self.eval()
#         frequency = frequency.lower()
#         assert frequency in ['lf', 'hf']

#         vq_model = self.vq_model_l if frequency == 'lf' else self.vq_model_h
#         decoder = self.decoder_l if frequency == 'lf' else self.decoder_h
#         zero_pad = zero_pad_high_freq if frequency == 'lf' else zero_pad_low_freq

#         zq = F.embedding(s, vq_model._codebook.embed)  # (b n d)
#         zq = vq_model.project_out(zq)  # (b n c)
#         zq = rearrange(zq, 'b n c -> b c n')  # (b c n) == (b c (h w))
#         H_prime = self.H_prime_l if frequency == 'lf' else self.H_prime_h
#         W_prime = self.W_prime_l if frequency == 'lf' else self.W_prime_h
#         zq = rearrange(zq, 'b c (h w) -> b c h w', h=H_prime, w=W_prime)

#         xhat = decoder(zq)

#         if return_representations:
#             return xhat, zq
#         else:
#             return xhat

#     def critical_reverse_sampling(self,
#                                   s_l: torch.Tensor,
#                                   unknown_number_in_the_beginning,
#                                   class_condition: Union[torch.Tensor, None],
#                                   kind: str,
#                                   s_h: torch.Tensor=None,
#                                   ):
#         """
#         s: sampled token sequence from the naive iterative decoding.
#         """
#         if kind == 'lf':
#             mask_token_ids = self.mask_token_ids['lf']
#             transformer = self.transformer_l
#             vq_model = self.vq_model_l
#             s = s_l
#             s_star = s.clone()
#             s_star_prev = s.clone()
#             temperature = self.choice_temperature_l
#         elif kind == 'hf':
#             mask_token_ids = self.mask_token_ids['hf']
#             transformer = self.transformer_h
#             vq_model = self.vq_model_h
#             s = s_h
#             s_star = s.clone()
#             s_star_prev = s.clone()
#             temperature = 0.
#         else:
#             raise ValueError

#         # compute the confidence scores for s_T
#         # the scores are used for the step retraction by iteratively removing unrealistic tokens.
#         confidence_scores = self.compute_confidence_score(kind, s_l, mask_token_ids, vq_model, transformer, class_condition, s_h=s_h)  # (b n)

#         # find s_{t*}
#         # t* denotes the step where unrealistic tokens have been removed.
#         t_star = self.T[kind]
#         logprob_prev = -torch.inf
#         for t in range(1, self.T[kind])[::-1]:
#             # masking ratio according to the masking scheduler
#             # ratio_t = 1. * (t + 1) / self.T[kind]  # just a percentage e.g. 1 / 12
#             ratio_tm1 = 1. * t / self.T[kind]  # tm1: t - 1
#             # mask_ratio_t = self.gamma(ratio_t)
#             mask_ratio_tm1 = self.gamma(ratio_tm1)  # tm1: t - 1

#             # mask length
#             # mask_len_t = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio_t), 1)
#             mask_len_tm1 = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio_tm1), 1)

#             # masking matrices: {True: masking, False: not-masking}
#             # masking_t = self.mask_by_random_topk(mask_len_t, confidence_scores, temperature=0., device=s.device)  # (b n)
#             masking_tm1 = self.mask_by_random_topk(mask_len_tm1, confidence_scores, temperature=temperature, device=s.device)  # (b n)
#             # masking = ~((masking_tm1.float() - masking_t.float()).bool())  # (b n); True for everything except the area of interest with False.

#             s_star = torch.where(masking_tm1, mask_token_ids, s)

#             # predict s_t given s_{t-1}
#             s_tm1 = torch.where(masking_tm1, mask_token_ids, s)  # (b n)
#             if kind == 'lf':
#                 logits = self.masked_prediction(transformer, class_condition, s_tm1)  # (b n k)
#             elif kind == 'hf':
#                 logits = self.masked_prediction(transformer, class_condition, s_l, s_tm1)  # (b n k)
#             prob = torch.nn.functional.softmax(logits, dim=-1)  # (b n K)
#             logprob = prob.clamp_min(1.e-5).log10()  # (b n k)
#             logprob = torch.gather(logprob, dim=-1, index=s.unsqueeze(-1)).squeeze(-1)  # (b n)
#             print('t:', t)
#             # print('masking_tm1:', masking_tm1)
#             # print('masking_tm1.int().mean():', masking_tm1.float().mean().item())
#             logprob = logprob[masking_tm1]
#             logprob = logprob.mean().cpu().detach().item()

#             # stopping criteria
#             if (t == self.T[kind]-1) or (logprob > logprob_prev - 0.05*logprob_prev):
#             # if (t != 1):
#                 logprob_prev = logprob
#                 t_star = t
#                 s_star_prev = s_star.clone()
#                 pass
#             else:
#                 break

         
#         print('t_star:', t_star)
#         return t_star, s_star_prev
    
    

#     def iterative_decoding_with_self_token_critic(self,
#                                                   t_star,
#                                                   s_star,
#                                                   kind: str,
#                                                   unknown_number_in_the_beginning,
#                                                   class_condition: Union[torch.Tensor, None],
#                                                   device,
#                                                   s_l=None
#                                                   ):
#         if kind == 'lf':
#             mask_token_ids = self.mask_token_ids['lf']
#             transformer = self.transformer_l
#             vq_model = self.vq_model_l
#             choice_temperature = self.choice_temperature_l
#         elif kind == 'hf':
#             mask_token_ids = self.mask_token_ids['hf']
#             transformer = self.transformer_h
#             vq_model = self.vq_model_h
#             choice_temperature = self.choice_temperature_h
#         else:
#             raise ValueError

#         s = s_star
#         for t in range(t_star, self.T[kind]):
#             if kind == 'lf':
#                 logits = self.masked_prediction(transformer, class_condition, s)  # (b n k)
#             elif kind == 'hf':
#                 logits = self.masked_prediction(transformer, class_condition, s_l, s)  # (b n k)
            
#             sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()  # (b n)
#             unknown_map = (s == mask_token_ids)  # which tokens need to be sampled; (b n)
#             sampled_ids = torch.where(unknown_map, sampled_ids, s)  # keep the previously-sampled tokens; (b n)

#             # create masking according to `t`
#             ratio = 1. * (t + 1) / self.T[kind]  # just a percentage e.g. 1 / 12
#             mask_ratio = self.gamma(ratio)

#             if kind == 'lf':
#                 selected_probs = self.compute_confidence_score(kind, sampled_ids, mask_token_ids, vq_model, transformer, class_condition)  # (b n)
#             elif kind == 'hf':
#                 selected_probs = self.compute_confidence_score(kind, s_l, mask_token_ids, vq_model, transformer, class_condition, s_h=sampled_ids)  # (b n)
#             _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
#             selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # assign inf probability to the previously-selected tokens; (b n)

#             mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # number of tokens that are to be masked;  (b,)
#             mask_len = torch.clip(mask_len, min=0.)  # `mask_len` should be equal or larger than zero.

#             # Adds noise for randomness
#             masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=choice_temperature * (1. - ratio), device=device)
#             # masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=0., device=device)

#             # Masks tokens with lower confidence.
#             s = torch.where(masking, mask_token_ids, sampled_ids)  # (b n)

#         return s


#     def compute_confidence_score(self, kind, s_l, mask_token_ids, vq_model, transformer, class_condition, s_h=None):
        
#         if kind == 'lf':
#             s = s_l
#         elif kind == 'hf':
#             s = s_h

#         confidence_scores = torch.zeros_like(s).float()  # (b n)
#         for n in range(confidence_scores.shape[-1]):
#             s_m = copy.deepcopy(s)  # (b n)
#             s_m[:, n] = mask_token_ids  # (b n); masking the n-th token to measure the confidence score for that token.
#             if kind == 'lf':
#                 logits = self.masked_prediction(transformer, class_condition, s_l)  # (b n k)
#             elif kind == 'hf':
#                 logits = self.masked_prediction(transformer, class_condition, s_l, s_h)  # (b n k)
#             prob = torch.nn.functional.softmax(logits, dim=-1)  # (b n K)
#             # logprob = prob.clamp_min(1e-5).log10()  # (b n k)

#             selected_prob = torch.gather(prob, dim=2, index=s.unsqueeze(-1)).squeeze(-1)  # (b n)
#             selected_prob = selected_prob[:, n]  # (b,)

#             confidence_scores[:, n] = selected_prob
#         return confidence_scores




