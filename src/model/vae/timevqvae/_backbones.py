from typing import Union
from einops import rearrange, repeat
import numpy as np
import torch
import torch.jit as jit
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm
from src.common._utils import exists
from ._utils import FeedForward, time_to_timefreq, timefreq_to_time, calculate_padding, l2norm


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



class Upscale(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, h_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            weight_norm(
                nn.Conv1d(
                    in_channels,
                    h_dim,
                    kernel_size=7,
                    stride=1,
                    dilation=1,
                    padding=calculate_padding(7, 1, 1),
                )
            ),
            nn.GELU(),
            nn.BatchNorm1d(h_dim),
            weight_norm(
                nn.Conv1d(
                    h_dim,
                    out_channels,
                    kernel_size=7,
                    stride=1,
                    dilation=2,
                    padding=calculate_padding(7, 1, 2),
                )
            ),
        )

    def forward(self, x, upscale_size: int):
        """
        x: (b n d)
        """
        x = rearrange(x, "b n d -> b d n")  # (b d n)
        x = F.interpolate(x, size=(upscale_size,), mode="nearest")  # (b d m)
        x = self.conv(x)  # (b d m)
        x = rearrange(x, "b d m -> b m d")
        return x




class SnakeActivation(jit.ScriptModule):
    """
    this version allows multiple values of `a` for different channels/num_features
    """

    def __init__(
        self, num_features: int, dim: int, a_base=0.2, learnable=True, a_max=0.5
    ):
        super().__init__()
        assert dim in [1, 2], "`dim` supports 1D and 2D inputs."

        if learnable:
            if dim == 1:  # (b d l); like time series
                a = np.random.uniform(
                    a_base, a_max, size=(1, num_features, 1)
                )  # (1 d 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            elif dim == 2:  # (b d h w); like 2d images
                a = np.random.uniform(
                    a_base, a_max, size=(1, num_features, 1, 1)
                )  # (1 d 1 1)
                self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        else:
            self.register_buffer("a", torch.tensor(a_base, dtype=torch.float32))

    @jit.script_method
    def forward(self, x):
        return x + (1 / self.a) * torch.sin(self.a * x) ** 2




class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        frequency_indepence: bool,
        mid_channels=None,
        dropout: float = 0.0,
    ):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        kernel_size = (1, 3) if frequency_indepence else (3, 3)
        padding = (0, 1) if frequency_indepence else (1, 1)

        layers = [
            SnakeActivation(
                in_channels, 2
            ),  # SnakyGELU(in_channels, 2), #SnakeActivation(in_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            weight_norm(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=padding,
                    groups=in_channels,
                )
            ),
            weight_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=1)),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(
                out_channels, 2
            ),  # SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            weight_norm(
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=padding,
                    groups=mid_channels,
                )
            ),
            weight_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=1)),
            nn.Dropout(dropout),
        ]
        self.convs = nn.Sequential(*layers)
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.proj(x) + self.convs(x)


class VQVAEEncBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, frequency_indepence: bool, dropout: float = 0.0
    ):
        super().__init__()

        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            weight_norm(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    stride=(1, 2),
                    padding=padding,
                    padding_mode="replicate",
                    groups=in_channels,
                )
            ),
            weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(
                out_channels, 2
            ),  # SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEDecBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, frequency_indepence: bool, dropout: float = 0.0
    ):
        super().__init__()

        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        self.block = nn.Sequential(
            weight_norm(
                nn.ConvTranspose2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    stride=(1, 2),
                    padding=padding,
                    groups=in_channels,
                )
            ),
            weight_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1)),
            nn.BatchNorm2d(out_channels),
            SnakeActivation(
                out_channels, 2
            ),  # SnakyGELU(out_channels, 2), #SnakeActivation(out_channels, 2), #nn.LeakyReLU(), #SnakeActivation(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.block(x)
        return out


class VQVAEEncoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(
        self,
        init_dim: int,
        hid_dim: int,
        num_channels: int,
        downsample_rate: int,
        n_resnet_blocks: int,
        kind: str,
        n_fft: int,
        frequency_indepence: bool,
        dropout: float = 0.3,
        **kwargs,
    ):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param bn: use of BatchNorm
        :param kwargs:
        """
        super().__init__()
        self.kind = kind
        self.n_fft = n_fft

        d = init_dim
        enc_layers = [
            VQVAEEncBlock(num_channels, d, frequency_indepence),
        ]
        d *= 2
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            enc_layers.append(VQVAEEncBlock(d // 2, d, frequency_indepence))
            for _ in range(n_resnet_blocks):
                enc_layers.append(ResBlock(d, d, frequency_indepence, dropout=dropout))
            d *= 2
        enc_layers.append(
            ResBlock(d // 2, hid_dim, frequency_indepence, dropout=dropout)
        )
        self.encoder = nn.Sequential(*enc_layers)

        self.is_num_tokens_updated = False
        self.register_buffer("num_tokens", torch.tensor(0))
        self.register_buffer("H_prime", torch.tensor(0))
        self.register_buffer("W_prime", torch.tensor(0))

    def forward(self, x):
        """
        :param x: (b c l)
        """
        in_channels = x.shape[1]
        x = time_to_timefreq(x, self.n_fft, in_channels)  # (b c h w)

        if self.kind == "lf":
            x = x[:, :, [0], :]  # (b c 1 w)
        elif self.kind == "hf":
            x = x[:, :, 1:, :]  # (b c h-1 w)

        out = self.encoder(x)  # (b c h w)
        out = F.normalize(out, dim=1)  # following hilcodec
        if not self.is_num_tokens_updated:
            self.H_prime = torch.tensor(out.shape[2])
            self.W_prime = torch.tensor(out.shape[3])
            self.num_tokens = self.H_prime * self.W_prime
            self.is_num_tokens_updated = True
        return out


class VQVAEDecoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(
        self,
        init_dim: int,
        hid_dim: int,
        num_channels: int,
        downsample_rate: int,
        n_resnet_blocks: int,
        input_length: int,
        kind: str,
        n_fft: int,
        x_channels: int,
        frequency_indepence: bool,
        dropout: float = 0.3,
        **kwargs,
    ):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param kwargs:
        """
        super().__init__()
        self.kind = kind
        self.n_fft = n_fft
        self.x_channels = x_channels

        kernel_size = (1, 4) if frequency_indepence else (3, 4)
        padding = (0, 1) if frequency_indepence else (1, 1)

        d = int(
            init_dim * 2 ** (int(round(np.log2(downsample_rate))) - 1)
        )  # enc_out_dim == dec_in_dim
        if round(np.log2(downsample_rate)) == 0:
            d = int(init_dim * 2 ** (int(round(np.log2(downsample_rate)))))
        dec_layers = [ResBlock(hid_dim, d, frequency_indepence, dropout=dropout)]
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            for _ in range(n_resnet_blocks):
                dec_layers.append(ResBlock(d, d, frequency_indepence, dropout=dropout))
            d //= 2
            dec_layers.append(VQVAEDecBlock(2 * d, d, frequency_indepence))
        dec_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    d,
                    d,
                    kernel_size=kernel_size,
                    stride=(1, 2),
                    padding=padding,
                    groups=d,
                ),
                nn.ConvTranspose2d(d, num_channels, kernel_size=1),
            )
        )
        dec_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    num_channels,
                    num_channels,
                    kernel_size=kernel_size,
                    stride=(1, 2),
                    padding=padding,
                ),
                nn.ConvTranspose2d(num_channels, num_channels, kernel_size=1),
            )
        )
        self.decoder = nn.Sequential(*dec_layers)

        self.interp = nn.Upsample(input_length, mode="linear")
        # self.linear = nn.Linear(input_length, input_length)  # though helpful, it consumes too much memory for long sequences

    def forward(self, x):
        out = self.decoder(x)  # (b c h w)

        if self.kind == "lf":
            zeros = (
                torch.zeros(
                    (out.shape[0], out.shape[1], self.n_fft // 2 + 1, out.shape[-1])
                )
                .float()
                .to(out.device)
            )
            zeros[:, :, [0], :] = out
            out = zeros
        elif self.kind == "hf":
            zeros = (
                torch.zeros(
                    (out.shape[0], out.shape[1], self.n_fft // 2 + 1, out.shape[-1])
                )
                .float()
                .to(out.device)
            )
            zeros[:, :, 1:, :] = out
            out = zeros
        out = timefreq_to_time(out, self.n_fft, self.x_channels)  # (b c l)

        out = self.interp(out)  # (b c l)
        # out = out + self.linear(out)  # (b c l)
        return out
    


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for dep_i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, dim_head=dim_head, heads=heads),
                        # MultiHeadDifferentialAttention(h=heads, d=dim_head, embedding_dim=dim),
                        # MultiheadDiffAttn(dim, dep_i+1, heads, dim_head),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x



class BidirectionalTransformer(nn.Module):
    def __init__(
        self,
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
        model_dropout: float = 0.3,
        emb_dropout: float = 0.3,
        **kwargs,
    ):
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
        assert kind in ["lf", "hf"], "invalid `kind`."
        self.kind = kind
        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim if kind == "lf" else 2 * embed_dim
        # out_dim = embed_dim
        self.emb_dropout = emb_dropout
        self.mask_token_ind = {"lf": codebook_sizes, "hf": codebook_sizes}

        # token embeddings
        self.tok_emb_l = nn.Embedding(
            codebook_sizes + 1, embed_dim
        )  # `+1` is for mask-token
        if kind == "hf":
            self.tok_emb_h = nn.Embedding(
                codebook_sizes + 1, embed_dim
            )  # `+1` is for mask-token

        # transformer
        self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.class_condition_emb = nn.Embedding(
            n_classes + 1, in_dim
        )  # `+1` is for no-condition
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = TransformerBlocks(
            dim=hidden_dim, depth=n_layers, dim_head=64, heads=heads, ff_mult=ff_mult
        )
        codebook_size = codebook_sizes if kind == "lf" else codebook_sizes
        self.pred_head = nn.Sequential(
            *[
                weight_norm(nn.Linear(in_features=hidden_dim, out_features=hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(hidden_dim, eps=1e-12),
                weight_norm(
                    nn.Linear(in_features=hidden_dim, out_features=codebook_size)
                ),
            ]
        )
        # self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size + 1))

        if kind == "hf":
            self.projector = Upscale(embed_dim, embed_dim, 2 * embed_dim)

    def class_embedding(
        self, class_condition: Union[None, torch.Tensor], batch_size: int, device
    ):
        cond_type = (
            "uncond" if isinstance(class_condition, type(None)) else "class-cond"
        )

        if cond_type == "uncond":
            class_uncondition = repeat(
                torch.Tensor([self.n_classes]).long().to(device),
                "i -> b i",
                b=batch_size,
            )  # (b 1)
            cls_emb = self.class_condition_emb(class_uncondition)  # (b 1 dim)
            return cls_emb
        elif cond_type == "class-cond":
            if self.training:
                ind = (
                    torch.rand(class_condition.shape).to(device) > self.p_unconditional
                )  # to enable classifier-free guidance
            else:
                ind = torch.ones_like(class_condition, dtype=torch.bool).to(device)
            class_condition = torch.where(
                ind, class_condition.long(), self.n_classes
            )  # (b 1)
            cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
            return cls_emb

    def _token_emb_dropout(
        self,
        s: torch.LongTensor,
        token_emb: torch.FloatTensor,
        freq_type: str,
        p: float,
    ):
        mask_ind = (s == self.mask_token_ind[freq_type])[:, :, None]  # (b n 1)
        token_emb_dropout = F.dropout(
            token_emb, p=p
        )  # (b n d); to make the prediction process more robust during sampling
        token_emb = torch.where(mask_ind, token_emb, token_emb_dropout)  # (b n d)
        return token_emb

    def forward_lf(self, s_M_l, class_condition: Union[None, torch.Tensor] = None):
        device = s_M_l.device

        token_embeddings = self.tok_emb_l(s_M_l)  # (b n dim)
        if self.training:
            token_embeddings = self._token_emb_dropout(
                s_M_l, token_embeddings, "lf", p=self.emb_dropout
            )  # (b n d)

        cls_emb = self.class_embedding(
            class_condition, s_M_l.shape[0], device
        )  # (b 1 dim)

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
            token_embeddings_l = self._token_emb_dropout(
                s_l, token_embeddings_l, "lf", p=self.emb_dropout
            )
            token_embeddings_h = self._token_emb_dropout(
                s_M_h, token_embeddings_h, "hf", p=self.emb_dropout
            )
        token_embeddings_l = self.projector(
            token_embeddings_l, upscale_size=token_embeddings_h.shape[1]
        )  # (b m dim)
        token_embeddings = torch.cat(
            (token_embeddings_l, token_embeddings_h), dim=-1
        )  # (b m 2*dim)

        cls_emb = self.class_embedding(
            class_condition, s_l.shape[0], device
        )  # (b 1 2*dim)

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

    def forward(
        self, s_M_l, s_M_h=None, class_condition: Union[None, torch.Tensor] = None
    ):
        """
        embed_ind: indices for embedding; (b n)
        class_condition: (b 1); if None, unconditional sampling is operated.
        """
        if self.kind == "lf":
            logits = self.forward_lf(s_M_l, class_condition)
        elif self.kind == "hf":
            logits = self.forward_hf(s_M_l, s_M_h, class_condition)
        else:
            raise ValueError
        return logits

