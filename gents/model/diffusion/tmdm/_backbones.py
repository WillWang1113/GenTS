
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._layers import DSAttention, AttentionLayer, DataEmbedding
from ._utils import make_beta_schedule

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        # out = gamma.view(-1, self.num_out) * out

        out = gamma.view(t.size()[0], -1, self.num_out) * out
        return out


class ConditionalGuidedModel(nn.Module):
    def __init__(self, config, MTS_args):
        super(ConditionalGuidedModel, self).__init__()
        n_steps = config.timesteps + 1
        self.cat_x = config.cat_x
        self.cat_y_pred = config.cat_y_pred
        data_dim = MTS_args.enc_in * 2

        self.lin1 = ConditionalLinear(data_dim, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, MTS_args.c_out)

    def forward(self, x, y_t, y_0_hat, t):
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=-1)
            else:
                eps_pred = torch.cat((y_t, x), dim=2)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=2)
            else:
                eps_pred = y_t
        if y_t.device.type == "mps":
            eps_pred = self.lin1(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            eps_pred = self.lin2(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            eps_pred = self.lin3(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

        else:
            eps_pred = F.softplus(self.lin1(eps_pred, t))
            eps_pred = F.softplus(self.lin2(eps_pred, t))
            eps_pred = F.softplus(self.lin3(eps_pred, t))
        eps_pred = self.lin4(eps_pred)
        return eps_pred


class Denoiser(nn.Module):
    """
    Vanilla Transformer
    """

    def __init__(self, configs):
        super(Denoiser, self).__init__()

        # with open(configs.configs_dir, "r") as f:
        #     config = yaml.unsafe_load(f)
        #     configs = dict2namespace(config)

        self.args = configs
        # self.device = device
        self.configs = configs

        self.model_var_type = "fixedlarge"
        self.num_timesteps = configs.timesteps
        # self.vis_step = configs.vis_step
        # self.num_figs = configs.num_figs
        self.dataset_object = None

        betas = make_beta_schedule(
            schedule=configs.beta_schedule,
            num_timesteps=self.num_timesteps,
            start=configs.beta_start,
            end=configs.beta_end,
        ).float()
        self.register_buffer('betas', betas)
        self.register_buffer('betas_sqrt', torch.sqrt(betas))
        # betas = self.betas = betas.float()
        # self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('one_minus_betas_sqrt', torch.sqrt(alphas))
        # self.alphas = alphas
        # self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.register_buffer('alphas_bar_sqrt', torch.sqrt(alphas_cumprod))
        self.register_buffer('one_minus_alphas_bar_sqrt', torch.sqrt(1.0 - alphas_cumprod))
        # self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        # self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if configs.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= (
                0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
            )
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]], dim=0
        )
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('posterior_mean_coeff_1', (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ))
        self.register_buffer('posterior_mean_coeff_2', (
            torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        ))
        # self.alphas_cumprod_prev = alphas_cumprod_prev
        # self.posterior_mean_coeff_1 = (
        #     betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # )
        # self.posterior_mean_coeff_2 = (
        #     torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        # )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_variance', posterior_variance)
        
        # self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            # self.logvar = betas.log()
            self.register_buffer('logvar', betas.log())
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            # self.logvar = posterior_variance.clamp(min=1e-20).log()
            self.register_buffer('logvar', posterior_variance.clamp(min=1e-20).log())

        self.tau = None  # precision fo test NLL computation

        # CATE MLP
        self.diffussion_model = ConditionalGuidedModel(configs, self.args)

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.CART_input_x_embed_dim,
            configs.embed,
            configs.freq,
            configs.dropout,
            add_pos=configs.emb_add_pos,
            add_temporal=configs.emb_add_temporal,
        )

        # a = 0

    def forward(self, x, x_mark, y, y_t, y_0_hat, t):
        enc_out = self.enc_embedding(x, x_mark)
        dec_out = self.diffussion_model(enc_out, y_t, y_0_hat, t)

        return dec_out



class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))  # BxExS
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            # The reason why we only import delta for the first attn_block of Encoder
            # is to integrate Informer into our framework, where row size of attention of Informer is changing each layer
            # and inconsistent to the sequence length of the initial input,
            # then no way to add delta to every row, so we make delta=0.0 (See our Appendix E.2)
            #
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Note that delta only used for Self-Attention(x_enc with x_enc)
        # and Cross-Attention(x_enc with x_dec),
        # but not suitable for Self-Attention(x_dec with x_dec)

        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(
                x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class Projector(nn.Module):
    """
    MLP to learn the De-stationary factors
    """

    def __init__(
        self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3
    ):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class NSFormer(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, configs):
        super(NSFormer, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            add_pos=configs.emb_add_pos,
            add_temporal=configs.emb_add_temporal,
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            add_pos=configs.emb_add_pos,
            add_temporal=configs.emb_add_temporal,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        DSAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

        self.tau_learner = Projector(
            enc_in=configs.enc_in,
            seq_len=configs.seq_len,
            hidden_dims=configs.p_hidden_dims,
            hidden_layers=configs.p_hidden_layers,
            output_dim=1,
        )
        self.delta_learner = Projector(
            enc_in=configs.enc_in,
            seq_len=configs.seq_len,
            hidden_dims=configs.p_hidden_dims,
            hidden_layers=configs.p_hidden_layers,
            output_dim=configs.seq_len,
        )

        self.z_mean = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.z_logvar = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
        )

        self.z_out = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
        )

    def KL_loss_normal(self, posterior_mean, posterior_logvar):
        KL = -0.5 * torch.mean(
            1 - posterior_mean**2 + posterior_logvar - torch.exp(posterior_logvar),
            dim=1,
        )
        return torch.mean(KL)

    def reparameterize(self, posterior_mean, posterior_logvar):
        posterior_var = posterior_logvar.exp()
        # take sample
        if self.training:
            posterior_mean = posterior_mean.repeat(100, 1, 1, 1)
            posterior_var = posterior_var.repeat(100, 1, 1, 1)
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
            z = z.mean(0)
        else:
            z = posterior_mean
        # z = posterior_mean
        return z

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = (
            torch.cat(
                [
                    x_enc[:, -self.label_len :, :],
                    torch.zeros_like(x_dec[:, -self.pred_len :, :]),
                ],
                dim=1,
            )
            .to(x_enc.device)
            .clone()
        )

        tau = self.tau_learner(
            x_raw, std_enc
        ).exp()  # B x S x E, B x 1 x E -> B x 1, positive scalar
        delta = self.delta_learner(x_raw, mean_enc)  # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(
            enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta
        )

        mean = self.z_mean(enc_out)
        logvar = self.z_logvar(enc_out)

        z_sample = self.reparameterize(mean, logvar)

        # dec_out = self.z_out(torch.cat([z_sample, dec_out], dim=-1))
        enc_out = self.z_out(z_sample)

        KL_z = self.KL_loss_normal(mean, logvar)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            tau=tau,
            delta=delta,
        )

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :], dec_out, KL_z, z_sample  # [B, L, D]
