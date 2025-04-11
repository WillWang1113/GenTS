import math
from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers.embed import DataEmbedding
from ...layers.transformer import NSFormer
from ...model.base import BaseModel
from ...utils.losses import log_normal
import numpy as np


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start**0.5, end**0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [
                min(
                    1
                    - (
                        math.cos(
                            ((i + 1) / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    )
                    / (
                        math.cos(
                            (i / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    ),
                    max_beta,
                )
                for i in range(num_timesteps)
            ]
        )
        if schedule == "cosine_reverse":
            betas = betas.flip(0)  # starts at max_beta then decreases fast
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [
                start
                + 0.5
                * (end - start)
                * (1 - math.cos(t / (num_timesteps - 1) * math.pi))
                for t in range(num_timesteps)
            ]
        )
    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


# Forward functions
def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    """
    y_0_hat: prediction of pre-trained guidance model; can be extended to represent
        any prior mean setting at timestep T.
    """
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    y_t = (
        sqrt_alpha_bar_t * y
        + (1 - sqrt_alpha_bar_t) * y_0_hat
        + sqrt_one_minus_alpha_bar_t * noise
    )
    return y_t


# Reverse function -- sample y_{t-1} given y_t
def p_sample(
    model, x, x_mark, y, y_0_hat, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt
):
    """
    Reverse diffusion process sampling -- one time step.

    y: sampled y at time step t, y_t.
    y_0_hat: prediction of pre-trained guidance model.
    y_T_mean: mean of prior distribution at timestep T.
    We replace y_0_hat with y_T_mean in the forward process posterior mean computation, emphasizing that
        guidance model prediction y_0_hat = f_phi(x) is part of the input to eps_theta network, while
        in paper we also choose to set the prior mean at timestep T y_T_mean = f_phi(x).
    """
    device = next(model.parameters()).device
    z = torch.randn_like(y)  # if t > 1 else torch.zeros_like(y)
    t = torch.tensor([t]).to(device)
    alpha_t = extract(alphas, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
    # y_t_m_1 posterior mean component coefficients
    gamma_0 = (
        (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    )
    gamma_1 = (
        (sqrt_one_minus_alpha_bar_t_m_1.square())
        * (alpha_t.sqrt())
        / (sqrt_one_minus_alpha_bar_t.square())
    )
    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
        sqrt_one_minus_alpha_bar_t.square()
    )
    eps_theta = model(x, x_mark, 0, y, y_0_hat, t).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = (
        1
        / sqrt_alpha_bar_t
        * (
            y
            - (1 - sqrt_alpha_bar_t) * y_T_mean
            - eps_theta * sqrt_one_minus_alpha_bar_t
        )
    )
    # posterior mean
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
    # posterior variance
    beta_t_hat = (
        (sqrt_one_minus_alpha_bar_t_m_1.square())
        / (sqrt_one_minus_alpha_bar_t.square())
        * (1 - alpha_t)
    )
    y_t_m_1 = y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)
    return y_t_m_1


# Reverse function -- sample y_0 given y_1
def p_sample_t_1to0(model, x, x_mark, y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt):
    device = next(model.parameters()).device
    t = torch.tensor([0]).to(
        device
    )  # corresponding to timestep 1 (i.e., t=1 in diffusion models)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = model(x, x_mark, 0, y, y_0_hat, t).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = (
        1
        / sqrt_alpha_bar_t
        * (
            y
            - (1 - sqrt_alpha_bar_t) * y_T_mean
            - eps_theta * sqrt_one_minus_alpha_bar_t
        )
    )
    y_t_m_1 = y_0_reparam.to(device)
    return y_t_m_1


def p_sample_loop(
    model, x, x_mark, y_0_hat, y_T_mean, n_steps, alphas, one_minus_alphas_bar_sqrt
):
    device = next(model.parameters()).device
    z = torch.randn_like(y_T_mean).to(device)
    cur_y = z + y_T_mean  # sample y_T
    y_p_seq = [cur_y]
    for t in reversed(range(1, n_steps)):  # t from T to 2
        y_t = cur_y
        cur_y = p_sample(
            model,
            x,
            x_mark,
            y_t,
            y_0_hat,
            y_T_mean,
            t,
            alphas,
            one_minus_alphas_bar_sqrt,
        )  # y_{t-1}
        y_p_seq.append(cur_y)
    assert len(y_p_seq) == n_steps
    y_0 = p_sample_t_1to0(
        model, x, x_mark, y_p_seq[-1], y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt
    )
    y_p_seq.append(y_0)
    return y_p_seq


# Evaluation with KLD
def kld(y1, y2, grid=(-20, 20), num_grid=400):
    y1, y2 = y1.numpy().flatten(), y2.numpy().flatten()
    p_y1, _ = np.histogram(y1, bins=num_grid, range=[grid[0], grid[1]], density=True)
    p_y1 += 1e-7
    p_y2, _ = np.histogram(y2, bins=num_grid, range=[grid[0], grid[1]], density=True)
    p_y2 += 1e-7
    return (p_y1 * np.log(p_y1 / p_y2)).sum()


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


class Model(nn.Module):
    """
    Vanilla Transformer
    """

    def __init__(self, configs):
        super(Model, self).__init__()

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


class TMDM(BaseModel):
    def __init__(
        self,
        seq_len,
        obs_len,
        seq_dim,
        emb_add_pos=True,
        emb_add_temporal=False,
        freq=None,
        emb_temporal_type=None,
        k_cond=1.0,
        k_z=1e-2,
        d_z=8,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        factor=3,
        dropout=0.05,
        activation="gelu",
        p_hidden_dims=[64, 64],
        p_hidden_layers=2,
        CART_input_x_embed_dim=32,
        n_diff_steps=100,
        beta_schedule='linear',
        beta_start=1e-4,
        beta_end=2e-2,
        cat_x=True,
        cat_y_pred=True,
        lr=1e-3,
        weight_decay=0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.condition = 'predict'
        args = Namespace(**self.hparams_initial)
        args.seq_len = obs_len
        args.pred_len = seq_len
        # ! label_len default half obs_len start
        args.label_len = obs_len // 2
        args.enc_in = seq_dim
        args.dec_in = seq_dim
        args.c_out = seq_dim
        args.output_attention = False
        args.timesteps = n_diff_steps
        args.embed = emb_temporal_type
        args.features = "S" if seq_dim == 1 else "M"

        self.args = args
        self.cond_pred_model = NSFormer(args)
        self.model = Model(args)

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        self.args.n_z_samples = n_sample
        self.args.n_z_samples_depart = kwargs.get("n_times", n_sample)
        self.args.test_batch_size = condition.shape[0]

        def store_gen_y_at_step_t(config, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = self.model.num_timesteps - idx
            gen_y = (
                y_tile_seq[idx].reshape(
                    config.test_batch_size,
                    int(config.n_z_samples / config.n_z_samples_depart),
                    (config.label_len + config.pred_len),
                    config.c_out,
                )
                # .cpu()
                # .numpy()
            )
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y
            else:
                gen_y_by_batch_list[current_t] = torch.concat(
                    [gen_y_by_batch_list[current_t], gen_y], dim=0
                )
            return gen_y

        gen_y_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]

        batch_x = condition
        batch_y = kwargs.get("seq")

        batch_x_mark = kwargs.get("obs_mark", torch.zeros_like(batch_x))
        batch_y_mark = kwargs.get("seq_mark", torch.zeros_like(batch_y))

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :])
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)

        _, y_0_hat_batch, _, z_sample = self.cond_pred_model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark
        )

        repeat_n = int(self.args.n_z_samples / self.args.n_z_samples_depart)
        y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
        y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
        y_T_mean_tile = y_0_hat_tile
        x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
        x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

        x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
        x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)

        gen_y_box = []
        for _ in range(self.args.n_z_samples_depart):
            y_tile_seq = p_sample_loop(
                self.model,
                x_tile,
                x_mark_tile,
                y_0_hat_tile,
                y_T_mean_tile,
                self.model.num_timesteps,
                self.model.alphas,
                self.model.one_minus_alphas_bar_sqrt,
            )

            gen_y = store_gen_y_at_step_t(
                config=self.args,
                # config_diff=self.model.diffusion_config,
                idx=self.model.num_timesteps,
                y_tile_seq=y_tile_seq,
            )
            gen_y_box.append(gen_y)
        outputs = torch.concat(gen_y_box, dim=1)

        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, :, -self.args.pred_len :, f_dim:]
        # shape: [bs, n_samples, pred_len, dims]
        outputs = outputs.permute(0, 2, 3, 1)
        # shape: [bs, pred_len, dims, n_samples]

        return outputs
    
    def _get_loss(self, batch):
        batch_x = batch["c"]
        batch_y = batch["seq"][:, self.args.obs_len - self.args.label_len :]

        # ! add time features or not?
        batch_x_mark = batch.get("obs_mark", torch.zeros_like(batch_x))
        batch_y_mark = batch.get("seq_mark", torch.zeros_like(batch_y))

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :])
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(
            self.device
        )
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        _, y_0_hat_batch, KL_loss, z_sample = self.cond_pred_model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark
        )

        # loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))
        loss_vae = torch.nn.functional.mse_loss(y_0_hat_batch, batch_y)
        # loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))

        loss_vae_all = loss_vae + self.args.k_z * KL_loss
        # y_0_hat_batch = z_sample

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(self.device)

        y_t_batch = q_sample(
            batch_y,
            y_T_mean,
            self.model.alphas_bar_sqrt,
            self.model.one_minus_alphas_bar_sqrt,
            t,
            noise=e,
        )
        output = self.model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)

        loss = (
            e[:, -self.args.pred_len :, :] - output[:, -self.args.pred_len :, :]
        ).square().mean() + self.args.k_cond * loss_vae_all
    
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        loss = self._get_loss(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        loss = self._get_loss(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.cond_pred_model.parameters()},
            ],
            lr=self.hparams.lr,
        )
