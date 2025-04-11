from argparse import Namespace
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.embed import series_decomp
from src.layers.misc import exists, extract_into_tensor
from src.model.base import BaseModel
from src.model.diffusion.vanilladiffusion import cosine_schedule as cosine_beta_schedule

# from models_diffusion.DDPM_CNNNet import *
# from models_diffusion.DDPM_diffusion_worker import *
# from layers.RevIN import *
# from src.layers.norm import RevIN
from ._dpm_solver import DPM_Solver, NoiseScheduleVP, model_wrapper


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class DPMSolverSampler(object):
    def __init__(self, model, diffusion_worker, **kwargs):
        super().__init__()
        self.model = model
        self.diffusion_worker = diffusion_worker
        # to_torch = (
        #     lambda x: x.clone()
        #     .detach()
        #     .to(torch.float32)
        #     .to(self.model.device)
        # )
        self.register_buffer(
            "alphas_cumprod",
            self.diffusion_worker.alphas_cumprod.clone().detach().to(torch.float32),
        )

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                    )
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        # sampling
        F, L = shape
        size = (batch_size, F, L)

        # print(f'Data shape for DPM-Solver sampling is {size}, sampling steps {S}')

        device = self.diffusion_worker.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            lambda x, t, c: self.model.forward(x, t, c),
            ns,
            model_type="x_start",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )

        dpm_solver = DPM_Solver(model_fn, ns, predict_x0=True, thresholding=False)
        x = dpm_solver.sample(
            img,
            steps=S,
            skip_type="time_uniform",
            method="multistep",
            order=2,
            lower_order_final=True,
        )

        return x.to(device), None


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class Conv1dWithInitialization(nn.Module):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)


class InputConvNetwork(nn.Module):
    def __init__(
        self,
        args,
        inp_num_channel,
        out_num_channel,
        num_layers=3,
        ddpm_channels_conv=None,
    ):
        super(InputConvNetwork, self).__init__()

        self.args = args

        self.inp_num_channel = inp_num_channel
        self.out_num_channel = out_num_channel

        kernel_size = 3
        padding = 1
        if ddpm_channels_conv is None:
            self.channels = args.ddpm_channels_conv
        else:
            self.channels = ddpm_channels_conv
        self.num_layers = num_layers

        self.net = nn.ModuleList()

        if num_layers == 1:
            self.net.append(
                Conv1dWithInitialization(
                    in_channels=self.inp_num_channel,
                    out_channels=self.out_num_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=True,
                )
            )
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    dim_inp = self.inp_num_channel
                else:
                    dim_inp = self.channels
                self.net.append(
                    Conv1dWithInitialization(
                        in_channels=dim_inp,
                        out_channels=self.channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=True,
                    )
                )
                (self.net.append(torch.nn.BatchNorm1d(self.channels)),)
                (self.net.append(torch.nn.LeakyReLU(0.1)),)
                self.net.append(torch.nn.Dropout(0.1, inplace=True))

            self.net.append(
                Conv1dWithInitialization(
                    in_channels=self.channels,
                    out_channels=self.out_num_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=True,
                )
            )

    def forward(self, x=None):
        out = x
        for m in self.net:
            out = m(out)

        return out


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        # print("1", np.shape(x))
        x = self.projection1(x)
        # print("2", np.shape(x))
        x = F.silu(x)
        x = self.projection2(x)
        # print("3", np.shape(x))
        x = F.silu(x)
        # 1 torch.Size([64, 128])
        # 2 torch.Size([64, 128])
        # 3 torch.Size([64, 128])
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class My_DiffusionUnet_v0(nn.Module):
    def __init__(self, args, num_vars, seq_len, pred_len, net_id=0):
        super(My_DiffusionUnet_v0, self).__init__()

        self.args = args

        self.num_vars = num_vars

        self.seq_len = seq_len
        self.label_len = args.label_len
        self.pred_len = pred_len

        # self.device = args.device

        self.net_id = net_id
        self.smoothed_factors = args.smoothed_factors
        self.num_bridges = len(self.smoothed_factors) + 1

        self.dim_diff_step = args.ddpm_dim_diff_steps
        # time_embed_dim = self.dim_diff_step
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=args.diff_steps,
            embedding_dim=self.dim_diff_step,
        )
        self.act = lambda x: x * torch.sigmoid(x)

        self.use_features_proj = False
        self.channels = args.ddpm_inp_embed
        if self.use_features_proj:
            self.feature_projection = nn.Sequential(
                nn.Linear(self.num_vars, self.channels),
            )
            self.input_projection = InputConvNetwork(
                args, self.channels, self.channels, num_layers=args.ddpm_layers_inp
            )
        else:
            self.input_projection = InputConvNetwork(
                args, self.num_vars, self.channels, num_layers=args.ddpm_layers_inp
            )

        self.dim_intermediate_enc = args.ddpm_channels_fusion_I
        self.enc_conv = InputConvNetwork(
            args,
            self.channels + self.dim_diff_step,
            self.dim_intermediate_enc,
            num_layers=args.ddpm_layers_I,
        )

        if self.args.individual:
            self.cond_projections = nn.ModuleList()

        if args.ablation_study_F_type == "Linear":
            if self.args.individual:
                for i in range(self.num_vars):
                    self.cond_projections.append(nn.Linear(self.seq_len, self.pred_len))
                    self.cond_projections[i].weight = nn.Parameter(
                        (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                    )
            else:
                self.cond_projection = nn.Linear(self.seq_len, self.pred_len)

        elif args.ablation_study_F_type == "CNN":
            if self.args.individual:
                for i in range(self.num_vars):
                    self.cond_projections.append(nn.Linear(self.seq_len, self.pred_len))
                    self.cond_projections[i].weight = nn.Parameter(
                        (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                    )
            else:
                self.cond_projection = nn.Linear(self.seq_len, self.pred_len)

            self.cnn_cond_projections = InputConvNetwork(
                args,
                self.num_vars,
                self.pred_len,
                num_layers=args.cond_ddpm_num_layers,
                ddpm_channels_conv=args.cond_ddpm_channels_conv,
            )
            self.cnn_linear = nn.Linear(self.seq_len, self.num_vars)

        if self.net_id == self.num_bridges - 1:
            if args.use_ar_init:
                self.combine_conv = InputConvNetwork(
                    args,
                    self.dim_intermediate_enc + 2 * self.num_vars,
                    self.num_vars,
                    num_layers=args.ddpm_layers_II,
                    ddpm_channels_conv=args.dec_channel_nums,
                )
            else:
                self.combine_conv = InputConvNetwork(
                    args,
                    self.dim_intermediate_enc + 1 * self.num_vars,
                    self.num_vars,
                    num_layers=args.ddpm_layers_II,
                    ddpm_channels_conv=args.dec_channel_nums,
                )
        else:
            if args.use_ar_init:
                self.combine_conv = InputConvNetwork(
                    args,
                    self.dim_intermediate_enc + 3 * self.num_vars,
                    self.num_vars,
                    num_layers=args.ddpm_layers_II,
                    ddpm_channels_conv=args.dec_channel_nums,
                )
            else:
                self.combine_conv = InputConvNetwork(
                    args,
                    self.dim_intermediate_enc + 2 * self.num_vars,
                    self.num_vars,
                    num_layers=args.ddpm_layers_II,
                    ddpm_channels_conv=args.dec_channel_nums,
                )

        args.modes1 = 10
        args.compression = 0
        args.ratio = 1
        args.mode_type = 0

    def forward(
        self, xt, timesteps, cond=None, ar_init=None, future_gt=None, mask=None
    ):
        # print("cond", np.shape(cond))
        if ar_init is None:
            ar_init = cond[:, :, -self.pred_len :]

        if self.net_id == self.num_bridges - 1:
            prev_scale_out = None
        else:
            prev_scale_out = cond[:, :, self.seq_len : self.seq_len + self.pred_len]
        cond = cond[:, :, : self.seq_len]

        # xt： B, N, H
        # timesteps: torch.Size([64])
        # cond B, N, L

        # diffusion_emb = timestep_embedding(timesteps, self.dim_diff_step)
        # diffusion_emb = self.time_embed(diffusion_emb)
        diffusion_emb = self.diffusion_embedding(timesteps.long())
        diffusion_emb = self.act(diffusion_emb)
        diffusion_emb = diffusion_emb.unsqueeze(-1).repeat(1, 1, np.shape(xt)[-1])

        if self.use_features_proj:
            xt = self.feature_projection(xt.permute(0, 2, 1)).permute(0, 2, 1)

        out = self.input_projection(xt)
        out = self.enc_conv(torch.cat([diffusion_emb, out], dim=1))

        if self.args.individual:
            pred_out = torch.zeros(
                [xt.size(0), self.num_vars, self.pred_len], dtype=xt.dtype
            ).to(xt.device)
            for i in range(self.num_vars):
                pred_out[:, i, :] = self.cond_projections[i](cond[:, i, :])
        else:
            pred_out = self.cond_projection(cond)

        if self.args.ablation_study_F_type == "CNN":
            temp_out = self.cnn_cond_projections(cond)
            pred_out += self.cnn_linear(temp_out).permute(0, 2, 1)

        # =====================================================================
        # mixing matrix

        if self.args.use_future_mixup and future_gt is not None:
            y_clean = future_gt[:, :, -self.pred_len :]
            if self.args.beta_dist_alpha > 0:
                rand_for_mask = np.random.beta(
                    self.args.beta_dist_alpha,
                    self.args.beta_dist_alpha,
                    size=np.shape(y_clean),
                )
                rand_for_mask = torch.tensor(rand_for_mask, dtype=torch.long).to(
                    xt.device
                )
            else:
                rand_for_mask = torch.rand_like(y_clean).to(xt.device)

                # ! Ablation study?
                # if self.args.ablation_study_masking_type == "hard":
                #     tau = self.args.ablation_study_masking_tau
                #     hard_indcies = rand_for_mask > tau
                #     data_random_hard_making = rand_for_mask
                #     data_random_hard_making[hard_indcies] = 1
                #     data_random_hard_making[~hard_indcies] = 0
                #     rand_for_mask = data_random_hard_making
                # if self.args.ablation_study_masking_type == "segment":
                #     tau = self.args.ablation_study_masking_tau
                #     segment_mask = torch.from_numpy(noise_mask(pred_out[:,0,:], masking_ratio=tau, lm=24)).to(yn.device)
                #     # print("masking_ratio", tau, torch.sum(segment_mask))
                #     segment_mask = segment_mask.unsqueeze(1).repeat(1, np.shape(pred_out)[1], 1)
                #     rand_for_mask = segment_mask.float()
            pred_out = (
                rand_for_mask * pred_out
                + (1 - rand_for_mask) * future_gt[:, :, -self.pred_len :]
            )

        # ar_init = None
        if self.args.use_ar_init:
            if self.net_id == self.num_bridges - 1:
                out = torch.cat([out, pred_out, ar_init], dim=1)
            else:
                out = torch.cat([out, pred_out, ar_init, prev_scale_out], dim=1)
        else:
            if self.net_id == self.num_bridges - 1:
                out = torch.cat([out, pred_out], dim=1)
            else:
                out = torch.cat([out, pred_out, prev_scale_out], dim=1)

        if self.args.use_residual:
            out = self.combine_conv(out) + pred_out
        else:
            out = self.combine_conv(out)

        # SHOULD BE  B, N, L
        return out


class RevIN(nn.Module):
    def __init__(
        self, args, num_features: int, eps=1e-5, affine=True, subtract_last=False
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.args = args
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "test_norm":
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            if self.args.subtract_short_terms:
                self.mean = torch.mean(
                    x[:, -self.args.label_len :, :], dim=dim2reduce, keepdim=True
                ).detach()
            else:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Diffusion_Worker(nn.Module):
    def __init__(self, args, u_net=None):
        super(Diffusion_Worker, self).__init__()

        self.args = args
        # self.device = args.device

        self.parameterization = args.parameterization
        assert self.parameterization in ["noise", "x_start"], (
            'currently only supporting "eps" and "x0"'
        )

        self.net = u_net

        self.diff_train_steps = args.diff_steps
        self.diff_test_steps = self.diff_train_steps

        # self.beta_start = 1e-4 # 1e4
        # self.beta_end = 2e-2

        try:
            self.beta_start = args.beta_start  # 1e4
            self.beta_end = args.beta_end
        except:
            self.beta_start = 1e-4  # 1e4
            self.beta_end = 2e-2

        try:
            self.beta_schedule = args.beta_schedule  # default "cosine"
        except:
            self.beta_schedule = "cosine"

        self.v_posterior = 0.0
        self.original_elbo_weight = 0.0
        self.l_simple_weight = 1

        self.loss_type = "l2"

        self.set_new_noise_schedule(
            None,
            self.beta_schedule,
            self.diff_train_steps,
            self.beta_start,
            self.beta_end,
        )

        self.clip_denoised = True

        self.total_N = len(self.alphas_cumprod)
        self.T = 1.0
        self.eps = 1e-5

        self.nn = u_net

    def set_new_noise_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        diff_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            if beta_schedule == "linear":
                betas = np.linspace(beta_start, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(beta_start**0.5, beta_end**0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - beta_start) / (np.exp(-betas) + 1) + beta_start
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)["betas"].numpy()
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = beta_start
        self.linear_end = beta_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "noise":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x_start":
            lvlb_weights = (
                0.8
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: self.scaling_noise * torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def train_forward(
        self, x0, x1, mask=None, condition=None, ar_init=None, return_mean=True
    ):
        x = x0
        # cond_ts = condition

        # Feed normalized inputs ith shape of [bsz, feas, seqlen]
        # both x and cond are two time seires

        B = np.shape(x)[0]
        t = (
            torch.randint(
                0,
                self.num_timesteps,
                size=[
                    (B + 1) // 2,
                ],
            )
            .long()
            .to(x.device)
        )
        t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)
        t = t[:B]

        noise = torch.randn_like(x)
        x_k = self.q_sample(x_start=x, t=t, noise=noise)

        model_out = self.net(
            x_k, t, cond=condition, ar_init=ar_init, future_gt=x0, mask=None
        )

        if self.parameterization == "noise":
            target = noise
        elif self.parameterization == "x_start":
            target = x
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported"
            )

        # in the submission version, we calculate time first, and then calculate variable
        # f_dim = -1 if self.args.features == 'MS' else 0

        # if self.args.features == "S":
        model_out = model_out.permute(0, 2, 1)
        target = target.permute(0, 2, 1)

        if return_mean:
            loss = (
                self.get_loss(model_out[:, :, :], target[:, :, :], mean=False)
                .mean(dim=2)
                .mean(dim=1)
            )
            loss_simple = loss.mean() * self.l_simple_weight

            loss_vlb = (self.lvlb_weights[t] * loss).mean()
            loss = loss_simple + self.original_elbo_weight * loss_vlb

        else:
            # loss = self.get_loss(model_out[:,:,:], target[:,:,:], mean=False).mean(dim=-1).mean(dim=0)
            # print(">>>", np.shape(loss))

            # ! Loss function = mse
            loss = F.mse_loss(model_out[:, :, :], target[:, :, :])
            # elif self.args.opt_loss_type == "smape":
            #     criterion = smape_loss()
            #     loss = criterion(model_out[:,:,:], target[:,:,:])
        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond_ts, ar_init, clip_denoised: bool):
        model_out = self.nn(
            x, t, cond=cond_ts, ar_init=ar_init, future_gt=None, mask=None
        )

        if self.parameterization == "noise":
            # model_out.clamp_(-10., 10.)
            model_out = self.predict_start_from_noise(x, t=t, noise=model_out)

        x_recon = model_out

        if clip_denoised:
            x_recon.clamp_(-self.args.our_ddpm_clip, self.args.our_ddpm_clip)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self, x, t, cond=None, ar_init=None, clip_denoised=True, repeat_noise=False
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, cond_ts=cond, ar_init=ar_init, clip_denoised=clip_denoised
        )

        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def ddpm_sampling(
        self,
        x1=None,
        mask=None,
        cond=None,
        ar_init=None,
        store_intermediate_states=False,
    ):
        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.

        x = x1
        b, d, _ = np.shape(x1)
        shape = (b, d, np.shape(x)[-1])
        timeseries = torch.randn(shape, device=x.device)
        intermediates = [timeseries.permute(0, 2, 1)]  # return bsz, seqlen, fea_dim

        for i in reversed(range(0, self.num_timesteps)):
            timeseries = self.p_sample(
                timeseries,
                torch.full((b,), i, device=x.device, dtype=torch.long),
                cond=cond,
                ar_init=ar_init,
                clip_denoised=self.clip_denoised,
            )
            if store_intermediate_states:
                intermediates.append(timeseries.permute(0, 2, 1))

        outs = timeseries

        return outs


class BaseMapping(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, args, seq_len=None, pred_len=None):
        super(BaseMapping, self).__init__()

        self.args = args
        # self.device = args.device

        if seq_len is None:
            self.seq_len = args.seq_len
            self.pred_len = args.pred_len
        else:
            self.seq_len = seq_len
            self.pred_len = pred_len

        # Decompsition Kernel Size
        # kernel_size = args.kernel_size
        self.individual = args.individual
        self.channels = args.num_vars

        if self.individual:
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

        else:
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.rev = (
            RevIN(
                args,
                args.num_vars,
                affine=args.affine,
                subtract_last=args.subtract_last,
            ).to(args.device)
            if args.use_window_normalization
            else None
        )

    def train_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, train_val=None):
        # return [Batch, Output length, Channel]

        if self.args.use_window_normalization:
            x_enc_i = self.rev(x_enc, "norm")  #
            # x_dec_i = self.rev(x_dec[:,-self.pred_len:,:], 'test_norm')
        else:
            x_enc_i = x_enc
            # x_dec_i = x_dec

        x = x_enc_i

        trend_init = x
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            trend_output = self.Linear_Trend(trend_init)

        x = trend_output
        outputs = x.permute(0, 2, 1)

        outputs = (
            self.rev(outputs, "denorm")
            if self.args.use_window_normalization
            else outputs
        )

        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -self.pred_len :, f_dim:]
        ground_truth = x_dec[:, -self.pred_len :, f_dim:]
        # ground_truth = x_dec[:, -self.pred_len :, f_dim:].to(self.device)

        # ! Loss function = MSE
        loss = F.mse_loss(outputs, ground_truth)

        # elif self.args.opt_loss_type == "smape":
        #     criterion = smape_loss()
        #     loss = criterion(outputs, ground_truth)
        return loss

    def test_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.args.use_window_normalization:
            x_enc_i = self.rev(x_enc, "norm")
            # x_dec_i = self.rev(x_dec[:,-self.pred_len:,:], 'test_norm')
        else:
            x_enc_i = x_enc
            # x_dec_i = x_dec

        x = x_enc_i

        trend_init = x
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            trend_output = self.Linear_Trend(trend_init)

        x = trend_output
        outputs = x.permute(0, 2, 1)

        outputs = (
            self.rev(outputs, "denorm")
            if self.args.use_window_normalization
            else outputs
        )

        f_dim = -1 if self.args.features == "MS" else 0
        return outputs[:, -self.pred_len :, f_dim:]


class MrDiff(BaseModel):
    """
    Decomposition-Linear
    """

    def __init__(
        self,
        seq_len,
        obs_len,
        seq_dim=1,
        n_diff_steps=100,
        smoothed_factors=[5, 25, 51],
        affine=False,
        subtract_last=True,
        use_window_normalization=True,
        use_future_mixup=True,
        use_ar_init=False,
        use_residual=True,
        type_sampler="dpm",
        lr=1e-3,
        weight_decay=0.00001,
        individual=True,
        ddpm_inp_embed=64,
        ddpm_layers_inp=10,
        ddpm_dim_diff_steps=256,
        ddpm_channels_conv=128,
        ddpm_channels_fusion_I=256,
        ddpm_layers_I=5,
        ddpm_layers_II=10,
        dec_channel_nums=256,
        cond_ddpm_num_layers=5,
        cond_ddpm_channels_conv=256,
        our_ddpm_clip=100,
        parameterization="x_start",
        beta_dist_alpha=-1,
        **kwargs,
    ):
        super(MrDiff, self).__init__()
        self.condition='predict'
        self.save_hyperparameters()
        args = Namespace(**self.hparams_initial)
        args.seq_len = obs_len
        args.pred_len = seq_len
        args.label_len = obs_len
        args.num_vars = seq_dim
        args.diff_steps = n_diff_steps
        args.features = "S" if seq_dim == 1 else "M"
        args.ablation_study_F_type = "Linear"
        args.device = self.device
        self.args = args
        # self.device = args.device

        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        self.num_vars = args.num_vars
        self.smoothed_factors = args.smoothed_factors
        self.linear_history_len = args.seq_len  # args.linear_history_lens

        self.num_bridges = len(args.smoothed_factors) + 1

        self.base_models = nn.ModuleList(
            [
                BaseMapping(
                    args, seq_len=self.linear_history_len, pred_len=self.pred_len
                )
                for i in range(self.num_bridges)
            ]
        )
        self.decompsitions = nn.ModuleList(
            [series_decomp(i) for i in self.smoothed_factors]
        )

        self.u_nets = nn.ModuleList(
            [
                My_DiffusionUnet_v0(
                    args, self.num_vars, self.seq_len, self.pred_len, net_id=i
                )
                for i in range(self.num_bridges)
            ]
        )

        self.diffusion_workers = nn.ModuleList(
            [Diffusion_Worker(args, self.u_nets[i]) for i in range(self.num_bridges)]
        )

        self.rev = (
            RevIN(
                args,
                args.num_vars,
                affine=args.affine,
                subtract_last=args.subtract_last,
            ).to(args.device)
            if args.use_window_normalization
            else None
        )

        if args.diff_steps < 100:
            args.type_sampler == "none"
        if args.type_sampler == "none":
            pass
        elif args.type_sampler == "dpm":
            assert self.args.parameterization == "x_start"
            self.samplers = [
                DPMSolverSampler(self.u_nets[i], self.diffusion_workers[i])
                for i in range(self.num_bridges)
            ]

    def obatin_multi_trends(self, batch_x):
        # batch_x: (B, N, L)

        batch_x = batch_x.permute(0, 2, 1)

        # batch_x: (B, L, N)
        # print("self.smoothed_factors", self.smoothed_factors)

        batch_x_trends = []
        batch_x_trend_0 = batch_x
        for i in range(self.num_bridges - 1):
            _, batch_x_trend = self.decompsitions[i](batch_x_trend_0)
            # print("batch_x_trend", np.shape(batch_x_trend))

            # plt.plot(batch_x_trend[0,0,:].cpu().numpy())

            batch_x_trends.append(batch_x_trend.permute(0, 2, 1))
            batch_x_trend_0 = batch_x_trend

        # plt.savefig("demo_haha.png")
        return batch_x_trends

    def train_forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        return_mean=True,
        meta_weights=None,
        train_val=False,
    ):
        if self.args.use_window_normalization:
            x_enc_i = self.rev(x_enc, "norm")
            x_dec_i = self.rev(x_dec[:, -self.pred_len :, :], "test_norm")
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec

        x_past = x_enc_i
        x_future = x_dec_i[:, -self.args.pred_len :, :]

        # print(">>>>>", np.shape(x_past), np.shape(x_future))

        future_trends = self.obatin_multi_trends(x_future.permute(0, 2, 1))
        # each trend: B, N, L
        future_trends = [trend_i.permute(0, 2, 1) for trend_i in future_trends]
        # each trend: B, L, N
        future_xT = torch.randn_like(x_future)
        future_trends.append(future_xT)

        # ==================================
        # history trends
        past_trends = self.obatin_multi_trends(x_past.permute(0, 2, 1))
        past_trends = [trend_i.permute(0, 2, 1) for trend_i in past_trends]

        # ==================================
        # ar-init trends
        ar_init_trends = []
        for i in range(self.num_bridges):
            if i == 0:
                future_trend_i = x_future
                past_trend_i = x_past
                future_trend_i = (
                    self.rev(future_trend_i, "denorm")
                    if self.args.use_window_normalization
                    else future_trend_i
                )
                past_trend_i = (
                    self.rev(past_trend_i, "denorm")
                    if self.args.use_window_normalization
                    else past_trend_i
                )
                linear_guess = (
                    self.base_models[i]
                    .test_forward(
                        past_trend_i[:, -self.linear_history_len :, :],
                        x_mark_enc,
                        future_trend_i,
                        x_mark_dec,
                    )
                    .detach()
                )
                linear_guess = (
                    self.rev(linear_guess, "test_norm")
                    if self.args.use_window_normalization
                    else linear_guess
                )
            else:
                future_trend_i = future_trends[i - 1]
                past_trend_i = past_trends[i - 1]
                future_trend_i = (
                    self.rev(future_trend_i, "denorm")
                    if self.args.use_window_normalization
                    else future_trend_i
                )
                past_trend_i = (
                    self.rev(past_trend_i, "denorm")
                    if self.args.use_window_normalization
                    else past_trend_i
                )
                linear_guess_i = (
                    self.base_models[i]
                    .test_forward(
                        past_trend_i[:, -self.linear_history_len :, :],
                        x_mark_enc,
                        future_trend_i,
                        x_mark_dec,
                    )
                    .detach()
                )
                linear_guess_i = (
                    self.rev(linear_guess_i, "test_norm")
                    if self.args.use_window_normalization
                    else linear_guess_i
                )
                ar_init_trends.append(linear_guess_i)

        total_loss = []
        for i in range(self.num_bridges):
            # X0 clean
            # X1: occupied [bsz, fea, seq_len]

            if i == 0:
                X1 = future_trends[0].permute(0, 2, 1)
                X0 = x_future.permute(0, 2, 1)

                cond = torch.cat([x_past.permute(0, 2, 1), X1], dim=-1)

                MASK = torch.ones((np.shape(X1)[0], self.num_vars, self.pred_len)).to(
                    X1.device
                )

                loss_i = self.diffusion_workers[i].train_forward(
                    X0,
                    X1,
                    mask=MASK,
                    condition=cond,
                    ar_init=linear_guess.permute(0, 2, 1),
                    return_mean=return_mean,
                )
                total_loss.append(loss_i)

            else:
                X1 = future_trends[i].permute(0, 2, 1)

                if i == self.num_bridges - 1:
                    cond = past_trends[i - 1].permute(0, 2, 1)
                else:
                    cond = torch.cat([past_trends[i - 1].permute(0, 2, 1), X1], dim=-1)

                X0 = future_trends[i - 1].permute(0, 2, 1)

                MASK = torch.ones((np.shape(X1)[0], self.num_vars, self.pred_len)).to(
                    X1.device
                )
                # print("past_trends[i-1]", np.shape(past_trends[i-1]), np.shape(ar_init_trends[i-1]))

                loss_i = self.diffusion_workers[i].train_forward(
                    X0,
                    X1,
                    mask=MASK,
                    condition=cond,
                    ar_init=ar_init_trends[i - 1].permute(0, 2, 1),
                    return_mean=return_mean,
                )
                total_loss.append(loss_i)

        if return_mean:
            total_loss = torch.stack(total_loss).mean()
        else:
            if meta_weights is not None:
                total_loss = torch.stack(total_loss).reshape(-1)
                train_loss_tmp = [w * total_loss[i] for i, w in enumerate(meta_weights)]
                total_loss = sum(train_loss_tmp)
            else:
                total_loss = torch.stack(total_loss).reshape(-1)

        return total_loss

    def test_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.args.use_window_normalization:
            x_enc_i = self.rev(x_enc, "norm")
            x_dec_i = self.rev(x_dec[:, -self.pred_len :, :], "test_norm")
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec

        x_past = x_enc_i
        x_future = x_dec_i[:, -self.args.pred_len :, :]

        future_trends = self.obatin_multi_trends(x_future.permute(0, 2, 1))
        # each trend: B, N, L
        future_trends = [trend_i.permute(0, 2, 1) for trend_i in future_trends]
        # each trend: B, L, N
        future_xT = torch.randn_like(x_future)
        future_trends.append(future_xT)

        # ==================================
        # history trends
        past_trends = self.obatin_multi_trends(x_past.permute(0, 2, 1))
        past_trends = [trend_i.permute(0, 2, 1) for trend_i in past_trends]

        # ==================================
        # ar-init trends
        ar_init_trends = []
        for i in range(self.num_bridges):
            if i == 0:
                future_trend_i = x_future
                past_trend_i = x_past
                future_trend_i = (
                    self.rev(future_trend_i, "denorm")
                    if self.args.use_window_normalization
                    else future_trend_i
                )
                past_trend_i = (
                    self.rev(past_trend_i, "denorm")
                    if self.args.use_window_normalization
                    else past_trend_i
                )
                linear_guess = self.base_models[i].test_forward(
                    past_trend_i[:, -self.linear_history_len :, :],
                    x_mark_enc,
                    future_trend_i,
                    x_mark_dec,
                )
                linear_guess = (
                    self.rev(linear_guess, "test_norm")
                    if self.args.use_window_normalization
                    else linear_guess
                )
            else:
                future_trend_i = future_trends[i - 1]
                past_trend_i = past_trends[i - 1]
                future_trend_i = (
                    self.rev(future_trend_i, "denorm")
                    if self.args.use_window_normalization
                    else future_trend_i
                )
                past_trend_i = (
                    self.rev(past_trend_i, "denorm")
                    if self.args.use_window_normalization
                    else past_trend_i
                )
                linear_guess_i = self.base_models[i].test_forward(
                    past_trend_i[:, -self.linear_history_len :, :],
                    x_mark_enc,
                    future_trend_i,
                    x_mark_dec,
                )
                linear_guess_i = (
                    self.rev(linear_guess_i, "test_norm")
                    if self.args.use_window_normalization
                    else linear_guess_i
                )
                ar_init_trends.append(linear_guess_i)

        B, nF, nL = np.shape(x_past)[0], self.num_vars, self.pred_len
        if self.args.features in ["MS"]:
            nF = 1
        shape = [nF, nL]

        # ==================================
        # history trends

        all_outs = []
        for i in range(self.args.sample_times):
            X1 = future_xT.permute(0, 2, 1)
            res_out = X1

            for j in reversed(range(0, self.num_bridges)):
                MASK = torch.ones(
                    (np.shape(x_past)[0], self.num_vars, self.pred_len)
                ).to(X1.device)

                if self.args.type_sampler == "none":
                    if j == 0:
                        cond = torch.cat([x_past.permute(0, 2, 1), X1], dim=-1)
                        X1 = self.diffusion_workers[j].ddpm_sampling(
                            X1,
                            mask=MASK,
                            cond=cond,
                            ar_init=linear_guess.permute(0, 2, 1),
                        )
                    else:
                        if j == self.num_bridges - 1:
                            cond = past_trends[j - 1].permute(0, 2, 1)
                        else:
                            # cond = torch.cat([x_past, X1], dim=-1)

                            cond = torch.cat(
                                [past_trends[j - 1].permute(0, 2, 1), X1], dim=-1
                            )

                        X1 = self.diffusion_workers[j].ddpm_sampling(
                            X1,
                            mask=MASK,
                            cond=cond,
                            ar_init=ar_init_trends[j - 1].permute(0, 2, 1),
                        )

                else:
                    start_code = torch.randn((B, nF, nL), device=X1.device)
                    if j == 0:
                        cond = torch.cat([x_past.permute(0, 2, 1), X1], dim=-1)
                        cA = cond
                        cB = linear_guess.permute(0, 2, 1)
                    else:
                        if j == self.num_bridges - 1:
                            cond = past_trends[j - 1].permute(0, 2, 1)
                        else:
                            cond = torch.cat(
                                [past_trends[j - 1].permute(0, 2, 1), X1], dim=-1
                            )
                        cA = cond
                        cB = ar_init_trends[j - 1].permute(0, 2, 1)

                    S = self.args.nfe
                    samples_ddim, _ = self.samplers[j].sample(
                        S=S,
                        conditioning=torch.cat([cA, cB], dim=-1),
                        batch_size=B,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=1.0,
                        unconditional_conditioning=None,
                        eta=0.0,
                        x_T=start_code,
                    )
                    X1 = samples_ddim

                if j == 0:
                    res_out = X1

            outs_i = res_out.permute(0, 2, 1).to(X1.device)

            outs_i = (
                self.rev(outs_i, "denorm")
                if self.args.use_window_normalization
                else outs_i
            )
            all_outs.append(outs_i)
        all_outs = torch.stack(all_outs, dim=-1)
        outputs = all_outs

        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -self.args.pred_len :, f_dim:]

        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        batch_x = batch["c"]

        batch_y = batch["seq"]

        loss = self.train_forward(batch_x, None, batch_y, None)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x = batch["c"]

        batch_y = batch["seq"]

        loss = self.train_forward(batch_x, None, batch_y, None)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        # return loss

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        self.args.sample_times = n_sample
        self.args.nfe = kwargs.get("nfe", 5)
        batch_x = condition
        batch_y = kwargs["seq"]
        preds = self.test_forward(batch_x, None, batch_y, None)
        # shape: [bs, seq_len, seq_dim, sample_times]
        return preds
