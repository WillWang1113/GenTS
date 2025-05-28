from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

from src.common._modules import series_decomp
from src.model.base import BaseModel

from ._backbones import (
    BaseMapping,
    Diffusion_Worker,
    DPMSolverSampler,
    My_DiffusionUnet_v0,
    RevIN,
)


class MrDiff(BaseModel):
    ALLOW_CONDITION = ["predict"]

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
        condition="predict",
        **kwargs,
    ):
        super(MrDiff, self).__init__(seq_len, seq_dim, condition, **kwargs)
        self.condition = "predict"
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
