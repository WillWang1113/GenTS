from argparse import Namespace
from typing import Dict, Tuple
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

from gents.evaluation.model_free.distribution_distance import WassersteinDistances
from gents.model.base import BaseModel
from gents.common._modules import FinalTanh, NeuralCDE, RNNLayer
from ._backbones import (
    Multi_Layer_ODENetwork,
    build_model_tabular_nonlinear,
    run_latent_ctfp_model5_2,
    create_regularization_fns,
)


class GTGAN(BaseModel):
    """`GT-GAN <https://proceedings.neurips.cc/paper_files/paper/2022/file/f03ce573aa8bce26f77b76f1cb9ee979-Paper-Conference.pdf>`__: General Purpose Time Series Synthesis with Generative Adversarial Networks

    Adapted from the `official codes <https://github.com/Jinsung-Jeon/GT-GAN>`__

    Args:
        seq_len (int): Target sequence length.
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given conditions, should be one of `ALLOW_CONDITION`. Defaults to None.
        hidden_size (int, optional): Hidden size of models. Defaults to 64.
        num_layers_r (int, optional): Recovery network layers. Defaults to 2.
        num_layers_d (int, optional): Discriminator network layers. Defaults to 2.
        num_layers_mlp (int, optional): Embedder ODE MLP layers. Defaults to 3.
        x_hidden (int, optional): Hidden size of data embedding layer in recovery and discriminator networks. Defaults to 48.
        last_activation_r (str, optional): Last activation function in recovery network. Choose from `["identity", "softplus", "tanh","sigmoid"]`. Defaults to "identity".
        last_activation_d (str, optional): Last activation function in discriminator network. Choose from `["identity", "softplus", "tanh","sigmoid"]`. Defaults to "sigmoid".
        solver (str, optional): Solver in generator network. Choose from `['euler','Euler', 'RK2', 'RK4', 'RK23', 'Sym12Async', 'RK12','Dopri5','dopri5','rk4','sym12async','adalf', 'fixedstep_sym12async','fixedstep_adalf']` Defaults to "sym12async".
        atol (float, optional): Absolute tolerance . Defaults to 1e-3.
        rtol (float, optional): Relative tolerance . Defaults to 1e-3.
        time_length (float, optional): Maximum ODE time (T) in generator. Defaults to 1.0.
        train_T (bool, optional): Whether ODE time is trainable in generator (CNF). Defaults to True.
        nonlinearity (str, optional): Nonlinearity in generator (CNF). Choose from `['tanh', 'relu', 'softplus', 'elu', 'swish', 'sigmoid', 'square', 'identity']`. Defaults to "softplus".
        step_size (float, optional): ODE solver step size in generator (CNF). Defaults to 0.1.
        first_step (float, optional): First step size for adaptive solvers in generator (CNF). Defaults to 0.16667.
        divergence_fn (str, optional): Divergence function in generator (CNF). Choose from `['approximate', 'brute_force']` Defaults to "approximate".
        residual (bool, optional): Residual connection in ODE function in generator (CNF). Defaults to False.
        rademacher (bool, optional): Sample from Rademacher distribution or Gaussian. Defaults to True.
        layer_type (str, optional): ODE layer type in generator (CNF). Choose from `['ignore', 'hyper', 'squash', 'concat', 'concat_v2', 'concatsquash', 'blend', 'concatcoord']` Defaults to "concat".
        reconstruction (float, optional): Loss coefficient for reconstruction. || x - decode(encode(x)) ||. Defaults to 0.01.
        kinetic_energy (float, optional): Loss coefficient for kinetic energy. int_t ||f||_2^2. Defaults to 0.5.
        jacobian_norm2 (float, optional): Loss coefficient for jacobian normalization. int_t ||df/dx||_F^2. Defaults to 0.1.
        directional_penalty (float, optional): Loss coefficient for directional penalty. int_t ||(df/dx)^T f||^2. Defaults to 0.01.
        total_deriv (float, optional): Loss coefficient for total derivation. int_t ||df/dt||^2. Defaults to None.
        activation (str, optional): Activation in CNF. Choose from `['exp', 'softplus', 'identity']`  Defaults to "exp".
        num_iwae_samples (int, optional): Number of samples to train IWAE encoder. Defaults to 1.
        num_blocks (int, optional): Number of stacked CNFs. Defaults to 1.
        batch_norm (bool, optional): Batch normalization. Defaults to False.
        bn_lag (float, optional): Batch normalization lag. Defaults to 0.0.
        cnf_hidden_dims (Tuple[int], optional): CNF hidden size list. Defaults to (32, 64, 64, 32).
        test_solver (str, optional): Generator ODE solver for inference. Defaults to None.
        test_atol (float, optional): Generator atol of ODE solver for inference. Defaults to 0.1.
        test_rtol (float, optional): Generator rtol of ODE solver for inference. Defaults to 0.1.
        test_step_size (float, optional): Generator step size of ODE solver for inference. Defaults to None.
        test_first_step (float, optional): Generator first step size of ODE solver for inference. Defaults to None.
        gamma (float, optional): Loss weight coefficient for loss_g_ue. Defaults to 1.0.
        log_time (int, optional): G update frequency. Defaults to 2.
        lr (Dict[str, float], optional): Learning rate for different networks. ER: embedder and recovery, G: generator, D: discriminator. Defaults to {"ER": 1e-3, "G": 1e-3, "D": 1e-3}.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """

    ALLOW_CONDITION = [None]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        hidden_size: int = 64,
        num_layers_r: int = 2,
        num_layers_d: int = 2,
        num_layers_mlp: int = 3,
        x_hidden: int = 48,
        last_activation_r: str = "identity",
        last_activation_d: str = "sigmoid",
        solver: str = "sym12async",
        atol: float = 1e-3,
        rtol: float = 1e-3,
        time_length: float = 1.0,
        train_T: bool = True,
        nonlinearity: str = "softplus",
        step_size: float = 0.1,
        first_step: float = 0.16667,
        divergence_fn: str = "approximate",
        residual: bool = False,
        rademacher: bool = True,
        layer_type: str = "concat",
        reconstruction: float = 0.01,
        kinetic_energy: float = 0.5,
        jacobian_norm2: float = 0.1,
        directional_penalty: float = 0.01,
        total_deriv: float = None,
        activation: str = "exp",
        num_iwae_samples: int = 1,
        num_blocks: int = 1,
        batch_norm: bool = False,
        bn_lag: float = 0.0,
        cnf_hidden_dims: Tuple[int] = (32, 64, 64, 32),
        test_solver: str = None,
        test_atol: float = 0.1,
        test_rtol: float = 0.1,
        test_step_size: float = None,
        test_first_step: float = None,
        gamma: float = 1.0,
        log_time: int = 2,
        lr: Dict[str, float] = {"ER": 1e-3, "G": 1e-3, "D": 1e-3},
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.args = Namespace(
            effective_shape=hidden_size, dims=cnf_hidden_dims, **self.hparams_initial
        )
        ode_func = FinalTanh(seq_dim, hidden_size, hidden_size, num_layers_mlp)
        self.embedder = NeuralCDE(
            func=ode_func,
            input_channels=seq_dim,
            hidden_channels=hidden_size,
            output_channels=hidden_size,
        )
        self.recovery = Multi_Layer_ODENetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=seq_dim,
            gru_input_size=hidden_size,
            x_hidden=x_hidden,
            num_layer=num_layers_r,
            last_activation=last_activation_r,
            delta_t=0.5,
        )
        regularization_fns, regularization_coeffs = create_regularization_fns(self.args)

        self.generator = build_model_tabular_nonlinear(
            self.args, self.args.effective_shape, regularization_fns=regularization_fns
        )
        self.supervisor = nn.Sequential(
            RNNLayer(hidden_size, hidden_size, hidden_size, num_layers=num_layers_r),
            nn.Sigmoid(),
        )
        self.discriminator = Multi_Layer_ODENetwork(
            input_size=seq_dim,
            hidden_size=hidden_size,
            output_size=1,
            gru_input_size=hidden_size,
            x_hidden=x_hidden,
            last_activation=last_activation_d,
            num_layer=num_layers_d,
            delta_t=0.5,
        )
        self.gamma = gamma

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        # for n, p in self.named_parameters():
        # print(n, p.device)
        # batch = dataset[dataset_size]
        # x = batch['data'].to(device)
        # train_coeffs = batch['inter']#.to(device)
        # original_x = batch['original_data'].to(device)
        # obs = kwargs.get("t")
        obs = torch.arange(self.hparams.seq_len).to(self.device).float()
        obs = repeat(obs, "t -> b t", b=n_sample)
        assert obs is not None
        assert obs.shape[0] == n_sample
        # x = x[:, :, :-1]
        z = torch.randn(n_sample, self.hparams.seq_len, self.args.effective_shape).to(
            self.device
        )
        time = torch.FloatTensor(list(range(self.hparams.seq_len))).to(self.device)

        # final_index = (torch.ones(n_sample) * self.hparams.seq_len - 1).to(self.device)

        ###########################################
        # time = torch.FloatTensor(list(range(24))).cuda()
        times = time
        times = times.unsqueeze(0)
        times = times.unsqueeze(2)
        times = times.repeat(n_sample, 1, 1)
        h_hat = run_latent_ctfp_model5_2(
            self.args, self.generator, z, times, self.device, z=True
        )
        # print(obs.device)
        # print(h_hat.device)
        # print(next(self.recovery.parameters()).device)
        x_hat = self.recovery(h_hat, obs)
        ###########################################
        # h_hat = run_latent_ctfp_model5_2(self.args, self.generator, z, times, self.device, z=True)
        ###################################
        # x_hat = self.recovery(h_hat, obs)
        return x_hat

    def configure_optimizers(self):
        optimizer_er = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=self.hparams.lr["ER"],
        )
        optimizer_gs = torch.optim.Adam(
            self.generator.parameters(), lr=self.hparams.lr["G"]
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.lr["D"]
        )
        return [optimizer_er, optimizer_gs, optimizer_d], []

    def training_step(self, batch, batch_idx):
        max_steps = self.trainer.max_epochs
        x = batch["seq"]
        x = x.masked_fill(~batch["data_mask"], float("nan"))

        # cond = batch.get("c", None)
        # if (cond is not None) and (self.condition == "impute"):
        #     x = x.masked_fill(cond.bool(), float("nan"))
        batch_size = x.shape[0]
        t = batch["t"]
        time = torch.arange(x.shape[1]).to(x)
        times = time.unsqueeze(0)
        times = times.unsqueeze(2)
        times = times.repeat(batch_size, 1, 1)
        final_index = (torch.ones(batch_size) * x.shape[1] - 1).to(self.device)

        optimizer_er, optimizer_gs, optimizer_d = self.optimizers()

        if (
            (self.current_epoch >= 0) and (self.current_epoch < int(1 / 2 * max_steps))
        ) or (
            (self.global_step >= 0)
            and (self.global_step < int(1 / 2 * self.trainer.max_steps))
        ):
            # self.toggle_optimizer(optimizer_er)
            # x = batch['data'].to(device)
            train_coeffs = batch["coeffs"]
            # original_x = batch['original_data'].to(device)
            obs = t
            # x = x[:, :, :-1]
            # time = torch.arange(x.shape[1]).to(x)
            # final_index = (torch.ones(batch_size) * x.shape[1] - 1).to(self.device)
            h = self.embedder(time, train_coeffs, final_index)
            x_tilde = self.recovery(h, obs)
            x_no_nan = x[~torch.isnan(x)]
            x_tilde_no_nan = x_tilde[~torch.isnan(x)]
            loss_e_t0 = self._loss_e_t0(x_tilde_no_nan, x_no_nan)
            loss_e_0 = self._loss_e_0(loss_e_t0)
            optimizer_er.zero_grad()
            self.manual_backward(loss_e_0)
            # loss_e_0.backward()
            optimizer_er.step()
            # self.untoggle_optimizer(optimizer_er)
            self.log("loss_e_0", loss_e_0)
        else:
            # self.toggle_optimizer(optimizer_d)
            for _ in range(2):
                # self.generator.train()
                # self.supervisor.train()
                # self.recovery.train()
                # self.toggle_optimizer(optimizer_d)

                # batch = dataset[batch_size]
                # x = batch['data'].to(device)
                train_coeffs = batch["coeffs"]
                # original_x = batch['original_data'].to(device)
                obs = t
                # x = x[:, :, :-1]
                z = torch.randn(batch_size, x.size(1), self.args.effective_shape).to(x)
                # time = torch.FloatTensor(list(range(24))).cuda()
                # final_index = (torch.ones(batch_size) * 23).cuda()
                h = self.embedder(time, train_coeffs, final_index)
                # times = time
                # times = times.unsqueeze(0)
                # times = times.unsqueeze(2)
                # times = times.repeat(obs.shape[0], 1, 1)
                h_hat = run_latent_ctfp_model5_2(
                    self.args, self.generator, z, times, self.device, z=True
                )
                x_real = self.recovery(h, obs)
                x_fake = self.recovery(h_hat, obs)
                y_fake = self.discriminator(x_fake, obs)
                y_real = self.discriminator(x_real, obs)
                loss_d = self._loss_d2(y_real, y_fake)

                if loss_d.item() > 0.15:
                    optimizer_d.zero_grad()
                    self.manual_backward(loss_d)
                    # loss_d.backward()
                    optimizer_d.step()
                    # torch.cuda.empty_cache()
                # self.untoggle_optimizer(optimizer_d)
                self.log("loss_d", loss_d)
                # self.untoggle_optimizer(optimizer_d)

                # self.toggle_optimizer(optimizer_er)
                #############Recovery######################
                h = self.embedder(time, train_coeffs, final_index)
                x_tilde = self.recovery(h, obs)

                x_no_nan = x[~torch.isnan(x)]
                x_tilde_no_nan = x_tilde[~torch.isnan(x)]
                loss_e_t0 = self._loss_e_t0(x_tilde_no_nan, x_no_nan)

                loss_e_0 = self._loss_e_0(loss_e_t0)
                loss_e = loss_e_0
                optimizer_er.zero_grad()
                self.manual_backward(loss_e)
                # loss_e.backward()
                optimizer_er.step()
                self.log("loss_e", loss_e)

                torch.cuda.empty_cache()
                # self.untoggle_optimizer(optimizer_er)

            # self.toggle_optimizer(optimizer_gs)
            if self.global_step % self.args.log_time == 0:
                # batch = dataset[batch_size]
                # x = batch['data'].to(device)
                train_coeffs = batch["coeffs"]  # .to(device)
                # original_x = batch['original_data'].to(device)
                obs = t
                # x = x[:, :, :-1]
                # time = torch.FloatTensor(list(range(24))).cuda()
                # final_index = (torch.ones(batch_size) * 23).cuda()

                h = self.embedder(time, train_coeffs, final_index)
                # times = time
                # times = times.unsqueeze(0)
                # times = times.unsqueeze(2)
                # times = times.repeat(obs.shape[0], 1, 1)
                #################################################
                if self.args.kinetic_energy is None:
                    loss_s, loss = run_latent_ctfp_model5_2(
                        self.args, self.generator, h, times, self.device, z=False
                    )
                    optimizer_gs.zero_grad()
                    self.manual_backward(loss_s)
                    # loss_s.backward()
                else:
                    loss_s, loss, reg_state = run_latent_ctfp_model5_2(
                        self.args, self.generator, h, times, self.device, z=False
                    )
                    optimizer_gs.zero_grad()
                    self.manual_backward(loss_s + reg_state)
                    # (loss_s+reg_state).backward()
                optimizer_gs.step()
                self.log("loss_s", loss_s)

            # batch = dataset[batch_size]
            # x = batch['data'].to(device)
            train_coeffs = batch["coeffs"]  # .to(device)
            # original_x = batch['original_data'].to(device)
            obs = t
            # x = x[:, :, :-1]
            # time = torch.FloatTensor(list(range(24))).cuda()
            # final_index = (torch.ones(batch_size) * 23).cuda()
            z = torch.randn(batch_size, x.size(1), self.args.effective_shape).to(
                self.device
            )
            h = self.embedder(time, train_coeffs, final_index)
            # times = time.unsqueeze(0)
            # times = times.unsqueeze(2)
            # times = times.repeat(obs.shape[0], 1, 1)
            h_hat = run_latent_ctfp_model5_2(
                self.args, self.generator, z, times, self.device, z=True
            )

            x_hat = self.recovery(h_hat, obs)

            x_no_nan = x[~torch.isnan(x)]
            x_hat_no_nan = x_hat[~torch.isnan(x)]

            y_fake = self.discriminator(x_hat, obs)
            loss_g_u = self._loss_g_u(y_fake)
            loss_g_v = self._loss_g_v(x_no_nan, x_hat_no_nan)
            loss_g = self._loss_g3(loss_g_u, loss_g_v)
            optimizer_gs.zero_grad()
            # loss_g.backward()
            self.manual_backward(loss_g)
            optimizer_gs.step()
            # self.untoggle_optimizer(optimizer_gs)
            self.log("loss_g", loss_g)

        # return super().training_step(*args, **kwargs)

    def validation_step(self, batch, batch_idx):
        batch_size = batch["seq"].shape[0]
        val_samples = self.sample(batch_size, batch.get("c", None))

        val_w_loss = (
            WassersteinDistances(
                batch["seq"].cpu().flatten(1).numpy(),
                val_samples.cpu().flatten(1).numpy(),
            )
            .marginal_distances()
            .mean()
        )
        self.log_dict({"val_loss": val_w_loss}, on_epoch=True, prog_bar=True)

    def _loss_e_t0(self, x_tilde, x):
        return F.mse_loss(x_tilde, x)

    def _loss_e_0(self, loss_e_t0):
        return torch.sqrt(loss_e_t0) * 10

    def _loss_d2(self, y_real, y_fake):
        loss_d_real = F.binary_cross_entropy_with_logits(
            y_real, torch.ones_like(y_real)
        )
        loss_d_fake = F.binary_cross_entropy_with_logits(
            y_fake, torch.zeros_like(y_fake)
        )
        return loss_d_real + loss_d_fake

    def _loss_g_u(self, y_fake):
        return F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))

    def _loss_g_u_e(self, y_fake_e):
        return F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

    def _loss_g_v(self, x_hat, x):
        loss_g_v1 = torch.mean(
            torch.abs(
                torch.sqrt(torch.var(x_hat, 0) + 1e-6)
                - torch.sqrt(torch.var(x, 0) + 1e-6)
            )
        )
        loss_g_v2 = torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
        return loss_g_v1 + loss_g_v2

    def _loss_g(self, loss_g_u, loss_g_u_e, loss_s, loss_g_v):
        return (
            loss_g_u
            + self.gamma * loss_g_u_e
            + 100 * torch.sqrt(loss_s)
            + 100 * loss_g_v
        )

    def _loss_g2(self, loss_g_u, loss_s, loss_g_v):
        return loss_g_u + loss_s + 100 * loss_g_v

    def _loss_g3(self, loss_g_u, loss_g_v):
        return loss_g_u + 100 * loss_g_v
