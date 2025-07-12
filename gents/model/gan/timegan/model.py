import torch
from torch import nn
from torch.nn import functional as F

from gents.model.base import BaseModel
from gents.common._modules import RNNLayer


class TimeGAN(BaseModel):
    """`Time-series Generative Adversarial Networks <https://proceedings.neurips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf>`__ 

    Adapted from the `official codes <https://github.com/jsyoon0823/TimeGAN>`__
    
    .. note::
        The orignial codes are based on Tensorflow, we adapt the source codes into pytorch.

    Args:
        seq_len (int): Target sequence length.
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to None.
        latent_dim (int, optional): Latent variable dimension. Defaults to 128.
        hidden_size (int, optional): Hidden size for embedder, recovery, generator, supervisor, and discriminator. Defaults to 128.
        num_layers (int, optional): RNN layers. Defaults to 1.
        rnn_type (str, optional): RNN types, choose from `["gru", "lstm", "rnn"]`. Defaults to "gru".
        gamma (float, optional): Loss coefficient for momount loss in Stage 3. Defaults to 1.0.
        eta (float, optional): Loss coefficient for momount loss in Stage 3. Defaults to 1.0.
        n_critic (int, optional): G/D update times. Defaults to 2.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """
    
    ALLOW_CONDITION = [None]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        latent_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        rnn_type: str = "gru",
        gamma: float = 1.0,
        eta: float = 1.0,
        n_critic: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition)
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.embedder = RNNLayer(
            seq_dim,
            hidden_size,
            latent_dim,
            num_layers,
            rnn_type=rnn_type,
            **kwargs,
        )

        self.recovery = RNNLayer(
            latent_dim,
            hidden_size,
            seq_dim,
            num_layers,
            rnn_type=rnn_type,
            **kwargs,
        )

        # Notice that the generator only produce latents,
        # and supervisor aims to supervise the mapping from t-1 to t
        self.generator = RNNLayer(
            seq_dim,
            hidden_size,
            latent_dim,
            num_layers,
            rnn_type=rnn_type,
            **kwargs,
        )
        self.supervisor = RNNLayer(
            latent_dim,
            hidden_size,
            latent_dim,
            num_layers,
            rnn_type=rnn_type,
            **kwargs,
        )

        self.discriminator = nn.Sequential(
            RNNLayer(
                latent_dim, hidden_size, 1, num_layers, rnn_type=rnn_type, **kwargs
            ),
            nn.Sigmoid(),
        )

    def configure_optimizers(self):
        e0_optim = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=self.hparams.lr,
        )
        e_optim = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=self.hparams.lr,
        )
        d_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
        )
        g_optim = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=self.hparams.lr,
        )
        gs_optim = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.supervisor.parameters()),
            lr=self.hparams.lr,
        )

        return [e0_optim, e_optim, d_optim, g_optim, gs_optim], []

    def training_step(self, batch, batch_idx):
        max_steps = self.trainer.max_epochs
        x = batch["seq"]
        # x = self._norm(batch["seq"], mode="norm")
        e0_optim, e_optim, d_optim, g_optim, gs_optim = self.optimizers()

        if (self.current_epoch >= 0) and (self.current_epoch < int(1 / 3 * max_steps)):
            # 1. Embedding network training
            self.toggle_optimizer(e0_optim)
            h = self.embedder(x)
            x_tilde = self.recovery(h)
            loss = 10 * F.mse_loss(x_tilde, x).sqrt()
            e0_optim.zero_grad()
            self.manual_backward(loss)
            e0_optim.step()
            self.log_dict({"stage1-recon_los": loss}, on_epoch=True, on_step=False)
            self.untoggle_optimizer(e0_optim)

        elif (self.current_epoch >= int(1 / 3 * max_steps)) and (
            self.current_epoch < int(2 / 3 * max_steps)
        ):
            # 2. Training only with supervised loss
            self.toggle_optimizer(gs_optim)
            h = self.embedder(x)
            h_hat_supervise = self.supervisor(h)
            loss = F.mse_loss(h[:, 1:], h_hat_supervise[:, :-1])
            gs_optim.zero_grad()
            self.manual_backward(loss)
            gs_optim.step()
            self.log_dict({"stage2-superv_los": loss}, on_epoch=True, on_step=False)
            self.untoggle_optimizer(gs_optim)

        else:
            # 3. Joint Training
            z = torch.rand_like(x) # ! original code: uniform distrib.

            for _ in range(self.hparams.n_critic):
                # Generator loss
                # 1. Adversarial loss
                self.toggle_optimizer(g_optim)
                h = self.embedder(x)
                h_hat_supervise = self.supervisor(h)

                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)

                y_fake = self.discriminator(h_hat).flatten(1)
                y_fake_e = self.discriminator(e_hat).flatten(1)

                x_hat = self.recovery(h_hat)

                g_loss_u = F.binary_cross_entropy(y_fake, torch.ones_like(y_fake))
                g_loss_u_e = F.binary_cross_entropy(y_fake_e, torch.ones_like(y_fake_e))

                # 2. Supervised loss
                g_loss_s = F.mse_loss(h[:, 1:], h_hat_supervise[:, :-1])

                # 3. Moment loss
                g_loss_v1 = torch.mean(
                    torch.abs((torch.mean(x_hat, [0])) - (torch.mean(x, [0])))
                )
                g_loss_v2 = torch.mean(
                    torch.abs(
                        torch.sqrt(torch.var(x_hat, [0]) + 1e-6)
                        - torch.sqrt(torch.var(x, [0]) + 1e-6)
                    )
                )

                g_loss_v = g_loss_v1 + g_loss_v2
                g_loss = (
                    g_loss_u
                    + g_loss_u_e * self.hparams_initial.gamma
                    + (torch.sqrt(g_loss_s) + g_loss_v) * self.hparams_initial.eta
                )

                # UPDATE generator
                g_optim.zero_grad()
                self.manual_backward(g_loss)
                g_optim.step()
                self.untoggle_optimizer(g_optim)

                # UPDATE embedder
                self.toggle_optimizer(e_optim)
                h = self.embedder(x)
                # h_hat_supervise = self.supervisor(h)
                x_tilde = self.recovery(h)
                e_loss = 10 * F.mse_loss(x_tilde, x).sqrt()
                e_loss = e_loss + 0.1 * F.mse_loss(
                    h[:, 1:], h_hat_supervise[:, :-1].detach()
                )
                e_optim.zero_grad()
                self.manual_backward(e_loss)
                e_optim.step()
                self.untoggle_optimizer(e_optim)

            # UPDATE discriminator
            # update_flag = (batch_idx + 1) % self.hparams_initial.n_critic == 0
            # if update_flag:
            self.toggle_optimizer(d_optim)

            y_real = self.discriminator(h.detach()).flatten(1)
            y_fake = self.discriminator(h_hat.detach()).flatten(1)
            y_fake_e = self.discriminator(e_hat.detach()).flatten(1)

            D_loss_real = F.binary_cross_entropy(y_real, torch.ones_like(y_real))
            D_loss_fake = F.binary_cross_entropy(
                y_fake,
                torch.zeros_like(y_fake),
            )
            D_loss_fake_e = F.binary_cross_entropy(y_fake_e, torch.zeros_like(y_fake_e))
            d_loss = (
                D_loss_real + D_loss_fake + D_loss_fake_e * self.hparams_initial.gamma
            )

            if d_loss > 0.15:
                d_optim.zero_grad()
                self.manual_backward(d_loss)
                d_optim.step()
            self.untoggle_optimizer(d_optim)

            self.log_dict(
                {
                    "stage3-g_loss": g_loss,
                    "stage3-e_loss": e_loss,
                    "stage3-d_loss": d_loss,
                },
                on_epoch=True,
                on_step=False,
            )

    # TODO: ES for GAN?
    def validation_step(self, batch, batch_idx): ...

    def _sample_impl(self, n_sample: int, condition: torch.Tensor = None, **kwargs):
        z = torch.rand(
            (n_sample, self.hparams_initial.seq_len, self.hparams_initial.seq_dim)
        ).to(next(self.parameters()))
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        samples = self.recovery(h_hat)
        return samples
        # return self._norm(samples, mode="denorm")

    # def on_fit_start(self):
    #     """TimeGAN will normalize time series into [0, 1] before training. After training, we will inverse normalize.
    #     """
    #     train_dl = self.trainer.datamodule.train_dataloader()
    #     all_x = torch.concat([batch["seq"] for batch in train_dl])
    #     self.min_x = all_x.min(dim=0).values.min(dim=0).values.to(self.device)
    #     self.max_x = all_x.max(dim=0).values.max(dim=0).values.to(self.device)

