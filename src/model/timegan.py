import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP


class GRULayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_size=128,
        out_dim=128,
        num_layers=1,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.gru = nn.GRU(
            in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.proj = MLP(hidden_size, [out_dim])

    def forward(self, x):
        h, _ = self.gru(x)
        emb = self.proj(h)
        return emb


class TimeGAN(L.LightningModule):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        gamma: float = 1.0,
        eta: float = 1.0,
        n_critic: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.embedder = GRULayer(seq_dim, hidden_size, latent_dim, num_layers, **kwargs)
        self.recovery = GRULayer(latent_dim, hidden_size, seq_dim, num_layers, **kwargs)
        self.generator = GRULayer(
            seq_dim, hidden_size, latent_dim, num_layers, **kwargs
        )
        self.supervisor = GRULayer(
            latent_dim, hidden_size, latent_dim, num_layers, **kwargs
        )
        self.discriminator = GRULayer(latent_dim, hidden_size, 1, num_layers, **kwargs)

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
            list(self.generator.parameters()) + list(self.discriminator.parameters()),
            lr=self.hparams.lr,
        )
        gs_optim = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.discriminator.parameters()),
            lr=self.hparams.lr,
        )

        return [e0_optim, e_optim, d_optim, g_optim, gs_optim], []

    def training_step(self, batch, batch_idx):
        max_epoch = self.trainer.max_epochs
        x = batch["seq"]
        e0_optim, e_optim, d_optim, g_optim, gs_optim = self.optimizers()

        if (self.current_epoch >= 0) and (self.current_epoch < int(1 / 3 * max_epoch)):
            # 1. Embedding network training
            self.toggle_optimizer(e0_optim)
            h = self.embedder(x)
            x_tilde = self.recovery(h)
            loss = F.mse_loss(x_tilde, x)
            e0_optim.zero_grad()
            self.manual_backward(loss)
            e0_optim.step()
            self.log_dict({"stage1-recon_los": loss}, on_epoch=True, on_step=False)
            self.untoggle_optimizer(e0_optim)

        elif (self.current_epoch >= int(1 / 3 * max_epoch)) and (
            self.current_epoch < int(2 / 3 * max_epoch)
        ):
            # 2. Training only with supervised loss
            self.toggle_optimizer(gs_optim)
            z = torch.randn_like(x)  # ! original code: uniform distrib.
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
            z = torch.randn_like(x)

            # Generator loss
            # 1. Adversarial loss
            self.toggle_optimizer(g_optim)
            h = self.embedder(x)
            h_hat_supervise = self.supervisor(h)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)
            y_fake = self.discriminator(h_hat)
            y_fake_e = self.discriminator(e_hat)
            x_hat = self.recovery(h_hat)

            g_loss_u = F.binary_cross_entropy_with_logits(
                y_fake, torch.ones_like(y_fake)
            )
            g_loss_u_e = F.binary_cross_entropy_with_logits(
                y_fake_e, torch.ones_like(y_fake_e)
            )

            # 2. Supervised loss
            g_loss_s = F.mse_loss(h[:, 1:], h_hat_supervise[:, :-1])

            # 3. Moment loss
            g_loss_v1 = torch.mean(torch.abs(x_hat.mean(dim=0) - x.mean(dim=0)))
            g_loss_v2 = torch.mean(
                torch.abs(
                    torch.sqrt(torch.var(x_hat, dim=0) + 1e-6)
                    - torch.sqrt(torch.var(x, dim=0) + 1e-6)
                )
            )
            g_loss_v = g_loss_v1 + g_loss_v2
            g_loss = (
                g_loss_u
                + g_loss_u_e * self.hparams_initial.gamma
                + (torch.sqrt(g_loss_s)
                + g_loss_v) * self.hparams_initial.eta
            )

            # UPDATE generator
            g_optim.zero_grad()
            self.manual_backward(g_loss)
            g_optim.step()
            self.untoggle_optimizer(g_optim)

            # UPDATE embedder
            self.toggle_optimizer(e_optim)
            h = self.embedder(x)
            x_tilde = self.recovery(h)
            e_loss = F.mse_loss(x_tilde, x)
            e_optim.zero_grad()
            self.manual_backward(e_loss)
            e_optim.step()
            self.untoggle_optimizer(e_optim)

            # UPDATE discriminator
            update_flag = (batch_idx + 1) % self.hparams_initial.n_critic == 0
            if update_flag:
                self.toggle_optimizer(d_optim)

                y_real = self.discriminator(h.detach())
                y_fake = self.discriminator(h_hat.detach())
                y_fake_e = self.discriminator(e_hat.detach())

                D_loss_real = F.binary_cross_entropy_with_logits(
                    torch.ones_like(y_real), y_real
                )
                D_loss_fake = F.binary_cross_entropy_with_logits(
                    torch.zeros_like(y_fake), y_fake
                )
                D_loss_fake_e = F.binary_cross_entropy_with_logits(
                    torch.zeros_like(y_fake_e), y_fake_e
                )
                d_loss = (
                    D_loss_real
                    + D_loss_fake
                    + D_loss_fake_e * self.hparams_initial.gamma
                )

                # if d_loss > 0.15:
                d_optim.zero_grad()
                self.manual_backward(d_loss)
                d_optim.step()
                self.untoggle_optimizer(d_optim)

            self.log_dict(
                {
                    "stage3-g_loss": g_loss,
                    "stage3-e_loss": e_loss,
                    "stage3-d_loss": d_loss if update_flag else 0.0,
                },
                on_epoch=True,
                on_step=False,
            )

    def validation_step(self, batch, batch_idx):
        pass

    # def sample(self, num_samples, current_device, **kwargs):
    #     cond = kwargs.get("condition")
    #     z = torch.randn(num_samples, self.hparams.latent_dim)
    #     z = z.to(current_device)

    #     samples = self(z, cond)
    #     return samples
