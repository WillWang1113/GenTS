import torch
from torch import nn
from torchvision.ops import MLP

from src.layers.rnn import RNNLayer
from torchdiffeq import odeint

from src.model.vae.vanillavae import VanillaVAE


class ODEfunc(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = MLP(*args, **kwargs)

    def forward(self, t, x):
        # t = t.expand(x.shape[0], 1)
        # return self.net(torch.cat([t, x], dim=1))
        return self.net(x)


class NeuralODE(VanillaVAE):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int,
        hidden_size_list: list = [64],
        num_layers: int = 1,
        ode_method: str = "dopri5",
        rnn_type: str = "gru",
        beta: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__(
            seq_len, seq_dim, latent_dim, hidden_size_list, beta, lr, weight_decay
        )
        self.save_hyperparameters()

        self.encoder = RNNLayer(
            seq_dim,
            hidden_size_list[-1],
            latent_dim * 2,
            num_layers,
            rnn_type=rnn_type,
            # **kwargs,
        )
        self.ode_fn = ODEfunc(
            in_channels=latent_dim,
            hidden_channels=hidden_size_list + [latent_dim],
            activation_layer=nn.ELU,
        )
        self.decoder = MLP(
            latent_dim,
            [seq_dim],
            # activation_layer=torch.nn.Identity,
        )
        self.ode_method = ode_method
        self.fc_logvar = None
        self.fc_mu = None
        self.latent_dim = latent_dim

    def encode(self, x, c=None, **kwargs):
        # t = kwargs.get(
        #     "t",
        #     torch.linspace(0, 2 * torch.pi, self.hparams_initial.seq_len).type_as(x),
        # ).reshape(1, -1, 1)
        # print(x.shape)
        # print(t.shape)
        # t = t.expand(x.shape[0], -1, -1)
        # xt = torch.concat([x, t], dim=-1)
        latents = self.encoder(x.flip(dims=[1]))
        mu = latents[:, -1, : self.latent_dim]
        logvar = latents[:, -1, self.latent_dim :]
        return latents, mu, logvar

    def decode(self, z, c=None, **kwargs):
        t = kwargs.get(
            "t",
            torch.linspace(0, 1.2 * torch.pi, self.hparams_initial.seq_len).type_as(z),
        )
        zs = odeint(
            self.ode_fn, z, t, method=self.ode_method, options={"dtype": torch.float32}
        ).permute(1, 0, 2)
        zs = self.decoder(zs)
        return zs

    def _sample_impl(self, n_sample, condition=None, t=None):
        z = torch.randn((n_sample, self.hparams_initial.latent_dim)).to(
            next(self.parameters())
        )
        t = (
            t.to(z)
            if t is not None
            else torch.linspace(0, 2 * torch.pi, self.hparams_initial.seq_len).to(z)
        )

        zs = odeint(
            self.ode_fn, z, t, method=self.ode_method, options={"dtype": torch.float32}
        ).permute(1, 0, 2)
        zs = self.decoder(zs)
        return zs

    # def configure_optimizers(self):
    #     return torch.optim.Adam(
    #         self.parameters(),
    #         lr=self.hparams_initial.lr,
    #         weight_decay=self.hparams_initial.weight_decay,
    #     )

    # def training_step(self, batch, batch_idx):
    #     x = batch["seq"]
    #     c = batch.get("c", None)

    #     # Run RNN in reverse mode
    #     latents = self.embedder(x[:, ::-1, :])

    #     # ODE
    #     z = odeint(
    #         self.ode_fn, latents, torch.linspace(0, 1, self.hparams_initial.seq_len)
    #     )

    #     # decode
    #     x_hat = self.decoder(z)

    #     loss = F.mse_loss(x_hat, x)
    #     self.log("train/loss", loss)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     pass

    # def _sample_impl(self, n_sample: int, condition: torch.Tensor = None):
    #     z = torch.rand(
    #         (n_sample, self.hparams_initial.seq_len, self.hparams_initial.seq_dim)
    #     ).to(next(self.parameters()))
    #     e_hat = self.generator(z)
    #     h_hat = self.supervisor(e_hat)
    #     samples = self.recovery(h_hat)
    #     return samples
    #     # return self._norm(samples, mode="denorm")

    # def on_fit_start(self):
    #     train_dl = self.trainer.datamodule.train_dataloader()
    #     all_x = torch.concat([batch["seq"] for batch in train_dl])
    #     self.min_x = all_x.min(dim=0).values.min(dim=0).values.to(self.device)
    #     self.max_x = all_x.max(dim=0).values.max(dim=0).values.to(self.device)

    # # def _norm(self, x, mode="norm"):
    # #     return (
    # #         (x - self.min_x) / (self.max_x - self.min_x)
    # #         if mode == "norm"
    # #         else (x * (self.max_x - self.min_x)) + (self.min_x)
    # #     )
