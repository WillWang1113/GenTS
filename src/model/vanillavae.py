import torch
from torch.nn import functional as F
from torchvision.ops import MLP

from src.layers.mlp import MLPDecoder, MLPEncoder

# from src.layers.conv import ConvDecoder, ConvEncoder
from .base import BaseModel


class VanillaVAE(BaseModel):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int,
        hidden_size_list: list = [64, 128, 256],
        beta: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        # print(self.hparams)
        self.hiddens = hidden_size_list.copy()
        self.encoder = MLPEncoder(seq_len, seq_dim, latent_dim, self.hiddens, **kwargs)
        self.hiddens.reverse()
        self.decoder = MLPDecoder(seq_len, seq_dim, latent_dim, self.hiddens, **kwargs)

        self.fc_mu = MLP(latent_dim, [latent_dim])
        self.fc_logvar = MLP(latent_dim, [latent_dim])

    def encode(self, x, c=None):
        # x = x
        latents = self.encoder(x)
        mu = self.fc_mu(latents)
        logvar = self.fc_logvar(latents)
        return latents, mu, logvar

    def decode(self, z, c=None):
        return self.decoder(z)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def _get_loss(self, batch):
        x = batch["seq"]
        c = batch.get("c", None)

        # encode
        latents, mu, logvar = self.encode(x, c)

        # reparameterize
        z = self.reparam(mu, logvar)

        # decode
        x_hat = self.decode(z, c)

        assert x.shape == x_hat.shape

        # reconstruction loss
        recons_loss = F.mse_loss(x_hat, x)

        # KL divergence loss
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        loss_dict = dict(
            recon_loss=recons_loss,
            kl_loss=self.hparams_initial.beta * kld_loss,
            loss=recons_loss + self.hparams_initial.beta * kld_loss,
        )
        return loss_dict
    
    def _sample_impl(self, n_sample, condition=None):
        z = torch.randn((n_sample, self.hparams_initial.latent_dim)).to(self.device)
        x_hat = self.decode(z, condition)
        return x_hat

    def training_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch)
        prefix = "train_"
        loss_dict = {prefix + key: value for key, value in loss_dict.items()}
        self.log_dict(loss_dict, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        return loss_dict[prefix+"loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch)
        prefix = "val_"
        loss_dict = {prefix + key: value for key, value in loss_dict.items()}
        self.log_dict(loss_dict, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        return loss_dict[prefix+"loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams_initial.lr,
            weight_decay=self.hparams_initial.weight_decay,
        )