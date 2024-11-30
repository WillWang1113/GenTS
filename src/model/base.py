from abc import ABC, abstractmethod
from lightning import LightningModule
import torch
from torch.nn import functional as F


class BaseModel(ABC, LightningModule):
    @abstractmethod
    def sample(self, n_sample):
        raise NotImplementedError()


class BaseVAE(ABC, LightningModule):
    @abstractmethod
    def encode(self, x, c=None):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, z, c=None):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x = batch["seq"]
        c = batch.get("c", None)

        # encode
        latents, mu, logvar = self.encode(x, c)

        # reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        # decode
        x_hat = self.decode(z, c)

        # loss
        loss = self._get_loss(x, x_hat, mu, logvar)

        self.log_dict({"train_loss": loss}, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["seq"]
        c = batch.get("c", None)

        # encode
        latents, mu, logvar = self.encode(x, c)

        # reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        # decode
        x_hat = self.decode(z, c)

        # loss
        loss = self._get_loss(x, x_hat, mu, logvar)
        self.log_dict({"val_loss": loss})

        return loss

    def _get_loss(self, x, x_hat, mu, logvar):
        assert x.shape == x_hat.shape
        recons_loss = F.mse_loss(x_hat, x)
        kld_loss = torch.mean(
            0.5
            * torch.sum(
                -self.hparams_initial.latent_dim - logvar + mu**2 + logvar.exp(), dim=1
            ),
            dim=0,
        )
        loss = recons_loss + self.hparams_initial.beta * kld_loss
        return loss


class BaseGAN(ABC, LightningModule):
    @abstractmethod
    def encode(self, x, c=None):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, z, c=None):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x = batch["seq"]
        c = batch.get("c", None)

        # encode
        latents, mu, logvar = self.encode(x, c)

        # reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        # decode
        x_hat = self.decode(z, c)

        # loss
        loss = self._get_loss(x, x_hat, mu, logvar)

        self.log_dict({"train_loss": loss}, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["seq"]
        c = batch.get("c", None)

        # encode
        latents, mu, logvar = self.encode(x, c)

        # reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        # decode
        x_hat = self.decode(z, c)

        # loss
        loss = self._get_loss(x, x_hat, mu, logvar)
        self.log_dict({"val_loss": loss})

        return loss

    def _get_loss(self, x, x_hat, mu, logvar):
        assert x.shape == x_hat.shape
        recons_loss = F.mse_loss(x_hat, x)
        kld_loss = torch.mean(
            0.5
            * torch.sum(
                -self.hparams_initial.latent_dim - logvar + mu**2 + logvar.exp(), dim=1
            ),
            dim=0,
        )
        loss = recons_loss + self.hparams_initial.beta * kld_loss
        return loss
