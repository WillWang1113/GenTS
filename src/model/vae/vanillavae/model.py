import torch
from torch.nn import functional as F
from torchvision.ops import MLP

from src.model.base import BaseModel
from src.common._losses import kl_loss
# from src.utils.losses import kl_loss

from src.common._modules import MLPDecoder, MLPEncoder


class VanillaVAE(BaseModel):
    """Vanilla Variational Autoencoder (VAE) model with MLP encoder and decoder."""

    ALLOW_CONDITION = [None, "predict", "impute", "class"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int = 128,
        hidden_size_list: list = [64, 128, 256],
        w_kl: float = 1e-4,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        condition: str = None,
        **kwargs,
    ):
        """
        Args:
            seq_len (int): Target sequence length
            seq_dim (int): Target sequence dimension, for univariate time series, set as 1
            latent_dim (int, optional): Latent dimension for z. Defaults to 128.
            hidden_size_list (list, optional): Hidden size for encoder and decoder. Defaults to [64, 128, 256].
            w_kl (float, optional): Loss weight of KL div. Defaults to 1e-4.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-5.
            condition (str, optional): Given conditions. Defaults to None.
        """

        super().__init__(seq_len, seq_dim, condition)
        self.save_hyperparameters()
        # print(self.hparams)
        self.hiddens = hidden_size_list.copy()
        self.encoder = MLPEncoder(seq_len, seq_dim, latent_dim, self.hiddens, **kwargs)
        self.hiddens.reverse()
        self.decoder = MLPDecoder(seq_len, seq_dim, latent_dim, self.hiddens, **kwargs)
        self.cond_net = None
        self.condition = condition
        if condition:
            if condition == "predict":
                assert kwargs.get("obs_len") is not None
                obs_len = kwargs.get("obs_len")
                self.cond_net = MLPEncoder(
                    obs_len, seq_dim, latent_dim, self.hiddens, **kwargs
                )
            elif condition == "impute":
                # assert kwargs.get('obs_len') is not None
                # obs_len = kwargs.get('obs_len')
                self.cond_net = MLPEncoder(
                    seq_len, seq_dim, latent_dim, self.hiddens, **kwargs
                )

        self.fc_mu = MLP(latent_dim, [latent_dim])
        self.fc_logvar = MLP(latent_dim, [latent_dim])

    def encode(self, x: torch.Tensor, c: torch.Tensor = None, **kwargs):
        # x = x
        latents = self.encoder(x)
        if (c is not None) and (self.cond_net is not None):
            if self.condition == "impute":
                c = x * (~c).int()
            c = c.to(x)
            cond_lats = self.cond_net(c)
            latents = latents + cond_lats
        mu = self.fc_mu(latents)
        logvar = self.fc_logvar(latents)
        return latents, mu, logvar

    def decode(self, z: torch.Tensor, c: torch.Tensor = None, **kwargs):
        if (c is not None) and (self.cond_net is not None):
            if self.condition == "impute":
                c = kwargs['seq'] * (~c).int()
            c = c.to(z)
            cond_lats = self.cond_net(c)
            z = z + cond_lats
        return self.decoder(z)

    def reparam(self, mu, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mu + eps * std
            return z
        else:
            return mu

    def _get_loss(self, batch):
        x = batch["seq"][:, -self.hparams_initial.seq_len :]
        batch_size = x.shape[0]

        # encode
        _, mu, logvar = self.encode(x, **batch)
        # print(latents.shape)
        # print(mu.shape)
        # print(logvar.shape)

        # reparameterize
        z = self.reparam(mu, logvar)

        # prior z distribution
        z_prior, mu_prior, logvar_prior = self.sample_prior(n_sample=z.shape[0])

        # decode
        x_hat = self.decode(z, **batch)

        assert x.shape == x_hat.shape

        # reconstruction loss
        recons_loss = F.mse_loss(x_hat, x)

        # KL divergence loss
        kld_loss = kl_loss(mu, logvar, mu_prior, logvar_prior)
        kld_loss = torch.sum(kld_loss) / batch_size

        loss_dict = dict(
            recon_loss=recons_loss,
            kl_loss=self.hparams_initial.w_kl * kld_loss,
            loss=recons_loss + self.hparams_initial.w_kl * kld_loss,
        )
        return loss_dict, z_prior, mu_prior, logvar_prior, z, mu, logvar

    def _sample_impl(self, n_sample, condition=None, **kwargs):
        if self.condition is None:
            z = torch.randn((n_sample, self.hparams_initial.latent_dim)).to(self.device)
            all_samples = self.decode(z, condition)
        else:
            all_samples = []
            # if self.condition == "impute":
            #     c = kwargs['seq'] * (~condition).int()
            # else:
            #     c = condition
                
            for i in range(n_sample):
                z = torch.randn(
                    (condition.shape[0], self.hparams_initial.latent_dim)
                ).to(self.device)
                x_hat = self.decode(z, **kwargs)
                all_samples.append(x_hat)
            all_samples = torch.stack(all_samples, dim=-1)

        return all_samples

    def training_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch)[0]
        prefix = "train_"
        loss_dict = {prefix + key: value for key, value in loss_dict.items()}
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)
        return loss_dict["train_loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self._get_loss(batch)[0]
        prefix = "val_"
        loss_dict = {prefix + key: value for key, value in loss_dict.items()}
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)
        return loss_dict

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams_initial.lr,
            weight_decay=self.hparams_initial.weight_decay,
        )

    def sample_prior(self, n_sample):
        z_prior = torch.randn((n_sample, self.hparams_initial.latent_dim)).to(
            self.device
        )
        mu_prior = torch.zeros_like(z_prior)
        logvar_prior = torch.zeros_like(z_prior)
        return z_prior, mu_prior, logvar_prior
