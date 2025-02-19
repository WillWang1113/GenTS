import torch
from torch import nn

from src.layers.mlp import MLPDecoder, MLPEncoder
from src.model.base import BaseGAN


class Generator(nn.Module):
    def __init__(
        self, seq_len, seq_dim, latent_dim, hidden_size_list=[256, 128, 64], **kwargs
    ):
        super().__init__()
        self.dec = MLPDecoder(seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs)

    def forward(self, z, c=None):
        return self.dec(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        seq_len,
        seq_dim,
        latent_dim,
        hidden_size_list=[64, 128, 256],
        last_sigmoid=False,
        **kwargs,
    ):
        super().__init__()
        self.enc = MLPEncoder(seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs)
        self.out_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, 1),
        )
        if last_sigmoid:
            self.out_mlp.append(nn.Sigmoid())

    def forward(self, x, c=None):
        latents = self.enc(x)
        validity = self.out_mlp(latents)
        return validity


class VanillaGAN(BaseGAN):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int,
        hidden_size_list: list = [64, 128, 256],
        beta: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        clip_value: float = 0.01,
        n_critic: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        self.discriminator = Discriminator(**self.hparams)
        self.hparams.hidden_size_list.reverse()
        self.generator = Generator(**self.hparams)
        self.criterionSource = nn.BCELoss()

    def training_step(self, batch):
        x = batch["seq"]
        c = batch.get("c", None)
        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(x.shape[0], self.hparams.latent_dim).type_as(x)

        # train generator
        if (self.global_step + 1) % (self.hparams.n_critic + 1) != 0:
            self.toggle_optimizer(optimizer_g)

            # generate images
            self.generated_imgs = self.generator(z, c)

            # adversarial loss is binary cross-entropy
            g_loss = -torch.mean(self.discriminator(self.generator(z, c), c))
            optimizer_g.zero_grad()
            self.manual_backward(g_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)
            loss_dict = {"g_loss": g_loss}
            self.log_dict(loss_dict)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples

        # discriminator loss is the average of these
        else:
            self.toggle_optimizer(optimizer_d)

            d_loss = -torch.mean(self.discriminator(x, c)) + torch.mean(
                self.discriminator(self.generator(z, c), c)
            )
            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

            for p in self.discriminator.parameters():
                p.data.clamp_(-self.hparams.clip_value, self.hparams.clip_value)

            loss_dict = {"d_loss": d_loss}
            self.log_dict(loss_dict)

    def validation_step(self, batch, batch_idx):
        x = batch["seq"]
        c = batch.get("c", None)
        z = torch.randn(x.shape[0], self.hparams.latent_dim).type_as(x)

        w_distance = torch.mean(self.discriminator(x, c)) - torch.mean(
            self.discriminator(self.generator(z, c), c)
        )

        self.log("val_loss", w_distance)

    def configure_optimizers(self):
        g_optim = torch.optim.RMSprop(
            self.generator.parameters(),
            lr=self.hparams.lr,
        )
        d_optim = torch.optim.RMSprop(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
        )
        return [g_optim, d_optim], []


    @torch.no_grad()
    def sample(self, n_sample, condition=None):
        self.eval()
        z = torch.randn((n_sample, self.hparams_initial.latent_dim)).to(self.device)
        samples = self.generator(z, condition)
        return samples
