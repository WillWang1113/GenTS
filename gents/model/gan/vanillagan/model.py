import torch

from gents.model.base import BaseModel
from ._backbones import Generator, Discriminator


class VanillaGAN(BaseModel):
    """Vanilla Wasserstein GAN with MLP Generator and Discriminator.
    
    For conditional generation, an extra MLP is used for embedding conditions.
    
    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given conditions, should be one of `ALLOW_CONDITION`. Defaults to None.
        latent_dim (int, optional): Latent dimension for z. Defaults to 128.
        hidden_size_list (list, optional): Hidden size for encoder and decoder. Defaults to [64, 128, 256].
        clip_value (float, optional): Gradient clip value. Defaults to 0.01.
        n_critic (int, optional): D/G update times. Defaults to 5.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """

    ALLOW_CONDITION = [None, "predict", "impute", "class"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        latent_dim: int = 128,
        hidden_size_list: list = [64, 128, 256],
        clip_value: float = 0.01,
        n_critic: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):

        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        gen_param = self.hparams_initial.copy()
        self.discriminator = Discriminator(**self.hparams_initial)
        gen_param['hidden_size_list'].reverse()
        self.generator = Generator(**gen_param)

    def training_step(self, batch, batch_idx):
        x = batch["seq"][:, -self.hparams_initial.seq_len :]
        c = batch.get("c")
        c = torch.nan_to_num(c) if self.condition == "impute" else c

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
            self.log_dict(loss_dict, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x = batch["seq"][:, -self.hparams_initial.seq_len :]
        c = batch.get("c")
        c = torch.nan_to_num(c) if self.condition == "impute" else c

        # if self.condition == "impute":
        #     c = x * (~c).int()
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

    # @torch.no_grad()
    def _sample_impl(self, n_sample, condition=None, **kwargs):
        if self.condition is None or self.condition == "class":
            z = torch.randn((n_sample, self.hparams_initial.latent_dim)).to(self.device)
            all_samples = self.generator(z, condition)
        else:
            # if self.condition == "impute":
            # c = kwargs["seq"] * (~condition).int()
            c = condition.to(self.device)
            c = torch.nan_to_num(c) if self.condition == "impute" else c
            all_samples = []
            for i in range(n_sample):
                z = torch.randn((c.shape[0], self.hparams_initial.latent_dim)).to(
                    self.device
                )
                samples = self.generator(z, c)
                all_samples.append(samples)
            all_samples = torch.stack(all_samples, dim=-1)

        return all_samples
