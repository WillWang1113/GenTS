import torch

from gents.model.base import BaseModel
from torch.nn import functional as F
from ._backbones import Generator, Discriminator


class RCGAN(BaseModel):
    """`Recurrent conditional GAN <https://arxiv.org/pdf/1706.02633>`__ 
    
    Adapted from the `official codes <https://github.com/ratschlab/RGAN/tree/master>`__
    
    .. note::
        The orignial codes are based on Tensorflow, we adapt the source codes into pytorch.

    
    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to None.
        latent_dim (int, optional): Latent variable dimension. Defaults to 128.
        num_layers (int, optional): RNN layers. Defaults to 1.
        class_emb_dim (int, optional): Embedding dimension for class labels. Defaults to 8.
        hidden_size (int, optional): Hidden size for RNN. Defaults to 128.
        n_critic (int, optional): G/D update times. Defaults to 1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """
    ALLOW_CONDITION = [None, "class"]
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        latent_dim: int = 128,
        num_layers: int = 1,
        class_emb_dim: int = 8,
        hidden_size: int = 128,
        n_critic: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):

        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False
        n_classes = self.class_num if self.condition == 'class' else 0
        self.seq_len = seq_len
        self.seq_dim = seq_dim

        # networks
        self.discriminator = Discriminator(
            seq_dim=seq_dim,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            class_emb_dim=class_emb_dim,
            n_classes=n_classes,
        )
        self.generator = Generator(
            seq_dim=seq_dim,
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            class_emb_dim=class_emb_dim,
            n_classes=n_classes,
        )

    def training_step(self, batch):
        x = batch["seq"]
        c = batch.get("c", None)
        if c is not None:
            c = c.squeeze(-1)
        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(x.shape[0], self.seq_len, self.hparams.latent_dim).type_as(x)

        self.toggle_optimizer(optimizer_d)
        y_real = self.discriminator(x, c)
        y_fake = self.discriminator(self.generator(z, c), c)

        d_loss = F.binary_cross_entropy_with_logits(
            y_real, torch.ones_like(y_real)
        ) + F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))

        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        # for p in self.discriminator.parameters():
        #     p.data.clamp_(-self.hparams.clip_value, self.hparams.clip_value)

        loss_dict = {"d_loss": d_loss}
        self.log_dict(loss_dict)

        for _ in range(self.hparams.n_critic):
            self.toggle_optimizer(optimizer_g)

            y_fake_new = self.discriminator(self.generator(z, c), c)

            # adversarial loss is binary cross-entropy
            g_loss = F.binary_cross_entropy_with_logits(
                y_fake_new, torch.ones_like(y_fake_new)
            )
            optimizer_g.zero_grad()
            self.manual_backward(g_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)
            loss_dict = {"g_loss": g_loss}
            self.log_dict(loss_dict)

    def validation_step(self, batch, batch_idx): ...

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        d_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return [g_optim, d_optim], []

    def _sample_impl(self, n_sample, condition=None, **kwargs):

        z = torch.randn((n_sample, self.seq_len, self.hparams.latent_dim)).to(
            self.device
        )
        samples = self.generator(z, condition)

        # if self.condition is None:
        #     z = torch.randn((n_sample, self.seq_len, self.hparams.latent_dim)).to(
        #         self.device
        #     )
        #     samples = self.generator(z, condition)
        # else:
        #     # class conditional
        #     all_samples = []
        #     for i in range(n_sample):
        #         z = torch.randn(
        #             (condition.shape[0], self.seq_len, self.hparams.latent_dim)
        #         ).to(self.device)
        #         sample = self.generator(z, condition)
        #         all_samples.append(sample)
        #     samples = torch.stack(all_samples, dim=-1)
        return samples
