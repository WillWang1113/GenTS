import torch
from torch import nn

from src.layers.rnn import RNNLayer
from src.model.base import BaseModel
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(
        self,
        seq_dim,
        latent_dim,
        hidden_size,
        num_layers,
        class_emb_dim,
        n_classes=0,
        **kwargs,
    ):
        super().__init__()
        self.dec = RNNLayer(
            latent_dim + class_emb_dim, hidden_size, seq_dim, num_layers, rnn_type="gru"
        )
        self.n_classes = n_classes
        self.emb = nn.Embedding(n_classes + 1, class_emb_dim)

    def forward(self, z, c=None):
        if c is None:
            c = torch.ones((z.shape[0], 1)) * self.n_classes
            c = c.to(z).int()
        cond = self.emb(c).expand(-1, z.shape[1], -1)
        z = torch.concat([z, cond], dim=-1)
        return self.dec(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        seq_dim,
        latent_dim,
        hidden_size,
        num_layers,
        class_emb_dim,
        n_classes=0,
        **kwargs,
    ):
        super().__init__()
        self.enc = RNNLayer(
            seq_dim + class_emb_dim, hidden_size, 1, num_layers, rnn_type="gru"
        )
        self.n_classes = n_classes
        self.emb = nn.Embedding(n_classes + 1, class_emb_dim)

    def forward(self, x, c=None):
        if c is None:
            c = torch.ones((x.shape[0], 1)) * self.n_classes
            c = c.to(x).int()
        cond = self.emb(c).expand(-1, x.shape[1], -1)
        z = torch.concat([x, cond], dim=-1)
        return self.enc(z)


class RCGAN(BaseModel):
    ALLOW_CONDITION = [None, "class"]
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int = 128,
        class_emb_dim: int = 8,
        hidden_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        num_layers: int = 1,
        # clip_value: float = 0.01,
        n_critic: int = 1,
        condition: str = None,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition)
        self.save_hyperparameters()
        self.automatic_optimization = False
        assert condition in [None, "class"]
        n_classes = 0
        if condition == "class":
            assert kwargs.get("n_classes", None) is not None
            n_classes = kwargs.get("n_classes")
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
        z = torch.randn((n_sample, self.seq_len, self.hparams_initial.latent_dim)).to(
            self.device
        )
        samples = self.generator(z, condition)
        return samples
