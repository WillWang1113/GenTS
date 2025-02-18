from pkg_resources import non_empty_lines
import torch
from torch import nn
from src.layers.conv import ConvEncoder, ConvDecoder
from src.layers.mlp import MLPDecoder, MLPEncoder
from src.model.base import BaseGAN
from src.utils.check import _condition_shape_check


class Generator(nn.Module):
    def __init__(
        self, seq_len, seq_dim, latent_dim, hidden_size_list=[256, 128, 64], **kwargs
    ):
        super().__init__()
        self.dec = MLPDecoder(seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs)
        # self.dec = ConvDecoder(seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs)

    def forward(self, z, c=None):
        return self.dec(z).permute(0, 2, 1)


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
        # self.enc = ConvEncoder(seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs)
        self.out_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, 1),
            # nn.Sigmoid(),
        )
        if last_sigmoid:
            self.out_mlp.append(nn.Sigmoid())

    def forward(self, x, c=None):
        # emb_cond = self.cond_embed(cond).reshape(-1, 1, self.seq_len)

        # concat_input = torch.concat([img, emb_cond], dim=1)
        latents = self.enc(x.permute(0, 2, 1))
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

    # def forward(self, z, cond):
    #     return self.generator(z, cond)

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

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            # valid = torch.ones(x.size(0), 1).type_as(x)
            # valid = valid

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

    # def on_validation_epoch_end(self):
    #     # Get sample reconstruction image
    #     test_input, test_label = next(iter(self.trainer.val_dataloaders))
    #     test_input = test_input.to(self.device)
    #     test_label = test_label.to(self.device)

    #     if test_label.shape[1] > 12:
    #         show_num = 3
    #     else:
    #         show_num = test_label.shape[1]

    #     fig, axs = plt.subplots(3, 3, figsize=[6.4 * 2, 4.8 * 2])
    #     axs = axs.flatten()
    #     z = self.vis_z.type_as(test_input)

    #     for i in range(len(axs)):
    #         choose = random.randint(0, len(test_input) - 1)
    #         samples = self(z[i], test_label[[choose]].repeat(10, 1))

    #         axs[i].plot(test_input[choose, 0, :].cpu(), c="red", label="data")
    #         axs[i].plot(samples[:, 0, :].T.cpu(), c="grey", alpha=0.5)
    #         axs[i].legend()
    #         axs[i].set(title=f"cond on: {test_label[choose].cpu().numpy()[:show_num]}")
    #     fig.tight_layout()
    #     fig.savefig(
    #         os.path.join(self.trainer.log_dir, f"valid_{self.current_epoch}.png")
    #     )
    #     plt.close()

    @torch.no_grad()
    def sample(self, n_sample, condition=None):
        self.eval()
        z = torch.randn((n_sample, self.hparams_initial.latent_dim)).to(self.device)
        # if condition is not None:
        #     _condition_shape_check(n_sample, condition)
        samples = self.generator(z, condition)
        return samples
