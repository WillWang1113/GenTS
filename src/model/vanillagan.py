import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
from torch import Tensor
import matplotlib.pyplot as plt
import random
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable

from src.utils import WassersteinDistances


L.seed_everything(3407, True)

# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0, 0.03)


class Generator(nn.Module):
    def __init__(
        self, in_channels, latent_dim, cond_in_channels, img_shape, hidden_dims=None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        current_seq_len = img_shape[-1] // (2 ** (len(hidden_dims)))
        self.latent_seq_len = current_seq_len
        self.decoder_input = nn.Linear(
            latent_dim + cond_in_channels, hidden_dims[0] * current_seq_len
        )

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_dims[-1], out_channels=in_channels, kernel_size=3, padding=1
            ),
        )
        self.hidden_dims = hidden_dims

    def forward(self, z, cond):
        z = torch.concat([z, cond], dim=-1)
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], self.latent_seq_len)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


class Discriminator(nn.Module):
    def __init__(self, in_channels, cond_in_channels, img_shape, hidden_dims=None):
        super().__init__()
        self.seq_len = img_shape[-1]

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        self.hidden_dims = hidden_dims
        seq_len = img_shape[-1]
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_in_channels, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], seq_len * 1),
        )

        in_channels += 1
        current_seq_len = seq_len
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
            current_seq_len = current_seq_len // 2
        self.latent_seq_len = current_seq_len

        self.encoder = nn.Sequential(*modules)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1] * current_seq_len, hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[-1], 1),
            # nn.Sigmoid(),
        )

    def forward(self, img, cond):
        emb_cond = self.cond_embed(cond).reshape(-1, 1, self.seq_len)

        concat_input = torch.concat([img, emb_cond], dim=1)
        result = self.encoder(concat_input)
        result = torch.flatten(result, start_dim=1)

        validity = self.out_mlp(result)
        return validity


class WGAN(L.LightningModule):
    def __init__(
        self,
        in_channels,
        cond_in_channels,
        seq_len,
        latent_dim: int = 128,
        lr: float = 0.001,
        b1: float = 0.5,
        b2: float = 0.999,
        # clip_value: float = 0.01,
        n_critic: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.vis_z = torch.randn(9, 10, latent_dim)

        # networks
        data_shape = (in_channels, seq_len)
        # self.generator = Generator(
        #     img_channels=in_channels, img_size=seq_len, latent_dim=latent_dim, cond_channels=cond_in_channels
        # )
        # self.discriminator = Discriminator(
        #     img_channels=in_channels, img_size=seq_len, cond_channels=cond_in_channels
        # )
        self.generator = Generator(
            in_channels=in_channels,
            cond_in_channels=cond_in_channels,
            latent_dim=self.hparams.latent_dim,
            img_shape=data_shape,
        )
        self.discriminator = Discriminator(
            in_channels=in_channels,
            cond_in_channels=cond_in_channels,
            img_shape=data_shape,
        )
        self.criterionSource = nn.BCELoss()
        # self.generator.apply(weights_init_normal)
        # self.discriminator.apply(weights_init_normal)

    def forward(self, z, cond):
        return self.generator(z, cond)

    def training_step(self, batch):
        imgs = batch['seq']
        cond = batch['c']
        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        clip_value = 0.01

        # train generator
        if (self.global_step + 1) % (self.hparams.n_critic + 1) != 0:
            # generate images
            self.generated_imgs = self(z, cond)

            # log sampled images

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = -torch.mean(self.discriminator(self(z, cond), cond))
            optimizer_g.zero_grad()
            self.manual_backward(g_loss)
            optimizer_g.step()
            tqdm_dict = {"g_loss": g_loss}
            self.log_dict(tqdm_dict, prog_bar=True)


        # train discriminator
        # Measure discriminator's ability to classify real from generated samples

        # discriminator loss is the average of these
        else:
            d_loss = -torch.mean(self.discriminator(imgs, cond)) + torch.mean(
                self.discriminator(self(z, cond), cond)
            )
            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()
            for p in self.discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            tqdm_dict = {"d_loss": d_loss}
            self.log_dict(tqdm_dict, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        imgs = batch['seq']
        cond = batch['c']
        
        # For generation
        latent = torch.randn(imgs.shape[0], self.hparams.latent_dim).to(self.device)
        fakeData = self.generator(latent, cond)
        wd = WassersteinDistances(
            imgs.flatten(1).cpu().numpy(), fakeData.flatten(1).cpu().numpy(), seed=9
        )
        val_loss = np.mean(wd.sliced_distances(100))

        # For forecasting
        # latent = torch.randn(imgs.shape[0], self.hparams.latent_dim).to(self.device)
        # fakeData = self.generator(latent, cond)
        # val_loss = torch.nn.functional.mse_loss(fakeData, imgs)
        
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        g_optim = torch.optim.RMSprop(
            self.generator.parameters(),
            lr=self.hparams.lr,
            # betas=(self.hparams.b1, self.hparams.b2),
        )
        d_optim = torch.optim.RMSprop(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            # betas=(self.hparams.b1, self.hparams.b2),
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

    def sample(self, num_samples, current_device, **kwargs):
        cond = kwargs.get("condition")
        z = torch.randn(num_samples, self.hparams.latent_dim)
        z = z.to(current_device)

        samples = self(z, cond)
        return samples
