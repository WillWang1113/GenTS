import numpy as np
import torch
from torch import nn
from torchvision.ops import MLP

from src.model.base import BaseModel

# Seems useless
# class PairwiseDiscriminator(nn.Module):
#     def __init__(self, n_channels, alpha):
#         super().__init__()
#         self.n_channels = n_channels
#         n_corr_values = n_channels * (n_channels - 1) // 2
#         layers = []
#         while np.log2(n_corr_values) > 1:
#             layers.append(nn.Linear(n_corr_values, n_corr_values // 2))
#             layers.append(nn.LeakyReLU(alpha))
#             layers.append(nn.Dropout(0.3))
#             n_corr_values = n_corr_values // 2
#         layers.append(nn.Linear(n_corr_values, 1))
#         layers.append(nn.Sigmoid())
#         self.classifier = nn.Sequential(*layers)

#         self.pairwise_correlation = torch.corrcoef
#         self.upper_triangle = lambda x: x[
#             torch.triu(torch.ones(n_channels, n_channels), diagonal=1) == 1
#         ]

#     def forward(self, x):
#         final_upper_trianle = []
#         for i in range(x.shape[0]):
#             pairwise_correlation = self.pairwise_correlation(x[i, :].transpose(0, 1))
#             upper_triangle = self.upper_triangle(pairwise_correlation)
#             final_upper_trianle.append(upper_triangle)
#         final_upper_trianle = torch.stack(final_upper_trianle)
#         return self.classifier(final_upper_trianle)


class LSTMDiscriminator(nn.Module):
    """Discriminator with LSTM"""

    def __init__(self, in_dim, hidden_size=256, num_layers=1, **kwargs):
        super(LSTMDiscriminator, self).__init__()

        self.hidden_dim = hidden_size

        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        out, _ = self.lstm(x)
        out = self.linear(out.view(x.size(0) * x.size(1), self.hidden_dim))
        out = out.view(x.size(0), x.size(1))
        return out


class Discriminator(nn.Module):
    def __init__(
        self, in_dim, hidden_size_list=[256, 128, 64, 1], dropout=0.3, **kwargs
    ):
        super().__init__()
        self.model = MLP(
            in_dim, hidden_size_list, activation_layer=nn.LeakyReLU, dropout=dropout
        )
        # last dropout layer pop
        self.model.pop(-1)
        self.model.append(torch.nn.Sigmoid())

    def forward(self, x):
        output = self.model(x.flatten(1))
        return output


class LSTMGenerator(nn.Module):
    """Generator with LSTM"""

    def __init__(self, latent_dim, seq_len, hidden_size=256, num_layers=1, **kwargs):
        super(LSTMGenerator, self).__init__()

        self.ts_dim = seq_len
        self.hidden_dim = hidden_size

        self.lstm = nn.LSTM(latent_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, seq_len)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        out, _ = self.lstm(x)
        out = self.linear(out.view(x.size(0) * x.size(1), self.hidden_dim))
        out = out.view(x.size(0), self.ts_dim)
        return out


class Generator(nn.Module):
    def __init__(
        self, latent_dim, seq_len, hidden_size_list=[256, 512], dropout=0.3, **kwargs
    ):
        super().__init__()
        self.model = MLP(
            latent_dim,
            hidden_size_list + [seq_len],
            norm_layer=nn.BatchNorm1d,
            activation_layer=torch.nn.LeakyReLU,
            dropout=dropout,
        )
        self.model.pop(-1)


    def forward(self, x):
        output = self.model(x)
        return output


class COSCIGAN(BaseModel):
    def __init__(
        self,
        seq_len,
        seq_dim,
        latent_dim,
        DG_type="LSTM",
        central_disc_type="MLP",
        gamma=5.0,
        lr={"G": 1e-3, "D": 1e-3, "CD": 1e-4},
        **kwargs,
    ):
        super().__init__()
        assert DG_type in ["MLP", "LSTM"]
        assert central_disc_type in ["MLP", "LSTM"]
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generators, self.discriminators = [], []
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.latent_dim = latent_dim
        if DG_type == "LSTM":
            gen_cls = LSTMGenerator
            dis_cls = LSTMDiscriminator
        else:
            gen_cls = Generator
            dis_cls = Discriminator
        for i in range(seq_dim):
            self.generators.append(gen_cls(latent_dim, seq_len, **kwargs))
            self.discriminators.append(dis_cls(seq_len, **kwargs))
        self.generators = nn.ModuleList(self.generators)
        self.discriminators = nn.ModuleList(self.discriminators)
        
        if central_disc_type == "LSTM":
            cd_cls = LSTMDiscriminator
        else:
            cd_cls = Discriminator
        self.central_disc = cd_cls(seq_dim * seq_len, **kwargs)
        self.gamma = gamma
        self.lr = lr
        self.loss_fn = nn.BCELoss()

    def configure_optimizers(self):
        optims = [
            torch.optim.Adam(
                self.central_disc.parameters(), lr=self.lr["CD"], betas=[0.5, 0.9]
            )
        ]
        for i in range(self.seq_dim):
            optims.append(
                torch.optim.Adam(
                    self.generators[i].parameters(), lr=self.lr["G"], betas=[0.5, 0.9]
                )
            )
            optims.append(
                torch.optim.Adam(
                    self.discriminators[i].parameters(),
                    lr=self.lr["D"],
                    betas=[0.5, 0.9],
                )
            )
        return optims, []

    def training_step(self, batch):
        optims = self.optimizers()
        optim_cd = optims[0]
        optim_gs = optims[1::2]
        optim_ds = optims[2::2]

        x = batch["seq"]
        batch_size = x.shape[0]

        z = torch.randn(batch_size, self.latent_dim).to(x)

        signal_group = [x[..., i] for i in range(self.seq_dim)]

        # Generate
        generated_samples = [self.generators[i](z) for i in range(self.seq_dim)]

        generated_samples_labels = torch.zeros((batch_size, 1)).to(x)
        real_samples_labels = torch.ones((batch_size, 1)).to(x)
        all_samples_labels = torch.concat(
            (real_samples_labels, generated_samples_labels)
        )

        # Data for training the discriminators
        all_samples_group = [
            torch.concat((signal_group[i], generated_samples[i]))
            for i in range(self.seq_dim)
        ]

        # Training the discriminators
        # outputs_D = []
        loss_D = 0
        for i in range(self.seq_dim):
            optim_ds[i].zero_grad()
            outputs_D = self.discriminators[i](all_samples_group[i])
            loss_Di = self.loss_fn(outputs_D, all_samples_labels)
            self.manual_backward(loss_Di, retain_graph=True)
            # loss_D[i].backward(retain_graph=True)
            optim_ds[i].step()
            loss_D += loss_Di
        loss_D = loss_D / self.seq_dim

        # Training the central discriminator
        group_generated = torch.concat(generated_samples, dim=-1)
        group_real = torch.concat(signal_group, dim=-1)

        all_samples_central = torch.concat((group_generated, group_real))
        all_samples_labels_central = torch.concat(
            (torch.zeros((batch_size, 1)), torch.ones((batch_size, 1)))
        ).to(x)

        # Training the central discriminator
        optim_cd.zero_grad()
        output_central_discriminator = self.central_disc(all_samples_central)
        loss_CD = self.loss_fn(output_central_discriminator, all_samples_labels_central)
        self.manual_backward(loss_CD, retain_graph=True)
        # loss_central_discriminator.backward(retain_graph=True)
        optim_cd.step()

        # Training the generators
        # outputs_G = {}
        loss_G_local = []
        loss_G = 0
        for i in range(self.seq_dim):
            optim_gs[i].zero_grad()
            outputs_Gi = self.discriminators[i](generated_samples[i])
            loss_G_locali = self.loss_fn(outputs_Gi, real_samples_labels)
            loss_G_local.append(loss_G_locali)

            generated_samples_new = []
            for j in range(self.seq_dim):
                samples_new = self.generators[j](z)

                if i != j:
                    samples_new = samples_new.detach()

                generated_samples_new.append(samples_new)

            all_generated_samples = torch.concat(generated_samples_new, dim=-1)

            samples_central_new = torch.concat((all_generated_samples, group_real))
            output_central_discriminator_new = self.central_disc(samples_central_new)
            loss_central_discriminator_new_i = self.loss_fn(
                output_central_discriminator_new, all_samples_labels_central
            )

            loss_Gi = loss_G_locali - self.gamma * loss_central_discriminator_new_i
            self.manual_backward(loss_Gi, retain_graph=True)
            optim_gs[i].step()
            loss_G += loss_Gi
        loss_G = loss_G / self.seq_dim

        self.log_dict(
            {
                "loss_G": loss_G,
                "loss_D": loss_D,
                "loss_CD": loss_CD,
            },
            on_epoch=True,
            on_step=False,
        )
        # TODO: ES for GAN?

    def validation_step(self, batch, batch_idx): ...

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        z = torch.randn((n_sample, self.latent_dim)).to(self.device)
        x = torch.stack([self.generators[i](z) for i in range(self.seq_dim)], dim=-1)
        return x
