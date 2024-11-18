import copy
import torch
from torch import nn
from torchvision.ops import MLP


class ConvEncoder(nn.Module):
    def __init__(self, seq_len, seq_dim, latent_dim, hidden_size_list=[64, 128, 256], **kwargs):
        super().__init__()
        current_seq_len = seq_len
        in_dim = seq_dim

        # Build Encoder
        modules = []
        for h_dim in hidden_size_list:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_dim = h_dim
            current_seq_len = current_seq_len // 2
        self.latent_seq_len = current_seq_len
        self.encoder = nn.Sequential(*modules)
        self.linear = nn.Linear(hidden_size_list[-1] * current_seq_len, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x.flatten(start_dim=1))
        return x


class ConvDecoder(nn.Module):
    def __init__(self, seq_len, seq_dim, latent_dim, hidden_size_list=[256, 128, 64], **kwargs):
        super().__init__()
        latent_seq_len = copy.deepcopy(seq_len)
        for i in range(len(hidden_size_list)):
            latent_seq_len = latent_seq_len // 2
        self.latent_seq_len = latent_seq_len

        self.decoder_input = nn.Linear(
            latent_dim, hidden_size_list[0] * latent_seq_len
        )
        self.hidden_size_list = hidden_size_list

        modules = []
        for i in range(len(hidden_size_list) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        hidden_size_list[i],
                        hidden_size_list[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm1d(hidden_size_list[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_size_list[-1],
                hidden_size_list[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm1d(hidden_size_list[-1]),
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_size_list[-1],
                out_channels=seq_dim,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, self.hidden_size_list[0], self.latent_seq_len)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x
