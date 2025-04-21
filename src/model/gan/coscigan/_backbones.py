import torch
from torch import nn
from torchvision.ops import MLP


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
