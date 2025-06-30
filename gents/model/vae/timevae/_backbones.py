import copy
import torch
from torch import nn
from torchvision.ops import MLP


class ConvEncoder(nn.Module):
    def __init__(
        self, seq_len, seq_dim, latent_dim, hidden_size_list=[64, 128, 256], **kwargs
    ):
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
        x = self.encoder(x.permute(0, 2, 1))
        x = self.linear(x.flatten(start_dim=1))
        return x


class ConvDecoder(nn.Module):
    def __init__(
        self,
        seq_len,
        seq_dim,
        latent_dim,
        hidden_size_list = [256, 128, 64],
        **kwargs,
    ):
        super().__init__()
        latent_seq_len = copy.deepcopy(seq_len)
        for i in range(len(hidden_size_list)):
            latent_seq_len = latent_seq_len // 2
        self.latent_seq_len = latent_seq_len

        self.decoder_input = nn.Linear(latent_dim, hidden_size_list[0] * latent_seq_len)
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
        return x.permute(0, 2, 1)



class TrendLayer(nn.Module):
    """
    The TrendLayer class is a neural network module that models the trend component of a time series.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        feat_dim (int): Dimensionality of the feature space.
        trend_poly (int): The order of the polynomial used to model the trend.
        seq_len (int): The length of the input sequence.
    """

    def __init__(self, latent_dim, feat_dim, trend_poly, seq_len):
        super(TrendLayer, self).__init__()
        self.feat_dim = feat_dim
        self.trend_poly = trend_poly
        self.seq_len = seq_len
        self.trend_dense1 = MLP(latent_dim, [feat_dim * trend_poly])
        self.trend_dense2 = MLP(feat_dim * trend_poly, [feat_dim * trend_poly])

    def forward(self, z):
        trend_params = self.trend_dense1(z)
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(
            -1, self.feat_dim, self.trend_poly
        )  # shape: N x D x P

        lin_space = torch.linspace(
            0, 1, self.seq_len
        )  # shape of lin_space: 1d tensor of length T
        poly_space = torch.stack(
            [lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0
        )  # shape: P x T
        poly_space = poly_space.to(z)

        trend_vals = torch.matmul(
            trend_params, poly_space.unsqueeze(0)
        )  # shape (N, D, T)
        trend_vals = trend_vals.permute(0, 2, 1)  # shape: (N, T, D)

        return trend_vals


class SeasonalLayer(nn.Module):
    def __init__(self, latent_dim, feat_dim, seq_len, custom_seas, **kwargs):
        super(SeasonalLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.custom_seas = custom_seas
        self.dense_layers = nn.ModuleList(
            [
                MLP(latent_dim, hidden_channels=[feat_dim * num_seasons])
                for i, (num_seasons, len_per_season) in enumerate(custom_seas)
            ]
        )

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = torch.arange(num_seasons).unsqueeze(1) + torch.zeros(
            (num_seasons, len_per_season), dtype=torch.int32
        )
        season_indexes = season_indexes.flatten()
        # Ensure the length matches seq_len
        season_indexes = season_indexes.repeat(self.seq_len // len_per_season + 1)[
            : self.seq_len
        ]
        return season_indexes

    def forward(self, z: torch.Tensor):
        N = z.shape[0]
        ones_tensor = torch.ones(
            [N, self.feat_dim, self.seq_len], dtype=torch.int32
        ).to(z.device)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)  # shape: (N, D * S)
            season_params = season_params.view(
                N, self.feat_dim, num_seasons
            )  # shape: (N, D, S)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            )  # shape: (T, )
            season_indexes_over_time = season_indexes_over_time.to(z.device)

            dim2_idxes = ones_tensor * season_indexes_over_time.view(
                1, 1, -1
            )  # shape: (N, D, T)
            season_vals = torch.gather(
                season_params, dim=2, index=dim2_idxes
            )  # shape (N, D, T)

            all_seas_vals.append(season_vals)

        all_seas_vals = torch.stack(all_seas_vals, dim=-1)  # shape: (N, D, T, S)
        all_seas_vals = all_seas_vals.sum(dim=-1)  # shape (N, D, T)
        all_seas_vals = all_seas_vals.permute(0, 2, 1)  # shape (N, T, D)
        return all_seas_vals


class TrendSeasonalDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        seq_len,
        seq_dim,
        trend_poly=0,
        custom_seas=None,
        **kwargs,
    ):
        super().__init__()
        self.seq_dim = seq_dim
        self.seq_len = seq_len
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        # self.use_residual_conn = use_residual_conn
        self.level_mlp = MLP(latent_dim, [seq_dim, seq_dim])

        if self.trend_poly is not None and self.trend_poly > 0:
            self.trend_vals = TrendLayer(latent_dim, seq_dim, trend_poly, seq_len)
        else:
            self.trend_vals = torch.nn.Identity()

        if self.custom_seas is not None and len(self.custom_seas) > 0:
            self.cust_seas_vals = SeasonalLayer(
                latent_dim=latent_dim,
                feat_dim=seq_dim,
                seq_len=seq_len,
                custom_seas=custom_seas,
            )
        else:
            self.cust_seas_vals = nn.Sequential(nn.Linear(latent_dim, seq_dim * seq_len))

    def forward(self, z):
        z_trend = self.trend_vals(z)
        z_seasonal = self.cust_seas_vals(z).reshape(-1, self.seq_len, self.seq_dim)
        return z_trend + z_seasonal

