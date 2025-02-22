import torch
from torch import nn
from torchvision.ops import MLP

from src.layers.conv import ConvDecoder, ConvEncoder
from src.model.vanillavae import VanillaVAE


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
            self.cust_seas_vals = torch.nn.Identity()

    def forward(self, z):
        z_trend = self.trend_vals(z)
        z_seasonal = self.cust_seas_vals(z)
        return z_trend + z_seasonal


# TODO：Implement TimeVAE BUG inherent
class TimeVAE(VanillaVAE):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int,
        hidden_size_list=[64, 128, 256],
        beta: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
        **kwargs,
    ):
        """
        hidden_layer_sizes: list of number of filters in convolutional layers in encoder and residual connection of decoder.
        trend_poly: integer for number of orders for trend component. e.g. setting trend_poly = 2 will include linear and quadratic term.
        custom_seas: list of tuples of (num_seasons, len_per_season).
            num_seasons: number of seasons per cycle.
            len_per_season: number of epochs (time-steps) per season.
        use_residual_conn: boolean value indicating whether to use a residual connection for reconstruction in addition to
        trend, generic and custom seasonalities.
        """

        super().__init__(
            seq_len, seq_dim, latent_dim, hidden_size_list, beta, lr, weight_decay
        )
        self.save_hyperparameters()
        # encoding
        self.encoder = ConvEncoder(seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs)
        hidden_size_list.reverse()
        self.decoder = ConvDecoder(seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs)
        self.trend_season_dec = TrendSeasonalDecoder(**self.hparams_initial)

    def encode(self, x, c=None):
        latents = self.encoder(x)
        mu = self.fc_mu(latents)
        logvar = self.fc_logvar(latents)
        return latents, mu, logvar

    def decode(self, z, c=None):
        return self.decoder(z) + self.trend_season_dec(z)
