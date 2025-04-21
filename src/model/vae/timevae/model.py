from ..vanillavae.model import VanillaVAE
from ._backbones import ConvDecoder, ConvEncoder, TrendSeasonalDecoder


class TimeVAE(VanillaVAE):
    ALLOW_CONDITION = [None]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int = 128,
        hidden_size_list=[64, 128, 256],
        beta: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        trend_poly=2,
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

        # override the encoder and decoder
        self.encoder = ConvEncoder(
            seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs
        )
        hidden_size_list.reverse()
        self.decoder = ConvDecoder(
            seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs
        )
        self.trend_season_dec = TrendSeasonalDecoder(**self.hparams_initial)

    def encode(self, x, c=None, **kwargs):
        latents = self.encoder(x)
        mu = self.fc_mu(latents)
        logvar = self.fc_logvar(latents)
        return latents, mu, logvar

    def decode(self, z, c=None, **kwargs):
        return self.decoder(z) + self.trend_season_dec(z)
