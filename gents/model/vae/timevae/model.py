from typing import List, Tuple

from ..vanillavae.model import VanillaVAE
from ._backbones import ConvDecoder, ConvEncoder, TrendSeasonalDecoder


class TimeVAE(VanillaVAE):
    """`TimeVAE <https://arxiv.org/abs/2111.08095>`__ for time series generation.
    
    Adapted from the `official codes <https://github.com/abudesai/timeVAE>`__
    
    .. note::
        The orignial codes are based on Tensorflow, we adapt the source codes into pytorch.
    
    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to None.
        latent_dim (int, optional): Latent variable dimension. Defaults to 128.
        hidden_size_list (list, optional): Hidden size for encoder and decoder. Defaults to [64, 128, 256].
        w_kl (float, optional): Loss weight of KL div. Defaults to 1e-4.
        trend_poly (int, optional): integer for number of orders for trend component. e.g. setting trend_poly = 2 will include linear and quadratic term.
        custom_seas (List[Tuple[int, int]], optional): list of tuples of (num_seasons, len_per_season). num_seasons: number of seasons per cycle. len_per_season: number of epochs (time-steps) per season.
        use_residual_conn (bool, optional): boolean value indicating whether to use a residual connection for reconstruction in addition to trend, generic and custom seasonalities.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """
    
    ALLOW_CONDITION = [None]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        latent_dim: int = 128,
        hidden_size_list=[64, 128, 256],
        w_kl: float = 5e-3,
        trend_poly: int = 2,
        custom_seas: List[Tuple[int, int]] = None,
        use_residual_conn: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        """
        Args:
            seq_len (int): Target sequence length
            seq_dim (int): Target sequence dimension, for univariate time series, set as 1
            condition (str, optional): Given conditions, allowing [None, 'predict', 'impute']. Defaults to None.
            latent_dim (int, optional): Latent dimension for z. Defaults to 128.
            hidden_size_list (list, optional): Hidden size for encoder and decoder. Defaults to [64, 128, 256].
            w_kl (float, optional): Loss weight of KL div. Defaults to 1e-4.
            trend_poly (int, optional): integer for number of orders for trend component. e.g. setting trend_poly = 2 will include linear and quadratic term.
            custom_seas (List[Tuple[int, int]], optional): list of tuples of (num_seasons, len_per_season). num_seasons: number of seasons per cycle. len_per_season: number of epochs (time-steps) per season.
            use_residual_conn (bool, optional): boolean value indicating whether to use a residual connection for reconstruction in addition to trend, generic and custom seasonalities.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        """

        super().__init__(
            seq_len,
            seq_dim,
            condition,
            latent_dim,
            hidden_size_list,
            w_kl,
            lr,
            weight_decay,
        )
        self.save_hyperparameters()

        # override the encoder and decoder
        self.encoder = ConvEncoder(
            seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs
        )
        # hidden_size_list.reverse()
        self.decoder = ConvDecoder(
            seq_len, seq_dim, latent_dim, hidden_size_list[::-1], **kwargs
        )
        self.trend_season_dec = TrendSeasonalDecoder(**self.hparams_initial)

    def _encode(self, x, c=None, **kwargs):
        latents = self.encoder(x)
        mu = self.fc_mu(latents)
        logvar = self.fc_logvar(latents)
        return latents, mu, logvar

    def _decode(self, z, c=None, **kwargs):
        return self.decoder(z) + self.trend_season_dec(z)
