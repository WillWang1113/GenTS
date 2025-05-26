from torch import nn

from src.common._modules import MLPDecoder, MLPEncoder


class Generator(nn.Module):
    def __init__(
        self, seq_len, seq_dim, latent_dim, hidden_size_list=[256, 128, 64], **kwargs
    ):
        super().__init__()
        self.dec = MLPDecoder(seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs)
        condition = kwargs.get("condition")
        if condition:
            if condition == "predict":
                # assert kwargs.get("obs_len") is not None
                obs_len = kwargs.get("obs_len")
                self.cond_net = MLPEncoder(
                    obs_len, seq_dim, latent_dim, hidden_size_list, **kwargs
                )
            elif condition == "impute":
                # assert kwargs.get('obs_len') is not None
                # obs_len = kwargs.get('obs_len')
                self.cond_net = MLPEncoder(
                    seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs
                )

    def forward(self, z, c=None):
        if (c is not None) and (self.cond_net is not None):
            z = z + self.cond_net(c)
        return self.dec(z)


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
        self.out_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, 1),
        )
        if last_sigmoid:
            self.out_mlp.append(nn.Sigmoid())

        condition = kwargs.get("condition")
        if condition:
            if condition == "predict":
                # assert kwargs.get("obs_len") is not None
                obs_len = kwargs.get("obs_len")
                self.cond_net = MLPEncoder(
                    obs_len, seq_dim, latent_dim, hidden_size_list, **kwargs
                )
            elif condition == "impute":
                # assert kwargs.get('obs_len') is not None
                # obs_len = kwargs.get('obs_len')
                self.cond_net = MLPEncoder(
                    seq_len, seq_dim, latent_dim, hidden_size_list, **kwargs
                )

    def forward(self, x, c=None):
        latents = self.enc(x)
        if (c is not None) and (self.cond_net is not None):
            latents = latents + self.cond_net(c)
        validity = self.out_mlp(latents)
        return validity
