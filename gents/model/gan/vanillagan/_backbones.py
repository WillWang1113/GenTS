from torch import nn

from gents.common._modules import get_backbone, LabelEmbedder


class Generator(nn.Module):
    def __init__(
        self, seq_len, seq_dim, latent_dim, backbone="mlp", backbone_params=None, **kwargs
    ):
        super().__init__()
        EncoderCls, DecoderCls = get_backbone(backbone)
        bp = backbone_params or {}

        self.dec = DecoderCls(seq_len, seq_dim, latent_dim, **bp, **kwargs)

        condition = kwargs.get("condition")
        if condition:
            if condition == "predict":
                obs_len = kwargs.get("obs_len")
                self.cond_net = EncoderCls(
                    obs_len, seq_dim, latent_dim, **bp, **kwargs
                )
            elif condition == "impute":
                self.cond_net = EncoderCls(
                    seq_len, seq_dim, latent_dim, **bp, **kwargs
                )
            elif condition == "super_resolution":
                lr_len = seq_len // kwargs.get("sr_factor")
                self.cond_net = EncoderCls(
                    lr_len, seq_dim, latent_dim, **bp, **kwargs
                )
            elif condition == "class":
                self.cond_net = LabelEmbedder(
                    kwargs.get("class_num"), latent_dim, dropout_prob=0.0
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
        backbone="mlp",
        backbone_params=None,
        last_sigmoid=False,
        **kwargs,
    ):
        super().__init__()
        EncoderCls, _ = get_backbone(backbone)
        bp = backbone_params or {}

        self.enc = EncoderCls(seq_len, seq_dim, latent_dim, **bp, **kwargs)
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
                obs_len = kwargs.get("obs_len")
                self.cond_net = EncoderCls(
                    obs_len, seq_dim, latent_dim, **bp, **kwargs
                )
            elif condition == "impute":
                self.cond_net = EncoderCls(
                    seq_len, seq_dim, latent_dim, **bp, **kwargs
                )
            elif condition == "super_resolution":
                lr_len = seq_len // kwargs.get("sr_factor")
                self.cond_net = EncoderCls(
                    lr_len, seq_dim, latent_dim, **bp, **kwargs
                )
            elif condition == "class":
                self.cond_net = LabelEmbedder(
                    kwargs.get("class_num"), latent_dim, dropout_prob=0.0
                )

    def forward(self, x, c=None):
        latents = self.enc(x)
        if (c is not None) and (self.cond_net is not None):
            latents = latents + self.cond_net(c)
        validity = self.out_mlp(latents)
        return validity
