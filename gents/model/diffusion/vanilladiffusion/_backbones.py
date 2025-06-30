import torch
import torch.nn as nn
import math
from gents.common._modules import MLPDecoder, MLPEncoder
from torchvision.ops import MLP

class TimestepEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
    

class Denoiser(nn.Module):
    def __init__(
        self,
        seq_len,
        seq_dim,
        latent_dim,
        hidden_size_list=[64, 128, 256],
        condition: str = None,
        **kwargs,
    ):
        super().__init__()

        self.enc = MLPEncoder(seq_len, seq_dim, latent_dim, [], **kwargs)
        self.net = nn.ModuleList(
            [
                MLP(
                    in_channels=latent_dim,
                    hidden_channels=[hidden_size_list[i], latent_dim],
                    dropout=kwargs.get("dropout", 0.0),
                )
                for i in range(len(hidden_size_list))
            ]
        )
        self.dec = MLPDecoder(seq_len, seq_dim, latent_dim, [], **kwargs)
        self.pe = TimestepEmbed(latent_dim)
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

    def forward(self, x, t, c=None):
        x = self.enc(x)
        t = self.pe(t)
        x = x + t
        if (c is not None) and (self.cond_net is not None):
            cond_lats = self.cond_net(c)
            x = x + cond_lats
        for layer in self.net:
            x = x + layer(x)
        return self.dec(x)



class DenoiserTransformer(nn.Module):
    def __init__(
        self,
        seq_len,
        seq_dim,
        latent_dim,
        n_head=4,
        num_layers=2,
        condition: str = None,
        **kwargs,
    ):
        super().__init__()
        
        self.pe = TimestepEmbed(latent_dim)
        self.embedder = nn.Linear(in_features=seq_dim, out_features=latent_dim)
        self.unembedder = nn.Linear(in_features=latent_dim, out_features=seq_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_head, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers
        )
        self.hidden_size=latent_dim
        
        if condition:
            if condition == "predict":
                # assert kwargs.get("obs_len") is not None
                obs_len = kwargs.get("obs_len")
                self.cond_net = MLPEncoder(
                    obs_len, seq_dim, latent_dim, [latent_dim], **kwargs
                )
            elif condition == "impute":
                # assert kwargs.get('obs_len') is not None
                # obs_len = kwargs.get('obs_len')
                self.cond_net = MLPEncoder(
                    seq_len, seq_dim, latent_dim, [latent_dim], **kwargs
                )

    def forward(self, x, t, c=None):
        x = self.embedder(x)
        t = self.pe(t).unsqueeze(1)
        x = x + t
        if (c is not None) and (self.cond_net is not None):
            cond_lats = self.cond_net(c)
            x = x + cond_lats.unsqueeze(1)
        x = self.backbone(x)
        return self.unembedder(x)


            



class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps.
    Courtesy of https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, d_model: int, scale: float = 30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.d_model = d_model
        self.W = nn.Parameter(
            torch.randn((d_model + 1) // 2) * scale, requires_grad=False
        )

        self.dense = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, use_time_axis: bool = True
    ) -> torch.Tensor:
        time_proj = timesteps[:, None] * self.W[None, :] * 2 * torch.pi
        embeddings = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)

        # Slice to get exactly d_model
        t_emb = embeddings[:, : self.d_model]  # (batch_size, d_model)

        if use_time_axis:
            t_emb = t_emb.unsqueeze(1)

        projected_emb: torch.Tensor = self.dense(t_emb)

        return x + projected_emb
