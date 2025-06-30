import torch
from torch import nn

from gents.common._modules import RNNLayer


class Generator(nn.Module):
    def __init__(
        self,
        seq_dim,
        latent_dim,
        hidden_size,
        num_layers,
        class_emb_dim,
        n_classes=0,
        **kwargs,
    ):
        super().__init__()
        self.dec = RNNLayer(
            latent_dim + class_emb_dim, hidden_size, seq_dim, num_layers, rnn_type="gru"
        )
        self.n_classes = n_classes
        self.emb = nn.Embedding(n_classes + 1, class_emb_dim)

    def forward(self, z, c=None):
        if c is None:
            c = torch.ones((z.shape[0], 1)) * self.n_classes
            c = c.to(z).long()

        cond = self.emb(c).unsqueeze(1).expand(-1, z.shape[1], -1)
        z = torch.concat([z, cond], dim=-1)
        return self.dec(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        seq_dim,
        latent_dim,
        hidden_size,
        num_layers,
        class_emb_dim,
        n_classes=0,
        **kwargs,
    ):
        super().__init__()
        self.enc = RNNLayer(
            seq_dim + class_emb_dim, hidden_size, 1, num_layers, rnn_type="gru"
        )
        self.n_classes = n_classes
        self.emb = nn.Embedding(n_classes + 1, class_emb_dim)

    def forward(self, x, c=None):
        if c is None:
            c = torch.ones((x.shape[0], 1)) * self.n_classes
            c = c.to(x).long()
        
        cond = self.emb(c).unsqueeze(1).expand(-1, x.shape[1], -1)
        z = torch.concat([x, cond], dim=-1)
        return self.enc(z)
