import math
from typing import Callable, List
import torch
import torch.nn as nn

torch.cuda.empty_cache()
n_condition = 1


class ScalarEmbedding(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim):
        super(ScalarEmbedding, self).__init__()
        self.embedding_layer_1 = nn.Linear(input_dim, seq_len)
        self.embedding_layer_2 = nn.Linear(seq_len, seq_len * hidden_dim)

    def forward(self, x):
        x = self.embedding_layer_1(x.float())
        x = self.embedding_layer_2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_value: float):
        super().__init__()
        self.max_value = max_value

        linear_dim = dim // 2
        periodic_dim = dim - linear_dim

        self.scale = torch.exp(
            -2
            * torch.arange(0, periodic_dim).float()
            * math.log(self.max_value)
            / periodic_dim
        )
        self.shift = torch.zeros(periodic_dim)
        self.shift[::2] = 0.5 * math.pi

        self.linear_proj = nn.Linear(1, linear_dim)

    def forward(self, t):
        periodic = torch.sin(t * self.scale.to(t) + self.shift.to(t))
        linear = self.linear_proj(t / self.max_value)
        return torch.cat([linear, periodic], -1)


class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: Callable = nn.ReLU(),
        final_activation: Callable = None,
    ):
        super().__init__()

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)

        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(activation)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self, seq_len, dim, hidden_dim, max_i, num_layers=8, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)

        self.input_proj = FeedForward(dim, [], hidden_dim)
        self.conditional_proj = ScalarEmbedding(seq_len, n_condition, hidden_dim)

        self.proj = FeedForward(
            4 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU()
        )

        self.enc_att = []
        self.i_proj = []
        for _ in range(num_layers):
            self.enc_att.append(
                nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
            )
            self.i_proj.append(nn.Linear(3 * hidden_dim, hidden_dim))
        self.enc_att = nn.ModuleList(self.enc_att)
        self.i_proj = nn.ModuleList(self.i_proj)

        self.output_proj = FeedForward(hidden_dim, [], dim)

    def forward(self, x, t, i, bm):
        shape = x.shape
        x = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)

        x = self.input_proj(x)
        t = self.t_enc(t)
        i = self.i_enc(i)
        bm = self.conditional_proj(bm.view(-1, 1)).view(-1, self.seq_len, self.hidden_dim)
        x = self.proj(torch.cat([x, t, i, bm], -1))
        # print(f"shape(x, t, i, bm): {x.shape, t.shape, i.shape, bm.shape}")

        for att_layer, i_proj in zip(self.enc_att, self.i_proj):
            y, _ = att_layer(query=x, key=x, value=x)
            x = x + torch.relu(y)

        x = self.output_proj(x)
        x = x.view(*shape)
        return x
