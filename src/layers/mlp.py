import copy
import torch
from torch import nn
from torchvision.ops import MLP


class MLPEncoder(nn.Module):
    def __init__(
        self, seq_len, seq_dim, latent_dim, hidden_size_list=[64, 128, 256], **kwargs
    ):
        super().__init__()
        # Build Encoder
        self.encoder = MLP(seq_len * seq_dim, hidden_size_list + [latent_dim])

    def forward(self, x):
        x = self.encoder(x.flatten(1))
        return x


class MLPDecoder(nn.Module):
    def __init__(
        self,
        seq_len,
        seq_dim,
        latent_dim,
        hidden_size_list=[256, 128, 64],
        **kwargs,
    ):
        super().__init__()
        self.seq_len=seq_len
        self.seq_dim=seq_dim
        # Build Decoder
        self.decoder = MLP(latent_dim, hidden_size_list + [seq_len * seq_dim])

    def forward(self, x):
        x = self.decoder(x)
        return x.reshape(-1, self.seq_len, self.seq_dim)
    
class FinalTanh(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(
            hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(
            hidden_hidden_channels, input_channels * hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels,
                         self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):

        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(
            *z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z
    