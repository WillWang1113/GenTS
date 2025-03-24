from torch import nn    
from torchvision.ops import MLP
import torch

class RNNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_size=128,
        out_dim=128,
        num_layers=1,
        dropout=0.0,
        rnn_type='gru',
        **kwargs,
    ):
        super().__init__()
        assert rnn_type in ['gru', 'lstm', 'rnn']
        if rnn_type == 'gru':
            rnn_class = nn.GRU
        elif rnn_type == 'lstm':
            rnn_class = nn.LSTM
        else:
            rnn_class = nn.RNN

        self.gru = rnn_class(
            in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        emb = self.proj(h)
        return emb
    
class FullGRUODECell_Autonomous(nn.Module):
    
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        #self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        #self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time of evaluation
            h        hidden state (current)

        Returns:
            Updated h
        """
        #xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh