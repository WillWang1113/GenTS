from torch import nn    
from torchvision.ops import MLP


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