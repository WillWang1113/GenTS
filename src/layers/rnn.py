from torch import nn    
from torchvision.ops import MLP


class GRULayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_size=128,
        out_dim=128,
        num_layers=1,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.gru = nn.GRU(
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