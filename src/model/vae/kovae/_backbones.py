import torch
import torch.nn as nn
from src.common._modules import FinalTanh, NeuralCDE


class VKEncoderIrregular(nn.Module):
    def __init__(self, args):
        super(VKEncoderIrregular, self).__init__()
        self.args = args
        self.z_dim = self.args.z_dim
        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim
        self.batch_norm = self.args.batch_norm
        self.num_layers = self.args.num_layers

        if self.batch_norm:
            self.b_norm = nn.BatchNorm1d(self.hidden_dim * 2)

        ode_func = FinalTanh(
            self.inp_dim, self.hidden_dim, self.hidden_dim, self.num_layers
        )
        self.emb = NeuralCDE(
            func=ode_func,
            input_channels=self.inp_dim,
            hidden_channels=self.hidden_dim,
            output_channels=self.hidden_dim,
        )
        self.rnn = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, time, train_coeffs, final_index):
        # encode
        h = self.emb(time, train_coeffs, final_index)
        h, _ = self.rnn(h)
        if self.batch_norm:
            h = self.b_norm(torch.permute(h, (0, 2, 1)))
            h = torch.permute(h, (0, 2, 1))  # permute back to b x s x c
        return h


class VKEncoder(nn.Module):
    def __init__(self, args, num_layers=3):
        super(VKEncoder, self).__init__()
        self.args = args
        self.z_dim = self.args.z_dim
        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim
        self.batch_norm = self.args.batch_norm

        if self.batch_norm:
            self.b_norm = nn.BatchNorm1d(self.hidden_dim * 2)

        self.rnn = nn.GRU(
            input_size=self.inp_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=args.num_layers,
            batch_first=True,
        )

    def forward(self, x):
        # encode
        h, _ = self.rnn(x)  # b x seq_len x channels
        if self.batch_norm:
            h = self.b_norm(torch.permute(h, (0, 2, 1)))
            h = torch.permute(h, (0, 2, 1))  # permute back to b x s x c
        return h


class VKDecoder(nn.Module):
    def __init__(self, args):
        super(VKDecoder, self).__init__()
        self.args = args
        self.z_dim = self.args.z_dim
        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim

        self.rnn = nn.GRU(
            input_size=self.z_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=args.num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(self.args.hidden_dim * 2, self.args.inp_dim)

    def forward(self, z):
        # decode
        h, _ = self.rnn(z)
        x_hat = self.linear(h)
        # x_hat = nn.functional.sigmoid(self.linear(h))
        return x_hat

