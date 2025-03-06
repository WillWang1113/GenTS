import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.vae.vanillavae import VanillaVAE
from src.layers.nde import NeuralCDE, natural_cubic_spline_coeffs
from src.layers.mlp import FinalTanh


class VKEncoderIrregular(nn.Module):
    def __init__(self, latent_dim, seq_dim, hidden_size, batch_norm, num_layers=3):
        super(VKEncoderIrregular, self).__init__()
        # self.args = args
        self.z_dim = latent_dim
        self.inp_dim = seq_dim
        self.hidden_dim = hidden_size
        self.batch_norm = batch_norm
        self.num_layers = num_layers

        if self.batch_norm:
            self.b_norm = nn.BatchNorm1d(self.hidden_dim * 2)

        ode_func = FinalTanh(
            self.inp_dim, self.hidden_dim, self.hidden_dim, self.num_layers
        )
        self.rnn = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )
        self.emb = NeuralCDE(
            func=ode_func,
            input_channels=self.inp_dim,
            hidden_channels=self.hidden_dim,
            output_channels=self.hidden_dim,
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
    def __init__(self, latent_dim, seq_dim, hidden_size, batch_norm, num_layers=3):
        super(VKEncoder, self).__init__()
        # self.args = args
        self.z_dim = latent_dim
        self.inp_dim = seq_dim
        self.hidden_dim = hidden_size
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.b_norm = nn.BatchNorm1d(self.hidden_dim * 2)

        self.rnn = nn.GRU(
            input_size=self.inp_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=num_layers,
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
    def __init__(self, latent_dim, seq_dim, hidden_size, num_layers):
        super(VKDecoder, self).__init__()
        # self.args = args
        self.z_dim = latent_dim
        self.inp_dim = seq_dim
        self.hidden_dim = hidden_size

        self.rnn = nn.GRU(
            input_size=self.z_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size * 2, seq_dim)

    def forward(self, z):
        # decode
        h, _ = self.rnn(z)
        x_hat = self.linear(h)

        # ! Originally, added a sigmoid output layer
        # x_hat = nn.functional.sigmoid(self.linear(h))

        return x_hat


class KoVAE(VanillaVAE):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int = 16,
        hidden_size: int = 20,
        num_layers: int = 3,
        batch_norm: bool = True,
        beta: float = 1e-3,
        gamma: float = 5e-4,
        pinv_solver: bool = False,
        koopman_nstep: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        condition: str = None,
        **kwargs,
    ):
        super().__init__(
            seq_len,
            seq_dim,
            latent_dim,
            beta=beta,
            lr=lr,
            weight_decay=weight_decay,
            condition=None,
        )
        self.koopman_nstep = koopman_nstep
        self.latent_dim = latent_dim  # latent
        self.seq_dim = seq_dim  # seq channel (multivariate features)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.gamma = gamma
        self.pinv_solver = pinv_solver
        self.missing_value = True if condition == "impute" else False

        if self.missing_value:
            self.encoder = VKEncoderIrregular(
                latent_dim, seq_dim, hidden_size, batch_norm, num_layers=num_layers
            )
        else:
            self.encoder = VKEncoder(
                latent_dim, seq_dim, hidden_size, batch_norm, num_layers=num_layers
            )
        self.decoder = VKDecoder(latent_dim, seq_dim, hidden_size, num_layers)

        # ----- Prior of content is a uniform Gaussian and Prior of motion is an LSTM
        self.z_prior_gru = nn.GRUCell(self.latent_dim, self.hidden_size)

        self.z_prior_mean = nn.Linear(self.hidden_size, self.latent_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_size, self.latent_dim)

        # ----- Posterior of sequence
        self.fc_mu = nn.Linear(self.hidden_size * 2, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_size * 2, self.latent_dim)


    def encode(self, x: torch.Tensor, c=None, **kwargs):
        if self.missing_value:
            time = torch.arange(self.seq_len).to(x)
            final_index = (torch.ones(x.shape[0]) * (self.seq_len-1)).to(x)
            mask = kwargs.get('mask')
            x_masked_nan = x.masked_fill(mask.bool(), float('nan'))
            x = natural_cubic_spline_coeffs(time, x_masked_nan)
            latents = self.encoder(time, x, final_index)
        else:
            latents = self.encoder(x)
        mu = self.fc_mu(latents)
        logvar = self.fc_logvar(latents)
        return latents, mu, logvar

    def compute_operator_and_pred(self, z):
        z_past, z_future = z[:, :-1], z[:, 1:]  # split latent

        # solve linear system (broadcast)
        if self.pinv_solver:
            Ct = torch.linalg.pinv(z_past.reshape(-1, self.latent_dim)) @ z_future.reshape(
                -1, self.latent_dim
            )

        else:
            # self.qr_solver
            Q, R = torch.linalg.qr(z_past.reshape(-1, self.latent_dim))
            B = Q.T @ z_future.reshape(-1, self.latent_dim)
            Ct = torch.linalg.solve_triangular(R, B, upper=True)

        # predict (broadcast)
        z_pred = z_past @ Ct

        err = 0.0
        z_hat = z_past
        for jj in range(self.koopman_nstep):
            z_hat = z_hat @ Ct
            err += F.mse_loss(z_hat[:, : -jj or None], z[:, (jj + 1) :]) / torch.norm(
                z_hat[:, : -jj or None], p="fro"
            )

        return Ct, z_pred, err

    def _get_loss(self, batch):
        loss_dict, z_prior, mu_prior, logvar_prior, z, mu, logvar = super()._get_loss(
            batch
        )

        Ct_prior, z_pred_prior, pred_err_prior = self.compute_operator_and_pred(z_prior)
        loss_dict["pred_loss"] = pred_err_prior * self.gamma
        loss_dict["loss"] += loss_dict["pred_loss"]
        return loss_dict, z_prior, mu_prior, logvar_prior, z, mu, logvar

    def _sample_impl(self, n_sample, condition=None, **kwargs):
        # sample from prior
        _, _, z_out = self.sample_prior(n_sample, random_sampling=True)
        x_rec = self.decoder(z_out)
        return x_rec

    # ------ sample z purely from learned LSTM prior with arbitrary seq ------
    def sample_prior(self, n_sample, random_sampling=True):
        batch_size = n_sample

        # z_out = None  # This will ultimately store all z_s in the format [batch_size, seq_len, z_dim]
        z_logvars, z_means, z_out = self.zeros_init(batch_size, self.seq_len)

        # initialize arbitrary input (zeros) and hidden states.
        z_t = torch.zeros(batch_size, self.latent_dim).to(z_out)
        h_t = torch.zeros(batch_size, self.hidden_size).to(z_out)

        for i in range(self.seq_len):
            h_t = self.z_prior_gru(z_t, h_t)

            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparam(z_mean_t, z_logvar_t, random_sampling)

            z_out[:, i] = z_t
            z_means[:, i] = z_mean_t
            z_logvars[:, i] = z_logvar_t

        return z_means, z_logvars, z_out

    def zeros_init(self, batch_size, seq_len):
        z_out = torch.zeros(batch_size, seq_len, self.latent_dim).to(self.device)
        z_means = torch.zeros(batch_size, seq_len, self.latent_dim).to(self.device)
        z_logvars = torch.zeros(batch_size, seq_len, self.latent_dim).to(self.device)
        return z_logvars, z_means, z_out
