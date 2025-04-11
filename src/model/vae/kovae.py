from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.base import BaseModel
from src.layers.nde import NeuralCDE

from src.layers.mlp import FinalTanh
from src.utils.losses import kl_loss



def reparameterize(mean, logvar, random_sampling=True):
    # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
    if random_sampling is True:
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        z = mean + eps * std
        return z
    else:
        return mean


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


class KoVAE(BaseModel):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int = 16,
        hidden_size: int = 20,
        num_layers: int = 3,
        batch_norm: bool = True,
        w_rec: float = 1.0,
        w_kl: float = 0.007,
        w_pred_prior: float = 0.005,
        pinv_solver: bool = False,
        koopman_nstep: int = 1,
        lr: float = 7e-4,
        weight_decay: float = 0.0,
        condition: str = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.koopman_nstep = koopman_nstep
        self.latent_dim = latent_dim  # latent
        self.seq_dim = seq_dim  # seq channel (multivariate features)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        # self.gamma = gamma
        self.pinv_solver = pinv_solver
        self.missing_value = True if condition == "impute" else False

        self.args = Namespace(
            z_dim=latent_dim,
            inp_dim=seq_dim,
            hidden_dim=hidden_size,
            batch_norm=batch_norm,
            num_layers=num_layers,
        )
        if self.missing_value:
            self.encoder = VKEncoderIrregular(self.args)
        else:
            self.encoder = VKEncoder(self.args)
        self.decoder = VKDecoder(self.args)

        # ----- Prior of content is a uniform Gaussian and Prior of motion is an LSTM
        self.z_prior_gru = nn.GRUCell(self.latent_dim, self.hidden_size)

        self.z_prior_mean = nn.Linear(self.hidden_size, self.latent_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_size, self.latent_dim)

        # ----- Posterior of sequence
        self.z_mean = nn.Linear(self.hidden_size * 2, self.latent_dim)
        self.z_logvar = nn.Linear(self.hidden_size * 2, self.latent_dim)

    def compute_operator_and_pred(self, z):
        z_past, z_future = z[:, :-1], z[:, 1:]  # split latent

        # solve linear system (broadcast)
        if self.pinv_solver:
            Ct = torch.linalg.pinv(
                z_past.reshape(-1, self.latent_dim)
            ) @ z_future.reshape(-1, self.latent_dim)

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

    def _sample_impl(self, n_sample, condition=None, **kwargs):
        # sample from prior
        z_mean_prior, z_logvar_prior, z_out = self.sample_prior(
            n_sample, self.seq_len, random_sampling=True
        )
        x_rec = self.decoder(z_out)
        return x_rec

    def loss(self, x, x_rec, Z_enc, Z_enc_prior):
        """
        :param x: The original sequence input
        :param x_rec: The reconstructed sequence
        :param Z_enc: Dictionary of posterior modeling {mean, logvar and sample}
        :param Z_enc_prior: Dictionary of prior modeling {mean, logvar and sample}
        :return: loss value
        """

        # PENALTIES
        a0 = self.hparams.w_rec
        a1 = self.hparams.w_kl
        a2 = self.hparams.w_pred_prior

        batch_size = x.size(0)

        z_post_mean, z_post_logvar, z_post = (
            Z_enc["mean"],
            Z_enc["logvar"],
            Z_enc["sample"],
        )
        z_prior_mean, z_prior_logvar, z_prior = (
            Z_enc_prior["mean"],
            Z_enc_prior["logvar"],
            Z_enc_prior["sample"],
        )
        ## Ap after sampling ##
        Ct_prior, z_pred_prior, pred_err_prior = self.compute_operator_and_pred(z_prior)

        loss = 0.0
        if self.hparams.w_rec:
            recon = F.mse_loss(x_rec, x, reduction="sum") / batch_size
            loss = a0 * recon
            agg_losses = [loss]

        if self.hparams.w_kl:
            kld_z = kl_loss(z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar)
            # kld_z = torch.clamp(kld_z, min=self.budget)
            kld_z = torch.sum(kld_z) / batch_size
            loss += a1 * kld_z
            agg_losses.append(kld_z)

        if self.hparams.w_pred_prior:
            loss += a2 * pred_err_prior
        agg_losses.append(pred_err_prior)

        agg_losses = [loss] + agg_losses
        return tuple(agg_losses)

    # ------ sample z purely from learned LSTM prior with arbitrary seq ------
    def sample_prior(self, n_sample, seq_len, random_sampling=True):
        batch_size = n_sample

        # z_out = None  # This will ultimately store all z_s in the format [batch_size, seq_len, z_dim]
        z_logvars, z_means, z_out = self.zeros_init(batch_size, seq_len)

        # initialize arbitrary input (zeros) and hidden states.
        z_t = torch.zeros(batch_size, self.latent_dim).to(self.device)
        h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)

        for i in range(seq_len):
            h_t = self.z_prior_gru(z_t, h_t)

            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = reparameterize(z_mean_t, z_logvar_t, random_sampling)

            z_out[:, i] = z_t
            z_means[:, i] = z_mean_t
            z_logvars[:, i] = z_logvar_t

        return z_means, z_logvars, z_out

    def zeros_init(self, batch_size, seq_len):
        z_out = torch.zeros(batch_size, seq_len, self.latent_dim).to(self.device)
        z_means = torch.zeros(batch_size, seq_len, self.latent_dim).to(self.device)
        z_logvars = torch.zeros(batch_size, seq_len, self.latent_dim).to(self.device)
        return z_logvars, z_means, z_out

    def forward(self, x, time=None, final_index=None):
        # ------------- ENCODING PART -------------
        if time is not None and final_index is not None:
            z = self.encoder(time, x, final_index)
        else:
            z = self.encoder(x)

        # variational part for input
        z_mean = self.z_mean(z)
        z_logvar = self.z_logvar(z)
        z_post = reparameterize(z_mean, z_logvar, random_sampling=True)

        Z_enc = {"mean": z_mean, "logvar": z_logvar, "sample": z_post}

        # # ------------- PRIOR PART -------------
        z_mean_prior, z_logvar_prior, z_out = self.sample_prior(
            z.size(0), self.seq_len, random_sampling=True
        )
        Z_enc_prior = {"mean": z_mean_prior, "logvar": z_logvar_prior, "sample": z_out}

        # pass z_post instead of z_pred
        x_rec = self.decoder(z_post)

        return x_rec, Z_enc, Z_enc_prior

    def training_step(self, batch, batch_idx):
        if self.missing_value:
            # imputation task

            x = batch["seq"]
            cond = batch['c']
            x = x.masked_fill(cond.bool(), float("nan"))
            
            x = batch['seq']
            train_coeffs = batch['inter']  # .to(device)
            time = torch.arange(x.shape[1]).to(x)
            final_index = (torch.ones(x.shape[0]) * (self.seq_len-1)).to(x)
            x_rec, Z_enc, Z_enc_prior = self(train_coeffs, time, final_index)

        else:
            X = batch["seq"]
            x_rec, Z_enc, Z_enc_prior = self(X)

        losses = self.loss(
            X, x_rec, Z_enc, Z_enc_prior
        )  # x_rec, x_pred_rec, z, z_pred_, Ct
        self.log_dict(
            {
                "train_loss": losses[0],
                "train_recon_loss": losses[1],
                "train_kl_loss": losses[2],
                "train_pred_loss": losses[3],
            },
            prog_bar=True, on_epoch=True
        )
        return losses[0]

    def validation_step(self, batch, batch_idx):
        if self.missing_value:
            # imputation task

            x = batch["seq"]
            cond = batch['c']
            x = x.masked_fill(cond.bool(), float("nan"))
            
            x = batch['seq']
            train_coeffs = batch['inter']  # .to(device)
            time = torch.arange(x.shape[1]).to(x)
            final_index = (torch.ones(x.shape[0]) * (self.seq_len-1)).to(x)
            x_rec, Z_enc, Z_enc_prior = self(train_coeffs, time, final_index)

        else:
            X = batch["seq"]
            x_rec, Z_enc, Z_enc_prior = self(X)

        losses = self.loss(
            X, x_rec, Z_enc, Z_enc_prior
        )  # x_rec, x_pred_rec, z, z_pred_, Ct
        self.log_dict(
            {
                "val_loss": losses[0],
                "val_recon_loss": losses[1],
                "val_kl_loss": losses[2],
                "val_pred_loss": losses[3],
            },
            prog_bar=True, on_epoch=True
        )
        # return losses[0]

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams_initial.lr,
            weight_decay=self.hparams_initial.weight_decay,
        )