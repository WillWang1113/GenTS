import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP

from src.layers.rnn import RNNLayer
from src.model.base import BaseModel
from torchdiffeq import odeint

from src.model.vanillavae import VanillaVAE
import numpy as np

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.nhidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class NeuralODE(BaseModel):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int,
        hidden_size_list: list = [64],
        num_layers: int = 1,
        ode_method: str = 'dopri5',
        rnn_type: str = "gru",
        beta: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = RecognitionRNN(
            latent_dim=latent_dim, obs_dim=seq_dim, nhidden=hidden_size_list[-1]
        )
        self.ode_fn = LatentODEfunc(
            latent_dim=latent_dim,
            nhidden=hidden_size_list[-1],
        )
        self.decoder = Decoder(latent_dim, seq_dim)
        self.ode_method = ode_method
        self.latent_dim = latent_dim

    # def encode(self, x, c=None, **kwargs):
    #     latents = self.encoder(x.flip(dims=[1]))
    #     mu = self.fc_mu(latents[:, -1, :])
    #     logvar = self.fc_logvar(latents[:, -1, :])
    #     return latents, mu, logvar

    # def decode(self, z, c=None, **kwargs):
    #     t = kwargs.get(
    #         "t",
    #         torch.linspace(0, 2 * torch.pi, self.hparams_initial.seq_len).type_as(z),
    #     )
    #     zs = odeint(self.ode_fn, z, t, method=self.ode_method).permute(1, 0, 2)
    #     zs = self.decoder(zs)
    #     return zs

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams_initial.lr,
            weight_decay=self.hparams_initial.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        x = batch["seq"]
        c = batch.get("c", None)
        t = batch.get("t", torch.linspace(0, 4 * torch.pi, self.hparams_initial.seq_len).type_as(x))

        # Run RNN in reverse mode
        # latents = self.embedder(x[:, ::-1, :])

        h = self.encoder.initHidden(x.shape[0]).to(x)
        for tt in reversed(range(x.size(1))):
            obs = x[:, tt, :]
            out, h = self.encoder.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(x)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.ode_fn, z0, t, method=self.ode_method).permute(1, 0, 2)
        pred_x = self.decoder(pred_z)
        
        # compute loss
        noise_std_ = torch.ones(pred_x.size()).to(x)
        noise_logvar = 2. * torch.log(noise_std_).to(x)
        logpx = log_normal_pdf(
            x, pred_x, noise_logvar).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(x)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        # print(logpx)
        # print(analytic_kl)
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        # loss.backward()
        # optimizer.step()
        # loss_meter.update(loss.item())
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x = batch["seq"]
        c = batch.get("c", None)
        t = batch.get("t", torch.linspace(0, 4 * torch.pi, self.hparams_initial.seq_len).type_as(x))
        
        # Run RNN in reverse mode
        # latents = self.embedder(x[:, ::-1, :])

        h = self.encoder.initHidden(x.shape[0]).to(x)
        for tt in reversed(range(x.size(1))):
            obs = x[:, tt, :]
            out, h = self.encoder.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(x)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.ode_fn, z0, t, method=self.ode_method).permute(1, 0, 2)
        pred_x = self.decoder(pred_z)
        
        # compute loss
        noise_std_ = torch.ones(pred_x.size()).to(x)
        noise_logvar = 2. * torch.log(noise_std_).to(x)
        logpx = log_normal_pdf(
            x, pred_x, noise_logvar).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(x)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        self.log("val_loss", loss,  on_step=True, on_epoch=False, logger=True, prog_bar=True)
        
        # loss.backward()
        # optimizer.step()
        # loss_meter.update(loss.item())
        return loss
    
    def _sample_impl(self, n_sample, condition=None):
        z = torch.randn((n_sample, self.hparams_initial.latent_dim)).to(self.device)
        x_hat = odeint(self.ode_fn, z, torch.linspace(0, 4 * np.pi, self.hparams_initial.seq_len), method=self.ode_method).permute(1, 0, 2)
        x_hat = self.decoder(x_hat)
        return x_hat


    #     # ODE
    #     z = odeint(
    #         self.ode_fn, latents, torch.linspace(0, 1, self.hparams_initial.seq_len)
    #     )

    #     # decode
    #     x_hat = self.decoder(z)

    #     loss = F.mse_loss(x_hat, x)
    #     self.log("train/loss", loss)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     pass

    # def _sample_impl(self, n_sample: int, condition: torch.Tensor = None):
    #     z = torch.rand(
    #         (n_sample, self.hparams_initial.seq_len, self.hparams_initial.seq_dim)
    #     ).to(next(self.parameters()))
    #     e_hat = self.generator(z)
    #     h_hat = self.supervisor(e_hat)
    #     samples = self.recovery(h_hat)
    #     return samples
    #     # return self._norm(samples, mode="denorm")

    # def on_fit_start(self):
    #     train_dl = self.trainer.datamodule.train_dataloader()
    #     all_x = torch.concat([batch["seq"] for batch in train_dl])
    #     self.min_x = all_x.min(dim=0).values.min(dim=0).values.to(self.device)
    #     self.max_x = all_x.max(dim=0).values.max(dim=0).values.to(self.device)

    # # def _norm(self, x, mode="norm"):
    # #     return (
    # #         (x - self.min_x) / (self.max_x - self.min_x)
    # #         if mode == "norm"
    # #         else (x * (self.max_x - self.min_x)) + (self.min_x)
    # #     )
