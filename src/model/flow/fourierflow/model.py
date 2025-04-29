# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""

This script contains the implementation for the spectral filter module of the Fourier flow

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from src.model.base import BaseModel
from ._spectral import SpectralFilter
from ._fourier import DFT, myDFT

from torch.distributions.multivariate_normal import MultivariateNormal

import sys

import warnings

warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class FourierFlow(BaseModel):
    ALLOW_CONDITION = [None]

    def __init__(
        self,
        seq_len,
        seq_dim,
        d_model=64,
        n_flows=10,
        condition=None,
        FFT=True,
        flip=True,
        normalize=False,
        lr=1e-3,
        weight_decay=1e-6,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition)
        self.save_hyperparameters()
        fft_size = seq_len
        self.k = int(fft_size / 2) + 1
        self.d = self.k * 2
        # self.d = fft_size
        self.fft_size = fft_size
        self.FFT = FFT
        self.normalize = normalize

        if flip:
            self.flips = [True if i % 2 else False for i in range(n_flows)]

        else:
            self.flips = [False for i in range(n_flows)]

        self.bijectors = nn.ModuleList(
            [
                SpectralFilter(
                    self.d, self.k, self.FFT, hidden=d_model, flip=self.flips[_]
                )
                for _ in range(n_flows)
            ]
        )

        self.FourierTransform = myDFT(N_fft=self.fft_size)
        # self.FourierTransform = DFT(N_fft=self.fft_size)

    def forward(self, x):
        if self.FFT:
            x = self.FourierTransform(x)[0]

            if self.normalize:
                x = (x - self.fft_mean) / self.fft_std

            x = x.flatten(1)
            # x = x.view(-1, self.d + 1)

        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):
            x, log_pz, lj = bijector(x, flip=f)
            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):
        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
            z = bijector.inverse(z, flip=f)

        if self.FFT:
            if self.normalize:
                z = z * self.fft_std.flatten(1) + self.fft_mean.flatten(1)
                # z = z * self.fft_std.view(-1, self.d + 1) + self.fft_mean.view(
                #     -1, self.d + 1
                # )

            z = self.FourierTransform.inverse(z)

        return z.detach()

    def training_step(self, batch, batch_idx):
        # Must be univariate time series
        X_train = batch["seq"].squeeze(-1)

        z, log_pz, log_jacob = self(X_train)
        loss = (-log_pz - log_jacob).mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Must be univariate time series
        X_train = batch["seq"].squeeze(-1)

        z, log_pz, log_jacob = self(X_train)
        loss = (-log_pz - log_jacob).mean()

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        if self.FFT:
            mu, cov = torch.zeros(self.d), torch.eye(self.d)
            # mu, cov = torch.zeros(self.d + 1), torch.eye(self.d + 1)

        else:
            mu, cov = torch.zeros(self.d), torch.eye(self.d)
        mu, cov = mu.to(self.device), cov.to(self.device)

        p_Z = MultivariateNormal(mu, cov)
        z = p_Z.rsample(sample_shape=(n_sample,))

        X_sample = self.inverse(z)

        return X_sample

        # return super()._sample_impl(n_sample, condition, **kwargs)

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)
        return {"optimizer": optim, "lr_scheduler": scheduler}

    def on_fit_start(self):
        dl = self.trainer.datamodule.train_dataloader()
        seq_data = dl.dataset.data
        assert seq_data.shape[-1] == 1
        seq_data = seq_data.squeeze(-1).to(self.device)
        
        X_train_spectral = self.FourierTransform(seq_data)[0]
        self.fft_mean = torch.mean(X_train_spectral, dim=0)
        self.fft_std = torch.std(X_train_spectral, dim=0)
        self.k = X_train_spectral.shape[-1]
        # self.d = seq_data.shape[1]
        self.d = self.k * 2
