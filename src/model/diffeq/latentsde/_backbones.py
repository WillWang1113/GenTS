# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a latent SDE on data from a stochastic Lorenz attractor.

Reproduce the toy example in Section 7.2 of https://arxiv.org/pdf/2001.01328.pdf

To run this file, first run the following to install extra requirements:
pip install fire

To run, execute:
python -m examples.latent_sde_lorenz
"""

import logging
import os
from typing import Sequence

# import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal

import torchsde


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class StochasticLorenz(object):
    """Stochastic Lorenz attractor.

    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self, a: Sequence = (10.0, 28.0, 8 / 3), b: Sequence = (0.1, 0.28, 0.3)
    ):
        super(StochasticLorenz, self).__init__()
        self.a = a
        self.b = b

    def f(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        a1, a2, a3 = self.a

        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3
        return torch.cat([f1, f2, f3], dim=1)

    def g(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        b1, b2, b3 = self.b

        g1 = x1 * b1
        g2 = x2 * b2
        g3 = x3 * b3
        return torch.cat([g1, g2, g3], dim=1)

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, normalize):
        """Sample data for training. Store data normalization constants if necessary."""
        xs = torchsde.sdeint(self, x0, ts)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std).add_(torch.randn_like(xs) * noise_std)
        return xs


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class LatentSDENet(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size, output_size):
        super(LatentSDENet, self).__init__()
        # Encoder.
        self.encoder = Encoder(
            input_size=data_size, hidden_size=hidden_size, output_size=context_size
        )
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid(),
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, output_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[:, i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler", flip=True):
        # Contextualization is only needed for posterior inference.
        if flip:
            ctx = self.encoder(torch.flip(xs, dims=(1,)))
        else:
            ctx = self.encoder(xs)
        ctx = torch.flip(ctx, dims=(1,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[:, 0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                (ctx,)
                + tuple(self.f_net.parameters())
                + tuple(self.g_nets.parameters())
                + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self,
                z0,
                ts,
                adjoint_params=adjoint_params,
                dt=1e-2,
                logqp=True,
                method=method,
            )
        else:
            zs, log_ratio = torchsde.sdeint(
                self, z0, ts, dt=1e-2, logqp=True, method=method
            )

        _xs = self.projector(zs)
        # xs_dist = Normal(loc=_xs, scale=noise_std)
        # log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        # return log_pxs, logqp0 + logqp_path
        return _xs, logqp0 + logqp_path

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(
            size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device
        )
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={"drift": "h"}, dt=1e-3, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs


def make_dataset(t0, t1, batch_size, noise_std, train_dir, device):
    data_path = os.path.join(train_dir, "lorenz_data.pth")
    if os.path.exists(data_path):
        data_dict = torch.load(data_path)
        xs, ts = data_dict["xs"], data_dict["ts"]
        logging.warning(f"Loaded toy data at: {data_path}")
        if xs.shape[1] != batch_size:
            raise ValueError(
                "Batch size has changed; please delete and regenerate the data."
            )
        if ts[0] != t0 or ts[-1] != t1:
            raise ValueError(
                "Times interval [t0, t1] has changed; please delete and regenerate the data."
            )
    else:
        _y0 = torch.randn(batch_size, 3, device=device)
        ts = torch.linspace(t0, t1, steps=100, device=device)
        xs = StochasticLorenz().sample(_y0, ts, noise_std, normalize=True)

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save({"xs": xs, "ts": ts}, data_path)
        logging.warning(f"Stored toy data at: {data_path}")
    return xs, ts


def main(
    batch_size=1024,
    latent_size=4,
    context_size=64,
    hidden_size=128,
    lr_init=1e-2,
    t0=0.0,
    t1=2.0,
    lr_gamma=0.997,
    num_iters=5000,
    kl_anneal_iters=1000,
    pause_every=50,
    noise_std=0.01,
    adjoint=False,
    train_dir="./dump/lorenz/",
    method="euler",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xs, ts = make_dataset(
        t0=t0,
        t1=t1,
        batch_size=batch_size,
        noise_std=noise_std,
        train_dir=train_dir,
        device=device,
    )
    latent_sde = LatentSDENet(
        data_size=3,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)
    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=lr_gamma
    )
    kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    # Fix the same Brownian motion for visualization.
    bm_vis = torchsde.BrownianInterval(
        t0=t0,
        t1=t1,
        size=(
            batch_size,
            latent_size,
        ),
        device=device,
        levy_area_approximation="space-time",
    )

    for global_step in tqdm.tqdm(range(1, num_iters + 1)):
        latent_sde.zero_grad()
        log_pxs, log_ratio = latent_sde(xs, ts, noise_std, adjoint, method)
        loss = -log_pxs + log_ratio * kl_scheduler.val
        loss.backward()
        optimizer.step()
        scheduler.step()
        kl_scheduler.step()
