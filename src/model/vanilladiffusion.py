from pkg_resources import non_empty_lines
import torch
from torch import nn
from src.layers.conv import ConvEncoder, ConvDecoder
from src.layers.mlp import MLPDecoder, MLPEncoder
from src.layers.transformer import DiT
from src.model.base import BaseDiffusion
from src.utils.check import _condition_shape_check
import torch.nn.functional as F
import math


def cosine_schedule(n_steps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = n_steps + 1
    x = torch.linspace(0, n_steps, steps)
    alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.001, 0.999)
    alphas = 1.0 - betas
    alphas_bars = torch.cumprod(alphas, dim=0)
    noise_schedule = dict(
        betas=betas,
        alphas=alphas,
        alpha_bars=alphas_bars,
        beta_bars=None,
    )
    return noise_schedule


def linear_schedule(n_steps, min_beta=1e-4, max_beta=2e-2):
    betas = torch.linspace(min_beta, max_beta, n_steps)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        "alpha_bars": alpha_bars.float(),
        "beta_bars": None,
        "alphas": alphas.float(),
        "betas": betas.float(),
    }


class VanillaDDPM(BaseDiffusion):
    """MovingAvg Diffusion."""

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int,
        patch_size: int = 16,
        n_layers=4,
        num_heads=16,
        mlp_ratio=4.0,
        # cond_dropout_prob=0.5,
        # cond_seq_len=None,
        # cond_seq_chnl=None,
        # norm=True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        noise_schedule: str = "cosine",
        # backbone_config: dict,
        # conditioner_config: dict,
        # ns_path: str,
        # norm=False,
        # lr=2e-4,
        # alpha=1e-5,
        T=1000,
        pred_x0=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # bb_class = getattr(src.backbone, backbone_config["name"])
        self.backbone = DiT(**self.hparams)
        self.seq_length = self.backbone.seq_length
        # self.norm = norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = F.mse_loss
        self.T = T
        self.pred_x0 = pred_x0
        # ! Notice:
        # in the schedule.py, alphas for x_0 to x_t
        # in the schedule.py, alphas_bars for x_t to x_t+1

        # in this DDPM.py, attribute alpha_bars for x_0 to x_t
        # in this DDPM.py, attribute alphas for x_t to x_t+1
        if noise_schedule == "cosine":
            noise_schedule = cosine_schedule(T)
        elif noise_schedule == "linear":
            noise_schedule = linear_schedule(T)
        else:
            raise ValueError(
                "VanillaDDPM only support for cosine or linear noise schedule."
            )

        self.register_buffer("alphas", noise_schedule["alphas"])
        self.register_buffer("betas", noise_schedule["betas"])
        self.register_buffer("alpha_bars", noise_schedule["alpha_bars"])
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x)
        mu_coeff = torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1)
        var_coeff = torch.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1, 1)
        x_noisy = mu_coeff * x + var_coeff * noise
        return x_noisy, noise

    def training_step(self, batch, batch_idx):
        x = batch.pop("seq")

        # if norm
        # x, (x_mean, x_std) = self._normalize(x) if self.norm else (x, (None, None))

        loss = self._get_loss(x, batch)

        log_dict = {"train_loss": loss}

        self.log_dict(
            log_dict, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0]
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.pop("seq")

        # if norm
        # x, (x_mean, x_std) = self._normalize(x) if self.norm else (x, (None, None))

        loss = self._get_loss(x, batch)
        log_dict = {
            "val_loss": loss,
        }

        self.log_dict(
            log_dict, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0]
        )
        return loss

    @torch.no_grad()
    def sample(self, n_sample, condition=None):
        # assert self._sample_ready
        # x_real = batch.pop("seq")
        # cond = batch.get("c", None)
        x = torch.randn(
            (n_sample, self.hparams_initial.seq_len, self.hparams_initial.seq_dim)
        ).type_as(next(iter(self.parameters())))
        # all_sample_x = []

        for t in range(self.T - 1, -1, -1):
            z = torch.randn_like(x)
            t_tensor = torch.tensor(t).repeat(x.shape[0]).type_as(x)
            eps_theta = self.backbone(x, t_tensor, condition)

            # if self.strategy == "ddpm":
            if self.pred_x0:
                if t > 0:
                    mu_pred = (
                        torch.sqrt(self.alphas[t]) * (1 - self.alpha_bars[t - 1]) * x
                        + torch.sqrt(self.alpha_bars[t - 1]) * self.betas[t] * eps_theta
                    )
                    mu_pred = mu_pred / (1 - self.alpha_bars[t])
                else:
                    mu_pred = eps_theta
            else:
                # TODO: BUG ON cosine
                # print(t)
                # print((1 - self.alphas[t]))
                # print(torch.sqrt(1 - self.alpha_bars[t]))
                # print(torch.sqrt(self.alphas[t]))
                # print((1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]))
                # print((eps_theta).flatten()[:5])
                # print(
                #     (
                #         x
                #         - (1 - self.alphas[t])
                #         / torch.sqrt(1 - self.alpha_bars[t])
                #         * eps_theta
                #     ).flatten()[:5]
                # )
                mu_pred = (
                    x
                    - (1 - self.alphas[t])
                    / torch.sqrt(1 - self.alpha_bars[t])
                    * eps_theta
                ) / torch.sqrt(self.alphas[t])
            if t == 0:
                sigma = 0
            else:
                sigma = torch.sqrt(
                    (1 - self.alpha_bars[t - 1])
                    / (1 - self.alpha_bars[t])
                    * self.betas[t]
                )
            # print(t)
            # print(sigma)
            x = mu_pred + sigma * z
            # print((x).flatten()[:5])
            

        return x

    def _get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        cond = condition.get("c", None)
        # sample t, x_T
        t = torch.randint(0, self.T, (batch_size,)).to(self.device)

        # corrupt data
        x_noisy, noise = self.degrade(x, t)

        # eps_theta
        # c = self._encode_condition(condition)
        eps_theta = self.backbone(x_noisy, t, cond)

        # compute loss
        if self.pred_x0:
            loss = F.mse_loss(eps_theta, x)
        else:
            loss = F.mse_loss(eps_theta, noise)

        # # regularizatoin
        # if (x_mean is not None) and (cond is not None):
        #     mean_loss = F.mse_loss(x_mean_hat, x_mean)
        # else:
        #     mean_loss = 0

        # # TODO: scale too small?
        # if (x_std is not None) and (cond is not None):
        #     # optimizing log std to stablize
        #     std_loss = F.mse_loss(x_std_hat, x_std)
        # else:
        #     std_loss = 0

        return loss
        # return loss, mean_loss, std_loss

    # def _init_noise(
    #     self,
    #     x: torch.Tensor,
    # ):
    # x_T = torch.randn_like(x)
    # if self.condition is None:
    #     if self.norm:
    #         self.init_mean = self.init_mu_dist.sample()
    #         _, (self.init_mean, self.init_std) = self._normalize(x)
    # elif self.condition == "sr":
    #     self.init_mean = x.mean(dim=1, keepdim=True)

    # return x_T

    # shape = (self.n_sample, x.shape[0], x.shape[1], x.shape[2])
    # return torch.randn(shape, device=x.device)

    # def config_sampling(
    #     self, n_sample, condition, strategy="ddpm", init_distribs=None, **kwargs
    # ):
    #     self.n_sample = n_sample
    #     assert condition in ["fcst", "sr", None]
    #     assert strategy in ["ddpm", "ddim"]
    #     self.condition = condition
    #     self.strategy = strategy
    #     if self.condition is None:
    #         if self.norm:
    #             assert init_distribs is not None
    #             self.init_mu_dist = init_distribs[0]
    #             self.init_std_dist = init_distribs[1]
    #     self._sample_ready = True

    # def _normalize(self, x):
    #     mean = torch.mean(x, dim=1, keepdim=True)
    #     stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-6)
    #     x_norm = (x - mean) / stdev
    #     return x_norm, (mean, stdev)
