import torch
import torch.nn.functional as F


from gents.model.base import BaseModel
from ._utils import linear_schedule, cosine_schedule
from ._dit import DiT


class VanillaDDPM(BaseModel):
    """Vanilla DDPM with DiT backbone.
    
    For conditional generation, an extra MLP is used for embedding conditions.

    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given conditions, allowing [None, 'predict', 'impute']. Defaults to None.
        d_model (int, optional): DiT model size. Defaults to 128.
        n_layers (int, optional): DiT depth. Defaults to 4.
        num_heads (int, optional): Attention heads in DiT. Defaults to 8.
        mlp_ratio (float, optional): Hidden size ratio of `d_model` in DiT block, i.e. `hidden_size = d_model * mlp_ratio`. Defaults to 4.0.
        patch_size (int, optional): Patchify length of time series, should be factors of `seq_len`, i.e. `seq_len % patch_len = 0`. Defaults to 16.
        noise_schedule (str, optional): Noise schedule of DDPM, ['cosine', 'linear']. Defaults to "cosine".
        n_diff_steps (int, optional): Total diffusion steps. Defaults to 1000.
        pred_x0 (bool, optional): Predict x_0 or noise. Defaults to True.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """

    ALLOW_CONDITION = [None, "predict", "impute", "class"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        d_model: int = 128,
        n_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        noise_schedule: str = "cosine",
        n_diff_steps: int = 1000,
        pred_x0: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ) -> None:
        """


        Args:
            seq_len (int): Target sequence length
            seq_dim (int): Target sequence dimension, for univariate time series, set as 1
            condition (str, optional): Given conditions, allowing [None, 'predict', 'impute']. Defaults to None.
            d_model (int, optional): DiT model size. Defaults to 128.
            n_layers (int, optional): DiT depth. Defaults to 4.
            num_heads (int, optional): Attention heads in DiT. Defaults to 8.
            mlp_ratio (float, optional): Hidden size ratio of `d_model` in DiT block, i.e. `hidden_size = d_model * mlp_ratio`. Defaults to 4.0.
            patch_size (int, optional): Patchify length of time series, should be factors of `seq_len`, i.e. `seq_len % patch_len = 0`. Defaults to 16.
            noise_schedule (str, optional): Noise schedule of DDPM, ['cosine', 'linear']. Defaults to "cosine".
            n_diff_steps (int, optional): Total diffusion steps. Defaults to 1000.
            pred_x0 (bool, optional): Predict x_0 or noise. Defaults to True.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-5.
            **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.

        """
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        if self.condition == "predict":
            cond_seq_len = self.obs_len
            cond_seq_chnl = seq_dim
            cond_n_class = None
        elif self.condition == "impute":
            cond_seq_len = seq_len
            cond_seq_chnl = seq_dim
            cond_n_class = None
        elif self.condition == "class":
            cond_seq_len = None
            cond_seq_chnl = None
            cond_n_class = self.class_num
        else:
            cond_seq_len = None
            cond_seq_chnl = None
            cond_n_class = None

        self.backbone = DiT(
            seq_channels=seq_dim,
            seq_length=seq_len,
            cond_seq_len=cond_seq_len,
            cond_seq_chnl=cond_seq_chnl,
            cond_n_class=cond_n_class,
            d_model=d_model,
            patch_size=patch_size,
            n_layers=n_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        # self.backbone = Denoiser(**self.hparams)
        # self.backbone = DenoiserTransformer(seq_len, seq_dim, latent_dim, condition=condition)
        self.loss_fn = F.mse_loss
        self.n_diff_steps = n_diff_steps
        self.pred_x0 = pred_x0
        if noise_schedule == "cosine":
            noise_schedule = cosine_schedule(n_diff_steps)
        elif noise_schedule == "linear":
            noise_schedule = linear_schedule(n_diff_steps)
        else:
            raise ValueError(
                "VanillaDDPM only support for cosine or linear noise schedule."
            )
        alphas = noise_schedule["alphas"]
        betas = noise_schedule["betas"]
        alphas_cumprod = noise_schedule["alpha_bars"]
        alphas_cumprod_prev = F.pad(
            noise_schedule["alpha_bars"][:-1], (1, 0), value=1.0
        )
        # self.register_buffer("alphas", noise_schedule["alphas"])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams_initial.lr,
            weight_decay=self.hparams_initial.weight_decay,
        )

    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x)
        mu_coeff = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        var_coeff = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        # mu_coeff = torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1)
        # var_coeff = torch.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1, 1)
        x_noisy = mu_coeff * x + var_coeff * noise
        return x_noisy, noise

    def training_step(self, batch, batch_idx):
        x = batch.pop("seq")[:, -self.hparams_initial.seq_len :]
        loss = self._get_loss(x, batch)
        log_dict = {"train_loss": loss}
        self.log_dict(
            log_dict, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0]
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.pop("seq")[:, -self.hparams_initial.seq_len :]
        loss = self._get_loss(x, batch)
        log_dict = {
            "val_loss": loss,
        }
        self.log_dict(
            log_dict, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0]
        )
        return loss

    def _sample_impl(self, n_sample, condition=None, **kwargs):
        # device = next(iter(self.parameters())).device
        if self.condition is None or self.condition == "class":
            x = torch.randn(
                (n_sample, self.hparams_initial.seq_len, self.hparams_initial.seq_dim)
            ).to(self.device)
            all_samples = self._sample_loop(x)
        else:
            # if self.condition == "impute":
            # condition = kwargs["seq"] * (~condition).int()
            condition = condition.to(self.device)
            condition = (
                torch.nan_to_num(condition) if self.condition == "impute" else condition
            )

            all_samples = []
            for i in range(n_sample):
                x = torch.randn(
                    (
                        condition.shape[0],
                        self.hparams_initial.seq_len,
                        self.hparams_initial.seq_dim,
                    ),
                    device=self.device,
                )
                x = self._sample_loop(x, condition)
                all_samples.append(x)
            all_samples = torch.stack(all_samples, dim=-1)

        return all_samples

    def _get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        c = condition.get("c")
        c = torch.nan_to_num(c) if self.condition == "impute" else c

        # if self.condition == "impute":
        #     cond = x * (~cond).int()
        # sample t, x_T
        t = torch.randint(0, self.n_diff_steps, (batch_size,)).to(self.device)

        # corrupt data
        x_noisy, noise = self.degrade(x, t)

        # eps_theta
        eps_theta = self.backbone(x_noisy, t, c)

        # compute loss
        if self.pred_x0:
            loss = F.mse_loss(eps_theta, x)
        else:
            loss = F.mse_loss(eps_theta, noise)

        return loss

    def _sample_loop(self, x, condition=None):
        for t in reversed(range(0, self.n_diff_steps)):
            z = torch.randn_like(x)
            t_tensor = torch.tensor(t).repeat(x.shape[0]).type_as(x)
            eps_theta = self.backbone(x, t_tensor, condition)

            if self.pred_x0:
                x0_hat = eps_theta
            else:
                x0_hat = (
                    self.sqrt_recip_alphas_cumprod[t] * x
                    - self.sqrt_recipm1_alphas_cumprod[t] * eps_theta
                )

            # x0_hat.clamp_(-1., 1.)

            posterior_mean = (
                self.posterior_mean_coef1[t] * x0_hat + self.posterior_mean_coef2[t] * x
            )
            # posterior_variance = self.posterior_variance[t]
            posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
            x = posterior_mean + (0.5 * posterior_log_variance_clipped).exp() * z

        return x

    # def sample_loop(self, x, condition=None):
    #     for t in range(self.n_diff_steps - 1, -1, -1):
    #         z = torch.randn_like(x)
    #         t_tensor = torch.tensor(t).repeat(x.shape[0]).type_as(x)
    #         eps_theta = self.backbone(x, t_tensor, condition)

    #         if self.pred_x0:
    #             x0_hat = eps_theta
    #         else:
    #             x0_hat = x - torch.sqrt(1 - self.alpha_bars[t]) * eps_theta
    #             x0_hat = x0_hat / torch.sqrt(self.alpha_bars[t])

    #         if t > 0:
    #             mu_pred = (
    #                 torch.sqrt(self.alphas[t]) * (1 - self.alpha_bars[t - 1]) * x
    #                 + torch.sqrt(self.alpha_bars[t - 1]) * self.betas[t] * x0_hat
    #             )
    #             mu_pred = mu_pred / (1 - self.alpha_bars[t])
    #         else:
    #             mu_pred = eps_theta

    #         if t == 0:
    #             sigma = 0
    #         else:
    #             sigma = torch.sqrt(
    #                 (1 - self.alpha_bars[t - 1])
    #                 / (1 - self.alpha_bars[t])
    #                 * self.betas[t]
    #             )
    #         x = mu_pred + sigma * z
    #     return x
