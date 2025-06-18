import torch
import torch.nn.functional as F


from src.model.base import BaseModel
from ._backbones import Denoiser, DenoiserTransformer
from ._utils import linear_schedule, cosine_schedule


class VanillaDDPM(BaseModel):
    """Vanilla DDPM with MLP backbone."""

    ALLOW_CONDITION = [None, "predict", "impute"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        latent_dim: int = 128,
        hidden_size_list: list = [128, 256, 512],
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
            latent_dim (int, optional): Latent dim. Defaults to 128.
            hidden_size_list (list, optional): Hidden size for Denoiser MLP. Defaults to [64, 128, 256].
            noise_schedule (str, optional): Noise schedule of DDPM, ['cosine', 'linear']. Defaults to "cosine".
            n_diff_steps (int, optional): Total diffusion steps. Defaults to 1000.
            pred_x0 (bool, optional): Predict x_0 or noise. Defaults to True.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        """
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        self.backbone = Denoiser(**self.hparams)
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

        self.register_buffer("alphas", noise_schedule["alphas"])
        self.register_buffer("betas", noise_schedule["betas"])
        self.register_buffer("alpha_bars", noise_schedule["alpha_bars"])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams_initial.lr,
            weight_decay=self.hparams_initial.weight_decay,
        )

    @torch.no_grad
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x)
        mu_coeff = torch.sqrt(self.alpha_bars[t]).reshape(-1, 1, 1)
        var_coeff = torch.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1, 1)
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
        if self.condition is None:
            x = torch.randn(
                (n_sample, self.hparams_initial.seq_len, self.hparams_initial.seq_dim)
            ).to(self.device)
            all_samples = self.sample_loop(x)
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
                x = self.sample_loop(x, condition)
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

    def sample_loop(self, x, condition=None):
        for t in range(self.n_diff_steps - 1, -1, -1):
            z = torch.randn_like(x)
            t_tensor = torch.tensor(t).repeat(x.shape[0]).type_as(x)
            eps_theta = self.backbone(x, t_tensor, condition)

            if self.pred_x0:
                x0_hat = eps_theta
            else:
                x0_hat = x - torch.sqrt(1 - self.alpha_bars[t]) * eps_theta
                x0_hat = x0_hat / torch.sqrt(self.alpha_bars[t])

            if t > 0:
                mu_pred = (
                    torch.sqrt(self.alphas[t]) * (1 - self.alpha_bars[t - 1]) * x
                    + torch.sqrt(self.alpha_bars[t - 1]) * self.betas[t] * x0_hat
                )
                mu_pred = mu_pred / (1 - self.alpha_bars[t])
            else:
                mu_pred = eps_theta

            if t == 0:
                sigma = 0
            else:
                sigma = torch.sqrt(
                    (1 - self.alpha_bars[t - 1])
                    / (1 - self.alpha_bars[t])
                    * self.betas[t]
                )
            x = mu_pred + sigma * z
        return x
