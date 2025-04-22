import torch
import torch.nn.functional as F


from src.model.base import BaseModel
from ._backbones import Denoiser
from ._utils import linear_schedule, cosine_schedule


class VanillaDDPM(BaseModel):
    ALLOW_CONDITION = [None, "predict", "impute"]
    """VanillaDDPM model with MLP backbone.

    Args:
        BaseDiffusion (_type_): _description_
    """

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int = 128,
        hidden_size_list: list = [64, 128, 256],
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        noise_schedule: str = "cosine",
        n_diff_steps=1000,
        pred_x0=True,
        condition: str = None,
        **kwargs,
    ) -> None:
        super().__init__(seq_len, seq_dim, condition)
        self.save_hyperparameters()
        # self.backbone = DiT(**self.hparams)
        self.backbone = Denoiser(**self.hparams)
        self.seq_length = seq_len
        # self.norm = norm
        # self.lr = lr
        # self.weight_decay = weight_decay
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
        self.save_hyperparameters()

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
        x = batch.pop("seq")
        loss = self._get_loss(x, batch)
        log_dict = {"train_loss": loss}
        self.log_dict(
            log_dict, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0]
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.pop("seq")
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
            if self.condition == "impute":
                print(condition.device)
                print(kwargs['seq'].device)
                condition = kwargs["seq"] * (~condition).int()
                condition = condition.to(self.device)

            all_samples = []
            for i in range(n_sample):
                x = torch.randn(
                    (
                        condition.shape[0],
                        self.hparams_initial.seq_len,
                        self.hparams_initial.seq_dim,
                    )
                ).type_as(next(iter(self.parameters())))
                x = self.sample_loop(x, condition)
                all_samples.append(x)
            all_samples = torch.stack(all_samples, dim=-1)

        return all_samples

    def _get_loss(self, x, condition: dict = None):
        batch_size = x.shape[0]
        cond = condition.get("c", None)
        if self.condition == "impute":
            cond = x * (~cond).int()
        # sample t, x_T
        t = torch.randint(0, self.n_diff_steps, (batch_size,)).to(self.device)

        # corrupt data
        x_noisy, noise = self.degrade(x, t)

        # eps_theta
        eps_theta = self.backbone(x_noisy, t, cond)

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
                if t > 0:
                    mu_pred = (
                        torch.sqrt(self.alphas[t]) * (1 - self.alpha_bars[t - 1]) * x
                        + torch.sqrt(self.alpha_bars[t - 1]) * self.betas[t] * eps_theta
                    )
                    mu_pred = mu_pred / (1 - self.alpha_bars[t])
                else:
                    mu_pred = eps_theta
            else:
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
            x = mu_pred + sigma * z
        return x
