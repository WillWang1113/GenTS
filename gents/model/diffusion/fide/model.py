import numpy as np
from ._backbones import TransformerModel
from ._utils import (
    get_betas,
    add_noise,
    get_gp_covariance,
    linear_decay,
    fitting_gev_and_sampling,
)
from gents.model.base import BaseModel
import torch


class FIDE(BaseModel):
    """`FIDE <https://openreview.net/pdf?id=5HQhYiGnYb>`_: Frequency-Inflated Conditional Diffusion Model for Extreme-Aware Time Series Generation

    Adapted from the `official codes <https://github.com/galib19/FIDE>`_
    
    .. note::
        Only support for univariate time series.
    
    .. warning::
        The original paper claimed an innovation on regularization on loss function. 
        However, in the original codes, the regularization term is detached from the computation graph, which may cause no gradients.
    
    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to None.
        n_diff_steps (int, optional): Total diffusion steps. Defaults to 100.
        d_model (int, optional): Model size. Defaults to 64.
        is_regularizer (bool, optional): Whether to add extreme value regularization. Defaults to True.
        high_freq_inflation_rate (float, optional): High frequency inflation rate. Should be greater than 1. Defaults to 1.1.
        percentage_of_freq_enhanced (float, optional): Percentage of frequencies that are inflated/enhanced. Defaults to 0.2.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-6.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.

    """

    ALLOW_CONDITION = [None]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int = 1,
        condition: str = None,
        n_diff_steps: int = 100,
        d_model: int = 64,
        is_regularizer: bool = True,
        high_freq_inflation_rate: float = 1.1,
        percentage_of_freq_enhanced: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        assert seq_dim == 1, "Only support univariate time series"
        betas = get_betas(n_diff_steps)
        alphas = torch.cumprod(1 - betas, dim=0)
        self.seq_len = seq_len
        self.diffusion_steps = n_diff_steps
        self.is_regularizer = is_regularizer
        self.model = TransformerModel(
            seq_len, dim=seq_dim, hidden_dim=d_model, max_i=n_diff_steps
        )
        self.high_freq_inflation_rate = high_freq_inflation_rate
        self.percentage_of_freq_enhanced = percentage_of_freq_enhanced
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)

    def training_step(self, batch, batch_idx):
        x = batch["seq"]
        bm, pos = torch.max(x, dim=1)
        inflated_x = self._high_freq_inflate(x)
        time = batch["t"].unsqueeze(-1)
        loss, ddpm_loss, reg_loss = self._get_loss(inflated_x, time, bm)
        self.log_dict(
            {
                "train_loss": loss,
                "ddpm_loss": ddpm_loss,
                "reg_loss": reg_loss,
            },
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["seq"]
        bm, pos = torch.max(x, dim=1)
        inflated_x = self._high_freq_inflate(x)
        time = batch["t"].unsqueeze(-1)
        loss, ddpm_loss, reg_loss = self._get_loss(inflated_x, time, bm)
        self.log_dict(
            {
                "val_loss": loss,
                "ddpm_loss": ddpm_loss,
                "reg_loss": reg_loss,
            },
            prog_bar=True,
        )
        return loss

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        bm_samples_gev = self.gev_model.rvs(size=n_sample)
        bm_samples_conditional = torch.tensor(bm_samples_gev, dtype=torch.float32).to(
            self.device
        )
        bm_samples_conditional = bm_samples_conditional.reshape(-1, 1, 1)
        t_grid = (
            torch.linspace(0, self.seq_len, self.seq_len).view(1, -1, 1).to(self.device)
        )
        samples_ddpm = self.sample_loop(
            t_grid.repeat(n_sample, 1, 1), bm_samples_conditional
        )

        return samples_ddpm

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _get_loss(self, x, t, bm):
        i = torch.randint(0, self.diffusion_steps, size=(x.shape[0],))
        i = i.view(-1, 1, 1).expand_as(x[..., :1]).to(x)

        x_noisy, noise = add_noise(x, t, i, self.alphas)
        pred_noise = self.model(x_noisy, t, i, bm)
        ddpm_loss = torch.sqrt(torch.mean((pred_noise - noise) ** 2))

        # ! reg_loss not backwarded
        if self.is_regularizer:
            lambda_1 = linear_decay(i[:, 0, :].reshape(-1, 1), self.diffusion_steps)
            pred_0 = x_noisy - pred_noise
            bm_pred, _ = torch.max(pred_0, dim=1)
            reg_loss = np.mean(self.gev_model.logpdf(bm_pred.detach().cpu().numpy()))
            reg_loss = -0.05 * torch.tensor(reg_loss, dtype=torch.float32).to(
                ddpm_loss.device
            )

            loss = ddpm_loss + reg_loss
        else:
            reg_loss = 0
            loss = ddpm_loss
        return loss, ddpm_loss, reg_loss

    def _high_freq_inflate(self, seq):
        real_data_fft = torch.fft.rfft(seq, dim=1)
        real_data_freq = torch.fft.rfftfreq(seq.shape[1])
        top_freq_enhanced = int(
            (real_data_fft.shape[1] * self.percentage_of_freq_enhanced)
        )

        high_freq_enhanced_fft_result = real_data_fft.clone()
        top_indices = torch.argsort(real_data_freq)[-top_freq_enhanced:]

        # Iterate over all datapoints along the second dimension
        for i in range(real_data_fft.shape[0]):
            high_freq_enhanced_fft_result[i, :, 0][top_indices] *= (
                self.high_freq_inflation_rate
            )

        real_data = torch.fft.irfft(high_freq_enhanced_fft_result, dim=1)
        return real_data

    def on_fit_start(self):
        dl = self.trainer.datamodule.train_dataloader()
        real_data = dl.dataset.data
        if self.is_regularizer:
            block_maxima_real_data_value, block_maxima_real_data_pos = torch.max(
                real_data, dim=1
            )
            block_maxima_real_data_value = block_maxima_real_data_value.reshape(
                -1, 1, 1
            )
            block_maxima_real_data_pos = block_maxima_real_data_pos.reshape(-1, 1, 1)
            block_maxima_real_data = block_maxima_real_data_value  # torch.cat((block_maxima_real_data_value, block_maxima_real_data_pos), dim=1)
            num_samples = block_maxima_real_data.shape[0]

            block_maxima_real_data_value = (
                block_maxima_real_data_value.cpu().numpy().reshape(-1)
            )
            block_maxima_real_data_pos = (
                block_maxima_real_data_pos.cpu().numpy().reshape(-1)
            )
            gev_model = fitting_gev_and_sampling(
                block_maxima_real_data_value, num_samples
            )
            self.gev_model = gev_model

    def sample_loop(self, t, bm_sample):
        cov = get_gp_covariance(t)
        L = torch.linalg.cholesky(cov)

        x = L @ torch.randn_like(t)

        for diff_step in reversed(range(0, self.diffusion_steps)):
            alpha = self.alphas[diff_step]
            beta = self.betas[diff_step]

            z = L @ torch.randn_like(t)

            i = torch.Tensor([diff_step]).expand_as(x[..., :1]).to(self.device)
            pred_noise = self.model(x, t, i, bm_sample)

            x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (
                1 - beta
            ).sqrt() + beta.sqrt() * z
        return x
