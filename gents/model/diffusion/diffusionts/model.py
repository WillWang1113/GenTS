import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import reduce
from tqdm.auto import tqdm
from ema_pytorch import EMA
from gents.common._utils import default, extract, identity
from gents.model.base import BaseModel

from ._backbones import Transformer
from ._utils import cosine_beta_schedule, linear_beta_schedule


class DiffusionTS(BaseModel):
    """`Diffusion-TS <https://openreview.net/pdf?id=4h1apFjO99>`__: Interpretable Diffusion for General Time Series Generation.

    Adapted from the `official codes <https://github.com/Y-debug-sys/Diffusion-TS>`__

    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to None.
        n_layer_enc (int, optional): Encoder layers. Defaults to 3.
        n_layer_dec (int, optional): Decoder layers. Defaults to 6.
        d_model (int, optional): Model size. Defaults to 128.
        n_diff_steps (int, optional): Total diffusion steps. Defaults to 1000.
        n_sample_steps (int, optional): Number of backward sample steps. Defaults to None.
        loss_type (str, optional): Loss function type. Choose from `['l1', 'l2']`. Defaults to "l1".
        beta_schedule (str, optional): Diffusion noise schedule. Choose from `['linear', 'cosine']`. Defaults to "cosine".
        n_heads (int, optional): Attention heads in transformer. Defaults to 4.
        mlp_hidden_times (int, optional): Hidden size ratio of `d_model` in Transformer, i.e. `hidden_size = d_model * mlp_hidden_times`. Defaults to 4.
        eta (float, optional): Coefficient of DDIM random noise. `eta=0` means deterministic sampling. Defaults to 0.0.
        attn_pd (float, optional): Attention dropout rate in Transformer. Defaults to 0.0.
        resid_pd (float, optional): MLP dropout rate in Transformer. Defaults to 0.0.
        kernel_size (int, optional): Kernel size of conv layer in Transformer. Defaults to None.
        padding_size (int, optional): Padding size of conv layer in Transformer. Defaults to None.
        use_ff (bool, optional): Whether to use Fourier Transform for regularization. Defaults to True.
        reg_weight (float, optional): Weight coefficient of Fourier loss. Defaults to None.
        ema_decay (float, optional): Exponential Moving Average (EMA) decay rate of model weights. Defaults to 0.995.
        ema_update_every (int, optional): EMA update interval. Defaults to 10.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """

    ALLOW_CONDITION = [None, "predict", "impute"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        n_layer_enc: int = 3,
        n_layer_dec: int = 6,
        d_model: int = 128,
        n_diff_steps: int = 1000,
        n_sample_steps: int = None,
        loss_type: str = "l1",
        beta_schedule: str = "cosine",
        n_heads: int = 4,
        mlp_hidden_times: int = 4,
        eta: float = 0.0,
        attn_pd: float = 0.0,
        resid_pd: float = 0.0,
        kernel_size: int = None,
        padding_size: int = None,
        use_ff: bool = True,
        reg_weight: float = None,
        ema_decay: float = 0.995,
        ema_update_every: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        **kwargs,
    ):
        super(DiffusionTS, self).__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()

        self.eta, self.use_ff = eta, use_ff
        self.total_seq_len = (
            seq_len + self.obs_len if condition == "predict" else seq_len
        )
        self.predict_length = seq_len
        self.feature_size = seq_dim
        self.ff_weight = default(reg_weight, math.sqrt(self.total_seq_len) / 5)

        self.model = Transformer(
            n_feat=seq_dim,
            n_channel=self.total_seq_len,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=self.total_seq_len,
            n_embd=d_model,
            conv_params=[kernel_size, padding_size],
            **kwargs,
        )

        if beta_schedule == "linear":
            betas = linear_beta_schedule(n_diff_steps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_diff_steps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (n_diff_steps,) = betas.shape
        self.num_timesteps = int(n_diff_steps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            n_sample_steps, n_diff_steps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= n_diff_steps
        self.fast_sampling = self.sampling_timesteps < n_diff_steps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate reweighting

        register_buffer(
            "loss_weight",
            torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100,
        )

        # TODO: change to torch.utils.swa_utils.AveragedModel
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every)
        
        

    def _predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def _q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _output(self, x, t, padding_masks=None):
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        return model_output

    def _model_predictions(self, x, t, clip_x_start=False, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones(
                x.shape[0], self.total_seq_len, dtype=bool, device=x.device
            )

        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )
        x_start = self._output(x, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self._predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def _p_mean_variance(self, x, t, clip_denoised=False):
        _, x_start = self._model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self._q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def _p_sample(self, x, t: int, clip_denoised=False, cond_fn=None, model_kwargs=None):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self._p_mean_variance(
            x=x, t=batched_times, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        if cond_fn is not None:
            model_mean = self._condition_mean(
                cond_fn,
                model_mean,
                model_log_variance,
                x,
                t=batched_times,
                model_kwargs=model_kwargs,
            )
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    @torch.no_grad()
    def _sample_uncond(self, shape):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img, _ = self._p_sample(img, t)
        return img

    @torch.no_grad()
    def _fast_sample_uncond(self, shape, clip_denoised=False):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self._model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    def _generate_mts(self, batch_size=16, model_kwargs=None, cond_fn=None):
        feature_size, seq_length = self.feature_size, self.total_seq_len
        if cond_fn is not None:
            sample_fn = (
                self._fast_sample_cond if self.fast_sampling else self._sample_cond
            )
            return sample_fn(
                (batch_size, seq_length, feature_size),
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )
        sample_fn = (
            self._fast_sample_uncond if self.fast_sampling else self._sample_uncond
        )
        return sample_fn((batch_size, seq_length, feature_size))

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def _q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self._q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        model_out = self._output(x, t, padding_masks)

        train_loss = self.loss_fn(model_out, target, reduction="none")

        fourier_loss = torch.tensor([0.0])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm="forward")
            fft2 = torch.fft.fft(target.transpose(1, 2), norm="forward")
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(
                torch.real(fft1), torch.real(fft2), reduction="none"
            ) + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction="none")
            train_loss += self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, "b ... -> b (...)", "mean")
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x, **kwargs):
        (
            b,
            c,
            n,
            device,
            feature_size,
        ) = *x.shape, x.device, self.feature_size
        assert n == feature_size, f"number of variable must be {feature_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)

    def _return_components(self, x, t: int):
        (
            b,
            c,
            n,
            device,
            feature_size,
        ) = *x.shape, x.device, self.feature_size
        assert n == feature_size, f"number of variable must be {feature_size}"
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self._q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def _fast_sample_infill(
        self,
        shape,
        target,
        sampling_timesteps,
        partial_mask=None,
        clip_denoised=False,
        model_kwargs=None,
    ):
        batch, device, total_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(
            time_pairs, desc="conditional sampling loop time step"
        ):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self._model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self._langevin_fn(
                sample=img,
                mean=pred_mean,
                sigma=sigma,
                t=time_cond,
                tgt_embs=target,
                partial_mask=partial_mask,
                **model_kwargs,
            )
            target_t = self._q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def _sample_infill(
        self,
        shape,
        target,
        partial_mask=None,
        clip_denoised=False,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="conditional sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self._p_sample_infill(
                x=img,
                t=t,
                clip_denoised=clip_denoised,
                target=target,
                partial_mask=partial_mask,
                model_kwargs=model_kwargs,
            )

        img[partial_mask] = target[partial_mask]
        return img

    def _p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=False,
        model_kwargs=None,
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = self._p_mean_variance(
            x=x, t=batched_times, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self._langevin_fn(
            sample=pred_img,
            mean=model_mean,
            sigma=sigma,
            t=batched_times,
            tgt_embs=target,
            partial_mask=partial_mask,
            **model_kwargs,
        )

        target_t = self._q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def _langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.0,
    ):
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self._output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = (
                        coef * ((mean - input_embs_param) ** 2 / 1.0).mean(dim=0).sum()
                    )
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = (
                        coef
                        * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
                    )
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss / sigma.mean()).mean(dim=0).sum()

                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter(
                    (
                        input_embs_param.data + coef_ * sigma.mean().item() * epsilon
                    ).detach()
                )

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample

    def _condition_mean(self, cond_fn, mean, log_variance, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x=x, t=t, **model_kwargs)
        new_mean = mean.float() + torch.exp(log_variance) * gradient.float()
        return new_mean

    def _condition_score(self, cond_fn, x_start, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)

        eps = self._predict_noise_from_start(x, t, x_start)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        pred_xstart = self._predict_start_from_noise(x, t, eps)
        model_mean, _, _ = self._q_posterior(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, pred_xstart

    def _sample_cond(self, shape, clip_denoised=False, model_kwargs=None, cond_fn=None):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img, x_start = self._p_sample(
                img,
                t,
                clip_denoised=clip_denoised,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
        return img

    def _fast_sample_cond(
        self, shape, clip_denoised=False, model_kwargs=None, cond_fn=None
    ):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self._model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if cond_fn is not None:
                _, x_start = self._condition_score(
                    cond_fn, x_start, img, time_cond, model_kwargs=model_kwargs
                )
                pred_noise = self._predict_noise_from_start(img, time_cond, x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    def training_step(self, batch, batch_idx):
        data = batch["seq"]
        loss = self(data, target=data)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch["seq"]
        loss = self(data, target=data)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _sample_impl(self, n_sample=1, condition: torch.Tensor = None, **kwargs):
        sampling_steps = kwargs.get("sampling_steps", 50)
        model_kwargs = {}
        model_kwargs["coef"] = kwargs.get("coef", 1e-1)
        model_kwargs["learning_rate"] = kwargs.get("stepsize", 1e-1)
        # self.model.load_state_dict(self.ema.ema_model.state_dict())

        if self.condition is None:
            sample = self._generate_mts(batch_size=n_sample)
            # samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            return sample
        else:
            x_shape = (condition.shape[0], self.total_seq_len, self.feature_size)
            # x = kwargs.get("seq", None)
            # assert x is not None, "x must be provided for sampling"
            # assert x.shape[1] == self.total_seq_len

            # t_m is mask
            # 0: missing
            # 1: observed
            if self.condition == "predict":
                t_m = torch.ones(x_shape).to(self.device)
                t_m[:, -self.predict_length :, :] = 0
                t_m = t_m.bool()
                target = torch.zeros(x_shape).to(self.device)
                target[:, : self.obs_len, :] = condition
            else:
                t_m = ~torch.isnan(condition)
                target = torch.nan_to_num(condition)

            all_samples = []
            for i in range(n_sample):
                if sampling_steps == self.num_timesteps:
                    sample = self._sample_infill(
                        shape=x_shape,
                        target=target,
                        # target=x * t_m,
                        partial_mask=t_m,
                        model_kwargs=model_kwargs,
                    )
                else:
                    sample = self._fast_sample_infill(
                        shape=x_shape,
                        target=target,
                        # target=x * t_m,
                        partial_mask=t_m,
                        model_kwargs=model_kwargs,
                        sampling_timesteps=sampling_steps,
                    )
                torch.cuda.empty_cache()
                all_samples.append(sample.detach())
            all_samples = torch.stack(all_samples, dim=-1)
            return all_samples[:, -self.predict_length :, :]

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update()

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.lr,
            betas=[0.9, 0.96],
            weight_decay=self.hparams.weight_decay,
        )
        return optim

    def on_fit_end(self):
        self.model.load_state_dict(self.ema.ema_model.state_dict())
