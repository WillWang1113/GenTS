import abc
import math
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchvision.ops import MLP
from tqdm import tqdm

from src.layers.embed import (
    GaussianFourierProjection,
    PositionalEncoding,
    TimeEncoding,
)
from src.utils.lr_schedule import get_cosine_schedule_with_warmup
from src.model.base import BaseModel
from src.utils.timefreq import idft, dft

SamplingOutput = namedtuple("SamplingOutput", ["prev_sample"])

@dataclass
class DiffusableBatch:
    X: torch.Tensor
    y: Optional[torch.Tensor] = None
    timesteps: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.X)

    @property
    def device(self) -> torch.device:
        return self.X.device


"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs. Adapted from https://github.com/yang-song/score_sde."""


class SDE(abc.ABC):
    """SDE abstract class for FourierDiffusion. Functions are designed for a mini-batch of inputs."""

    noise_scaling: bool = True

    def __init__(self, eps: float = 1e-5):
        """Construct an SDE.
        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        # self.noise_scaling = fourier_noise_scaling
        self.eps = eps
        self.G: Optional[torch.Tensor] = None

    @property
    def T(self) -> float:
        """End time of the SDE."""
        return 1.0

    @abc.abstractmethod
    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""

    @abc.abstractmethod
    def step(
        self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor
    ) -> SamplingOutput: ...

    def set_noise_scaling(self, max_len: int) -> None:
        """Finish the initialization of the scheduler by setting G (scaling diagonal)

        Args:
            max_len (int): number of time steps of the time series
        """

        G = torch.ones(max_len)
        if self.noise_scaling:
            G = 1 / (math.sqrt(2)) * G
            # Double the variance for the first component
            G[0] *= math.sqrt(2)
            # Double the variance for the middle component if max_len is even
            if max_len % 2 == 0:
                G[max_len // 2] *= math.sqrt(2)

        self.G = G  # Tensor of size (max_len)
        self.G_matrix = torch.diag(G)  # Tensor of size (max_len, max_len)
        assert G.shape[0] == max_len

    def set_timesteps(self, num_diffusion_steps: int) -> None:
        self.timesteps = torch.linspace(1.0, self.eps, num_diffusion_steps)
        self.step_size = self.timesteps[0] - self.timesteps[1]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x0 = original_samples
        mean, _ = self.marginal_prob(x0, timesteps)

        # Note that the std is not used here because the noise has been scaled prior to calling the function
        sample = mean + noise
        return sample

    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        if self.G is None:
            self.set_noise_scaling(shape[1])
        assert self.G is not None

        # Reshape the G matrix to be (1, max_len, max_len)
        scaling_matrix = self.G_matrix.view(
            -1, self.G_matrix.shape[0], self.G_matrix.shape[1]
        )

        z = torch.randn(*shape)
        # Return G@z where z \sim N(0,I)
        return torch.matmul(scaling_matrix, z)


class VEScheduler(SDE):
    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        # fourier_noise_scaling: bool = False,
        eps: float = 1e-5,
    ):
        """Construct a Variance Exploding SDE.
        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(eps=eps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor
    ]:  # perturbation kernel P(X(t)|X(0)) parameters
        if self.G is None:
            self.set_noise_scaling(x.shape[1])
        assert self.G is not None

        sigma_min = torch.tensor(self.sigma_min).type_as(t)
        sigma_max = torch.tensor(self.sigma_max).type_as(t)
        std = (sigma_min * (sigma_max / sigma_min) ** t).view(-1, 1) * self.G.to(
            x.device
        )
        mean = x
        return mean, std

    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        # In the case of VESDE, the prior is scaled by the maximum noise std
        return self.sigma_max * super().prior_sampling(shape)

    def step(
        self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor
    ) -> SamplingOutput:
        """Single denoising step, used for sampling.

        Args:
            model_output (torch.Tensor): output of the score model
            timestep (torch.Tensor): timestep
            sample (torch.Tensor): current sample to be denoised

        Returns:
            SamplingOutput: _description_
        """
        sqrt_derivative = (
            self.sigma_min
            * math.sqrt(2 * math.log(self.sigma_max / self.sigma_min))
            * (self.sigma_max / self.sigma_min) ** (timestep)
        )

        diffusion = torch.diag_embed(sqrt_derivative * self.G).to(device=sample.device)

        # Compute drift for the reverse: f(x,t) - G(x,t)G(x,t)^{T}*score
        drift = -(
            torch.matmul(diffusion * diffusion, model_output)
        )  # Notice that the drift of the forward is 0

        # Sample noise
        z = torch.randn_like(sample)
        assert self.step_size > 0
        x = (
            sample
            - drift * self.step_size  # - sign because of reverse time
            + torch.sqrt(self.step_size) * torch.matmul(diffusion, z)
        )
        output = SamplingOutput(prev_sample=x)
        return output


class VPScheduler(SDE):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        # fourier_noise_scaling: bool = True,
        eps: float = 1e-5,
    ):
        """Construct a Variance Preserving SDE.
        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
          G: tensor of size max_len
        """
        super().__init__(eps=eps)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # first check if G has been init.
        if self.G is None:
            self.set_noise_scaling(x.shape[1])
        assert self.G is not None

        # Compute -1/2*\int_0^t \beta(s) ds
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )

        mean = (
            torch.exp(log_mean_coeff[(...,) + (None,) * len(x.shape[1:])]) * x
        )  # mean: (batch_size, max_len, n_channels)

        std = torch.sqrt(
            (1.0 - torch.exp(2.0 * log_mean_coeff.view(-1, 1)))
        ) * self.G.to(x.device)  # std: (batch_size, max_len)

        return mean, std

    def get_beta(self, timestep: float) -> float:
        return self.beta_0 + timestep * (self.beta_1 - self.beta_0)

    def step(
        self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor
    ) -> SamplingOutput:
        """Single denoising step, used for sampling.

        Args:
            model_output (torch.Tensor): output of the score model
            timestep (torch.Tensor): timestep
            sample (torch.Tensor): current sample to be denoised

        Returns:
            SamplingOutput: _description_
        """
        beta = self.get_beta(timestep)
        diffusion = torch.diag_embed(math.sqrt(beta) * self.G).to(device=sample.device)

        # Compute drift
        drift = -0.5 * beta * sample - (
            torch.matmul(diffusion * diffusion, model_output)
        )

        # Sample noise
        z = torch.randn_like(sample)
        assert self.step_size > 0
        x = (
            sample
            - drift * self.step_size
            + torch.sqrt(self.step_size) * torch.matmul(diffusion, z)
        )
        output = SamplingOutput(prev_sample=x)
        return output


# Courtesy of https://github.com/yang-song/score_sde_pytorch/blob/main/losses.py
def get_sde_loss_fn(
    scheduler: SDE,
    train: bool,
    reduce_mean: bool = True,
    likelihood_weighting: bool = False,
) -> Callable[[nn.Module, DiffusableBatch], torch.Tensor]:
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model: nn.Module, batch: DiffusableBatch) -> torch.Tensor:
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        if train:
            model.train()
        else:
            model.eval()

        X = batch.X
        y = batch.y
        timesteps = batch.timesteps

        # Sample a time step uniformly from [eps, T]
        if timesteps is None:
            timesteps = (
                torch.rand(X.shape[0], device=X.device) * (scheduler.T - scheduler.eps)
                + scheduler.eps
            )

        # Sample the gaussian noise
        z = torch.randn_like(X)  # (batch_size, max_len, n_channels)

        _, std = scheduler.marginal_prob(X, timesteps)  # (batch_size, max_len)
        var = std**2  # (batch_size, max_len)

        std_matrix = torch.diag_embed(std)  # (batch_size, max_len, max_len)
        inverse_std_matrix = torch.diag_embed(1 / std)  # (batch_size, max_len, max_len)

        # compute Sigma^{1/2}z to be used for forward sampling: noise is x(t)
        noise = torch.matmul(std_matrix, z)  # (batch_size, max_len, n_channels)

        # compute Sigma^{-1/2}z to be used for the loss: target_noise is grad log p(x(t)|x(0))
        target_noise = torch.matmul(
            inverse_std_matrix, z
        )  # (batch_size, max_len, n_channels)

        # Do the perturbation
        X_noisy = scheduler.add_noise(
            original_samples=X, noise=noise, timesteps=timesteps
        )

        noisy_batch = DiffusableBatch(X=X_noisy, y=y, timesteps=timesteps)

        # Compute the score function
        score = model(noisy_batch)

        if not likelihood_weighting:
            # lambda(t) = E[||\grad log p(x(t)|x(0))||^2]

            # Compute 1/tr(\Sigma^{-1})
            weighting_factor = 1.0 / torch.sum(1.0 / var, dim=1)  # (batch_size,)
            assert weighting_factor.shape == (X.shape[0],)

            # 1/tr(\Sigma^{-1}) * ||s + \Sigma^{-1/2}z||^2
            losses = weighting_factor.view(-1, 1, 1) * torch.square(
                score + target_noise
            )

            # No relative minus size because:
            # log(p(x(t)|x(0))) = -1/2 * (x(t) -mean)^{T} Cov^{-1} (x(t) - mean) + C
            # grad log(p(x(t)|x(0))) = (-1) * Cov^{-1} (x(t) - mean)

            # Reduction
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)  # type: ignore

        else:
            # Compute the Mahalanobis distance, cf. https://arxiv.org/pdf/2111.13606.pdf + https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf

            # 1) s - \grad log p(x)
            difference = score + target_noise  # (batch_size, max_len, n_channels)

            # 2) Sigma(s - \grad log p(x))
            scaled_difference = torch.matmul(std_matrix, difference)

            # 3) Compute the loss
            losses = torch.square(scaled_difference)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)  # type: ignore

        loss = torch.mean(losses)
        return loss

    return loss_fn


class FourierDiffusion(BaseModel):
    ALLOW_CONDITION = [None]
    scale_noise: bool = True

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        noise_schedule: str = "vpsde",
        # fourier_noise_scaling: bool = True,
        hidden_size: int = 72,
        num_layers: int = 10,
        n_head: int = 4,
        n_diff_steps: int = 1000,
        lr: float = 1e-4,
        likelihood_weighting: bool = False,
        condition: str = None,
        **kwargs
    ) -> None:
        super().__init__(seq_len, seq_dim, condition)
        # Hyperparameters
        self.max_len = seq_len
        self.n_channels = seq_dim

        # assert noise_schedule in ["vpsde", "vesde"]
        # self.noise_scheduler = noise_schedule
        self.noise_scheduler = (
            VPScheduler() if noise_schedule == "vpsde" else VEScheduler()
        )
        self.num_warmup_steps = n_diff_steps // 10
        self.n_diff_steps = n_diff_steps
        self.lr_max = lr
        self.d_model = hidden_size
        # self.scale_noise = fourier_noise_scaling
        # self.sample_batch_size = sample_batch_size

        # Loss function
        self.likelihood_weighting = likelihood_weighting
        self.training_loss_fn, self.validation_loss_fn = self.set_loss_fn()

        # Model components
        self.pos_encoder = PositionalEncoding(d_model=hidden_size, max_len=self.max_len)
        self.time_encoder = self.set_time_encoder()
        self.embedder = nn.Linear(in_features=seq_dim, out_features=hidden_size)
        self.unembedder = nn.Linear(in_features=hidden_size, out_features=seq_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_head, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers
        )

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()

    def forward(self, batch: DiffusableBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.max_len,
            self.n_channels,
        ), (
            f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"
        )

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        # Channel embedding
        X = self.embedder(X)

        # Add positional encoding
        X = self.pos_encoder(X)

        # Add time encoding
        X = self.time_encoder(X, timesteps)

        # Backbone
        X = self.backbone(X)

        # Channel unembedding
        X = self.unembedder(X)

        assert isinstance(X, torch.Tensor)

        return X

    def training_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        batch = DiffusableBatch(X=dft(batch["seq"]))
        loss = self.training_loss_fn(self, batch)

        self.log_dict(
            {"train/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=True,
        )
        return loss

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        batch = DiffusableBatch(X=dft(batch["seq"]))
        loss = self.validation_loss_fn(self, batch)
        self.log_dict(
            {"val/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr_max)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.n_diff_steps,
        )
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def set_loss_fn(
        self,
    ) -> tuple[
        Callable[[nn.Module, DiffusableBatch], torch.Tensor],
        Callable[[nn.Module, DiffusableBatch], torch.Tensor],
    ]:
        # Depending on the scheduler, get the right loss function

        if isinstance(self.noise_scheduler, SDE):
            training_loss_fn = get_sde_loss_fn(
                scheduler=self.noise_scheduler,
                train=True,
                likelihood_weighting=self.likelihood_weighting,
            )
            validation_loss_fn = get_sde_loss_fn(
                scheduler=self.noise_scheduler,
                train=False,
                likelihood_weighting=self.likelihood_weighting,
            )

            return training_loss_fn, validation_loss_fn

        else:
            raise NotImplementedError(
                f"Scheduler {self.noise_scheduler} not implemented yet, cannot set loss function."
            )

    def set_time_encoder(self) -> TimeEncoding | GaussianFourierProjection:
        if isinstance(self.noise_scheduler, SDE):
            return GaussianFourierProjection(d_model=self.d_model)

        else:
            raise NotImplementedError(
                f"Scheduler {self.noise_scheduler} not implemented yet, cannot set time encoder."
            )

    def reverse_diffusion_step(self, batch: DiffusableBatch) -> torch.Tensor:
        # Get X and timesteps
        X = batch.X
        timesteps = batch.timesteps

        # Check the validity of the timestep (current implementation assumes same time for all samples)
        assert timesteps is not None and timesteps.size(0) == len(batch)
        assert torch.min(timesteps) == torch.max(timesteps)

        # Predict score for the current batch
        score = self(batch)
        # Apply a step of reverse diffusion
        output = self.noise_scheduler.step(
            model_output=score, timestep=timesteps[0].item(), sample=X
        )

        X_prev = output.prev_sample
        assert isinstance(X_prev, torch.Tensor)

        return X_prev

    def _sample_impl(
        self,
        n_sample: int,
        condition: Optional[int] = None,
        sample_batch_size: Optional[int] = 32,
        num_diffusion_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Set the score model in eval mode and move it to GPU
        # self.eval()

        # If the number of diffusion steps is not provided, use the number of training steps
        num_diffusion_steps = (
            self.n_diff_steps if num_diffusion_steps is None else num_diffusion_steps
        )
        self.noise_scheduler.set_timesteps(num_diffusion_steps)

        # Create the list that will store the samples
        all_samples = []

        # Compute the required amount of batches
        num_batches = max(1, n_sample // sample_batch_size)

        for batch_idx in tqdm(
            range(num_batches),
            desc="Sampling",
            unit="batch",
            leave=False,
            colour="blue",
        ):
            # Compute the batch size
            batch_size = min(
                n_sample - batch_idx * sample_batch_size,
                sample_batch_size,
            )
            # Sample from noise distribution
            X = self.sample_prior(batch_size)

            # Perform the diffusion step by step
            for t in self.noise_scheduler.timesteps:
                # Define timesteps for the batch
                timesteps = torch.full(
                    (batch_size,),
                    t,
                    dtype=(torch.long if isinstance(t.item(), int) else torch.float32),
                    device=self.device,
                    requires_grad=False,
                )
                # Create diffusable batch
                batch = DiffusableBatch(X=X, y=condition, timesteps=timesteps)
                # Return denoised X

                X = self.reverse_diffusion_step(batch)

            # Add the samples to the list
            all_samples.append(X.cpu())
        all_samples = torch.cat(all_samples, dim=0)
        return idft(all_samples)

    def sample_prior(self, batch_size: int) -> torch.Tensor:
        # Sample from the prior distribution
        if isinstance(self.noise_scheduler, SDE):
            X = self.noise_scheduler.prior_sampling(
                (batch_size, self.max_len, self.n_channels)
            ).to(device=self.device)

        else:
            raise NotImplementedError("Scheduler not recognized.")

        assert isinstance(X, torch.Tensor)
        return X


class FourierDiffusionMLP(FourierDiffusion):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        noise_schedule: str = "vpsde",
        # fourier_noise_scaling: bool = True,
        hidden_size: int = 72,
        d_mlp: int = 1024,
        num_layers: int = 10,
        n_diff_steps: int = 1000,
        lr: float = 1e-4,
        likelihood_weighting: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            seq_dim=seq_dim,
            seq_len=seq_len,
            noise_schedule=noise_schedule,
            # fourier_noise_scaling=fourier_noise_scaling,
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_head=1,
            n_diff_steps=n_diff_steps,
            lr=lr,
            likelihood_weighting=likelihood_weighting,
        )

        # Change the components that should be different in our score model
        self.embedder = nn.Linear(
            in_features=seq_len * seq_dim, out_features=hidden_size
        )
        self.unembedder = nn.Linear(
            in_features=hidden_size, out_features=seq_len * seq_dim
        )

        self.backbone = nn.ModuleList(  # type: ignore
            [
                MLP(
                    in_channels=hidden_size,
                    hidden_channels=[d_mlp, hidden_size],
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        self.pos_encoder = None

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()

    def forward(self, batch: DiffusableBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.max_len,
            self.n_channels,
        ), (
            f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"
        )

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        # Flatten the tensor
        # X = rearrange(X, "b t c -> b (t c)")
        X = torch.flatten(X, start_dim=1)

        # Channel embedding
        X = self.embedder(X)

        # Add time encoding
        X = self.time_encoder(X, timesteps, use_time_axis=False)

        # Backbone
        for layer in self.backbone:  # type: ignore
            X = X + layer(X)

        # Channel unembedding
        X = self.unembedder(X)

        # Unflatten the tensor
        X = X.view(-1, self.max_len, self.n_channels)
        # X = rearrange(X, "b (t c) -> b t c", t=self.max_len, c=self.n_channels)

        assert isinstance(X, torch.Tensor)

        return X


class FourierDiffusionLSTM(FourierDiffusion):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        noise_schedule: str = "vpsde",
        # fourier_noise_scaling: bool = True,
        hidden_size: int = 72,
        num_layers: int = 3,
        n_diff_steps: int = 1000,
        lr: float = 1e-3,
        likelihood_weighting: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            seq_dim=seq_dim,
            seq_len=seq_len,
            noise_schedule=noise_schedule,
            # fourier_noise_scaling=fourier_noise_scaling,
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_head=1,
            n_diff_steps=n_diff_steps,
            lr=lr,
            likelihood_weighting=likelihood_weighting,
        )

        # Change the components that should be different in our score model
        self.backbone = nn.ModuleList(  # type: ignore
            [
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.pos_encoder = None

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()

    def forward(self, batch: DiffusableBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.max_len,
            self.n_channels,
        ), (
            f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"
        )

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        # Channel embedding
        X = self.embedder(X)

        # Add time encoding
        X = self.time_encoder(X, timesteps)

        # Backbone
        for layer in self.backbone:  # type: ignore
            X = X + layer(X)[0]

        # Channel unembedding
        X = self.unembedder(X)

        assert isinstance(X, torch.Tensor)

        return X
