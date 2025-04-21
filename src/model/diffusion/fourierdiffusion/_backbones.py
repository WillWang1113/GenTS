import abc
import math
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        # Learnable Embedding matrix to map time steps to embeddings
        self.embedding = nn.Embedding(
            num_embeddings=max_len, embedding_dim=d_model, max_norm=math.sqrt(d_model)
        )  # (max_len, d_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a positional encoding to the tensor x.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, max_len, d_emb) to which the positional encoding should be added

        Returns:
            torch.Tensor: Tensor with an additional positional encoding
        """
        position = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, max_len)
        pe = self.embedding(position)  # (1, max_len, d_emb)
        x = x + pe
        return x


class TimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_time: int, use_time_axis: bool = True):
        super().__init__()

        # Learnable Embedding matrix to map time steps to embeddings
        self.embedding = nn.Embedding(
            num_embeddings=max_time, embedding_dim=d_model, max_norm=math.sqrt(d_model)
        )  # (max_time, d_emb)
        self.use_time_axis = use_time_axis

    def forward(
        self, x: torch.Tensor, timesteps: torch.LongTensor, use_time_axis: bool = True
    ) -> torch.Tensor:
        """Adds a time encoding to the tensor x.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, max_len, d_emb) to which the time encoding should be added
            timesteps (torch.LongTensor): Tensor of shape (batch_size,) containing the current timestep for each sample in the batch

        Returns:
            torch.Tensor: Tensor with an additional time encoding
        """
        t_emb = self.embedding(timesteps)  # (batch_size, d_model)
        if use_time_axis:
            t_emb = t_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        assert isinstance(t_emb, torch.Tensor)
        return x + t_emb


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps.
    Courtesy of https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, d_model: int, scale: float = 30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.d_model = d_model
        self.W = nn.Parameter(
            torch.randn((d_model + 1) // 2) * scale, requires_grad=False
        )

        self.dense = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, use_time_axis: bool = True
    ) -> torch.Tensor:
        time_proj = timesteps[:, None] * self.W[None, :] * 2 * torch.pi
        embeddings = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)

        # Slice to get exactly d_model
        t_emb = embeddings[:, : self.d_model]  # (batch_size, d_model)

        if use_time_axis:
            t_emb = t_emb.unsqueeze(1)

        projected_emb: torch.Tensor = self.dense(t_emb)

        return x + projected_emb


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
