from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchvision.ops import MLP
from tqdm import tqdm


from gents.model.base import BaseModel
from ._utils import idft, dft, get_cosine_schedule_with_warmup
from ._backbones import (
    VEScheduler,
    VPScheduler,
    SDE,
    get_sde_loss_fn,
    DiffusableBatch,
    GaussianFourierProjection,
    PositionalEncoding,
    TimeEncoding,
)


class FourierDiffusion(BaseModel):
    """`Time Series Diffusion in the Frequency Domain <https://arxiv.org/pdf/2402.05933>`__

    Adapted from the `official codes <https://github.com/JonathanCrabbe/FourierDiffusion>`__

    Args:
        seq_len (int): Target sequence length
        seq_dim (int, optional): Target sequence dimension. Only for univariate time series Defaults to 1.
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to None.
        noise_schedule (str, optional): Diffusion noise schedule. Choose from `['vpsde', 'vesde']` Defaults to "vpsde".
        hidden_size (int, optional): Model size of transformer layers. Defaults to 72.
        num_layers (int, optional): Transformer layers. Defaults to 10.
        n_head (int, optional): Attention heads. Defaults to 4.
        n_diff_steps (int, optional): Total diffusion steps. Defaults to 1000.
        likelihood_weighting (bool, optional): If `True`, weight the mixture of score matching losses according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in the original paper. Defaults to False.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """

    ALLOW_CONDITION = [None]
    _scale_noise: bool = True

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        noise_schedule: str = "vpsde",
        # fourier_noise_scaling: bool = True,
        hidden_size: int = 72,
        num_layers: int = 10,
        n_head: int = 4,
        n_diff_steps: int = 1000,
        likelihood_weighting: bool = False,
        lr: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()
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
        self.training_loss_fn, self.validation_loss_fn = self._set_loss_fn()

        # Model components
        self.pos_encoder = PositionalEncoding(d_model=hidden_size, max_len=self.max_len)
        self.time_encoder = self._set_time_encoder()
        self.embedder = nn.Linear(in_features=seq_dim, out_features=hidden_size)
        self.unembedder = nn.Linear(in_features=hidden_size, out_features=seq_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_head, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers
        )

        

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
            {"train_loss": loss},
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
            {"val_loss": loss},
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

    def _set_loss_fn(
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

    def _set_time_encoder(self) -> TimeEncoding | GaussianFourierProjection:
        if isinstance(self.noise_scheduler, SDE):
            return GaussianFourierProjection(d_model=self.d_model)

        else:
            raise NotImplementedError(
                f"Scheduler {self.noise_scheduler} not implemented yet, cannot set time encoder."
            )

    def _reverse_diffusion_step(self, batch: DiffusableBatch) -> torch.Tensor:
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
        sample_batch_size: Optional[int] = None,
        num_diffusion_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Set the score model in eval mode and move it to GPU
        # self.eval()
        if sample_batch_size is None:
            sample_batch_size = n_sample
        
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
            X = self._sample_prior(batch_size)

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

                X = self._reverse_diffusion_step(batch)

            # Add the samples to the list
            all_samples.append(X.cpu())
        all_samples = torch.cat(all_samples, dim=0)
        return idft(all_samples)

    def _sample_prior(self, batch_size: int) -> torch.Tensor:
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
    """FourierDiffusion with MLP backbone

    Args:
        seq_len (int): Target sequence length
        seq_dim (int, optional): Target sequence dimension. Only for univariate time series Defaults to 1.
        noise_schedule (str, optional): Diffusion noise schedule. Choose from `['vpsde', 'vesde']` Defaults to "vpsde".
        hidden_size (int, optional): Model size of transformer layers. Defaults to 72.
        num_layers (int, optional): Transformer layers. Defaults to 10.
        n_diff_steps (int, optional): Total diffusion steps. Defaults to 1000.
        likelihood_weighting (bool, optional): If `True`, weight the mixture of score matching losses according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in the original paper. Defaults to False.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
        
    """
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
        likelihood_weighting: bool = False,
        lr: float = 1e-4,
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
    """FourierDiffusion with LSTM backbone

    Args:
        seq_len (int): Target sequence length
        seq_dim (int, optional): Target sequence dimension. Only for univariate time series Defaults to 1.
        noise_schedule (str, optional): Diffusion noise schedule. Choose from `['vpsde', 'vesde']` Defaults to "vpsde".
        hidden_size (int, optional): Model size of transformer layers. Defaults to 72.
        num_layers (int, optional): Transformer layers. Defaults to 10.
        n_diff_steps (int, optional): Total diffusion steps. Defaults to 1000.
        likelihood_weighting (bool, optional): If `True`, weight the mixture of score matching losses according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in the original paper. Defaults to False.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        noise_schedule: str = "vpsde",
        # fourier_noise_scaling: bool = True,
        hidden_size: int = 72,
        num_layers: int = 3,
        n_diff_steps: int = 1000,
        likelihood_weighting: bool = False,
        lr: float = 1e-3,
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
