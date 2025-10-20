from typing import Dict
import torchcde
from gents.model.base import BaseModel
from ._backbones import Generator, Discriminator
import torch
from torch.optim.swa_utils import AveragedModel


class SDEGAN(BaseModel):
    """`Neural SDEs as Infinite-Dimensional GANs <https://arxiv.org/pdf/2102.03657>`__
    
    Adapted from the `official codes <https://github.com/google-research/torchsde/blob/master/examples/sde_gan.py>`__
    
    .. note::
        SDEGAN allows for irregular data.
    
    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to None.
        initial_noise_size (int, optional): Gaussian distribution dimension that used for sampling. Defaults to 5.
        noise_size (int, optional): Diffusion size for SDE. Defaults to 3.
        hidden_size (int, optional): latent SDE size. Defaults to 16.
        d_model (int, optional): MLP size. Defaults to 16.
        n_layers (int, optional): MLP layers. Defaults to 1.
        init_mult1 (float, optional): Scale ratio for Generator's encoder initial parameter. Defaults to 3.
        init_mult2 (float, optional): Scale ratio for Generator's neural SDE initial parameter. Defaults to 0.5.
        swa_step_start (int, optional): Start step for stochastic weight average (SWA) of model parameters. Defaults to 5000.
        lr (Dict[str, float], optional): Learning rate for different networks. G: generator, D: discriminator.. Defaults to {"G": 2e-4, "D": 1e-3}.
        weight_decay (float, optional): Weight decay. Defaults to 0.01.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
        
    """

    # Allow for irregular time series as inputs, but not for imputation.
    ALLOW_CONDITION = [None]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        initial_noise_size: int = 5,
        noise_size: int = 3,
        hidden_size: int = 16,
        d_model: int = 16,
        n_layers: int = 1,
        init_mult1: float = 3,
        init_mult2: float = 0.5,
        swa_step_start: int = 5000,
        lr: Dict[str, float] = {"G": 2e-4, "D": 1e-3},
        weight_decay: float = 0.01,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        self.seq_len = seq_len
        self.automatic_optimization = False
        self.generator = Generator(
            seq_dim, initial_noise_size, noise_size, hidden_size, d_model, n_layers
        )
        self.discriminator = Discriminator(seq_dim, hidden_size, d_model, n_layers)
        self.averaged_generator = AveragedModel(self.generator)
        self.averaged_discriminator = AveragedModel(self.discriminator)
        self.swa_step_start = swa_step_start
        with torch.no_grad():
            for param in self.generator._initial.parameters():
                param *= init_mult1
            for param in self.generator._func.parameters():
                param *= init_mult2

    def training_step(self, batch, batch_idx):
        generator_optimiser, discriminator_optimiser = self.optimizers()

        ts = torch.linspace(
            0, self.seq_len - 1, self.seq_len, device=batch["seq"].device
        )
        real_samples = torch.concat([batch["t"].unsqueeze(-1), batch["coeffs"]], dim=-1)

        # ts = batch["t"]

        generated_samples = self.generator(ts, real_samples.shape[0])
        generated_score = self.discriminator(generated_samples)
        real_score = self.discriminator(real_samples)
        loss = generated_score - real_score

        generator_optimiser.zero_grad()
        discriminator_optimiser.zero_grad()

        self.manual_backward(loss)
        for param in self.generator.parameters():
            param.grad *= -1

        generator_optimiser.step()
        discriminator_optimiser.step()

        ###################
        # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
        # LipSwish activation functions).
        ###################
        with torch.no_grad():
            for module in self.discriminator.modules():
                if isinstance(module, torch.nn.Linear):
                    lim = 1 / module.out_features
                    module.weight.clamp_(-lim, lim)

        # Stochastic weight averaging typically improves performance.
        if self.global_step > self.swa_step_start:
            self.averaged_generator.update_parameters(self.generator)
            self.averaged_discriminator.update_parameters(self.discriminator)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        batch_size = batch["seq"].shape[0]
        ts = torch.linspace(
            0, self.seq_len - 1, self.seq_len, device=batch["seq"].device
        )
        real_samples = torch.concat([batch["t"].unsqueeze(-1), batch["coeffs"]], dim=-1)

        if self.global_step > self.swa_step_start:
            generator = self.generator
            discriminator = self.discriminator

        else:
            generator = self.averaged_generator.module
            discriminator = self.averaged_discriminator.module

        generated_samples = generator(ts, batch_size)
        generated_score = discriminator(generated_samples)
        real_score = discriminator(real_samples)
        loss = generated_score - real_score

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        # total_unaveraged_loss = evaluate_loss(ts, batch_size, train_dataloader, generator, discriminator)
        # if step > swa_step_start:
        #     total_averaged_loss = evaluate_loss(ts, batch_size, train_dataloader, averaged_generator.module,
        #                                         averaged_discriminator.module)
        #     # trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
        #     #                 f"Loss (averaged): {total_averaged_loss:.4f}")
        # else:
        #     # trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        ts = torch.linspace(0, self.seq_len - 1, self.seq_len, device=self.device)
        generated_samples = self.generator(ts, n_sample).cpu()
        generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(ts)
        generated_samples = generated_samples[..., 1:]
        return generated_samples

    def configure_optimizers(self):
        generator_optimiser = torch.optim.Adadelta(
            self.generator.parameters(),
            lr=self.hparams.lr["G"],
            weight_decay=self.hparams.weight_decay,
        )
        discriminator_optimiser = torch.optim.Adadelta(
            self.discriminator.parameters(),
            lr=self.hparams.lr["D"],
            weight_decay=self.hparams.weight_decay,
        )

        return [generator_optimiser, discriminator_optimiser], []

    def on_fit_end(self):
        self.generator.load_state_dict(self.averaged_generator.module.state_dict())
        self.discriminator.load_state_dict(
            self.averaged_discriminator.module.state_dict()
        )
        # return super().on_fit_end()
