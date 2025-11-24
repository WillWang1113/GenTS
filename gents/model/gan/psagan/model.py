from copy import deepcopy
from math import log2
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gents.evaluation.model_free.distribution_distance import WassersteinDistances
from gents.evaluation.model_free.errors import crps
from gents.model.base import BaseModel
from ._backbones import ProDiscriminator, ProGenerator


class PSAGAN(BaseModel):
    """`PSA-GAN <https://openreview.net/pdf?id=Ix_mh42xq5w>`__: PROGRESSIVE SELF ATTENTION GANS FOR SYNTHETIC TIME SERIES

    Adapted from the `official codes <https://github.com/mbohlkeschneider/psa-gan>`__

    .. note::
        The orignial codes are based on Gluonts, we adapt the source codes into our framework.

    Args:
        seq_len (int): Target sequence length. In PSAGAN, `seq_len` should be power of 2 and greater than 8, e.g. 16, 32,
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given conditions, should be one of `ALLOW_CONDITION`. Defaults to None.
        time_feat_dim (int, optional): Time-related feature dimension, e.g. calendar features. 0 for no extra time features. Defaults to 0.
        ks_conv (int, optional): Kernel size for conv layer. Defaults to 3.
        ks_query (int, optional): Kernel size for attention query conv layer. Defaults to 1.
        ks_key (int, optional): Kernel size for attention key conv layer. Defaults to 1.
        ks_value (int, optional): Kernel size for attention query value layer. Defaults to 1.
        hidden_size (int, optional): Hidden size for G and D. Defaults to 32.
        depth_schedule (list, optional): At which epochs the model gets deeper. Starting with `seq_len=8`, `8*2^len(depth_schedule)` should be equal to `seq_len` Defaults to [5, 10, 15].
        epoch_fade_in (int, optional): Each time the model gets deeper, how many epochs it takes to finish deepening, should be less the the gap between two depth schedule. Defaults to 2.
        n_critic (int, optional): G/D update times. Defaults to 2.
        lr (Dict[str, float], optional): Dict of learning rates for G and D. Defaults to {"G": 1e-4, "D": 1e-4}.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.
    """

    ALLOW_CONDITION = [None, "predict"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        time_feat_dim: int = 0,
        hidden_size: int = 32,
        ks_conv: int = 3,
        ks_query: int = 1,
        ks_key: int = 1,
        ks_value: int = 1,
        # depth_schedule: list = [5, 10],
        epoch_fade_in: int = 2,
        n_critic: int = 2,
        lr: Dict[str, float] = {"G": 1e-4, "D": 1e-4},
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.context_len = self.obs_len if condition == "predict" else 0
        self.orig_seq_len = deepcopy(seq_len)
        if log2(seq_len) % 1 != 0:
            self.need_init_interp = True
            seq_len = 2 ** (int(log2(seq_len)) + 1)
        else:
            self.need_init_interp = False
        self.seq_len = seq_len
        # print("orig seq len:", self.orig_seq_len)
        # print("orig seq len:", self.seq_len)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = ProGenerator(
            seq_len,
            seq_dim,
            time_feat_dim,
            hidden_size=hidden_size,
            context_length=self.context_len,
            ks_conv=ks_conv,
            ks_key=ks_key,
            ks_query=ks_query,
            ks_value=ks_value,
            **kwargs,
        )
        self.discriminator = ProDiscriminator(
            seq_len,
            seq_dim,
            time_feat_dim,
            hidden_size=hidden_size,
            ks_conv=ks_conv,
            ks_key=ks_key,
            ks_query=ks_query,
            ks_value=ks_value,
            **kwargs,
        )
        self.seq_dim = seq_dim
        self.time_feat_dim = time_feat_dim
        self.depth_schedule = [i * 5 for i in range(1, int(log2(seq_len)) - 2)]
        # print(self.depth_schedule)
        # self.depth_schedule = depth_schedule
        self.condition = condition
        self.depth = 0
        self.pretrain_schedule = []
        self.nb_epoch_fade_in_new_layer = epoch_fade_in
        for k in self.depth_schedule:
            self.pretrain_schedule.append((k, k + epoch_fade_in))
        self.nb_stage = len(self.depth_schedule) if self.depth_schedule else 0
        self.loss_fn = nn.MSELoss()

    def _residual(self):
        if self.nb_stage >= 0:
            if len(self.pretrain_schedule) > 0:
                self.start_epoch_test = self.pretrain_schedule[0][0]
                self.end_epoch_test = self.pretrain_schedule[0][1]
                if (
                    self.end_epoch_test
                    > self.trainer.current_epoch
                    > self.start_epoch_test
                ):
                    self.start_epoch = self.pretrain_schedule[0][0]
                    self.end_epoch = self.pretrain_schedule[0][1]
                    self.pretrain_schedule.pop(0)
        try:
            if self.end_epoch >= self.trainer.current_epoch >= self.start_epoch:
                residual_factor = self._linear_interpolation(
                    self.start_epoch, self.end_epoch, self.trainer.current_epoch
                )
                self.generator.residual_factor = residual_factor
                self.discriminator.residual_factor = residual_factor

                return True
            else:
                return False
        except Exception:
            return False

    def _increase_depth(self):
        # Piece of code to compute at which depth the tensor should flow
        if self.nb_stage > 0:
            self.update_epoch = self.depth_schedule[0]
            if self.trainer.current_epoch > self.update_epoch:
                self.depth += 1
                self.nb_stage -= 1
                self.depth_schedule.pop(0)

    def _linear_interpolation(self, alpha, beta, x):
        assert beta > alpha
        return (x - alpha) / (beta - alpha)

    def _momment_loss(self, preds, target):
        std_loss = torch.abs(preds.std(dim=2) - target.std(dim=2)).mean()
        mean_loss = torch.abs(preds.mean(dim=2) - target.mean(dim=2)).mean()
        momment_loss = std_loss + mean_loss
        return momment_loss

    def on_train_epoch_start(self):
        """On each epoch start, check whether to increase model depth. If needed, then increase."""
        self._increase_depth()
        self.residual = self._residual()

    def training_step(self, batch, batch_idx):
        x = batch["seq"].permute(0, 2, 1)[..., -self.seq_len :]
        c = batch.get("c", None)

        if self.need_init_interp:
            new_len = 2 ** (int(log2(x.shape[2])) + 1)
            x = F.interpolate(x, size=new_len, mode="linear")

        # if c is not None:
        #     c = c.permute(0, 2, 1)
        if self.time_feat_dim > 0:
            assert batch.get("time_feat", None) is not None
        tf = batch.get("time_feat", None)
        optimizer_g, optimizer_d = self.optimizers()

        # set model growing params
        # self._increase_depth()
        # residual = self._residual()

        # sample noise
        z = torch.randn_like(x)

        # train generator
        if (self.global_step + 1) % (self.hparams.n_critic + 1) != 0:
            self.toggle_optimizer(optimizer_g)

            # generate images
            gen_fake = self.generator(
                z, time_feat=tf, depth=self.depth, residual=self.residual, context=c
            )

            reduce_factor = int(log2(self.seq_len)) - int(log2(gen_fake.size(2)))
            reduced_target = F.avg_pool1d(
                x,
                kernel_size=2**reduce_factor,
            )

            y_fake_gen = self.discriminator(
                gen_fake, time_feat=tf, depth=self.depth, residual=self.residual
            )
            loss_g = self.loss_fn(y_fake_gen, torch.ones_like(y_fake_gen)) / 2
            # print(loss_g)
            # add momment loss
            loss_g = loss_g + self._momment_loss(gen_fake, reduced_target)

            # if self.scaling_penalty != 0:
            #     self.run.score_g += self._scaling_penalty(generated.squeeze(1))

            # adversarial loss is binary cross-entropy
            # g_loss = -torch.mean(self.discriminator(self.generator(z, c), c))
            optimizer_g.zero_grad()
            self.manual_backward(loss_g)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)
            loss_dict = {"g_loss": loss_g}
            self.log_dict(
                loss_dict,
                on_epoch=True,
                on_step=False,
            )

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples

        # discriminator loss is the average of these
        else:
            self.toggle_optimizer(optimizer_d)
            with torch.no_grad():
                gen_fake = self.generator(
                    z, time_feat=tf, depth=self.depth, residual=self.residual, context=c
                )
            reduce_factor = int(log2(self.seq_len)) - int(log2(gen_fake.size(2)))
            reduced_target = F.avg_pool1d(
                x,
                kernel_size=2**reduce_factor,
            )

            y_fake = self.discriminator(
                gen_fake, time_feat=tf, depth=self.depth, residual=self.residual
            )
            y_real = self.discriminator(
                reduced_target, time_feat=tf, depth=self.depth, residual=self.residual
            )

            d_loss = self.loss_fn(y_real, torch.ones_like(y_real)) + self.loss_fn(
                y_fake, torch.zeros_like(y_fake)
            )
            d_loss = d_loss / 2
            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

            # for p in self.discriminator.parameters():
            #     p.data.clamp_(-self.hparams.clip_value, self.hparams.clip_value)

            loss_dict = {"d_loss": d_loss}
            self.log_dict(
                loss_dict,
                on_epoch=True,
                on_step=False,
            )

    def validation_step(self, batch, batch_idx):
        batch_size = batch["seq"].shape[0]

        if self.condition is None:
            val_samples = self.sample(batch_size)
            val_loss = (
                WassersteinDistances(
                    batch["seq"].cpu().flatten(1).numpy(),
                    val_samples.cpu().flatten(1).numpy(),
                )
                .marginal_distances()
                .mean()
            )
        elif self.condition == "predict":
            val_samples = self.sample(50, batch.get("c", None))
            val_loss = crps(batch["seq"], val_samples)

        self.log_dict({"val_loss": val_loss}, on_epoch=True, prog_bar=True)

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        time_feat = kwargs.get("time_feat", None)

        if ((self.time_feat_dim > 0) and (time_feat is None)) or (
            (self.time_feat_dim == 0) and (time_feat is not None)
        ):
            raise ValueError(
                "Given time feats but model does not use them or vice versa"
            )

        # Unconditional
        if self.condition is None:
            noise = torch.randn((n_sample, self.seq_dim, self.seq_len)).to(self.device)
            samples = self.generator(x=noise, time_feat=time_feat, context=None)
            if self.need_init_interp:
                return F.interpolate(
                    samples,
                    size=self.orig_seq_len,
                    mode="linear",
                ).permute(0, 2, 1)
            else:
                return samples.permute(0, 2, 1)
        else:
            # Forecasting
            all_samples = []
            for i in range(n_sample):
                noise = torch.randn(
                    (condition.shape[0], self.seq_dim, self.seq_len)
                ).to(self.device)
                sample = self.generator(x=noise, time_feat=time_feat, context=condition)
                if self.need_init_interp:
                    sample = F.interpolate(
                        sample,
                        size=self.orig_seq_len,
                        mode="linear",
                    ).permute(0, 2, 1)
                else:
                    sample = sample.permute(0, 2, 1)
                all_samples.append(sample)
            return torch.stack(all_samples, dim=-1)

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr["G"],
            weight_decay=self.hparams.weight_decay,
        )
        d_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr["D"],
            weight_decay=self.hparams.weight_decay,
        )
        return [g_optim, d_optim], []
        # return super().configure_optimizers()
