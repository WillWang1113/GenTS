from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base import BaseModel
from ._backbones import ProDiscriminator, ProGenerator


class PSAGAN(BaseModel):
    ALLOW_CONDITION = [None, "predict"]

    def __init__(
        self,
        seq_len,
        seq_dim,
        time_feat_dim=0,
        hidden_size: int = 32,
        condition=None,
        depth_schedule: list = [5, 10, 15],
        epoch_fade_in: int = 2,
        lr: dict = {"G": 1e-4, "D": 1e-4},
        weight_decay: float = 1e-5,
        n_critic: int = 2,
        # gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        if condition == "predict":
            # context_len = kwargs.get("obs_len", None)
            # assert context_len is not None
            self.context_len = self.obs_len
        else:
            self.context_len = 0

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = ProGenerator(
            seq_len,
            seq_dim,
            time_feat_dim,
            hidden_size=hidden_size,
            context_length=self.context_len,
            **kwargs,
        )
        self.discriminator = ProDiscriminator(
            seq_len, seq_dim, time_feat_dim, hidden_size=hidden_size, **kwargs
        )
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.time_feat_dim = time_feat_dim
        self.depth_schedule = depth_schedule
        self.condition = condition
        self.depth = 0
        self.pretrain_schedule = []
        self.nb_epoch_fade_in_new_layer = epoch_fade_in
        for k in depth_schedule:
            self.pretrain_schedule.append((k, k + epoch_fade_in))
        self.nb_stage = len(depth_schedule) if depth_schedule else 0
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
        self._increase_depth()
        self.residual = self._residual()

    def training_step(self, batch, batch_idx):
        x = batch["seq"].permute(0, 2, 1)[..., -self.seq_len :]
        c = batch.get("c", None)
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
            return self.generator(x=noise, time_feat=time_feat, context=None).permute(
                0, 2, 1
            )
        else:
            # Forecasting
            all_samples = []
            for i in range(n_sample):
                noise = torch.randn(
                    (condition.shape[0], self.seq_dim, self.seq_len)
                ).to(self.device)
                sample = self.generator(
                    x=noise, time_feat=time_feat, context=condition
                ).permute(0, 2, 1)
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
