from typing import List

import torch
from torch import nn

from ._backbones import MADE, Reverse, FlowSequential, BatchNormFlow

# from src.layers.conv import ConvDecoder, ConvEncoder
from src.model.base import BaseModel


class VanillaMAF(BaseModel):
    ALLOW_CONDITION = [None, 'predict', 'impute']
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int = 128,
        hidden_size_list: List[int] = [64, 128, 256],
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        condition: str = None,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition)
        self.save_hyperparameters()
        modules = []
        if condition:
            if condition == "predict":
                assert kwargs.get("obs_len") is not None
                obs_len = kwargs.get("obs_len")
                cond_input = obs_len * seq_dim
            elif condition == "impute":
                cond_input = seq_dim * seq_len
        else:
            cond_input = None
        for i in range(len(hidden_size_list)):
            modules += [
                MADE(seq_dim * seq_len, hidden_size_list[i], cond_input),
                BatchNormFlow(seq_dim * seq_len),
                Reverse(seq_dim * seq_len),
            ]
        self.flow = FlowSequential(*modules)
        for module in self.flow.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.fill_(0)

    def training_step(self, batch, batch_idx):
        x = batch["seq"]
        c = batch.get("c", None)
        if c is not None:
            if self.condition == "impute":
                c = x * (~c).int()
            elif self.condition == 'predict':
                x = x[:,-self.hparams_initial.seq_len:]
            c = c.flatten(1).to(x)
            
        x = x.flatten(1)
        
        loss = -self.flow.log_probs(x, c).mean()
        loss_dict = {"train_loss": loss}
        self.log_dict(
            loss_dict, on_step=True, on_epoch=False, logger=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["seq"]
        c = batch.get("c", None)
        if c is not None:
            if self.condition == "impute":
                c = x * (~c).int()
            elif self.condition == 'predict':
                x = x[:,-self.hparams_initial.seq_len:]
            c = c.flatten(1).to(x)
        x = x.flatten(1)
        
        loss = -self.flow.log_probs(x, c).mean()
        loss_dict = {"val_loss": loss}
        self.log_dict(
            loss_dict, on_step=True, on_epoch=False, logger=True, prog_bar=True
        )
        return loss

    def _sample_impl(self, n_sample, condition=None, **kwargs):
        if self.condition is None:
            all_samples = self.flow.sample(n_sample, None).reshape(
                n_sample, self.hparams.seq_len, self.hparams.seq_dim
            )
        else:
            all_samples = []
            if self.condition == "impute":
                assert kwargs.get("seq", None) is not None, "provide full sequence"
                c = kwargs["seq"] * (~condition).int()
            else:
                c = condition
            
            c = c.flatten(1)

            for i in range(n_sample):
                x_hat = self.flow.sample(c.shape[0], c).reshape(
                    c.shape[0], self.hparams.seq_len, self.hparams.seq_dim
                )
                all_samples.append(x_hat)
            all_samples = torch.stack(all_samples, dim=-1)

        return all_samples

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams_initial.lr,
            weight_decay=self.hparams_initial.weight_decay,
        )
