from typing import List

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP

from src.layers.norm import BatchNormFlow
from src.layers.flow import MADE, Reverse, FlowSequential
from src.layers.mlp import MLPEncoder, MLPDecoder

# from src.layers.conv import ConvDecoder, ConvEncoder
from ..base import BaseModel

import math
import types


class VanillaMAF(BaseModel):
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
        super().__init__()
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
        x = batch["seq"].flatten(1)
        c = batch.get("c", None)
        if c is not None:
            c = c.flatten(1)
        loss = -self.flow.log_probs(x, c).mean()
        loss_dict = {"train_loss": loss}
        self.log_dict(
            loss_dict, on_step=True, on_epoch=False, logger=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["seq"].flatten(1)
        c = batch.get("c", None)
        if c is not None:
            c = c.flatten(1)
        loss = -self.flow.log_probs(x, c).mean()
        loss_dict = {"val_loss": loss}
        self.log_dict(
            loss_dict, on_step=True, on_epoch=False, logger=True, prog_bar=True
        )
        return loss

    def _sample_impl(self, n_sample, c=None):
        if c is not None:
            c = c.flatten(1)
        return self.flow.sample(n_sample, c).reshape(n_sample, -1, self.hparams.seq_dim)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams_initial.lr,
            weight_decay=self.hparams_initial.weight_decay,
        )
