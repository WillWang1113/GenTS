from typing import List

import torch
from torch import nn

from ._backbones import MADE, Reverse, FlowSequential, BatchNormFlow

# from src.layers.conv import ConvDecoder, ConvEncoder
from src.model.base import BaseModel


class VanillaMAF(BaseModel):
    """Vanilla Masked Autoregressive Flow with MADE."""

    ALLOW_CONDITION = [None, "predict", "impute", "class"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        hidden_size_list: List[int] = [64, 128, 256],
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        """_summary_

        Args:
            seq_len (int): Target sequence length
            seq_dim (int): Target sequence dimension, for univariate time series, set as 1
            condition (str, optional): Given conditions, allowing [None, 'predict', 'impute']. Defaults to None.
            hidden_size_list (List[int], optional): Hidden size for MADE. Defaults to [64, 128, 256].
            lr (float, optional): Learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        """
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        modules = []
        if condition == 'predict':
            cond_seq_len = self.obs_len
            cond_input = cond_seq_len * seq_dim
        elif condition == 'impute':
            cond_seq_len = seq_len
            cond_input = cond_seq_len * seq_dim
        elif condition == 'class':
            cond_input = self.class_num
        # if condition:
        #     # if condition == "predict":
        #     #     assert kwargs.get("obs_len") is not None
        #     #     obs_len = kwargs.get("obs_len")
        #     #     cond_input = obs_len * seq_dim
        #     # elif condition == "impute":
        #     #     cond_input = seq_dim * seq_len
        #     cond_seq_len = self.obs_len if condition == "predict" else seq_len
        #     cond_input = cond_seq_len * seq_dim
        # else:
        #     cond_input = None
        for i in range(len(hidden_size_list)):
            modules += [
                MADE(seq_dim * seq_len, hidden_size_list[i], cond_input, condition),
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
        c = batch.get("c")
        if c is not None:
            if self.condition == "impute":
                c = torch.nan_to_num(c).flatten(1).to(x)
            elif self.condition == "predict":
                x = x[:, -self.hparams_initial.seq_len :].flatten(1)
            

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
                c = torch.nan_to_num(c).flatten(1).to(x)
            elif self.condition == "predict":
                x = x[:, -self.hparams_initial.seq_len :].flatten(1)
            
            # c = c.flatten(1).to(x)
        x = x.flatten(1)

        loss = -self.flow.log_probs(x, c).mean()
        loss_dict = {"val_loss": loss}
        self.log_dict(
            loss_dict, on_step=True, on_epoch=False, logger=True, prog_bar=True
        )
        return loss

    def _sample_impl(self, n_sample, condition=None, **kwargs):
        if self.condition is None or self.condition == 'class':
            all_samples = self.flow.sample(n_sample, None).reshape(
                n_sample, self.hparams.seq_len, self.hparams.seq_dim
            )
        else:
            all_samples = []
            # if self.condition == "impute":
            #     assert kwargs.get("seq", None) is not None, "provide full sequence"
            #     c = kwargs["seq"] * (~condition).int()
            # else:
            #     c = condition
            c = torch.nan_to_num(condition) if self.condition == "impute" else condition
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
