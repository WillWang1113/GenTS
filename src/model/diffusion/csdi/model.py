import numpy as np
import torch
import torch.nn as nn

from src.model.base import BaseModel
from ._backbones import CSDI_Forecasting, CSDI_Physio


class CSDI(BaseModel):
    ALLOW_CONDITION = ["predict", "impute"]

    def __init__(
        self,
        seq_len,
        seq_dim,
        condition="impute",
        n_diff_steps=50,
        n_layers=4,
        d_model=64,
        nheads=8,
        diffusion_embedding_dim=128,
        schedule="quad",
        beta_start=0.0001,
        beta_end=0.5,
        timeemb=128,
        featureemb=16,
        target_strategy="random",
        lr=1e-3,
        weight_decay=1e-6,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        config = dict(
            model=dict(
                is_unconditional=0,
                timeemb=timeemb,
                featureemb=featureemb,
                target_strategy=target_strategy,
                num_sample_features=64,
            ),
            diffusion=dict(
                layers=n_layers,
                channels=d_model,
                nheads=nheads,
                diffusion_embedding_dim=diffusion_embedding_dim,
                beta_start=beta_start,
                beta_end=beta_end,
                num_steps=n_diff_steps,
                schedule=schedule,
                is_linear=False,
            ),
        )

        if condition == "impute":
            assert config["model"]["target_strategy"] in ["random", "mix", "hist"]
            self.model = CSDI_Physio(config, seq_dim)
        else:
            config["model"]["target_strategy"] = "test"
            self.model = CSDI_Forecasting(config, seq_dim)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _rebuild_batch(self, batch):
        total_seq = batch["seq"]
        timepoints = batch["t"]
        batch_size = total_seq.shape[0]
        if self.condition == "predict":
            observed_mask = torch.ones_like(total_seq)
            target_mask = torch.ones_like(total_seq)
            target_mask[:, -self.hparams.seq_len :] = 0.0
            train_batch = dict(
                observed_data=total_seq,
                observed_mask=observed_mask,
                gt_mask=target_mask,
                timepoints=timepoints,
                feature_id=torch.arange(total_seq.shape[-1])
                .unsqueeze(0)
                .expand(batch_size, -1)
                .to(total_seq.device)
                * 1.0,
            )
        else:
            observed_mask = ~torch.isnan(total_seq)
            target_mask = torch.isnan(batch["c"])
            target_mask = ~target_mask
            # print(target_mask[0,:,0])
            train_batch = dict(
                observed_data=torch.nan_to_num(batch['c']),
                observed_mask=observed_mask,
                gt_mask=target_mask,
                timepoints=timepoints,
            )
        return train_batch

    def training_step(self, batch, batch_idx):
        train_batch = self._rebuild_batch(batch)
        loss = self.model(train_batch)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )
        return loss

    def val_dataloader(self, batch, batch_idx):
        train_batch = self._rebuild_batch(batch)
        loss = self.model(train_batch, is_train=0)
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )
        return loss

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        batch_size = condition.shape[0]
        timepoints = kwargs.get("t")
        if timepoints is None:
            timepoints = torch.arange(self.seq_len).to(self.device)

        if self.condition == "predict":
            seq_shape = (condition.shape[0], self.obs_len + self.seq_len, self.seq_dim)
            observed_data = torch.concat(
                [
                    condition,
                    torch.zeros((condition.shape[0], self.seq_len, self.seq_dim)).to(self.device),
                ],
                dim=1,
            )
            observed_mask = torch.ones(seq_shape).to(self.device)
            target_mask = torch.ones(seq_shape).to(self.device)
            target_mask[:, -self.seq_len :] = 0.0
            test_batch = dict(
                observed_data=observed_data,
                observed_mask=observed_mask,
                gt_mask=target_mask,
                timepoints=timepoints.to(self.device),
                feature_id=torch.arange(self.seq_dim)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .to(self.device)
                * 1.0,
            )
        else:
            seq_shape = (condition.shape[0], self.seq_len, self.seq_dim)

            observed_mask = kwargs.get(
                "observed_mask", torch.ones(seq_shape).to(self.device)
            )
            target_mask = torch.isnan(condition)
            target_mask = ~target_mask
            # print(target_mask[0,:,0])
            test_batch = dict(
                observed_data=torch.nan_to_num(condition),
                observed_mask=observed_mask,
                gt_mask=target_mask,
                timepoints=timepoints,
            )

        test_batch = self._rebuild_batch(kwargs)
        samples, observed_data, target_mask, observed_mask, observed_tp = (
            self.model.evaluate(test_batch, n_samples=n_sample)
        )
        return samples[:, -self.hparams.seq_len :]
