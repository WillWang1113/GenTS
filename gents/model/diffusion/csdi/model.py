import torch

from gents.model.base import BaseModel
from ._backbones import CSDI_Forecasting, CSDI_Physio


class CSDI(BaseModel):
    """`CSDI <https://arxiv.org/pdf/2107.03502>`__: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation

    Adapted from the `official codes <https://github.com/ermongroup/CSDI>`__

    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to 'impute'.
        n_diff_steps (int, optional): Total diffusion steps. Defaults to 50.
        n_layers (int, optional): Residual block layers. Defaults to 4.
        d_model (int, optional): Model size. Defaults to 64.
        nheads (int, optional): Attention heads. Defaults to 8.
        diffusion_embedding_dim (int, optional): Embedding dim of diffusion steps. Defaults to 128.
        schedule (str, optional): Diffusion noise schedule. Choose from `['quad', 'linear']` Defaults to "quad".
        beta_start (float, optional): First step noise schedule. Defaults to 0.0001.
        beta_end (float, optional): Last step noise schedule. Defaults to 0.5.
        timeemb (int, optional): Embedding dim for time steps of time series. Defaults to 128.
        featureemb (int, optional): Embedding dim for sequence dimension of time series. Defaults to 16.
        target_strategy (str, optional): Missing data strategy used for simulating training. Choose from `['random', 'mix']` Defaults to "random".
        num_sample_features (int, optional): The number of time series dimensions randomly sampled for training. If greater than `seq_dim`, then all channels are used. Defaults to 64.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-6.
    """

    ALLOW_CONDITION = ["predict", "impute"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = "impute",
        n_diff_steps: int = 50,
        n_layers: int = 4,
        d_model: int = 64,
        nheads: int = 8,
        diffusion_embedding_dim: int = 128,
        schedule: str = "quad",
        beta_start: float = 0.0001,
        beta_end: float = 0.5,
        timeemb: int = 128,
        featureemb: int = 16,
        target_strategy: str = "random",
        num_sample_features: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
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
                num_sample_features=num_sample_features,
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
            observed_mask = batch["data_mask"]

            # observed_mask = ~torch.isnan(total_seq)
            target_mask = torch.isnan(batch["c"])
            target_mask = ~target_mask
            # print(target_mask[0,:,0])
            train_batch = dict(
                observed_data=total_seq.masked_fill(~observed_mask, 0.0),
                # observed_data=torch.nan_to_num(batch['c']),
                observed_mask=observed_mask,
                gt_mask=target_mask,
                timepoints=timepoints,
            )
        return train_batch

    def training_step(self, batch, batch_idx):
        train_batch = self._rebuild_batch(batch)
        loss = self.model(train_batch)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        train_batch = self._rebuild_batch(batch)
        loss = self.model(train_batch, is_train=0)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
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
                    torch.zeros((condition.shape[0], self.seq_len, self.seq_dim)).to(
                        self.device
                    ),
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
                "data_mask", torch.ones(seq_shape).to(self.device)
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

        # test_batch = self._rebuild_batch(kwargs)
        samples, observed_data, target_mask, observed_mask, observed_tp = (
            self.model.evaluate(test_batch, n_samples=n_sample)
        )
        return samples[:, -self.hparams.seq_len :]
