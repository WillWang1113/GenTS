from argparse import Namespace
import logging
from gents.model.base import BaseModel
from ._utils import CustomDecayLR
from ._create_latent_ode_model import create_LatentODE_model
import torch


class LatentODE(BaseModel):
    """`Latent ODEs <https://arxiv.org/abs/1907.03907>`__ for Irregularly-Sampled Time Series

    Adapted from the `official codes <https://github.com/YuliaRubanova/latent_ode>`__

    .. note::
        LatentODE allows for irregular data. Imputation in Latent ODE is only interpolation, 
        i.e. only impute the missing steps that all channels are missing.

    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str, optional): Given condition type, should be one of `ALLOW_CONDITION`. Defaults to None.
        latent_dim (int, optional): Latent variable dimension. Defaults to 6.
        z0_encoder (str, optional): Type of encoder. Choose from `['odernn','rnn]` Defaults to "odernn".
        rec_layers (int, optional): Number of encoding (recognition) layers. Defaults to 1.
        rec_dim (int, optional): Number of encoding (recognition) layer size. Defaults to 20.
        gen_layers (int, optional): Number of generation layers. Defaults to 1.
        d_model (int, optional): Model size. Defaults to 32.
        obsrv_std (float, optional): Std of Gaussian distribution for observed data. Used for computing likelihood loss. Defaults to 0.01.
        poisson (bool, optional): Whether to use poisson distribution. Defaults to False.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        **kwargs: Arbitrary keyword arguments, e.g. obs_len, class_num, etc.

    """

    ALLOW_CONDITION = [None, "impute", "predict"]

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str = None,
        latent_dim: int = 6,
        z0_encoder: str = "odernn",
        rec_layers: int = 1,
        rec_dim: int = 20,
        gen_layers: int = 1,
        d_model: int = 32,
        obsrv_std: float = 0.01,
        poisson: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        if self.condition == "impute":
            logging.warning(
                "LatentODE only support on interpolation, a sub-kind of imputation"
            )
        self.save_hyperparameters()
        assert z0_encoder in ["odernn", "rnn"]
        # device = kwargs.get("device", self.device)
        self.seq_len = seq_len

        args = Namespace(
            latents=latent_dim,
            units=d_model,
            gru_units=d_model,
            rec_dims=rec_dim,
            **self.hparams_initial,
        )
        self.kl_coef = 1e-6

        self.model = create_LatentODE_model(args, seq_dim, obsrv_std)

    def training_step(self, batch, batch_idx):
        batch_dict = self._reform_batch(batch)
        loss_dict = self.model.compute_all_losses(
            batch_dict, n_traj_samples=3, kl_coef=self.kl_coef
        )
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        batch_dict = self._reform_batch(batch)
        loss_dict = self.model.compute_all_losses(
            batch_dict, n_traj_samples=3, kl_coef=self.kl_coef
        )
        loss_dict = {f'val_{k}':v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True)

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        total_seq_len = self.seq_len
        if self.condition == "predict":
            total_seq_len += self.obs_len
        # t = torch.linspace(
        #     20.0 / total_seq_len,
        #     20.0,
        #     total_seq_len,
        # ).to(self.device)
        t = kwargs["t"][0]

        if self.condition == "predict":
            total_seq_len += self.obs_len
            tp_to_predict = t[self.obs_len :]
            observed_tp = t[: self.obs_len]
            observed_data = condition
            observed_mask = kwargs.get("data_mask")[:, : self.obs_len]

            trajs, _ = self.model.get_reconstruction(
                tp_to_predict, observed_data, observed_tp, observed_mask, n_sample
            )
            trajs = trajs.permute(1, 2, 3, 0)

        elif self.condition == "impute":
            data = torch.nan_to_num(condition)
            mask = 1 - torch.isnan(condition).float()

            tp_to_predict = t
            observed_tp = t
            observed_data = data
            observed_mask = mask

            trajs, _ = self.model.get_reconstruction(
                tp_to_predict, observed_data, observed_tp, observed_mask, n_sample
            )
            trajs = trajs.permute(1, 2, 3, 0)
        else:
            trajs = self.model.sample_traj_from_prior(t, n_sample).squeeze(1)

        return trajs

    def _reform_batch(self, batch):
        x = batch["seq"]
        # t = torch.linspace(20.0 / x.shape[1], 20.0, x.shape[1]).to(x)
        t = batch["t"][0]
        if self.condition == "predict":
            # mask = torch.ones_like(batch["c"])
            data_dict = {
                "tp_to_predict": t[self.obs_len :],
                "observed_tp": t[: self.obs_len],
                "observed_data": batch["c"],
                "data_to_predict": batch["seq"][:, self.obs_len :, :],
                # observed_mask = mask over observed_data,
                # 1 is observed, 0 is not
                "observed_mask": batch["data_mask"][:, : self.obs_len, :].float(),
                # mask_predicted_data = mask over data_to_predict (for loss computing),
                # 1 is observed, 0 is not
                "mask_predicted_data": batch["data_mask"][:, self.obs_len :, :].float(),
                "labels": None,
                "mode": "extrap",
            }

        else:
            # mask = torch.ones_like(x)
            data_dict = {
                "tp_to_predict": t,
                "observed_tp": t,
                "observed_data": x.masked_fill(~batch["data_mask"], 0.0),
                "data_to_predict": x.masked_fill(~batch["data_mask"], 0.0),
                "observed_mask": batch["data_mask"].float(),
                "mask_predicted_data": batch["data_mask"].float(),
                "labels": None,
                "mode": "interp",
            }

        ## get batch data
        batch_dict = {
            "observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "observed_mask": None,
            "mask_predicted_data": None,
            "labels": None,
        }

        # remove the time points where there are no observations in this batch
        non_missing_tp = torch.sum(data_dict["observed_mask"], (0, 2)) != 0.0
        batch_dict["observed_data"] = data_dict["observed_data"][:, non_missing_tp]
        batch_dict["observed_tp"] = data_dict["observed_tp"][non_missing_tp]
        # print("observed data")
        # print(batch_dict["observed_data"].size())

        if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
            batch_dict["observed_mask"] = data_dict["observed_mask"][:, non_missing_tp]

        batch_dict["data_to_predict"] = data_dict["data_to_predict"]
        batch_dict["tp_to_predict"] = data_dict["tp_to_predict"]

        non_missing_tp = torch.sum(data_dict["mask_predicted_data"], (0, 2)) != 0.0
        batch_dict["data_to_predict"] = data_dict["data_to_predict"][:, non_missing_tp]
        batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][non_missing_tp]

        # print("data_to_predict")
        # print(batch_dict["data_to_predict"].size())

        if ("mask_predicted_data" in data_dict) and (
            data_dict["mask_predicted_data"] is not None
        ):
            batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"][
                :, non_missing_tp
            ]

        if ("labels" in data_dict) and (data_dict["labels"] is not None):
            batch_dict["labels"] = data_dict["labels"]

        batch_dict["mode"] = data_dict["mode"]

        return batch_dict

    def configure_optimizers(self):
        optim = torch.optim.Adamax(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = CustomDecayLR(optim)
        return {"optimizer": optim, "lr_scheduler": lr_scheduler}

    def on_train_batch_start(self, batch, batch_idx):
        wait_until_kl_inc = 10
        total_step = (
            self.trainer.num_training_batches * self.trainer.max_epochs
            if self.trainer.max_epochs > 0
            else self.trainer.max_steps
        )
        if (self.global_step // total_step) < wait_until_kl_inc:
            self.kl_coef = 1e-6
        else:
            self.kl_coef = 1 - 0.99 ** (
                batch_idx // self.trainer.num_training_batches - wait_until_kl_inc
            )
