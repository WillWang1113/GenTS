from argparse import Namespace
from src.model.base import BaseModel
from ._utils import CustomDecayLR
from ._create_latent_ode_model import create_LatentODE_model
import torch


class LatentODE(BaseModel):
    ALLOW_CONDITION = [None, "impute", "predict"]

    def __init__(
        self,
        seq_len,
        seq_dim,
        latent_dim=6,
        z0_encoder="rnn",
        rec_layers=1,
        rec_dim=20,
        gen_layers=1,
        d_model=32,
        obsrv_std=0.01,
        poisson=False,
        condition=None,
        lr=1e-3,
        **kwargs,
    ):
        super().__init__(seq_len, seq_dim, condition, **kwargs)
        self.save_hyperparameters()
        # device = kwargs.get("device", self.device)
        self.seq_len = seq_len

        if condition == "predict":
            self.obs_len = kwargs.get("obs_len", None)
            assert self.obs_len is not None
            assert self.obs_len > 0

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
        self.log_dict(loss_dict)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        batch_dict = self._reform_batch(batch)
        loss_dict = self.model.compute_all_losses(
            batch_dict, n_traj_samples=3, kl_coef=self.kl_coef
        )
        self.log_dict(loss_dict)

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        total_seq_len = self.seq_len
        if self.condition == "predict":
            total_seq_len += self.obs_len
        t = torch.linspace(
            20.0 / total_seq_len,
            20.0,
            total_seq_len,
        ).to(self.device)

        if self.condition == "predict":
            total_seq_len += self.obs_len
            tp_to_predict = t[self.obs_len :]
            observed_tp = t[: self.obs_len]
            observed_data = condition
            observed_mask = torch.ones_like(observed_data)

            trajs, _ = self.model.get_reconstruction(
                tp_to_predict, observed_data, observed_tp, observed_mask, n_sample
            )
            trajs = trajs.permute(1, 2, 3, 0)
        # ! condition for impute should be data, not masks
        elif self.condition == "impute":
            data = kwargs.get("seq").clone()
            data[condition] = 0.0
            mask = 1 - condition.float()

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
        # ! force override !
        t = torch.linspace(20.0 / x.shape[1], 20.0, x.shape[1]).to(x)
        if self.condition == "predict":
            mask = torch.ones_like(batch["c"])
            batch_dict = {
                "tp_to_predict": t[self.obs_len :],
                "observed_tp": t[: self.obs_len],
                "observed_data": batch["c"],
                "data_to_predict": batch["seq"][:, self.obs_len :, :],
                "observed_mask": mask,
                "mask_predicted_data": None,
                "labels": None,
                "mode": "extrap",
            }
            # print(batch_dict['observed_data'].shape)
            # print(batch_dict['data_to_predict'].shape)
        elif self.condition == "impute":
            mask = batch["c"]
            # 1: missing
            # 0: non-missing
            data = x.clone()
            data[mask] = 0.0
            mask = 1 - mask.float()

            batch_dict = {
                "tp_to_predict": t,
                "observed_tp": t,
                "observed_data": data,
                "data_to_predict": data,
                "observed_mask": mask,
                "mask_predicted_data": None,
                "labels": None,
                "mode": "interp",
            }
        else:
            mask = torch.ones_like(x)
            batch_dict = {
                "tp_to_predict": t,
                "observed_tp": t,
                "observed_data": x,
                "data_to_predict": x,
                "observed_mask": mask,
                "mask_predicted_data": None,
                "labels": None,
                "mode": "interp",
            }
        return batch_dict

    def configure_optimizers(self):
        optim = torch.optim.Adamax(self.model.parameters(), lr=self.hparams.lr)
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
