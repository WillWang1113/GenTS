import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torchcde
from einops import repeat
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        seq_len,
        seq_dim,
        condition,
        batch_size,
        inference_batch_size,
        data_dir: Path | str = Path.cwd() / "data",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.condition = condition
        self.obs_len = None
        self.missing_rate = None
        self.missing_kind = None
        self.batch_size = batch_size
        self.infer_bs = inference_batch_size
        self.data_dir = data_dir / self.dataset_name

        assert condition in [None, "predict", "impute", "class"]
        if condition == "predict":
            assert kwargs.get("obs_len", None) is not None
            self.obs_len = kwargs.get("obs_len")
        elif condition == "impute":
            assert kwargs.get("missing_rate", None) is not None
            assert kwargs.get("missing_type", None) is not None
            assert kwargs.get("missing_type", None) in ["random", "block"]
            self.missing_rate = kwargs.get("missing_rate")
            self.missing_type = kwargs.get("missing_type")
        self.total_seq_len = (
            seq_len + self.obs_len if self.obs_len is not None else seq_len
        )

    def setup(self, stage):
        # tsl = total seq len
        # tsd = total seq dim
        data = torch.load(
            os.path.join(
                self.data_dir, f"data_tsl{self.total_seq_len}_tsd{self.seq_dim}.pt"
            )
        )
        cond = None
        if self.condition is not None:
            cond = torch.load(
                os.path.join(
                    self.data_dir,
                    f"{self.condition}_cond_tsl{self.total_seq_len}_tsd{self.seq_dim}.pt",
                )
            )
        add_cond = cond is not None

        # train/val/test
        num_train = int(len(data) * 0.7)
        num_vali = int(len(data) * 0.2)
        num_test = len(data) - num_train - num_vali

        starts = dict(
            fit=0,
            validate=num_train,
            test=len(data) - num_test,
        )
        ends = dict(fit=num_train, validate=num_train + num_vali, test=len(data))

        if stage == "fit":
            train_data = data[starts["fit"] : ends["fit"]]
            val_data = data[starts["validate"] : ends["validate"]]
            train_cond, val_cond = None, None
            if add_cond:
                train_cond = cond[starts["fit"] : ends["fit"]]
                val_cond = cond[starts["validate"] : ends["validate"]]

            self.train_ds = TSDataset(
                train_data, train_cond, cond_type=self.condition, **self.hparams
            )
            self.val_ds = TSDataset(
                val_data, val_cond, cond_type=self.condition, **self.hparams
            )

        if stage == "test":
            test_data = data[starts["test"] : ends["test"]]
            test_cond = None
            if add_cond:
                test_cond = cond[starts["test"] : ends["test"]]
            self.test_ds = TSDataset(
                test_data, test_cond, cond_type=self.condition, **self.hparams
            )

    def prepare_data(self) -> None:
        data_file_pth = (
            self.data_dir / f"data_tsl{self.total_seq_len}_tsd{self.seq_dim}.pt"
        )
        cond_file_pth = (
            self.data_dir
            / f"{self.condition}_cond_tsl{self.total_seq_len}_tsd{self.seq_dim}.pt"
        )
        exist_data = data_file_pth.exists()
        exist_cond = cond_file_pth.exists()

        if (not exist_data) or (not exist_cond):
            logging.info(f"Downloading {self.dataset_name} dataset in {self.data_dir}.")
            os.makedirs(self.data_dir, exist_ok=True)

            # get data and save
            # data shape: [num_samples, total_seq_len, seq_dim (+1)],
            # +1: if time idx last dim
            # cond shape: [num_samples, ...]
            data, cond = self.get_data()
            torch.save(data, data_file_pth)
            if cond is not None:
                torch.save(
                    cond,
                    cond_file_pth,
                )

    def prepare_cond(self, data: torch.Tensor, class_cond: torch.Tensor = None):
        # Condition save
        if self.condition == "class":
            assert class_cond is not None
            cond = class_cond
        elif self.condition == "predict":
            cond = data[:, : self.obs_len, :]
        # TODO: giving condition=data with nan?
        elif self.condition == "impute":
            mask = torch.ones_like(data)
            # if self.missing_type == "random":
            # 1: missing
            # 0: non-missing
            mask = torch.rand_like(data) < self.missing_rate
            missing_data = data.masked_fill(mask.bool(), float("nan"))
            # # TODO: delete block missing?
            # elif self.missing_type == "block":
            #     delta = int(self.total_seq_len * self.missing_rate)
            #     rand_start = torch.randint(
            #         0, self.total_seq_len - delta, (data.shape[0],)
            #     )
            #     end = rand_start + delta
            #     t = torch.arange(self.total_seq_len).view(1, -1)  # shape: [1, 200]
            #     mask_2d = (t >= rand_start.view(-1, 1)) & (
            #         t <= end.view(-1, 1)
            #     )  # shape [1000, 200]
            #     mask = mask_2d.unsqueeze(-1).expand_as(data)
            #     mask = ~mask
            cond = missing_data
        else:
            cond = None
        return cond

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.infer_bs)

    @abstractmethod
    def get_data(self): ...

    @property
    @abstractmethod
    def dataset_name(self): ...


class TSDataset(Dataset):
    """Time series dataset. A batch conatins:

    A batch
        "seq": (B, T, C) Target time series window
        "t": (B, T) Time index at each time step in the window.
        Could be either the last axis of input data, or default [0,1,...,T-1]
        "c": (B, T/OBS, C) Condition. Empty if unconditional.
        "coeffs": (B, T, C) Coefficients of cubic spline of NCDE-related models.
        Empty if add_coeffs is False.
        "chnl_id": (1,) channel id if channel_independent is True
    """

    def __init__(
        self,
        data: torch.Tensor,
        cond: torch.Tensor = None,
        cond_type: str = None,
        add_coeffs: str = None,
        time_idx_last: bool = False,
        channel_independent: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert data.dim() == 3
        assert add_coeffs in [None, "linear", "cubic_spline"]
        assert cond_type in [None, "predict", "impute"]
        self.data = data
        self.data_shape = data.shape
        self.sample_chnl = False
        # if input TS is multivariate and treat channel independent, we need to sample a channel
        if (self.data.shape[-1] > 1) and channel_independent:
            self.sample_chnl = True

        if time_idx_last:
            self.data = self.data[:, :, :-1]
            self.time_idx = self.data[:, :, -1]
        else:
            self.time_idx = torch.arange(data.shape[1]).float()
            self.time_idx = repeat(self.time_idx, "t -> b t", b=data.shape[0])

        self.cond = cond
        self.cond_shape = None
        if cond is not None:
            self.cond_shape = tuple(cond.shape[1:])
        if add_coeffs is not None:
            if add_coeffs == "linear":
                interp_fn = torchcde.linear_interpolation_coeffs
            else:
                interp_fn = torchcde.natural_cubic_spline_coeffs

            # from torchcde import natural_cubic_spline_coeffs, linear_interpolation_coeffs

            t = torch.arange(data.shape[1]).float()

            if (cond_type == "impute") and (cond is not None):
                data_nan = cond
                # data_nan = data.masked_fill(cond.bool(), float("nan"))
            else:
                data_nan = data

            self.coeffs = interp_fn(data_nan, t)
        else:
            self.coeffs = None

    def __getitem__(self, index):
        if self.sample_chnl:
            chnl = torch.randint(0, self.data_shape[-1], (1,))
            batch_dict = dict(
                seq=self.data[index, :, chnl], t=self.time_idx[index], chnl_id=chnl
            )
        else:
            chnl = ...
            batch_dict = dict(seq=self.data[index, :, :], t=self.time_idx[index])

        if self.cond_shape is not None:
            batch_dict["c"] = self.cond[index, ..., chnl]
        if self.coeffs is not None:
            batch_dict["coeffs"] = self.coeffs[index, :, chnl]
        return batch_dict

    def __len__(self):
        return len(self.data)
