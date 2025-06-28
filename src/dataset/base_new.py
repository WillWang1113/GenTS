from calendar import c
import copy
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torchcde
from einops import repeat
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_url, download_and_extract_archive


class BaseDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str,
        batch_size: int,
        inference_batch_size: int,
        max_time: float = None,
        add_coeffs: str = None,
        irregular_dropout: float = 0.0,
        data_dir: Path | str = Path.cwd() / "data",
        **kwargs,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.max_time = max_time
        self.add_coeffs = add_coeffs
        self.condition = condition
        self.obs_len = None
        self.missing_rate = None
        self.missing_kind = None
        self.batch_size = batch_size
        self.infer_bs = inference_batch_size
        self.root_dir = data_dir
        self.data_dir = os.path.join(data_dir, self.dataset_name)
        self.irregular_dropout = irregular_dropout
        self.kwargs = kwargs

        assert condition in [None, "predict", "impute", "class"]
        if condition == "predict":
            assert kwargs.get("obs_len", None) is not None
            self.obs_len = kwargs.get("obs_len")
        elif condition == "impute":
            assert kwargs.get("missing_rate", None) is not None
            self.missing_rate = kwargs.get("missing_rate")
            self.missing_type = kwargs.get("missing_type")
        self.total_seq_len = (
            seq_len + self.obs_len if self.obs_len is not None else seq_len
        )

    def setup(self, stage):
        # tsl = total seq len
        # tsd = total seq dim
        (data, data_mask, class_label) = torch.load(
            os.path.join(
                self.data_dir,
                f"data_tsl{self.total_seq_len}_tsd{self.seq_dim}_ir{self.irregular_dropout}.pt",
            )
        )

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
            train_data_mask = data_mask[starts["fit"] : ends["fit"]]
            val_data_mask = data_mask[starts["validate"] : ends["validate"]]
            if class_label is not None:
                train_class_label = class_label[starts["fit"] : ends["fit"]]
                val_class_label = class_label[starts["validate"] : ends["validate"]]
            else:
                train_class_label, val_class_label = None, None
            self.train_ds = TSDataset(
                train_data,
                train_data_mask,
                train_class_label,
                condition=self.condition,
                max_time=self.max_time,
                add_coeffs=self.add_coeffs,
                **self.kwargs,
            )
            self.val_ds = TSDataset(
                val_data,
                val_data_mask,
                val_class_label,
                condition=self.condition,
                max_time=self.max_time,
                add_coeffs=self.add_coeffs,
                **self.kwargs,
            )

        if stage == "test":
            test_data = data[starts["test"] : ends["test"]]
            test_data_mask = data_mask[starts["test"] : ends["test"]]
            if class_label is not None:
                test_class_label = class_label[starts["test"] : ends["test"]]
            else:
                test_class_label = None
            self.test_ds = TSDataset(
                test_data,
                test_data_mask,
                test_class_label,
                condition=self.condition,
                max_time=self.max_time,
                add_coeffs=self.add_coeffs,
                **self.kwargs,
            )

    def prepare_data(self) -> None:
        data_file_pth = os.path.join(
            self.data_dir,
            f"data_tsl{self.total_seq_len}_tsd{self.seq_dim}_ir{self.irregular_dropout}.pt",
        )

        exist_data = os.path.exists(data_file_pth)

        if not exist_data:
            logging.info(f"Downloading {self.dataset_name} dataset in {self.data_dir}.")
            os.makedirs(self.data_dir, exist_ok=True)

            ###############################################
            # data_tuple:
            # 'data': slided time series sequences  (N, Total SL, C), if originally have NaNs, set to zeros
            # 'data_mask': observance mask in boolean (N, Total SL, C) if original dataset has NaN,
            # 'class_label': (optional) multi-class time series int (N, 1), e.g. electricity theft, patients, etc.
            (data, data_mask, class_label) = self.get_data()
            assert not data.isnan().any()
            # if data.isnan().any():
            #     logging.warning(
            #         "Data contains NaN values. They will be replaced with zeros."
            #     )
            #     # data = data.nan_to_num(0.0)
            torch.save((data, data_mask, class_label), data_file_pth)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.infer_bs)

    @abstractmethod
    def get_data(self) -> Tuple[torch.Tensor]: ...

    @property
    @abstractmethod
    def dataset_name(self) -> str: ...


class TSDataset(Dataset):
    """Time series dataset. A batch conatins:

    A batch
        "seq": (B, T, C) Target time series window
        "t": (B, T) Time index at each time step in the window. default [0,1,...,T-1]
        "c": (B, T/OBS, C) Condition. Empty if unconditional.
        "coeffs": (B, T, C) Coefficients of cubic spline of NCDE-related models.
        Empty if add_coeffs is False.
        "chnl_id": (1,) channel id if channel_independent is True
    """

    def __init__(
        self,
        data: torch.Tensor,
        data_mask: torch.BoolTensor = None,
        class_label: torch.IntTensor = None,
        condition: str = None,
        max_time: int = None,
        add_coeffs: str = None,
        **kwargs,
    ):
        super().__init__()
        assert data.dim() == 3
        assert add_coeffs in [None, "linear", "cubic_spline"]
        assert condition in [None, "predict", "impute", "class"]
        self.data = data
        self.class_labels = class_label
        # data_mask indicates original missing values
        # if data_mask=None, assume no missing values (1=observed, 0=missing)
        # if data_mask is not None, NaNs in the original data are set as 0
        self.data_mask = (
            torch.ones_like(data).bool() if data_mask is None else data_mask
        )

        self.data_shape = data.shape
        total_seq_len = data.shape[1]

        ############# set time index #############
        if max_time is None:
            time_idx = torch.arange(total_seq_len).float()
        else:
            time_idx = torch.arange(total_seq_len).float() / total_seq_len * max_time
        self.time_idx = repeat(time_idx, "t -> b t", b=data.shape[0])

        ############# set conditions #############
        if condition == "predict":
            # basic check
            obs_len = kwargs.get("obs_len")
            if obs_len is None:
                raise ValueError("obs_len should be provided for prediction.")
            elif obs_len < 0:
                raise ValueError("obs_len should be greater than 0.")
            # condition data is look-back window
            cond_data = data[:, :obs_len]

        # # time step missing
        # elif condition == "impute":
        #     # basic check
        #     missing_rate = kwargs.get("missing_rate")
        #     if missing_rate is None:
        #         raise ValueError("missing_rate > 1 should be provided for prediction.")
        #     elif missing_rate > 1 or missing_rate < 0:
        #         raise ValueError("missing_rate > 1 should be in (0, 1).")

        #     missing_tp = torch.rand_like(self.time_idx[0]) < missing_rate

        #     # 0: missing, 1: observed
        #     masks = self.data_mask.clone()
        #     masks[:, missing_tp] = False

        #     # condition_mask = masks.reshape(self.data_mask.shape)

        #     # condition data is partially oberved data
        #     # (allows original data contains missing values)
        #     # will have a target mask for loss computation
        #     cond_data = data.clone()
        #     cond_data[~masks] = torch.nan

        # random missing at all timesetps and all channels
        elif condition == "impute":
            # basic check
            missing_rate = kwargs.get("missing_rate")
            if missing_rate is None:
                raise ValueError("missing_rate > 1 should be provided for prediction.")
            elif missing_rate > 1 or missing_rate < 0:
                raise ValueError("missing_rate > 1 should be in (0, 1).")

            # 0: missing, 1: observed
            masks = self.data_mask.reshape(-1).clone()

            # manually set missing values from observed data
            obs_indices = torch.where(masks)[0].tolist()
            miss_indices = torch.randperm(len(obs_indices))[
                : (int)(len(obs_indices) * missing_rate)
            ].tolist()
            masks[miss_indices] = False
            condition_mask = masks.reshape(self.data_mask.shape)

            # condition data is partially oberved data
            # (allows original data contains missing values)
            # will have a target mask for loss computation
            cond_data = data.clone()
            cond_data[~condition_mask] = torch.nan
        elif condition == "class":
            # condition data is class label
            cond_data = class_label.long()
        else:
            cond_data = None
        self.cond_data = cond_data

        ############# add coeffs (optional) #############
        # For KoVAE, GTGAN, and SDEGAN
        if add_coeffs is not None:
            if add_coeffs == "linear":
                interp_fn = torchcde.linear_interpolation_coeffs
            else:
                interp_fn = torchcde.natural_cubic_spline_coeffs

            t = torch.arange(data.shape[1]).float()

            if data_mask is None:
                data_nan = data
            else:
                data_nan = data.clone()
                data_nan[~data_mask] = torch.nan

            coeffs = interp_fn(data_nan, t)
        else:
            coeffs = None
        self.coeffs = coeffs

    def __getitem__(self, index):
        batch_dict = dict(
            seq=self.data[index],
            t=self.time_idx[index],
            data_mask=self.data_mask[index],
        )
        if self.cond_data is not None:
            batch_dict["c"] = self.cond_data[index]
        if self.coeffs is not None:
            batch_dict["coeffs"] = self.coeffs[index]

        return batch_dict

    def __len__(self):
        return len(self.data)


class WebDownloadDataModule(BaseDataModule):
    D: int = ...
    index_col: str | int = ...
    url: str = ...
    csv_dir: str = ...
    data_source: str = ...

    def __init__(
        self,
        seq_len: int,
        select_seq_dim: List[int | str] = None,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        scale: bool = True,
        inference_batch_size: int = 1024,
        max_time: float = None,
        add_coeffs: str = None,
        irregular_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            seq_len,
            len(select_seq_dim) if isinstance(select_seq_dim, list) else self.D,
            condition,
            batch_size,
            inference_batch_size,
            max_time,
            add_coeffs,
            irregular_dropout,
            data_dir,
            **kwargs,
        )
        self.scale = scale
        self.select_seq_dim = select_seq_dim
        if select_seq_dim is not None:
            assert max(select_seq_dim) < self.D

    def get_data(self):
        if self.data_source == "zip":
            download_and_extract_archive(
                self.url, self.data_dir, self.data_dir, "archive.zip"
            )
            csv_pth = os.path.join(self.data_dir, self.csv_dir)
        else:
            download_url(self.url, self.data_dir, filename="archive.csv")
            csv_pth = os.path.join(self.data_dir, "archive.csv")
        df = pd.read_csv(csv_pth, index_col=self.index_col)

        # select dimensions
        if self.select_seq_dim is not None:
            if isinstance(self.select_seq_dim[0], str):
                df = df[self.select_seq_dim]
            elif isinstance(self.select_seq_dim[0], int):
                df = df.iloc[:, self.select_seq_dim]

        data_raw = df.values.astype(np.float32)
        n_window = data_raw.shape[0] - self.total_seq_len + 1
        n_trainval_window = int(n_window * 0.7) + int(n_window * 0.2)
        n_trainval_timesteps = n_trainval_window + self.total_seq_len - 1

        # scale
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data_raw[:n_trainval_timesteps])
            data_raw = self.scaler.transform(data_raw)
        data_samples = torch.from_numpy(copy.deepcopy(data_raw))
        data_original = torch.from_numpy(copy.deepcopy(data_raw))

        if self.irregular_dropout > 0:
            generator = torch.Generator().manual_seed(42)
            removed_points = (
                torch.randperm(data_samples.shape[0], generator=generator)[
                    : int(data_samples.shape[0] * self.irregular_dropout)
                ]
                .sort()
                .values
            )
            data_samples[removed_points] = float("nan")

        # slide window
        data_orignial_window, data_samples_window = [], []
        for i in range(len(data_original) - self.total_seq_len + 1):
            data_orignial_window.append(data_original[i : i + self.total_seq_len, :])
            data_samples_window.append(data_samples[i : i + self.total_seq_len, :])
        data_samples_window = torch.stack(data_samples_window, dim=0).float()

        # data: (N, Total SL, C), raw data WITHOUT NaNs
        # data_mask: (N, Total SL, C), boolean mask of raw data, 0=missing, 1=observed
        # class_label: (N,), class label if available, None otherwise
        data = torch.stack(data_orignial_window, dim=0).float()

        data_mask = ~torch.isnan(data_samples_window)
        class_label = None

        return data, data_mask, class_label
