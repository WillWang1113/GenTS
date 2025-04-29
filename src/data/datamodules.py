from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from .dataset import TSDataset
import numpy as np


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
        data = torch.load(
            os.path.join(self.data_dir, f"data_sl{self.total_seq_len}.pt")
        )
        cond = None
        if self.condition is not None:
            cond = torch.load(
                os.path.join(
                    self.data_dir, f"{self.condition}_cond_sl{self.total_seq_len}.pt"
                )
            )
        add_cond = cond is not None

        # train/val/test
        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test

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

            self.train_ds = TSDataset(train_data, train_cond, **self.hparams)
            self.val_ds = TSDataset(val_data, val_cond, **self.hparams)

        if stage == "test":
            test_data = data[starts["test"] : ends["test"]]
            test_cond = None
            if add_cond:
                test_cond = cond[starts["test"] : ends["test"]]
            self.test_ds = TSDataset(test_data, test_cond, **self.hparams)

    def prepare_data(self) -> None:
        data_file_pth = self.data_dir / f"data_sl{self.total_seq_len}.pt"
        cond_file_pth = (
            self.data_dir / f"{self.condition}_cond_sl{self.total_seq_len}.pt"
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

    def prepare_cond(self, data, class_cond=None):
        # Condition save
        if self.condition == "class":
            assert class_cond is not None
            cond = class_cond
        elif self.condition == "predict":
            cond = data[:, : self.obs_len, :]
        elif self.condition == "impute":
            mask = torch.ones_like(data)
            if self.missing_type == "random":
                mask = torch.rand_like(data) < self.missing_rate
            # TODO: delete block missing?
            elif self.missing_type == "block":
                delta = int(self.total_seq_len * self.missing_rate)
                rand_start = torch.randint(
                    0, self.total_seq_len - delta, (self.num_samples,)
                )
                end = rand_start + delta
                t = torch.arange(self.total_seq_len).view(1, -1)  # shape: [1, 200]
                mask_2d = (t >= rand_start.view(-1, 1)) & (
                    t <= end.view(-1, 1)
                )  # shape [1000, 200]
                mask = mask_2d.unsqueeze(-1).expand_as(data)
                mask = ~mask
            cond = mask
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


class Spiral2D(BaseDataModule):
    def __init__(
        self,
        seq_len: int = 200,
        num_samples: int = 1000,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        scale: bool = True,
        inference_batch_size: int = 1024,
        add_coeffs: bool = False,
        time_idx_last: bool = False,
        channel_independent: bool = False,
        **kwargs,
    ):
        super().__init__(
            seq_len,
            2,
            condition,
            batch_size,
            inference_batch_size,
            data_dir,
            add_coeffs=add_coeffs,
            time_idx_last=time_idx_last,
            channel_independent=channel_independent,
            **kwargs,
        )
        self.num_samples = num_samples

    def get_data(self):
        t = torch.linspace(0, 4 * torch.pi, self.total_seq_len).float()
        curves = []
        labels = []
        for _ in range(self.num_samples):
            a = torch.rand(1).item() * 0.5  # Initial radius
            b = torch.rand(1).item() * 0.2  # Growth rate

            direction = torch.randint(0, 2, (1,)).item()  # 0=clockwise, 1=ccw

            r = a + b * t
            if direction == 0:
                x = r * torch.cos(t)
                y = r * torch.sin(t)
            else:
                x = -r * torch.cos(t)
                y = r * torch.sin(t)

            x += torch.randn_like(x) * 0.01
            y += torch.randn_like(y) * 0.01

            curve = torch.stack([x, y], dim=1)
            curves.append(curve)
            labels.append(direction)
        data, class_cond = torch.stack(curves), torch.tensor(labels).unsqueeze(-1)

        # Condition save
        cond = self.prepare_cond(data, class_cond)

        return data, cond

    @property
    def dataset_name(self) -> str:
        return "2DSpiral"


class SineND(BaseDataModule):
    def __init__(
        self,
        seq_len: int = 200,
        seq_dim: int = 1,
        num_samples: int = 1000,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        scale: bool = True,
        inference_batch_size: int = 1024,
        add_coeffs: bool = False,
        time_idx_last: bool = False,
        channel_independent: bool = False,
        **kwargs,
    ):
        super().__init__(
            seq_len,
            seq_dim,
            condition,
            batch_size,
            inference_batch_size,
            data_dir,
            add_coeffs=add_coeffs,
            time_idx_last=time_idx_last,
            channel_independent=channel_independent,
            **kwargs,
        )
        self.num_samples = num_samples

    def get_data(self):
        # Initialize the output
        data = list()

        # Generate sine data
        for i in range(self.num_samples):
            # Initialize each time-series
            temp = list()
            # For each feature
            for k in range(self.seq_dim):
                # Randomly drawn frequency and phase
                freq = np.random.uniform(0.05, 0.4)
                phase = np.random.uniform(0, 1.5)

                # Generate sine signal based on the drawn frequency and phase
                temp_data = [np.sin(freq * j + phase) for j in range(self.seq_len)]
                temp.append(temp_data)

            # Align row/column
            temp = np.transpose(np.asarray(temp))
            # Normalize to [0,1]
            temp = (temp + 1) * 0.5
            # Stack the generated data
            data.append(temp)
        data = np.array(data)
        data = torch.from_numpy(data).float()

        # Condition save
        cond = self.prepare_cond(data, None)

        return data, cond

    @property
    def dataset_name(self) -> str:
        return "SineND"
