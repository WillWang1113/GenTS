import abc
import logging
import os
from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule


class TSDataset(Dataset):
    def __init__(self, data, cond=None):
        super().__init__()
        self.data = data
        self.cond = cond
        self.cond_shape = None
        if cond is not None:
            self.cond_shape = tuple(cond.shape[1:])

    def __getitem__(self, index):
        if self.cond_shape is not None:
            return dict(seq=self.data[index], c=self.cond[index])
        else:
            return dict(seq=self.data[index])

    def __len__(self):
        return len(self.data)


# class TSDataModule(LightningDataModule, abc.ABC):
#     def __init__(
#         self,
#         seq_len: int,
#         batch_size: int,
#         data_dir: Path | str = Path.cwd() / "data",
#         condition: str = None,
#         scale: bool = True,
#         inference_batch_size: int = 1024,
#         **kwargs,
#     ):
#         super().__init__()
#         assert condition in [None, "predict", "impute"]
#         self.data_dir = data_dir
#         self.seq_len = seq_len
#         self.dataset_cls = DS_type[condition] if condition is not None else TSDataset
#         self.batch_size = batch_size
#         self.scale = scale
#         self.kwargs = kwargs
#         self.infer_bs = inference_batch_size
#         self.total_seq_len = (
#             seq_len if condition != "predict" else seq_len + kwargs.get("obs_len")
#         )

#     @abc.abstractmethod
#     def get_data(self) -> None: ...

#     @property
#     @abc.abstractmethod
#     def dataset_name(self) -> str: ...

#     def setup(self, stage):
#         if stage == "fit":
#             self.train_ds = None
#             self.val_ds = None

#         if stage == "predict":
#             self.pred_ds = None

#         if stage == "test":
#             self.test_ds = None

#     def prepare_data(self) -> None:
#         if not self.data_dir.exists():
#             logging.info(f"Downloading {self.dataset_name} dataset in {self.data_dir}.")
#             os.makedirs(self.data_dir)
#             self.get_data()

#     def train_dataloader(self):
#         return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_ds, batch_size=self.batch_size)

#     def predict_dataloader(self):
#         return DataLoader(self.test_ds, batch_size=self.infer_bs)

#     def test_dataloader(self):
#         return DataLoader(self.test_ds, batch_size=self.infer_bs)


class SynDataModule(LightningDataModule):
    def __init__(
        self,
        seq_len: int = 200,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        scale: bool = True,
        inference_batch_size: int = 1024,
        num_samples: int = 1000,
        **kwargs,
    ):
        super().__init__()

        self.num_samples = num_samples
        self.data_dir = data_dir / self.dataset_name
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.scale = scale
        self.kwargs = kwargs
        self.infer_bs = inference_batch_size
        self.obs_len = None
        self.missing_rate = None
        self.missing_kind = None
        self.condition = condition

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

    def prepare_data(self) -> None:
        if (not (self.data_dir / "data.pt").exists()) or (
            not (self.data_dir / f"{self.condition}_cond.pt").exists()
        ):
            logging.info(f"Downloading {self.dataset_name} dataset in {self.data_dir}.")
            os.makedirs(self.data_dir, exist_ok=True)
            data, cond = self.get_data()
            torch.save(data, self.data_dir / "data.pt")
            if cond is not None:
                torch.save(cond, self.data_dir / f"{self.condition}_cond.pt")

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
                mask = torch.rand_like(data) > self.missing_rate
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

    @property
    def dataset_name(self) -> str:
        return "2DSpiral"

    def setup(self, stage):
        data = torch.load(os.path.join(self.data_dir, "data.pt"))
        cond = None
        if self.condition is not None:
            cond = torch.load(os.path.join(self.data_dir, f"{self.condition}_cond.pt"))
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

            self.train_ds = TSDataset(train_data, train_cond)
            self.val_ds = TSDataset(val_data, val_cond)

        if stage == "predict":
            test_data = data[starts["test"] : ends["test"]]
            test_cond = None
            if add_cond:
                test_cond = cond[starts["test"] : ends["test"]]
            self.pred_ds = TSDataset(test_data, test_cond)

        if stage == "test":
            test_data = data[starts["test"] : ends["test"]]
            test_cond = None
            if add_cond:
                test_cond = cond[starts["test"] : ends["test"]]
            self.test_ds = TSDataset(test_data, test_cond)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.infer_bs)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.infer_bs)


# OLD VERSION

# import abc
# import logging
# import os
# from pathlib import Path
# from typing import Optional
# from matplotlib import pyplot as plt
# import torch
# from torch.utils.data import DataLoader, Dataset
# from lightning import LightningDataModule


# class TSDataset(Dataset):
#     def __init__(self, data: torch.Tensor, seq_len: int):
#         self.seq_len = seq_len
#         self.data = data
#         self.total_len = seq_len

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         seq_x = self.data[s_begin:s_end]
#         data_dict = {"seq": seq_x}
#         return data_dict

#     def __len__(self):
#         return len(self.data) - self.total_len + 1


# class PredictDataset(TSDataset):
#     def __init__(self, data, seq_len, obs_len):
#         super().__init__(data, seq_len)
#         self.obs_len = obs_len
#         self.total_len += obs_len

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.obs_len

#         seq_x = self.data[s_begin:s_end]
#         seq_y = self.data[s_end : s_end + self.seq_len]
#         data_dict = {"c": seq_x, "seq": seq_y}
#         return data_dict


# class ImputeDataset(TSDataset):
#     def __init__(self, data, seq_len, missing_type, missing_rate):
#         super().__init__(data, seq_len)
#         assert missing_type in ["random", "block"]
#         assert 0 <= missing_rate <= 1
#         self.missing_type = missing_type
#         self.missing_rate = missing_rate

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len

#         seq_x = self.data[s_begin:s_end]
#         mask = torch.ones_like(seq_x)
#         if self.missing_type == "random":
#             mask = torch.rand_like(seq_x) > self.missing_rate
#         elif self.missing_type == "block":
#             rand_start = torch.randint(0, self.seq_len, (1,))
#             mask[rand_start : rand_start + int(self.seq_len * self.missing_rate)] = 0

#         data_dict = {"mask": mask, "seq": seq_x, "c": seq_x * mask}
#         return data_dict


# DS_type = {
#     # "synthesis": TSDataset,
#     "predict": PredictDataset,
#     "impute": ImputeDataset,
# }


# class TSDataModule(LightningDataModule, abc.ABC):
#     def __init__(
#         self,
#         seq_len: int,
#         batch_size: int,
#         data_dir: Path | str = Path.cwd() / "data",
#         condition: str = None,
#         scale: bool = True,
#         inference_batch_size: int = 1024,
#         **kwargs,
#     ):
#         super().__init__()
#         assert condition in [None, "predict", "impute"]
#         self.data_dir = data_dir
#         self.seq_len = seq_len
#         self.dataset_cls = DS_type[condition] if condition is not None else TSDataset
#         self.batch_size = batch_size
#         self.scale = scale
#         self.kwargs = kwargs
#         self.infer_bs = inference_batch_size
#         self.total_seq_len = (
#             seq_len if condition != "predict" else seq_len + kwargs.get("obs_len")
#         )

#     @abc.abstractmethod
#     def get_data(self) -> None: ...

#     @property
#     @abc.abstractmethod
#     def dataset_name(self) -> str: ...

#     def setup(self, stage):
#         if stage == "fit":
#             self.train_ds = None
#             self.val_ds = None

#         if stage == "predict":
#             self.pred_ds = None

#         if stage == "test":
#             self.test_ds = None

#     def prepare_data(self) -> None:
#         if not self.data_dir.exists():
#             logging.info(f"Downloading {self.dataset_name} dataset in {self.data_dir}.")
#             os.makedirs(self.data_dir)
#             self.get_data()

#     def train_dataloader(self):
#         return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_ds, batch_size=self.batch_size)

#     def predict_dataloader(self):
#         return DataLoader(self.test_ds, batch_size=self.infer_bs)

#     def test_dataloader(self):
#         return DataLoader(self.test_ds, batch_size=self.infer_bs)


# class SineDataModule(TSDataModule):
#     def __init__(
#         self,
#         seq_len: int,
#         batch_size: int,
#         data_dir: Path | str = Path.cwd() / "data",
#         condition: str = None,
#         scale: bool = True,
#         inference_batch_size: int = 1024,
#         **kwargs,
#     ):
#         super().__init__(
#             seq_len,
#             batch_size,
#             data_dir,
#             condition,
#             scale,
#             inference_batch_size,
#             **kwargs,
#         )

#     def get_data(self):
#         t = torch.linspace(0, 24 * torch.pi, 1152).float()
#         data = torch.cos(t) + 2 * torch.sin(t * 2) + torch.randn_like(t) * 0.1
#         data = (data - data.min()) / (data.max() - data.min())
#         data = data.reshape(-1, 1)
#         torch.save(data, self.data_dir / "sine.pt")

#     @property
#     def dataset_name(self) -> str:
#         return "sine"

#     def setup(self, stage):
#         data = torch.load(os.path.join(self.data_dir, f"{self.dataset_name}.pt"))
#         # train/val/test
#         num_train = int(len(data) * 0.7)
#         num_test = int(len(data) * 0.2)
#         num_vali = len(data) - num_train - num_test

#         starts = dict(
#             fit=0,
#             validate=num_train - self.total_seq_len,
#             test=len(data) - num_test - self.total_seq_len,
#         )
#         ends = dict(fit=num_train, validate=num_train + num_vali, test=len(data))

#         if stage == "fit":
#             train_data = data[starts["fit"] : ends["fit"]]
#             val_data = data[starts["validate"] : ends["validate"]]
#             self.train_ds = self.dataset_cls(train_data, self.seq_len, **self.kwargs)
#             self.val_ds = self.dataset_cls(val_data, self.seq_len, **self.kwargs)

#         if stage == "predict":
#             test_data = data[starts["test"] : ends["test"]]
#             self.pred_ds = self.dataset_cls(test_data, self.seq_len, **self.kwargs)

#         if stage == "test":
#             test_data = data[starts["test"] : ends["test"]]
#             self.test_ds = self.dataset_cls(test_data, self.seq_len, **self.kwargs)
