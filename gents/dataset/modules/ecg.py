import os

import numpy as np
import torch
from torchvision.datasets.utils import download_and_extract_archive

# from src.dataset.base import BaseDataModule
from ..base import BaseDataModule


class ECG(BaseDataModule):
    """`ECG5000 dataset <https://www.timeseriesclassification.com/description.php?Dataset=ECG5000>`__.
    Raw data has already be scaled.

    .. note::
        This is a univariate time series dataset, i.e. `seq_dim = 1`.

    .. note::
        This is a fixed-length dataset, i.e. for each time series `total_seq_len=140`, is fixed.
        For `predict` condition, following the rule that `total_seq_len = obs_len + seq_len <= 140`.

    .. note::
        Class labels of ECG are patient statuses (in total 5 labels).

    Attributes:
        L (int): Total sequence length, fixed to 140.
        D (int): Number of variates, fixed to 140.
        n_classes (int): Total number of class labels, fixed to 5.
        urls (str): `download link <https://zenodo.org/records/4656719/files/kdd_cup_2018_dataset_with_missing_values.zip>`__

    Args:
        seq_len (int, optional): Target sequence length, fixed to 140.
        seq_dim (int, optional): Target sequence dimensions, fixed to 1.
        batch_size (int, optional): Training and validation batch size. Defaults to 32.
        data_dir (str, optional): Directory to save the data file (default name: `"data_tsl{total_seq_len}_tsd{seq_dim}_ir{irregular_dropout}.pt"`). Defaults to "data".
        condition (str, optional): Possible condition type, choose from [None, 'predict','impute', 'class']. None standards for unconditional generation.
        inference_batch_size (int, optional): Testing batch size. Defaults to 1024.
        max_time (float, optional): Time step index [0, 1, ..., `total_seq_len` - 1] will be automatically generated. If `max_time` is given, then scale the time step index, [0, ..., `max_time`]. Defaults to None.
        add_coeffs (str, optional): Include interpolation coefficients or not. Needed for `KoVAE`, `GTGAN` and `SDEGAN`. Choose from `[None, 'linear', 'cubic_spline']`. If `None`, don't include. Defaults to None.
        irregular_dropout (float, optional): Dropout rate to similate irregular time series data by randomly dropout some time steps in the original data. Set between `[0.0, 1.0]` Defaults to 0.0.
        **kwargs: Additional arguments for the model
    """

    L = 140
    D = 1
    n_classes = 5

    url = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"

    def __init__(
        self,
        seq_len: int = 140,
        seq_dim: int = 1,
        batch_size: int = 32,
        data_dir: str = "./data",
        condition: str = None,
        inference_batch_size: int = 1024,
        max_time: float = 1.0,
        add_coeffs: str = None,
        irregular_dropout: float = 0.0,
        **kwargs,
    ):
        # seq_len = self.L - kwargs.get('obs_len', 0)
        # assert seq_len > 0 and seq_len <= self.L, f'total seq_len should be less than {self.L} for ECG dataset'

        super().__init__(
            seq_len,
            1,
            condition,
            batch_size,
            inference_batch_size,
            max_time,
            add_coeffs,
            irregular_dropout,
            data_dir,
            **kwargs,
        )
        if seq_dim != 1:
            raise ValueError("ECG is a univariate time series dataset")
        if self.total_seq_len != self.L:
            print(f'Original dataset fix seq_len=140, we will interpolate to seq_len={seq_len}')
            # raise ValueError("ECG has fixed total seq_len = 140")

    def get_data(self):
        download_and_extract_archive(
            self.url, self.data_dir, self.data_dir, "archive.zip"
        )
        orig_train = np.loadtxt(os.path.join(self.data_dir, "ECG5000_TRAIN.txt"))
        orig_test = np.loadtxt(os.path.join(self.data_dir, "ECG5000_TEST.txt"))
        all_data = np.concatenate([orig_test, orig_train])
        all_data = torch.from_numpy(all_data).float()
        shuffle_idx = torch.randperm(len(all_data))
        all_data = all_data[shuffle_idx]

        data = all_data[:, 1:].unsqueeze(-1)
        if self.seq_len != self.L:
            data = torch.nn.functional.interpolate(
                data.permute(0, 2, 1), size=self.seq_len, mode="linear"
            ).permute(0,2,1)
        data_mask = ~torch.isnan(data)
        class_label = all_data[:, 0] - 1.0
        return data, data_mask, class_label

    @property
    def dataset_name(self) -> str:
        return "ECG"
