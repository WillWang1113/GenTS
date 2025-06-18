import os
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets.utils import download_and_extract_archive

# from src.dataset.base import BaseDataModule
from src.dataset.base_new import BaseDataModule


class ECG(BaseDataModule):
    L = 140
    D = 1

    url = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"

    def __init__(
        self,
        seq_len: int = 140,
        seq_dim: int = 1,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        inference_batch_size: int = 1024,
        max_time: float = 1.0,
        add_coeffs: str = None,
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
            data_dir,
            **kwargs,
        )
        if seq_dim != 1:
            raise ValueError('ECG is a univariate time series dataset')
        if self.total_seq_len != self.L:
            raise ValueError('ECG has fixed total seq_len = 140')
            

    def get_data(self):
        download_and_extract_archive(self.url, self.data_dir, self.data_dir, 'archive.zip')
        orig_train = np.loadtxt(os.path.join(self.data_dir, 'ECG5000_TRAIN.txt'))
        orig_test = np.loadtxt(os.path.join(self.data_dir, 'ECG5000_TEST.txt'))
        all_data = np.concatenate([orig_test, orig_train])
        all_data = torch.from_numpy(all_data).float()
        data = all_data[:, 1:].unsqueeze(-1)
        data_mask = ~torch.isnan(data)
        class_label = all_data[:, 0]
        
        
        return data, data_mask, class_label

    @property
    def dataset_name(self) -> str:
        return "ECG"
