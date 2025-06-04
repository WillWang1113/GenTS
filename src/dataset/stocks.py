from io import StringIO
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
import torch

from src.dataset.base import BaseDataModule


class Stocks(BaseDataModule):
    def __init__(
        self,
        seq_len: int = 24,
        select_seq_dim: List[int | str] = None,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        scale: bool = True,
        inference_batch_size: int = 1024,
        add_coeffs: str = None,
        time_idx_last: bool = False,
        channel_independent: bool = False,
        **kwargs,
    ):
        super().__init__(
            seq_len,
            len(select_seq_dim) if isinstance(select_seq_dim, list) else select_seq_dim,
            condition,
            batch_size,
            inference_batch_size,
            data_dir,
            add_coeffs=add_coeffs,
            time_idx_last=time_idx_last,
            channel_independent=channel_independent,
            **kwargs,
        )
        self.scale = scale
        self.select_seq_dim = select_seq_dim
        self.D = 6
        if select_seq_dim is not None:
            assert max(select_seq_dim) < self.D

    def get_data(self):
        pre_download_dir = self.data_dir / f"{self.dataset_name}_raw.csv"
        if pre_download_dir.exists():
            # Load data from local file if it exists
            df = pd.read_csv(pre_download_dir)
        else:
            # Download stock data
            url = "https://raw.githubusercontent.com/jsyoon0823/TimeGAN/refs/heads/master/data/stock_data.csv"
            headers = {"Authorization": "Test"}
            response = requests.get(url, headers=headers)
            df = pd.read_csv(StringIO(response.text))
            df.to_csv(pre_download_dir, index=False)

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
        data_raw = torch.from_numpy(data_raw)

        # slide window
        data = []
        for i in range(len(data_raw) - self.total_seq_len + 1):
            data.append(data_raw[i : i + self.total_seq_len, :])
        data = torch.stack(data, dim=0).float()

        # Condition save
        cond = self.prepare_cond(data, None)

        return data, cond

    @property
    def dataset_name(self) -> str:
        return "Stocks"
