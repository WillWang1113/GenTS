import os
from functools import reduce
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torchvision.datasets.utils import download_and_extract_archive

# from src.dataset.base import BaseDataModule
from gents.dataset._monash_utils import convert_tsf_to_dataframe
from gents.dataset.base_new import BaseDataModule


class AirQuality(BaseDataModule):
    attributes = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]
    D = len(attributes)

    urls = "https://zenodo.org/records/4656719/files/kdd_cup_2018_dataset_with_missing_values.zip"

    def __init__(
        self,
        seq_len: int = 24,
        select_seq_dim: List[int | str] = None,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        scale: bool = True,
        inference_batch_size: int = 1024,
        max_time: float = 1.0,
        add_coeffs: str = None,
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
            data_dir,
            **kwargs,
        )
        self.scale = scale
        self.select_seq_dim = select_seq_dim

        if select_seq_dim is not None:
            assert max(select_seq_dim) < self.D

    def get_data(self):
        download_and_extract_archive(
            self.urls, self.data_dir, self.data_dir, "archive.zip"
        )

        (
            df,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        ) = convert_tsf_to_dataframe(
            os.path.join(self.data_dir, "kdd_cup_2018_dataset_with_missing_values.tsf")
        )
        df = df[df["city"] == "Beijing"]

        station_id = 0
        data, data_mask, class_label = [], [], []
        for i, sub_df in df.groupby(["station"]):
            station_df = []
            for j, attr in enumerate(self.attributes):
                loc = sub_df["air_quality_measurement"] == attr
                single_ts = sub_df[loc]["series_value"].iloc[0].to_numpy()
                station_aq_df = pd.DataFrame(single_ts, columns=[attr])
                station_aq_df = station_aq_df.astype(float)
                station_aq_df["time"] = pd.date_range(
                    sub_df["start_timestamp"].iloc[0],
                    freq="1H",
                    periods=len(station_aq_df),
                )
                station_df.append(station_aq_df)
            station_df = reduce(
                lambda left, right: pd.merge(left, right, on=["time"], how="outer"),
                station_df,
            )
            station_df = station_df.set_index("time")
            station_data, station_data_mask, station_label = self.slide_window(
                station_df, station_id
            )
            station_id += 1
            data.append(station_data)
            data_mask.append(station_data_mask)
            class_label.append(station_label)

        data = torch.concat(data)
        data_mask = torch.concat(data_mask)
        class_label = torch.concat(class_label)

        return data, data_mask, class_label

    @property
    def dataset_name(self) -> str:
        return "AirQuality"

    def slide_window(self, df: pd.DataFrame, station_id: int):
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
        data_mask = torch.isnan(data)
        class_label = torch.ones(data.shape[0]).float() * station_id

        return torch.nan_to_num(data), data_mask, class_label
