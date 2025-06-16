from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# from src.dataset.base import BaseDataModule
from src.dataset.base_new import BaseDataModule
from torchvision.datasets.utils import download_url
import os
import tarfile
import re


# adapted from CSDI (https://github.com/ermongroup/CSDI/blob/main/dataset_physio.py)


def get_idlist(root_dir, patient_set="a"):
    read_dir = os.path.join(root_dir, f"set-{patient_set}")
    patient_id = []
    for filename in os.listdir(read_dir):
        match = re.search("\d{6}", filename)
        if match:
            patient_id.append(match.group())
    patient_id = np.sort(patient_id)
    return patient_id


class Physionet(BaseDataModule):
    L = 48 * 60
    attributes = [
        "DiasABP",
        "HR",
        "Na",
        "Lactate",
        "NIDiasABP",
        "PaO2",
        "WBC",
        "pH",
        "Albumin",
        "ALT",
        "Glucose",
        "SaO2",
        "Temp",
        "AST",
        "Bilirubin",
        "HCO3",
        "BUN",
        "RespRate",
        "Mg",
        "HCT",
        "SysABP",
        "FiO2",
        "K",
        "GCS",
        "Cholesterol",
        "NISysABP",
        "TroponinT",
        "MAP",
        "TroponinI",
        "PaCO2",
        "Platelets",
        "Urine",
        "NIMAP",
        "Creatinine",
        "ALP",
    ]
    D = len(attributes)

    urls = [
        "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download",
        "https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download",
    ]

    outcome_urls = ["https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt"]

    def __init__(
        self,
        agg_minutes: int = 60,
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
        seq_len = self.L // agg_minutes - kwargs.get('obs_len', 0)
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
        self.agg_time_interval = f"{agg_minutes}min"
        # self.D = len(self.attributes)

        if select_seq_dim is not None:
            assert max(select_seq_dim) < self.D
        

    def get_data(self):
        raw_folder = self.data_dir
        for url in self.outcome_urls:
            filename = url.rpartition("/")[2]
            download_url(url, raw_folder, filename, None)

        for url in self.urls:
            filename = url.rpartition("/")[2]
            download_url(url, raw_folder, filename, None)
            tar = tarfile.open(os.path.join(raw_folder, filename), "r:gz")
            tar.extractall(raw_folder)
            tar.close()

            print("Processing {}...".format(filename))

        observed_values_list = []
        observed_masks_list = []
        # gt_masks_list = []

        for patient_set in ["a", "b"]:
            idlist = get_idlist(self.data_dir, patient_set)
            for id_ in tqdm(idlist):
                try:
                    observed_values, observed_masks = self.parse_patient_data(
                        id_, patient_set
                    )
                    observed_values_list.append(observed_values)
                    observed_masks_list.append(observed_masks)
                    # gt_masks_list.append(gt_masks)
                    
                except Exception as e:
                    print(id_, e)
                    continue
        observed_values = np.array(observed_values_list)
        observed_masks = np.array(observed_masks_list)
        # gt_masks = np.array(gt_masks_list)

        # calc mean and std and normalize values
        # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
        tmp_values = observed_values.reshape(-1, self.D)
        tmp_masks = observed_masks.reshape(-1, self.D)
        mean = np.zeros(self.D)
        std = np.zeros(self.D)
        for k in range(self.D):
            c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
            mean[k] = c_data.mean()
            std[k] = c_data.std()
        if self.scale:
            observed_values = (observed_values - mean) / std * observed_masks

        data = torch.from_numpy(observed_values).float()
        data_mask = torch.from_numpy(observed_masks).bool()
        # only has class_label for set_a
        class_label = None
        
        # select dimensions
        if self.select_seq_dim is not None:
            if isinstance(self.select_seq_dim[0], str):
                chnl_select = [self.attributes.index(ssd) for ssd in self.select_seq_dim]
            elif isinstance(self.select_seq_dim[0], int):
                chnl_select = self.select_seq_dim
            data = data[..., chnl_select]
            data_mask = data_mask[..., chnl_select]
        
        return data, data_mask, class_label

    @property
    def dataset_name(self) -> str:
        return "Physionet"

    def parse_patient_data(self, id_, patient_set="a"):
        read_dir = os.path.join(self.data_dir, "set-{}/{}.txt".format(patient_set, id_))
        df = pd.read_csv(read_dir)

        # add artifical time stamps for aggregation
        df[["h", "m"]] = df["Time"].str.split(":", expand=True).astype(float)
        df["day"] = df["h"] // 24 + 1
        df["h"] = df["h"] % 24
        df["year"] = 2001  # fake year, doesnt matter
        df["month"] = 12  # fake month, doesnt matter
        df["Time"] = pd.to_datetime(df[["year", "month", "day", "h", "m"]])
        df = df[["Time", "Parameter", "Value"]]
        df = df.pivot_table(index='Time', columns='Parameter', values='Value', aggfunc='mean')

        # complete table with all attributes and set NaNs
        col_to_add = set(self.attributes) - set(df.columns.tolist())
        col_to_add = list(col_to_add)
        df[col_to_add] = np.NaN
        df = df[self.attributes]
        assert self.attributes == df.columns.tolist()

        # aggregation
        # data shape: usually [24*N_t, C]
        # mask shape=data shape, 0:missing, 1:observed
        df_agg = df.resample(self.agg_time_interval).mean()
        
        # extend to full 2 days, and limit to 2 days
        df_agg = df_agg.reset_index()
        full_time = pd.date_range('2001-12-01 00:00:00', "2001-12-03 00:00:00", freq=self.agg_time_interval, inclusive='left')
        full_time = pd.DataFrame(full_time, columns=['Time'])
        df_final = full_time.merge(df_agg, how='outer')
        df_final = df_final.set_index('Time')
        df_final = df_final[:'2001-12-02']
        
        observed_values = df_final.values
        observed_masks = ~np.isnan(observed_values)
        

        # # randomly set some percentage as ground-truth
        # masks = observed_masks.reshape(-1).copy()
        # obs_indices = np.where(masks)[0].tolist()
        # miss_indices = np.random.choice(
        #     obs_indices, (int)(len(obs_indices) * self.missing_rate), replace=False
        # )
        # masks[miss_indices] = False
        # gt_masks = masks.reshape(observed_masks.shape)

        observed_values = np.nan_to_num(observed_values)
        
        # observed_masks = observed_masks.astype("float32")
        # gt_masks = gt_masks.astype("float32")

        # return observed_values, observed_masks, gt_masks
        return observed_values, observed_masks
