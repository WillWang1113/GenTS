import os
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule


class TSDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.data = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end
        # r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        data_dict = {"seq": seq_x}
        return data_dict

    def __len__(self):
        return len(self.data) - self.seq_len + 1


class TSDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, seq_len: int, condition=None):
        super().__init__()
        # self.data = data
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.condition = condition
        self.batch_size = batch_size

    def prepare_data(self):
        # Load, scale, condition

        # ! TO BE DELETED
        t = torch.linspace(0, 64 * torch.pi, 1024*2).float()
        data = torch.cos(t) +  torch.sin(2 * t)
        data = (data - data.min()) / (data.max() - data.min())
        data = data.reshape(-1, 1)
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, "test.ckpt")
        if not os.path.exists(file_path):
            torch.save(data, file_path)

    def setup(self, stage):
        data = torch.load(os.path.join(self.data_dir, "test.ckpt"))
        # train/val/test
        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test

        starts = dict(
            fit=0,
            validate=num_train - self.seq_len,
            test=len(data) - num_test - self.seq_len,
        )
        ends = dict(fit=num_train, validate=num_train + num_vali, test=len(data))
        # border1s = [0, num_train - self.seq_len, len(self.data) - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_vali, len(self.data)]

        if stage == "fit":
            train_data = data[starts["fit"] : ends["fit"]]
            val_data = data[starts["validate"] : ends["validate"]]
            self.train_ds = TSDataset(train_data, self.seq_len)
            self.val_ds = TSDataset(val_data, self.seq_len)

        if stage == "predict":
            test_data = data[starts["test"] : ends["test"]]
            self.pred_ds = TSDataset(test_data, self.seq_len)

        if stage == "test":
            test_data = data[starts["test"] : ends["test"]]
            self.test_ds = TSDataset(test_data, self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.train_ds, batch_size=max(self.batch_size, 512))
