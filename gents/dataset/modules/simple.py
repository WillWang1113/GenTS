from pathlib import Path

import numpy as np
import torch

from ..base import BaseDataModule


class Spiral2D(BaseDataModule):
    """Simulated 2D spiral curves with clock-wise or counter clock-wise direction.
    For one curve, :math:`x_1(t)=\pm r(t) \cos(t), x_2(t)=r(t) \sin(t)` where :math:`r(t)=a + bt, a \sim \mathcal{U}[0, 0.5), b \sim \mathcal{U}[0, 0.2)`.

    Args:
        seq_len (int, optional): Target sequence length. Defaults to 200.
        num_samples (int, optional): Number of total simulated curves. Defaults to 1000.
        batch_size (int, optional): Training and validation batch size. Defaults to 32.
        data_dir (Path | str, optional): Directory to save the data file (default name: `"data_tsl{total_seq_len}_tsd{seq_dim}_ir{irregular_dropout}.pt"`). Defaults to Path.cwd()/"data".
        condition (str, optional): Possible condition type, choose from [None, 'predict','impute', 'class']. None standards for unconditional generation.
        inference_batch_size (int, optional): Testing batch size. Defaults to 1024.
        max_time (float, optional): Time step index [0, 1, ..., `total_seq_len` - 1] will be automatically generated. If `max_time` is given, then scale the time step index, [0, ..., `max_time`]. Defaults to None.
        add_coeffs (str, optional): Include interpolation coefficients or not. Needed for `KoVAE`, `GTGAN` and `SDEGAN`. Choose from `[None, 'linear', 'cubic_spline']`. If `None`, don't include. Defaults to None.
        irregular_dropout (float, optional): Dropout rate to similate irregular time series data by randomly dropout some time steps in the original data. Set between `[0.0, 1.0]` Defaults to 0.0.
    """
    def __init__(
        self,
        seq_len: int = 200,
        num_samples: int = 1000,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        inference_batch_size: int = 1024,
        max_time: float = None,
        add_coeffs: str = None,
        irregular_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            seq_len,
            2,
            condition,
            batch_size,
            inference_batch_size,
            max_time,
            add_coeffs,
            irregular_dropout,
            data_dir,
            **kwargs,
        )
        self.num_samples = num_samples
        self.random_dropout = irregular_dropout
        assert irregular_dropout >= 0 and irregular_dropout < 1

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
        data, class_label = torch.stack(curves), torch.tensor(labels)

        # Condition save
        # cond = self.prepare_cond(data, class_cond)
        data_mask = torch.ones_like(data)
        if self.random_dropout > 0:
            mask = torch.bernoulli(
                torch.full(
                    (data.shape[0], data.shape[1]),
                    1 - self.random_dropout,
                    device=x.device,
                )
            ).unsqueeze(-1)

            data_mask = data_mask * mask
        data_mask = data_mask.bool()

        # data = data.masked_fill(~data_mask, 0.0)

        return data, data_mask.bool(), class_label

    @property
    def dataset_name(self) -> str:
        return "2DSpiral"


class SineND(BaseDataModule):
    """Simulated sine waves with `N` dimensions. For each dimension, :math:`x(t)=\sin(at+b), a \sim \mathcal{U}[0.05, 0.4], b \sim \mathcal{U}[0., 1.5]`.

    Args:
        seq_len (int, optional): Target sequence length. Defaults to 200.
        seq_dim (int, optional): Total simulated dimensions. Defaults to 200.
        num_samples (int, optional): Number of total simulated curves. Defaults to 1000.
        batch_size (int, optional): Training and validation batch size. Defaults to 32.
        data_dir (Path | str, optional): Directory to save the data file (default name: `"data_tsl{total_seq_len}_tsd{seq_dim}_ir{irregular_dropout}.pt"`). Defaults to Path.cwd()/"data".
        condition (str, optional): Possible condition type, choose from [None, 'predict','impute', 'class']. None standards for unconditional generation.
        inference_batch_size (int, optional): Testing batch size. Defaults to 1024.
        max_time (float, optional): Time step index [0, 1, ..., `total_seq_len` - 1] will be automatically generated. If `max_time` is given, then scale the time step index, [0, ..., `max_time`]. Defaults to None.
        add_coeffs (str, optional): Include interpolation coefficients or not. Needed for `KoVAE`, `GTGAN` and `SDEGAN`. Choose from `[None, 'linear', 'cubic_spline']`. If `None`, don't include. Defaults to None.
        irregular_dropout (float, optional): Dropout rate to similate irregular time series data by randomly dropout some time steps in the original data. Set between `[0.0, 1.0]` Defaults to 0.0.
    """
    def __init__(
        self,
        seq_len: int = 200,
        seq_dim: int = 1,
        num_samples: int = 1000,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        inference_batch_size: int = 1024,
        max_time: float = None,
        add_coeffs: str = None,
        irregular_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            seq_len,
            seq_dim,
            condition,
            batch_size,
            inference_batch_size,
            max_time,
            add_coeffs,
            irregular_dropout,
            data_dir,
            **kwargs,
        )
        self.num_samples = num_samples
        self.random_dropout = irregular_dropout
        assert irregular_dropout >= 0 and irregular_dropout < 1

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
                temp_data = [
                    np.sin(freq * j + phase) for j in range(self.total_seq_len)
                ]
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
        data_mask = torch.ones_like(data)
        if self.random_dropout > 0:
            mask = torch.bernoulli(
                torch.full(
                    (data.shape[0], data.shape[1]),
                    1 - self.random_dropout,
                    device=data.device,
                )
            ).unsqueeze(-1)

            data_mask = data_mask * mask
        data_mask = data_mask.bool()
        # data = data.masked_fill(~data_mask, 0.0)
        class_label = None
        return data, data_mask.bool(), class_label

    @property
    def dataset_name(self) -> str:
        return "SineND"
