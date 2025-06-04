from pathlib import Path

import numpy as np
import torch

from src.dataset.base import BaseDataModule


class Spiral2D(BaseDataModule):
    def __init__(
        self,
        seq_len: int = 200,
        num_samples: int = 1000,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        inference_batch_size: int = 1024,
        add_coeffs: str = None,
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
        inference_batch_size: int = 1024,
        add_coeffs: str = None,
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
                temp_data = [np.sin(freq * j + phase) for j in range(self.total_seq_len)]
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
