import os
from pathlib import Path
from typing import List

import numpy as np
import torch

from ..base import BaseDataModule


class MuJoCo(BaseDataModule):
    """MuJoCo data set with hopper standing task. 

    .. note::
        Require `Deepmind Control Suite` to run. (`pip install dm_control`)

    Attributes:
        D (int): Total number of variates, 14.
        
    Args:
        seq_len (int): Target sequence length
        select_seq_dim (List[int]): Subset of all sequence channels. Could be a `list` of `int` indicating the chosen channel indice. If `None`, use all channels. Defaults to None.
        num_samples (int, optional): Number of total simulated curves. Defaults to 5000.
        batch_size (int): Training and validation batch size.
        data_dir (str, optional): Directory to save the data file (default name: `"data_tsl{total_seq_len}_tsd{seq_dim}_ir{irregular_dropout}.pt"`). Defaults to Path.cwd()/"data".
        condition (str): Possible condition type, choose from [None, 'predict','impute', 'class']. None standards for unconditional generation.
        inference_batch_size (int): Testing batch size.
        max_time (float, optional): Time step index [0, 1, ..., `total_seq_len` - 1] will be automatically generated. If `max_time` is given, then scale the time step index, [0, ..., `max_time`]. Defaults to None.
        add_coeffs (str, optional): Include interpolation coefficients or not. Needed for `KoVAE`, `GTGAN` and `SDEGAN`. Choose from `[None, 'linear', 'cubic_spline']`. If `None`, don't include. Defaults to None.
        irregular_dropout (float, optional): Dropout rate to similate irregular time series data by randomly dropout some time steps in the original data. Set between `[0.0, 1.0]` Defaults to 0.0.
        **kwargs: Additional arguments for the model
    """
    D = 14
    def __init__(
        self,
        seq_len: int = 200,
        select_seq_dim: List[int] = None,
        num_samples: int = 5000,
        batch_size: int = 32,
        data_dir: str = "./data",
        condition: str = None,
        inference_batch_size: int = 1024,
        max_time: float = 1.0,
        add_coeffs: str = None,
        irregular_dropout: float = 0.0,
        train_val_test: List[float] = [0.7, 0.2, 0.1],
        **kwargs,
    ):
        super().__init__(
            seq_len,
            len(select_seq_dim) if select_seq_dim is not None else self.D,
            condition,
            batch_size,
            inference_batch_size,
            max_time,
            add_coeffs,
            irregular_dropout,
            data_dir,
            train_val_test,
            **kwargs,
        )
        self.num_samples = num_samples
        self.select_seq_dim = select_seq_dim
        if select_seq_dim is not None:
            assert max(select_seq_dim) < self.D
        self.random_dropout = irregular_dropout
        assert irregular_dropout >= 0 and irregular_dropout < 1


    def get_data(self):
        
        # pre_download_dir = os.path.join(self.data_dir, f"{self.dataset_name}_raw.pt")
        # if os.path.exists(pre_download_dir):
        #     # Load data from local file if it exists
        #     data = torch.load(pre_download_dir)
        # else:
            
        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception(
                "Deepmind Control Suite is required to generate the dataset."
            ) from e

        env = suite.load("hopper", "stand")
        physics = env.physics

        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(123)

        data = np.zeros((self.num_samples, self.total_seq_len, self.D))
        for i in range(self.num_samples):
            with physics.reset_context():
                # x and z positions of the hopper. We want z > 0 for the hopper to stay above ground.
                physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
                physics.data.qpos[2:] = np.random.uniform(
                    -2, 2, size=physics.data.qpos[2:].shape
                )
                physics.data.qvel[:] = np.random.uniform(
                    -5, 5, size=physics.data.qvel.shape
                )
            for t in range(self.total_seq_len):
                data[i, t, : self.D // 2] = physics.data.qpos
                data[i, t, self.D // 2 :] = physics.data.qvel
                physics.step()

        # Restore RNG.
        np.random.set_state(st0)
        data = torch.from_numpy(data).float()
        # torch.save(data, pre_download_dir)
        
        if self.select_seq_dim is not None:
            data = data[..., self.select_seq_dim]

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
        return "MuJoCo"
