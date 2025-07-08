from pathlib import Path
from typing import List

import numpy as np
import torch

from gents.dataset.base import BaseDataModule
# from src.dataset.base import BaseDataModule


class MuJoCo(BaseDataModule):
    D = 14
    def __init__(
        self,
        seq_len: int = 200,
        select_seq_dim: List[int] = None,
        num_samples: int = 5000,
        batch_size: int = 32,
        data_dir: Path | str = Path.cwd() / "data",
        condition: str = None,
        inference_batch_size: int = 1024,
        max_time: float = 1.0,
        add_coeffs: str = None,
        irregular_dropout: float = 0.0,
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
            **kwargs,
        )
        self.num_samples = num_samples
        self.select_seq_dim = select_seq_dim
        if select_seq_dim is not None:
            assert max(select_seq_dim) < self.D
        self.random_dropout = irregular_dropout
        assert irregular_dropout >= 0 and irregular_dropout < 1

        

    def get_data(self):
        
        pre_download_dir = self.data_dir / f"{self.dataset_name}_raw.pt"
        if pre_download_dir.exists():
            # Load data from local file if it exists
            data = torch.load(pre_download_dir)
        else:
            
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
            torch.save(data, pre_download_dir)
        
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
