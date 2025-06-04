from pathlib import Path
from typing import List

import numpy as np
import torch

from src.dataset.base import BaseDataModule


class MuJoCo(BaseDataModule):
    def __init__(
        self,
        seq_len: int = 200,
        select_seq_dim: List[int] = None,
        num_samples: int = 5000,
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
            len(select_seq_dim) if select_seq_dim is not None else 14,
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
        self.select_seq_dim = select_seq_dim
        self.D = 14
        if select_seq_dim is not None:
            assert max(select_seq_dim) < self.D

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
        cond = self.prepare_cond(data, None)

        return data, cond

    @property
    def dataset_name(self) -> str:
        return "MuJoCo"
