from abc import ABC, abstractmethod
from typing import Optional, Literal
from lightning import LightningModule
import torch
from torch.nn import functional as F


class BaseModel(ABC, LightningModule):
    # def __init__(
    #     self,
    #     seq_len: int,
    #     seq_dim: int,
    #     condition: Optional[Literal["predict", "impute", "class"]] = None,
    #     lr: Optional[float] = 1e-3,
    #     **kwargs,
    # ):
    #     super().__init__()
    #     self.seq_len = seq_len
    #     self.seq_dim = seq_dim
    #     self.condition = self.condition

    @torch.no_grad()  # wrap with torch.no_grad()
    def sample(self, n_sample: int = 1, condition=None, **kwargs):
        """Generate samples from the generative model"""
        self.eval()
        return self._sample_impl(n_sample, condition, **kwargs)

    @abstractmethod
    def _sample_impl(self, n_sample: int = 1, condition=None, **kwargs) -> torch.Tensor:
        """Actual implementation of the sampling process"""
