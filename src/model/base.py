from abc import ABC, abstractmethod
from typing import Optional, Literal
from lightning import LightningModule
import torch
from torch.nn import functional as F

def _condition_shape_check(n_sample, condition, cond_type):
    assert n_sample >=1
    if cond_type is None:
    
        if condition.shape[0] == 1:
            condition = condition.repeat(
                n_sample, *[1 for _ in range(len(condition.shape) - 1)]
            )
        elif condition.shape[0] == n_sample:
            pass
        else:
            raise ValueError(
                "The batch size of the given condition should be the same as n_sample or just 1."
            )
 
    return condition


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
        condition = _condition_shape_check(n_sample, condition, self.condition)
        self.eval()
        return self._sample_impl(n_sample, condition, **kwargs)

    @abstractmethod
    def _sample_impl(self, n_sample: int = 1, condition=None, **kwargs) -> torch.Tensor:
        """Actual implementation of the sampling process"""
