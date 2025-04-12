from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Literal, Union
from lightning import LightningModule
import torch
from torch.nn import functional as F


def _condition_shape_check(n_sample:int, condition:torch.Tensor, cond_type:str):
    if n_sample < 1:
        raise ValueError("n_sample should be greater than 0.")
    
    if cond_type == 'class':
        cond_batch_size = condition.shape[0]
            
        if cond_batch_size == 1 and n_sample > 1:
            condition = condition.repeat(
                n_sample, *[1 for _ in range(len(condition.shape) - 1)]
            )
        elif cond_batch_size == n_sample:
            pass
        else:
            raise ValueError(
                "The batch size of the given condition should be the same as n_sample or just 1."
            )
    elif cond_type in ['predict', 'impute']:
        if condition is None:
            raise ValueError(
                "Condition should not be None for prediction."
            )
    
    elif cond_type is None: ...
        # if condition is not None:
        #     raise ValueError(
        #         "Condition should be None for unconditional generation."
        #     )
    
    
    assert n_sample >= 1
    if cond_type is not None:
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
    """Base class for generative models in PyTorch Lightning"""

    ALLOW_CONDITION = ...

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str,
        **kwargs,
    ):
        """Base class for generative models in PyTorch Lightning

        Args:
            seq_len (int): Target sequence length
            seq_dim (int): Target sequence dimension, for univariate time series, set as 1
            condition (str): Possible condition type, choose from [None, 'predict','impute', 'class']. None standards for unconditional generation.
            **kwargs: Additional arguments for the model

        Raises:
            ValueError: Condition type not allowed for this model
        """
        super().__init__()
        if condition not in self.ALLOW_CONDITION:
            raise ValueError(
                f"Condition '{condition}' not allowed. Choose from {self.ALLOW_CONDITION}"
            )
        self.condition = condition

    @torch.no_grad()  # wrap with torch.no_grad()
    def sample(self, n_sample: int = 1, condition=None, **kwargs):
        """Generate samples from the generative model"""
        condition = _condition_shape_check(n_sample, condition, self.condition)
        self.eval()
        return self._sample_impl(n_sample, condition, **kwargs)

    @abstractmethod
    def _sample_impl(self, n_sample: int = 1, condition=None, **kwargs) -> torch.Tensor:
        """Actual implementation of the sampling process"""
