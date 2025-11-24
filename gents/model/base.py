from abc import ABC, abstractmethod

import torch
from lightning import LightningModule


def _condition_shape_check(
    n_sample: int, condition: torch.Tensor | int, cond_type: str
):
    if n_sample < 1:
        raise ValueError("n_sample should be greater than 0.")

    if cond_type == "class":
        if isinstance(condition, int):
            condition = torch.ones(n_sample) * condition
            condition = condition.long()

        elif isinstance(condition, torch.Tensor):
            if condition.ndim == 1:
                if condition.shape[0] == n_sample:
                    pass
                else:
                    raise ValueError(
                        "The batch size of the given condition should be the same as n_sample or just 1."
                    )
            else:
                raise ValueError(
                    "Condition for classification should be a 1D tensor or an integer."
                )
        else:
            raise ValueError(
                "The batch size of the given condition should be the same as n_sample or just 1."
            )
    elif cond_type in ["predict", "impute"]:
        if condition is None:
            raise ValueError("Condition should not be None for prediction.")

    elif cond_type is None:
        if n_sample < 1:
            raise ValueError(
                "n_sample should be greater than 0 for unconditional generation."
            )
        # if condition is not None:
        #     raise ValueError(
        #         "Condition should be None for unconditional generation."
        #     )

    return condition


class BaseModel(ABC, LightningModule):
    """Base class for time series generative models in PyTorch Lightning.

    Args:
        seq_len (int): Target sequence length
        seq_dim (int): Target sequence dimension, for univariate time series, set as 1
        condition (str): Possible condition type, choose from [None, 'predict','impute', 'class']. None standards for unconditional generation.
        **kwargs: Additional arguments for the model
    """

    ALLOW_CONDITION = ...

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        condition: str,
        **kwargs,
    ):
        super().__init__()

        # check
        if condition not in self.ALLOW_CONDITION:
            raise ValueError(
                f"Condition '{condition}' not allowed. Choose from {self.ALLOW_CONDITION}"
            )
        if condition == "predict":
            obs_len = kwargs.get("obs_len")
            if obs_len is None:
                raise ValueError("obs_len should be provided for prediction.")
            elif obs_len < 0:
                raise ValueError("obs_len should be greater than 0.")
            self.obs_len = obs_len

        if condition == "class":
            class_num = kwargs.get("class_num")
            if class_num is None:
                raise ValueError("class_num should be provided for classification.")
            elif class_num < 2:
                raise ValueError("class_num should be greater than 2.")
            self.class_num = class_num

        self.condition = condition

    @torch.no_grad()  # wrap with torch.no_grad()
    def sample(
        self, n_sample: int = 1, condition: torch.Tensor | int = None, **kwargs
    ) -> torch.Tensor:
        """Sample time series from trained model in evaluation mode.

        Args:
            n_sample (int, optional): The number of samples. Defaults to 1.
            condition (torch.Tensor | int, optional): Condition tensor in shape (batch_size, seq_len or obs_len, seq_dim). If the model is class label, then could be int.  Defaults to None.
            kwargs: Additional arguments for the sampling process. E.g. data_mask, t, etc.

        Returns:
            torch.Tensor: sampled time series of shape (n_sample, seq_len, seq_dim) for unconditional generation, or (batch_size, seq_len, seq_dim, n_sample) for conditional generation.
        """
        condition = _condition_shape_check(n_sample, condition, self.condition)
        self.eval()
        return self._sample_impl(n_sample, condition, **kwargs)

    @abstractmethod
    def _sample_impl(self, n_sample: int = 1, condition=None, **kwargs) -> torch.Tensor:
        """Actual implementation of the sampling process"""

    # TODO: implement predict_step
    # TODO: also the inference setup logic
    # def predict_step(self, batch, batch_idx):
    #     if self.condition is None or self.condition == "class":
    #         return self.sample(
    #             n_sample=batch["seq"].shape[0], condition=batch.get("c", None), **batch
    #         )
    #     else:
    #         return self.sample(
    #             self.hparams.get("n_sample", 10), condition=batch["c"], **batch
    #         )

    