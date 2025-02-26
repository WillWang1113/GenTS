from abc import ABC, abstractmethod
from lightning import LightningModule
import torch
from torch.nn import functional as F


class BaseModel(ABC, LightningModule):
    
    @torch.no_grad()  # wrap with torch.no_grad()
    def sample(self, n_sample: int = 1, condition=None, **kwargs):
        """ Generate samples from the generative model """
        self.eval()
        return self._sample_impl(n_sample, condition, **kwargs)

    @abstractmethod
    def _sample_impl(self, n_sample: int = 1, condition=None, **kwargs) -> torch.Tensor:
        """ Actual implementation of the sampling process """
        


# class BaseVAE(BaseModel):

#     @abstractmethod
#     def encode(self, x, c=None):
#         raise NotImplementedError()

#     @abstractmethod
#     def decode(self, z, c=None):
#         raise NotImplementedError()


# class BaseGAN(BaseModel):
#     pass


# class BaseDiffusion(BaseModel):
#     @abstractmethod
#     def degrade(self, x):
#         raise NotImplementedError()
