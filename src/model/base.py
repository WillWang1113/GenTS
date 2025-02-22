from abc import ABC, abstractmethod
from lightning import LightningModule
import torch
from torch.nn import functional as F


class BaseModel(ABC, LightningModule):
    @torch.no_grad()  # wrap with torch.no_grad()
    def sample(self, *args, **kwargs):
        self.eval()
        # actual implementation
        return self._sample_impl(*args, **kwargs)

    @abstractmethod
    def _sample_impl(self, *args, **kwargs):
        raise NotImplementedError()


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
