from abc import ABC, abstractmethod
from lightning import LightningModule
import torch
from torch.nn import functional as F


class BaseModel(ABC, LightningModule):
    @abstractmethod
    def sample(self, n_sample, c=None):
        raise NotImplementedError()
    

class BaseVAE(BaseModel):

    @abstractmethod
    def encode(self, x, c=None):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, z, c=None):
        raise NotImplementedError()


class BaseGAN(BaseModel):
    pass


class BaseDiffusion(BaseModel):
    @abstractmethod
    def degrade(self, x):
        raise NotImplementedError()
