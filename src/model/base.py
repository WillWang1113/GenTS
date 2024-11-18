from abc import ABC, abstractmethod
from lightning import LightningModule


class BaseModel(ABC, LightningModule):
    
    @abstractmethod
    def sample(self, n_sample):
        raise NotImplementedError()