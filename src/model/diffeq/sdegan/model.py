from src.model.base import BaseModel


class NeuralSDE(BaseModel):
    def __init__(self, seq_len, seq_dim, condition, **kwargs):
        super().__init__(seq_len, seq_dim, condition, **kwargs)

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def _sample_impl(self, n_sample=1, condition=None, **kwargs):
        return super()._sample_impl(n_sample, condition, **kwargs)

    def configure_optimizers(self):
        return super().configure_optimizers()
