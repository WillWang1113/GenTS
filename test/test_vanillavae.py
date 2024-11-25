from lightning import Trainer
from src.model.vanillavae import VanillaVAE
from src.model.timevae import TimeVAE
from src.data.dataloader import TSDataModule
import torch


hparams = dict(
    seq_len=96,
    seq_dim=3,
    latent_dim=128,
    beta=0.001,
    lr=0.001,
    weight_decay=0.00001,
    hidden_size_list=[64, 128, 256],
    trend_poly=2, custom_seas=[(4, 96//4), (2, 96//2)]
    
)
model = TimeVAE(**hparams)
dm = TSDataModule("test_data/", 32, 96)
trainer = Trainer(devices=[0], max_epochs=3)
trainer.fit(model, dm)
