from lightning import Trainer

# from src.model import VanillaVAE
from src.model.vanillagan import VanillaGAN
from src.model.vanillavae import VanillaVAE

# from src.model.timevae import TimeVAE
# from src.model.timegan import TimeGAN
from src.data.dataloader import TSDataModule

# import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)
import matplotlib.pyplot as plt

hparams = dict(
    seq_len=96,
    seq_dim=1,
    latent_dim=128,
    beta=1e-5,
    lr=1e-4,
    weight_decay=1e-4,
    hidden_size_list=[64, 128, 256],
    trend_poly=2,
    custom_seas=[(4, 96 // 4), (2, 96 // 2)],
    hidden_size=128,
)
# mc = ModelCheckpoint(
#     save_top_k=1,
#     dirpath=save_dir,
#     monitor="val_loss",
#     save_last=True,
# )
# b = VanillaVAE()
# model = TimeVAE(**hparams)
# model = TimeGAN(**hparams)
# model = TimeGAN(**hparams)
# model = VanillaGAN(**hparams)
model = VanillaVAE(**hparams)
dm = TSDataModule("test_data/", 32, 96)
trainer = Trainer(devices=[1], max_epochs=50)
trainer.fit(model, dm)

samples = model.sample(10, None)

print(samples.shape)

plt.plot(samples[0].flatten().cpu())
plt.savefig("test.png")
