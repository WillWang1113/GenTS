from lightning import Trainer

# from src.model import VanillaVAE
from src.model.vanillagan import VanillaGAN
from src.model.vanillavae import VanillaVAE
from src.model.vanilladiffusion import VanillaDDPM

# from src.model.timevae import TimeVAE
# from src.model.timegan import TimeGAN
from src.data.dataloader import TSDataModule
from torch.utils.data import DataLoader, Dataset


# import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)
import matplotlib.pyplot as plt

batch_size = 32
hparams = dict(
    seq_len=96,
    seq_dim=1,
    latent_dim=128,
    beta=1e-4,
    lr=1e-3,
    weight_decay=1e-4,
    hidden_size_list=[512, 512, 1024],
    trend_poly=2,
    custom_seas=[(4, 96 // 4), (2, 96 // 2)],
    hidden_size=128,
    n_critic=1,
    noise_schedule="cosine",
    T=200,
    pred_x0=True,
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
model = VanillaDDPM(**hparams)
# model = VanillaVAE(**hparams)
dm = TSDataModule("test_data/", batch_size, 96)
# dm.setup('test')
# dl = DataLoader(dm.test_ds, batch_size=max(batch_size, 512))
# batch = next(iter(dl))['seq']

trainer = Trainer(devices=[0], max_epochs=200)
trainer.fit(model, dm)

# print(next(iter(model.parameters())).dtype)
samples = model.sample(5, None)

# print(samples)


plt.plot(samples.cpu().squeeze().T)
# plt.plot(samples[1].flatten().cpu())
# plt.plot(batch[0].flatten().cpu())
plt.savefig(f"{model._get_name()}.png")
