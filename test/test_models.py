from lightning import Trainer

# from src.model import VanillaVAE
from src.model.timegan import TimeGAN
from src.model.timevae import TimeVAE
# from src.model.timewgan import TimeWGAN
from src.model.vanillagan import VanillaGAN
from src.model.vanillamaf import VanillaMAF
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

batch_size = 128
seq_len = 96
hparams = dict(
    seq_len=seq_len,
    seq_dim=1,
    latent_dim=128,
    beta=1e-4,
    lr=1e-3,
    eta=100,
    gamma=1,
    weight_decay=1e-4,
    hidden_size_list=[64, 128, 256],
    trend_poly=1,
    custom_seas=[(4, 96 // 4), (2, 96 // 2)],
    hidden_size=96,
    n_critic=2,
    noise_schedule="cosine",
    T=200,
    pred_x0=True,
    num_layers=3,
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
# model = VanillaDDPM(**hparams)
# model = VanillaMAF(**hparams)
# model = VanillaVAE(**hparams)
# model = TimeVAE(**hparams)
model = TimeGAN(**hparams)
dm = TSDataModule("test_data/", batch_size, seq_len)
# dm.setup('test')
# dl = DataLoader(dm.test_ds, batch_size=max(batch_size, 512))
# batch = next(iter(dl))['seq']

trainer = Trainer(devices=[1], max_epochs=5000)
trainer.fit(model, dm)

# for name, param in model.named_parameters():
    # print(name, "\t", param.shape, "\t", param.dtype, "\t", param.device)
# print(next(iter(model.parameters())).dtype)
model.to('cuda:1')
samples = model.sample(5, None)

# print(samples)


plt.plot(samples.cpu().squeeze().T)
# plt.plot(samples[1].flatten().cpu())
# plt.plot(batch[0].flatten().cpu())
plt.savefig(f"{model._get_name()}.png")
