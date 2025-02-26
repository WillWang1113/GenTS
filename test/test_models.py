from lightning import Trainer
import torch

# from src.model import VanillaVAE
# from src.model.neuralode_old import NeuralODE
from src.model.diffeq.neuralode import NeuralODE
from src.model.gan.timegan import TimeGAN
from src.model.vae.timevae import TimeVAE

# from src.model.timewgan import TimeWGAN
from src.model.gan.vanillagan import VanillaGAN
from src.model.flow.vanillamaf import VanillaMAF
from src.model.vae.vanillavae import VanillaVAE
from src.model.diffusion.vanilladiffusion import VanillaDDPM
from src.model.diffusion.fourierdiffusion import FourierDiffusionMLP, FourierDiffusion
from torchdiffeq import odeint
from src.model.diffusion._scheduler import VPScheduler
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
seq_len = 96
hparams = dict(
    seq_len=seq_len,
    seq_dim=1,
    # num_training_steps=100,
    # noise_scheduler=VPScheduler(),
)

# MY CODE
# hparams = dict(
#     seq_len=seq_len,
#     seq_dim=1,
#     latent_dim=8,
#     beta=1e-1,
#     lr=1e-3,
#     eta=100,
#     gamma=1,
#     weight_decay=1e-4,
#     hidden_size_list=[32],
#     # hidden_size_list=[64, 128, 256],
#     trend_poly=1,
#     custom_seas=[(4, 96 // 4), (2, 96 // 2)],
#     hidden_size=96,
#     n_critic=2,
#     # ode_method="rk4",
#     noise_schedule="cosine",
#     T=200,
#     pred_x0=True,
#     num_layers=1,
# )


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
# model = TimeGAN(**hparams)
# model = NeuralODE(**hparams)
model = FourierDiffusion(**hparams)
# model = NeuralODE.load_from_checkpoint(
#     "lightning_logs/version_3/checkpoints/epoch=99-step=200.ckpt"
# )
dm = TSDataModule("test_data/", batch_size, seq_len)
# dm.prepare_data()
# dm.setup("fit")
# print(batch)
trainer = Trainer(devices=[1], max_epochs=100)
trainer.fit(model, dm)

# for name, param in model.named_parameters():
# print(name, "\t", param.shape, "\t", param.dtype, "\t", param.device)
# print(next(iter(model.parameters())).dtype)
# model.to("cuda:1")

samples = model.sample(n_sample=3, )

plt.plot(samples.cpu().squeeze().T)
plt.savefig(f"{model._get_name()}.png")
