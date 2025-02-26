from lightning import Trainer
import torch

# from src.model import VanillaVAE
# from src.model.neuralode_old import NeuralODE
from src.model.diffeq.neuralode import NeuralODE
from src.model.timegan import TimeGAN
from src.model.timevae import TimeVAE

# from src.model.timewgan import TimeWGAN
from src.model.vanillagan import VanillaGAN
from src.model.flow.vanillamaf import VanillaMAF
from src.model.vanillavae import VanillaVAE
from src.model.diffusion.vanilladiffusion import VanillaDDPM
from torchdiffeq import odeint

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

batch_size = 512
seq_len = 100
hparams = dict(
    seq_len=seq_len,
    seq_dim=1,
    latent_dim=8,
    beta=1e-1,
    lr=1e-3,
    eta=100,
    gamma=1,
    weight_decay=1e-4,
    hidden_size_list=[32],
    # hidden_size_list=[64, 128, 256],
    trend_poly=1,
    custom_seas=[(4, 96 // 4), (2, 96 // 2)],
    hidden_size=96,
    n_critic=2,
    # ode_method="rk4",
    noise_schedule="cosine",
    T=200,
    pred_x0=True,
    num_layers=1,
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
# model = TimeGAN(**hparams)
# model = NeuralODE(**hparams)
model = NeuralODE.load_from_checkpoint(
    "lightning_logs/version_3/checkpoints/epoch=99-step=200.ckpt"
)
dm = TSDataModule("test_data/", batch_size, seq_len)
# dm.prepare_data()
# dm.setup("fit")
# print(batch)
trainer = Trainer(devices=[1], max_epochs=100)
trainer.fit(model, dm)

# for name, param in model.named_parameters():
# print(name, "\t", param.shape, "\t", param.dtype, "\t", param.device)
# print(next(iter(model.parameters())).dtype)
model.to("cuda:1")
model.eval()
batch = next(iter(dm.train_ds))
# dm.setup("test")
# dl = DataLoader(dm.train_ds, batch_size=1, shuffle=False)
# with torch.no_grad():
#     for batch in dl:
#         batch = {k: v.to("cuda:1") for k, v in batch.items()}
# t = torch.linspace(0, 2.4 * torch.pi, seq_len).to("cuda:1")
# latents, mu, logvar = model.encode(batch["seq"].to("cuda:1").unsqueeze(0))
# z = model.reparam(mu, logvar)

# samples = odeint(
#     model.ode_fn,
#     z,
#     t,
#     options={"dtype": torch.float32},
# ).permute(1, 0, 2)
# samples = model.decoder(samples)
#         break
t = torch.linspace(0, 4 * torch.pi, 1000)
samples = model.sample(3, None, t)

print(samples.shape)

t =t.cpu()
plt.plot(t.cpu(), samples.detach().cpu().squeeze().T)
plt.plot(
    torch.linspace(0, 1.2 * torch.pi, seq_len), batch["seq"].cpu().squeeze().T, lw=3
)
plt.plot(t, torch.cos(t) + torch.sin(t * 2), lw=3, ls="--")
# plt.plot(samples[1].flatten().cpu())
# plt.plot(batch[0].flatten().cpu())
plt.savefig(f"{model._get_name()}.png")
