from matplotlib import pyplot as plt
import numpy as np
import torch

# from gents.dataset.ecg import ECG
from gents.dataset import Spiral2D
from gents.model import VanillaDDPM
from gents.dataset import SineND
from gents.evaluation import tsne_visual
from lightning import Trainer, seed_everything

from gents.model.flow.vanillamaf.model import VanillaMAF
from gents.model.gan.rcgan.model import RCGAN
from gents.model.gan.vanillagan.model import VanillaGAN
from gents.model.vae.timevqvae.model import TimeVQVAE
from gents.model.vae.vanillavae.model import VanillaVAE

seed_everything(9)
# setup dataset and model
dm = Spiral2D(
    # num_samples = 5000,
    seq_len=24,
    batch_size=64,
    # data_dir=".data",
    condition="class",
)
# model = VanillaVAE(
#     seq_len=dm.seq_len,
#     seq_dim=dm.seq_dim,
#     class_num=2,
#     condition="class",
# )
model = TimeVQVAE(
    seq_len=dm.seq_len,
    seq_dim=dm.seq_dim,
    class_num=2,
    condition="class",
    # cfg_scale=1.0,
)

# training (on CPU for example)
trainer = Trainer(max_steps=3000, devices=[0])
trainer.fit(model, dm)

# testing
dm.setup("fit")

# test_class = 0
real_data = torch.cat([batch["seq"] for batch in dm.train_dataloader()])  # [N, 64, 2]
real_class = torch.cat([batch["c"] for batch in dm.train_dataloader()])  # [N, 2]
# real_cond = torch.cat([batch["c"] for batch in dm.train_dataloader()])  # [N, 2]

# real_data = real_data[real_class == test_class]
gen_data = model.sample(n_sample=len(real_data), condition=real_class)  # [N, 64, 2]

print(real_data.shape, gen_data.shape)
# visualization with tsne
tsne_visual(real_data.cpu().numpy(), gen_data.cpu().numpy(), real_class.cpu(), save_root="tsne.png")

fig, axs = plt.subplots(2)
axs[0].plot(real_data[0, :, :])
axs[1].plot(gen_data[0, :, :])
fig.savefig("test_pred_x0_false.png")
