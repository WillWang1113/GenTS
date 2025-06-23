from matplotlib import pyplot as plt
import torch
from src.dataset.ecg import ECG
from src.model import VanillaDDPM
from src.dataset import SineND
from src.evaluation import qualitative_visual
from lightning import Trainer

from src.model.flow.vanillamaf.model import VanillaMAF
from src.model.gan.rcgan.model import RCGAN
from src.model.gan.vanillagan.model import VanillaGAN
from src.model.vae.timevqvae.model import TimeVQVAE
from src.model.vae.vanillavae.model import VanillaVAE

# setup dataset and model
dm = ECG(batch_size=32, condition='class')
model = VanillaDDPM(
    patch_size=14,
    latent_dim=128,
    seq_len=dm.seq_len,
    seq_dim=dm.seq_dim, class_num=dm.n_classes, condition='class', pred_x0=False
)

# training (on CPU for example)
trainer = Trainer(max_epochs=100, devices=[0])
trainer.fit(model, dm)

# testing
dm.setup("test")

test_class = 2
real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
real_class = torch.cat([batch["c"] for batch in dm.test_dataloader()])  # [N, 2]

real_data = real_data[real_class == test_class]
gen_data = model.sample(n_sample=len(real_data), condition=test_class)  # [N, 64, 2]

print(real_data.shape, gen_data.shape)
# visualization with tsne
# qualitative_visual(real_data, gen_data, analysis="tsne", save_root="tsne.png")

fig, axs = plt.subplots(2)
axs[0].plot(real_data[0, :, :])
axs[1].plot(gen_data[0, :, :])
fig.savefig("test_pred_x0_false.png")
