from matplotlib import pyplot as plt
import torch
from src.dataset.energy import Energy
from src.dataset.sine import Spiral2D
from src.dataset.stocks import Stocks
from src.model import VanillaDDPM
from src.dataset import SineND
from src.evaluation import tsne_visual
from lightning import Trainer

from src.model.diffeq.latentode.model import LatentODE
from src.model.diffeq.ls4.model import LS4
from src.model.vae.kovae.model import KoVAE
from src.model import GTGAN

# setup dataset and model
dm = Spiral2D(
    # obs_len=24,
    seq_len=24,
    # num_samples=100,
    batch_size=64,
    irregular_dropout=0.2,
    # condition="predict",
)
model = LS4(
    latent_dim=128,
    # obs_len=dm.obs_len,
    seq_len=dm.seq_len,
    seq_dim=dm.seq_dim,
    condition="impute",
)
for b in model.named_buffers():
    print(b[0], b[1].device)

# training (on CPU for example)
trainer = Trainer(max_epochs=1, devices=[0])
trainer.fit(model, dm)
   
for b in model.named_buffers():
    print(b[0], b[1].device)
# testing
dm.setup("test")
real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
real_data_mask = torch.cat(
    [batch["data_mask"] for batch in dm.test_dataloader()]
)  # [N, 64, 2]
t = torch.cat([batch["t"] for batch in dm.test_dataloader()])  # [N, 64, 2]

# model.cuda()
gen_data = model.sample(
    n_sample=10,
    condition=real_data.masked_fill(~real_data_mask, float("nan"))[[3]],
    t=t,
    data_mask=real_data_mask[[3]],
)  # [N, 64, 2]

# visualization with tsne
# qualitative_visual(real_data, gen_data, analysis="tsne", save_root="tsne.png")

fig, axs = plt.subplots(2)
axs[0].plot(real_data.masked_fill(~real_data_mask, float("nan"))[3, :, :].cpu())
axs[1].plot(gen_data[0, ..., 0].cpu())
axs[1].plot(gen_data[0, ..., 1].cpu())
axs[1].plot(gen_data[0, ..., 2].cpu())
fig.savefig("test_pred_x0_false.png")
