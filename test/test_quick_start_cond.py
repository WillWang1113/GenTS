from matplotlib import pyplot as plt
import torch
from gents.dataset.energy import Energy
from gents.dataset.simple import Spiral2D
from gents.dataset.stocks import Stocks
from gents.evaluation.visualization.visual import predict_visual
from gents.model import VanillaDDPM
from gents.dataset import SineND
from gents.evaluation import tsne_visual
from lightning import Trainer

from gents.model.diffeq.latentode.model import LatentODE
from gents.model.diffeq.ls4.model import LS4
from gents.model.vae.kovae.model import KoVAE
from gents.model import GTGAN, TMDM, ImagenTime

# setup dataset and model
dm = Spiral2D(
    obs_len=24,
    seq_len=24,
    # num_samples=100,
    batch_size=64,
    # irregular_dropout=0.2,
    condition="predict",
)
model = LS4(
    # latent_dim=128,
    # d_model=128,
    # d_ff=1024,
    # delay=3,
    # embedding=16,
    obs_len=dm.obs_len,
    seq_len=dm.seq_len,
    seq_dim=dm.seq_dim,
    # ch_mult=[1,2],
    # attn_resolution=[4,2],
    condition="predict",
)

# training (on CPU for example)
trainer = Trainer(max_epochs=100)
trainer.fit(model, dm)
   

# testing
dm.setup("test")
real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
real_data_mask = torch.cat(
    [batch["data_mask"] for batch in dm.test_dataloader()]
)  # [N, 64, 2]
t = torch.cat([batch["t"] for batch in dm.test_dataloader()])  # [N, 64, 2]
cond_data = torch.cat([batch["c"] for batch in dm.test_dataloader()])  # [N, 64, 2]
# model.cuda()
gen_data = model.sample(
    n_sample=10,
    condition=cond_data,
    # condition=real_data.masked_fill(~real_data_mask, float("nan"))[[3]],
    t=t[0],
    data_mask=real_data_mask,
    # seq=real_data
)  # [N, 64, 2]

print(real_data.shape)
print(gen_data.shape)
# visualization with tsne
# qualitative_visual(real_data, gen_data, analysis="tsne", save_root="tsne.png")

# fig, axs = plt.subplots(2)
# axs[0].plot(real_data.masked_fill(~real_data_mask, float("nan"))[3, :, :].cpu())
# axs[1].plot(gen_data[0, ..., 0].cpu())
# axs[1].plot(gen_data[0, ..., 1].cpu())
# axs[1].plot(gen_data[0, ..., 2].cpu())
# fig.savefig("test_pred_x0_false.png")

predict_visual(real_data, gen_data, real_data_mask, save_root='test_TMDM.png')