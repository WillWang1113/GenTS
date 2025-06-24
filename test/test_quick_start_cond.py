from matplotlib import pyplot as plt
import torch
from src.dataset.energy import Energy
from src.dataset.stocks import Stocks
from src.model import VanillaDDPM
from src.dataset import SineND
from src.evaluation import qualitative_visual
from lightning import Trainer

from src.model.diffeq.latentode.model import LatentODE
from src.model.diffeq.ls4.model import LS4
from src.model.vae.kovae.model import KoVAE
from src.model import GTGAN

# setup dataset and model
dm = Stocks(
    seq_len=24,
    select_seq_dim=[1, 2, 3],
    batch_size=64,
    irregular_dropout=0.2,
    condition="impute", missing_rate=0.1
)
model = LS4(
    latent_dim=128, seq_len=dm.seq_len, seq_dim=dm.seq_dim, condition="impute"
)

# training (on CPU for example)
trainer = Trainer(max_steps=4, devices=[0])
trainer.fit(model, dm)

# testing
dm.setup("test")
real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
real_data_mask = torch.cat(
    [batch["data_mask"] for batch in dm.test_dataloader()]
)  # [N, 64, 2]
t = torch.cat([batch["t"] for batch in dm.test_dataloader()])  # [N, 64, 2]

model.cuda()
gen_data = model.sample(
    n_sample=10,
    condition=real_data.masked_fill(~real_data_mask, float("nan")).cuda(),
    t=t.cuda(),
    data_mask=real_data_mask.cuda()
)  # [N, 64, 2]

# visualization with tsne
# qualitative_visual(real_data, gen_data, analysis="tsne", save_root="tsne.png")

fig, axs = plt.subplots(2)
axs[0].plot(real_data.masked_fill(~real_data_mask, float("nan"))[0, :, :].cpu())
axs[1].plot(gen_data[0, ..., 0].cpu())
fig.savefig("test_pred_x0_false.png")
