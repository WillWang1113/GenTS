from matplotlib import pyplot as plt
import numpy as np
import torch
# from gents.dataset.energy import Energy
from gents.dataset.modules.ecg import ECG
# from gents.dataset.stocks import Stocks
from gents.dataset.modules.weather import Weather
from gents.model import VanillaDDPM
from gents.dataset import SineND
from gents.evaluation import tsne_visual
from lightning import Trainer

from gents.model.diffeq.latentsde.model import LatentSDE
from gents.model.vae.kovae.model import KoVAE
from gents.model import GTGAN

# setup dataset and model
dm = Weather(
    seq_len=24,
    select_seq_dim=np.arange(16).tolist(),
    batch_size=64,
    # irregular_dropout=0.2,
    # add_coeffs="cubic_spline",
)
model = LatentSDE(seq_len=dm.seq_len, seq_dim=dm.seq_dim)

# training (on CPU for example)
trainer = Trainer(max_epochs=10, gradient_clip_val=1.0, devices=[0])
trainer.fit(model, dm)

# testing
dm.setup("test")
real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
gen_data = model.sample(n_sample=len(real_data))  # [N, 64, 2]

# visualization with tsne
# qualitative_visual(real_data, gen_data, analysis="tsne", save_root="tsne.png")

fig, axs = plt.subplots(2)
axs[0].plot(real_data[0, :, :])
axs[1].plot(gen_data[1, :, :])
fig.savefig("test_pred_x0_false.png")
