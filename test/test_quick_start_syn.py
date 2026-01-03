from matplotlib import pyplot as plt
import numpy as np
import torch
# from gents.dataset.energy import Energy
from gents.dataset.modules.stocks import Stocks
from gents.dataset.modules.ecg import ECG
# from gents.dataset.stocks import Stocks
from gents.dataset.modules.weather import Weather
from gents.evaluation.model_based.ds import discriminative_score
from gents.model import VanillaDDPM
from gents.dataset import SineND
from gents.evaluation import tsne_visual
from lightning import Trainer, seed_everything

from gents.model.diffeq.latentsde.model import LatentSDE
from gents.model.diffeq.ls4.model import LS4
from gents.model.vae.kovae.model import KoVAE
from gents.model import GTGAN
from gents.model.vae.timevae.model import TimeVAE

seed_everything(9)
# setup dataset and model
dm = SineND(
    seq_len=48,
    seq_dim=2,
    select_seq_dim=None,
    batch_size=64,
    # irregular_dropout=0.2,
    # add_coeffs="cubic_spline",
)
model = LS4(seq_len=dm.seq_len, seq_dim=dm.seq_dim, enc_pool=[1], dec_pool=[1], prior_pool=[1])

# training (on CPU for example)
# trainer = Trainer(max_epochs=1, devices=[0])
# trainer.fit(model, dm)

# testing
dm.setup("fit")

for batch in dm.train_dataloader():
    print(batch['t'].shape)
    print(batch['t'][0])
    break

# real_data = torch.cat([batch["seq"] for batch in dm.train_dataloader()])  # [N, 64, 2]
# gen_data = model.sample(n_sample=len(real_data))  # [N, 64, 2]

# tsne_visual(real_data.numpy(), gen_data.numpy(), save_root="tsne.png")
# d_score = discriminative_score(real_data.numpy(), gen_data.numpy(), device="cpu")
# print(d_score)
# fig, axs = plt.subplots(2)
# axs[0].plot(real_data[0, :, :])
# axs[1].plot(gen_data[1, :, :])
# fig.savefig("test_pred_x0_false.png")
