from matplotlib import pyplot as plt
import torch
from src.dataset.energy import Energy
from src.dataset.stocks import Stocks
from src.model import VanillaDDPM
from src.dataset import SineND
from src.evaluation import qualitative_visual
from lightning import Trainer

from src.model.vae.kovae.model import KoVAE
from src.model import GTGAN

# setup dataset and model
dm = Stocks(
    seq_len=24,
    select_seq_dim=[1, 2, 3],
    batch_size=64,
    irregular_dropout=0.2,
    add_coeffs="cubic_spline",
)
model = GTGAN(latent_dim=128, seq_len=dm.seq_len, seq_dim=dm.seq_dim)

# training (on CPU for example)
trainer = Trainer(max_epochs=20, devices=[0])
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
