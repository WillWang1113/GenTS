from matplotlib import pyplot as plt
import torch
from src.model import VanillaDDPM
from src.dataset import SineND
from src.evaluation import qualitative_visual
from lightning import Trainer

# setup dataset and model
dm = SineND(seq_len=64, seq_dim=2, batch_size=32)
model = VanillaDDPM(
    seq_len=64, seq_dim=2, pred_x0=False, noise_schedule="cosine", n_diff_steps=1000
)

# training (on CPU for example)
trainer = Trainer(max_epochs=300, devices=[0])
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
