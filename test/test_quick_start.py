from matplotlib import pyplot as plt
import torch
from src.model import VanillaDDPM
from src.dataset import SineND, Stocks, Energy
from src.evaluation import visualization
from lightning import Trainer

# setup
dm = Energy(seq_len=24, select_seq_dim=[1,3], batch_size=64)
model = VanillaDDPM(seq_len=24, seq_dim=dm.seq_dim)

# training (on CPU for example)
trainer = Trainer(max_epochs=20, accelerator="cpu")
trainer.fit(model, dm)

# testing
dm.setup("test")
real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
gen_data = model.sample(n_sample=len(real_data))  # [N, 64, 2]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(real_data[0, :, :], label="Real Data")
ax[0].set_title("Real Data")
ax[1].plot(gen_data[0, :, :], label="Generated Data")
ax[1].set_title("Generated Data")
ax[0].legend()
ax[1].legend()
fig.savefig('generated_vs_real.png')

visualization(real_data, gen_data, analysis="tsne")
