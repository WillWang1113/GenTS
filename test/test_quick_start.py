from matplotlib import pyplot as plt
import torch
from src.model import VanillaDDPM
from src.dataset import SineND, Stocks, Energy, MuJoCo
from src.evaluation import visualization
from lightning import Trainer

# setup
dm = MuJoCo(seq_len=200, select_seq_dim=[4,8, 10], batch_size=64)
model = VanillaDDPM(seq_len=200, seq_dim=dm.seq_dim)

# training (on CPU for example)
trainer = Trainer(max_epochs=100, accelerator="cpu")
trainer.fit(model, dm)

# testing
dm.setup("test")
real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
gen_data = model.sample(n_sample=len(real_data))  # [N, 64, 2]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(real_data[0, :, :])
ax[0].set_title("Real Data")
ax[1].plot(gen_data[0, :, :])
ax[1].set_title("Generated Data")
ax[0].legend()
ax[1].legend()
fig.savefig('generated_vs_real.png')

visualization(real_data, gen_data, analysis="tsne")



# # import ...
# from src.evaluation import crps, mse

# # Forecasting
# dm = SineND(obs_len=64, seq_len=64, seq_dim=2, batch_size=64, condition='predict')
# model = VanillaDDPM(seq_len=64, seq_dim=2, obs_len=64, condition='predict')

# """Imputation"""
# # dm = SineND(missing_rate=0.2, seq_len=64, seq_dim=2, batch_size=64)
# # model = VanillaDDPM(seq_len=64, seq_dim=2, obs_len=64, condition='impute')

# # training (same before)
# trainer = Trainer(max_epochs=100, accelerator="cpu")
# trainer.fit(model, dm)

# # testing
# dm.setup("test")
# real_data, gen_data = [], []

# for batch in dm.test_dataloader():
#     print(batch['seq'].shape, batch['c'].shape)  # [N, 64, 2], [N, 64, 2]
    
# for batch in dm.test_dataloader():
#     real_data.append(batch['seq'][:,-64:])  # [N, 64, 2]

#     # in conditional generation, n_sample means how many inference times
#     gen_data.append(model.sample(n_sample=5, condition=batch['c']))  # [N, 64, 2, 100]
# real_data = torch.concat(real_data)
# gen_data = torch.concat(gen_data)


# fig, ax = plt.subplots(1)
# ax.plot(real_data[0, :, :], label="Real Data")
# ax.plot(gen_data[0, :, :, 0], label="Generated Data")
# ax.set_title("Real Data")
# fig.savefig('generated_vs_real.png')

# # visualization with tsne
# # print(crps(real_data, gen_data))