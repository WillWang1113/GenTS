from matplotlib import pyplot as plt
import torch
from src.dataset.electricity import Electricity
from src.dataset.ett import ETTh1, ETTh2, ETTm2
from src.dataset.exchange import Exchange
from src.dataset.traffic import Traffic
from src.evaluation.visual import imputation_visual, predict_visual
from src.model import VanillaDDPM
from src.dataset import SineND, Stocks, Energy, MuJoCo, Spiral2D
from src.dataset.physionet import Physionet
from src.evaluation import qualitative_visual
from lightning import Trainer

from src.model.diffeq.latentode.model import LatentODE
from src.model.diffusion.csdi.model import CSDI
from src.model.diffusion.diffusionts.model import DiffusionTS
from src.model.diffeq.ls4.model import LS4
# setup
# dm = SineND(seq_dim=2, seq_len=48, batch_size=64)
# model = LS4(seq_len=dm.seq_len, seq_dim=dm.seq_dim, latent_dim=5)

# # training (on CPU for example)
# trainer = Trainer(max_epochs=500, accelerator="cpu")
# trainer.fit(model, dm)

# # testing
# dm.setup("test")
# real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
# gen_data = model.sample(n_sample=len(real_data))  # [N, 64, 2]

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].plot(real_data[0, :, :])
# ax[0].set_title("Real Data")
# ax[1].plot(gen_data[0, :, :])
# ax[1].set_title("Generated Data")
# ax[0].legend()
# ax[1].legend()
# fig.savefig('generated_vs_real.png')

# visualization(real_data, gen_data, analysis="tsne")



# # import ...
# from src.evaluation import crps, mse

# Forecasting
# dm = SineND(obs_len=64, seq_len=64, seq_dim=2, batch_size=64, condition='predict')
# model = VanillaDDPM(seq_len=64, seq_dim=2, obs_len=64, condition='predict')

"""Imputation"""
obs_len = 100
dm = Traffic(select_seq_dim=[1,3,4],seq_len=100, batch_size=64, condition='predict', obs_len=obs_len, max_time=1.0)
# dm = Physionet(select_seq_dim=[4,8, 10], batch_size=64, condition='predict', obs_len=24)
# dm = SineND(missing_rate=0.2, seq_len=64, seq_dim=2, batch_size=64)
# model = LS4(seq_len=dm.seq_len, seq_dim=dm.seq_dim, condition='predict', enc_n_layers=2, enc_ff_layers=1, dec_n_layers=2, dec_ff_layers=1, prior_n_layers=2,)
model = VanillaDDPM(seq_len=dm.seq_len, seq_dim=dm.seq_dim, obs_len=dm.obs_len, condition='predict')

# training (same before)
trainer = Trainer(max_epochs=5, devices=[0])
trainer.fit(model, dm)
model.cuda()
# testing
dm.setup("test")
real_data, gen_data = [], []
data_mask, cond_mask = [], []
for batch in dm.test_dataloader():
    for k in batch:
        batch[k] = batch[k].cuda()
    real_data.append(batch['seq'])  # [N, 64, 2]
    # in conditional generation, n_sample means how many inference times
    gen_data.append(model.sample(n_sample=10, condition=batch['c'], **batch))  # [N, 64, 2, 100]
    data_mask.append(batch['data_mask'])
    cond_mask.append(~torch.isnan(batch['c']))
real_data = torch.concat(real_data).detach().cpu()
gen_data = torch.concat(gen_data).detach().cpu()
# gen_data = torch.concat([real_data[:,:dm.obs_len], gen_data], dim=1)
data_mask = torch.concat(data_mask).detach().cpu()
cond_mask = data_mask.clone()
cond_mask[:, dm.obs_len:] = 0.0
print(gen_data.shape)
print(real_data.shape)
print()
predict_visual(real_data, gen_data, data_mask, save_root=f'./test_{dm.condition}.png')

# fig, ax = plt.subplots(1)
# ax.plot(real_data[0, :, :], label="Real Data")
# ax.plot(gen_data[0, :, :, 0], label="Generated Data")
# ax.set_title("Real Data")
# fig.savefig('generated_vs_real.png')

# visualization with tsne
# print(crps(real_data, gen_data))