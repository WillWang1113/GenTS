from math import sqrt
from lightning import Trainer
import src.model
from src.data.dataloader import SineDataModule
import matplotlib.pyplot as plt

model_names = src.model.__all__
# TODO: iter all, Model Capability
conditions = ["synthesis", "predict", "impute"]
batch_size = 64
seq_len = 96
dm = SineDataModule(seq_len, batch_size)
# dm.setup('fit')



# MY CODE
hparams = dict(
    seq_len=seq_len,
    seq_dim=1,
    lr=1e-3,
)
n_row = int(sqrt(len(model_names)))
n_col = len(model_names) // n_row + 1
fig, axs = plt.subplots(n_row, n_col, figsize=[2 * n_col, 2 * n_row], sharex=True)
axs = axs.flatten()
for i in range(len(model_names)):
    test_model_cls = getattr(src.model, model_names[i])
    test_model = test_model_cls(**hparams)
    trainer = Trainer(devices=[1], max_epochs=10)
    trainer.fit(test_model, dm)
    samples = test_model.sample(3).squeeze().T.cpu().numpy()
    axs[i].plot(samples)
    axs[i].set_title(model_names[i])
fig.suptitle("Model Comparison")
fig.tight_layout()
fig.savefig("test_model.png", bbox_inches='tight')
