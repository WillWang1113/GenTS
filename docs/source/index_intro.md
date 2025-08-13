# GenTS: a Library for Generative Time Series Analysis

`GenTS` is an open-source library for researchers to conduct generative time series analysis, which covers **(class) synthesis**, **forecasting** and **imputation**. 

Based on `lightning`, `GenTS` provides a modular code base to benchmark different models on various datasets in a easy way. Specifically, we feature on:

- **State-of-the-art Models**: Generative time series models from top conferences in 5 recent years, including Diffusions, Flows, GANs, etc.
- **Multi-domain Datasets**: Various time series datasets from energy, health, and other domains.
- **Comprehensive Evaluation**: Both quantitative and qualitative methods for fidelity, usefulness, and accuracy.


## Installation
We recommand to first create a virtual environment, and activate the environment. Then you can install the necessary libraries by running the following command.
```bash
conda create -n gents python=3.10
conda activate gents
pip install -r requirements.txt
```

## Quick start
A minimal example of (unconditional) time series synthesis:
```python
import torch
from src.model import VanillaDDPM
from src.dataset import SineND
from src.evaluation import tsne_visual
from lightning import Trainer

# setup dataset and model
dm = SineND(seq_len=64, seq_dim=2, batch_size=64)
model = VanillaDDPM(seq_len=dm.seq_len, seq_dim=dm.seq_dim)

# training (on CPU for example)
trainer = Trainer(max_epochs=100, accelerator="cpu")
trainer.fit(model, dm)

# testing
dm.setup("test")
real_data = torch.cat([batch["seq"] for batch in dm.test_dataloader()])  # [N, 64, 2]
gen_data = model.sample(n_sample=len(real_data))  # [N, 64, 2]

# visualization with tsne
tsne_visual(real_data, gen_data, save_root="tsne.png")
```

We also support for conditional time series generation, including forecasting, imputation, and class generation. Please refer to `tutorials/` for detailed examples.

## Citation
If you enjoy or benefit from using `GenTS`, a citation to this repository will be greatly appreciated.
```BibTeX
@misc{wang2025a,
title={GenTS: a Library for Generative Time Series Analysis},
author={Chenxi Wang},
year={2025},
url={https://github.com/WillWang1113/GenTS}
}
```
