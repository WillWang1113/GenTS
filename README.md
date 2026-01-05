# GenTS: A Comprehensive Benchmark Library for Generative Time Series Models

[[Doc]](https://willwang1113.github.io/GenTS/)


`GenTS` is an open-source library for researchers to benchmark generative time series models, which covers both unconditional and conditional generation tasks, i.e. **(class label-guided) synthesis**, **forecasting** and **imputation**. 

Based on `torch` and `lightning`, `GenTS` provides a modular code base for benchmarking generative time series models on various datasets in a unified way. Specifically, we feature on:

- **Multi-domain Datasets**: Various time series datasets from energy, health, and other domains.
- **State-of-the-art Models**: Generative time series models from top conferences in 5 recent years, including Diffusions, VAEs, GANs, etc.
- **Panoramic Evaluation**: Model-free and model-based metrics are provided, along with supportive functions for visualization.


## Installation
We recommand to first create a virtual environment, and activate the environment. Then you can install the necessary libraries by running the following command.
```bash
conda create -n gents python=3.10
conda activate gents
pip install -r requirements.txt
pip install -e .
```

## Quick start
A minimal example of (unconditional) time series generation:
```python
import torch
from gents.model import VanillaDDPM
from gents.dataset import SineND
from gents.evaluation import tsne_visual
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

We also provide tutorials for different use case of generative time series models, including common conditional generation tasks (forecasting, imputation, class label-guided synthesis) and others. Please refer to `tutorials/` or our documents for detailed examples.


## Model zoo
GenTS included 25+ state-of-the-art generative time series models (and their variants), with different capabilities. Our model zoo will be  updated in a regular basis. 


| Name                  | Model Type | Synthesis          | Forecasting        | Imputation         | Class label        |
| --------------------- | ---------- | ------------------ | ------------------ | ------------------ | ------------------ |
| VanillaVAE            | VAE        | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| TimeVAE               | VAE        | :white_check_mark: |                    |                    |                    |
| TimeVQVAE             | VAE        | :white_check_mark: |                    |                    | :white_check_mark: |
| KoVAE                 | VAE        | :white_check_mark: |                    |                    |                    |
| VanillaGAN            | GAN        | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| TimeGAN               | GAN        | :white_check_mark: |                    |                    |                    |
| GT-GAN                | GAN        | :white_check_mark: |                    |                    |                    |
| PSA-GAN       | GAN        | :white_check_mark: |                    |                    |                    |
| RCGAN                 | GAN        | :white_check_mark: |                    |                    | :white_check_mark: |
| VanillaMAF            | Flow       | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Fourier Flow     | Flow       | :white_check_mark: |                    |                    |                    |
| VanillaDDPM           | Diffusion  | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| CSDI                  | Diffusion  |                    | :white_check_mark: | :white_check_mark: |                    |
| Diffusion-TS          | Diffusion  | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |
| TMDM                  | Diffusion  |                    | :white_check_mark: |                    |                    |
| FourierDiffusion      | Diffusion  | :white_check_mark: |                    |                    |                    |
| ImagenTime            | Diffusion  | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |
| FIDE         | Diffusion  | :white_check_mark: |                    |                    |                    |
| Latent ODE w. ODE-RNN | Diff. Eq.  | :white_check_mark: | :white_check_mark: | interpolation      |                    |
| Latent ODE w. RNN     | Diff. Eq.  | :white_check_mark: | :white_check_mark: | interpolation      |                    |
| Latent SDE            | Diff. Eq.  | :white_check_mark: |                    |                    |                    |
| SDEGAN                | Diff. Eq.  | :white_check_mark: |                    |                    |                    |
| LS4                   | Diff. Eq.  | :white_check_mark: | :white_check_mark: | interpolation      |                    |


## Datasets
GenTS preset over 10 widely used time series generation datasets, from multiple domains and resoulutions. Some of them come naturally with missing values and class labels, supporting to benchmark different kinds of models.

| Name        | Resolution     | Dimension | Missing value      | Class label | Domain      |
| ----------- | -------------- | --------- | ------------------ | ----------- | ----------- |
| SineND      | continuous     | N         | -                  | -           | Physics     |
| Spiral2D    | continuous     | 2         | -                  | 2           | Physics     |
| Stocks      | 1 day          | 6         | -                  | -           | Financial   |
| Energy      | 10 min         | 28        | -                  | -           | Energy      |
| ETT         | 1 hour/15 min  | 7         | -                  | -           | Energy      |
| Electricity | 1 hour         | 321       | -                  | -           | Energy      |
| Traffic     | 1 hour         | 862       | -                  | -           | Traffic     |
| Exchange    | 1 day          | 8         | -                  | -           | Financial   |
| MoJoCo      | continuous     | 14        | -                  | -           | Physics     |
| Physionet   | 1 min - 1 hour | 35        | :white_check_mark: | 2           | Healthcare  |
| ECG         | ~700 Hz        | 1         | -                  | 5           | Healthcare  |
| Air quality | 1 hour         | 6         | :white_check_mark: | -           | Environment |
| Weather     | 10 min         | 6         | -                  | -           | Environment |



## Custormization
Our `BaseModel` and datamodules  `BaseData` are based on `lightning`,

One can easily join `GenTS` following the `lightning.LightningModule` and `lightning.DataModule`. Please refer to our document's [tutorial on custormization](https://willwang1113.github.io/GenTS/tutorials/customization.html) for details, or [this website](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html).



## Citation
If you enjoy or benefit from using `GenTS`, a citation to this repository will be greatly appreciated.
```BibTeX
@misc{wang2025a,
title={GenTS: A Comprehensive Benchmark Library for Generative Time Series Models},
author={Chenxi Wang},
year={2025},
url={https://github.com/WillWang1113/GenTS}
}
```
