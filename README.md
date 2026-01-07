# GenTS: A Comprehensive Benchmark Library for Generative Time Series Models


`GenTS` is an open-source library for researchers to conduct generative time series analysis, which covers **(class label-guided) synthesis**, **forecasting** and **imputation**. 

## ✨ Highlights
Based on `lightning`, `GenTS` provides a modular code base to benchmark different models on various datasets in a easy way. Specifically, we feature on:

- **State-of-the-art Models**: Generative time series models from top conferences in 5 recent years, including Diffusions, Flows, GANs, etc.
- **Multi-domain Datasets**: Various time series datasets from energy, health, and other domains.
- **Comprehensive Evaluation**: Both quantitative and qualitative methods for fidelity, usefulness, and accuracy.

## 📖 Documents
For detailed tutorial of our library and API information, please refer to the [our documents](https://willwang1113.github.io/GenTS/).

## ⚙️ Installation
We recommand to first create a virtual environment, and activate the environment. Then you can install the necessary libraries by running the following command.
```bash
conda create -n gents python=3.10
conda activate gents
pip install -r requirements.txt
pip install -e .
```

## 🚖 Quick start
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

We also provide tutorials for different use case of generative time series models, including common conditional generation tasks (forecasting, imputation, class label-guided synthesis) and others. Please refer to tutorials/ or our documents for detailed examples.

The scripts for running benchmark experiments are located in the `test/` folder, incluing training and sampling for (class label-guided) synthesis, imputation, and forecasting. Please feel free to start with our scripts.


##  🏰 Model zoo
GenTS included 25+ state-of-the-art generative time series models (and their variants), with different capabilities. Our model zoo will be updated in a regular basis.

| 🤖 Name             | 🗂️ Model Type | Synthesis          | Forecasting        | Imputation         | Class label        |
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
| Latent ODE | Diff. Eq.  | :white_check_mark: | :white_check_mark: | :white_check_mark:      |                    |                   |
| Latent SDE            | Diff. Eq.  | :white_check_mark: |                    |                    |                    |
| SDEGAN                | Diff. Eq.  | :white_check_mark: |                    |                    |                    |
| LS4                   | Diff. Eq.  | :white_check_mark: | :white_check_mark: | :white_check_mark:      |                    |

<!-- | mr-diff               | Diffusion  |                    | :white_check_mark: | -->

<!--          |   Name    |      Model Type       |     Condition     |    Application     | Finish? |
| :--------------------: | :-------: | :-------------------: | :---------------: | :----------------: |
|       VanillaVAE       |    VAE    |           -           |        Syn        | :white_check_mark: |
|        TimeVAE         |    VAE    |           -           |        Syn        | :white_check_mark: |
|       TimeVQVAE        |    VAE    |      class label      |        Syn        | :white_check_mark: |
|         KoVAE          |    VAE    |           -           |    Syn(irreg)     | :white_check_mark: |
|       VanillaGAN       |    GAN    |           -           |        Syn        | :white_check_mark: |
|        TimeGAN         |    GAN    |           -           |        Syn        | :white_check_mark: |
|      AST **(-!)**      |    GAN    |  :white_check_mark:   |    Fcst(point)    | :white_check_mark: |
|       COSCI-GAN        |    GAN    |           -           |        Syn        | :white_check_mark: |
|         GT-GAN         |    GAN    |           -           |    Syn(irreg)     | :white_check_mark: |
| PSA-GAN **(-G)** $^1$  |    GAN    |           -           |        Syn        | :white_check_mark: |
|         RCGAN          |    GAN    |           -           |        Syn        | :white_check_mark: |
|       VanillaMAF       |   Flow    |           -           | Syn, Fcst, Impute | :white_check_mark: |
|   Fourier Flow $^1$    |   Flow    |           -           |        Syn        | :white_check_mark: |
|   LSTM-MAF **(-G)**    |   Flow    |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|      VanillaDDPM       | Diffusion |           -           |        Syn        | :white_check_mark: |
|          CSDI          | Diffusion |  :white_check_mark:   |   Fcst, Impute    | :white_check_mark: |
|      Diffusion-TS      | Diffusion |  :white_check_mark:   | Syn, Fcst, Impute | :white_check_mark: |
|          TMDM          | Diffusion |  :white_check_mark:   |       Fcst        | :white_check_mark: |
|        mr-diff         | Diffusion |  :white_check_mark:   |       Fcst        | :white_check_mark: |
|          RATD          | Diffusion |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|    FourierDiffusion    | Diffusion |           -           |        Syn        | :white_check_mark: |
|       ImagenTime       | Diffusion |  :white_check_mark:   |     Syn, Fcst     | :white_check_mark: |
|      D3M **(-M)**      | Diffusion |           -           |     Fcst, Imp     |   :white_circle:   |
|  TimeWeaver **(-M)**   | Diffusion |  :white_check_mark:   |        Syn        |   :white_circle:   |
| FTS-Diffusion **(-M)** | Diffusion |           -           |        Syn        |   :white_circle:   |
|     FIDE **(-!)**      | Diffusion |     block maxima      |        Syn        | :white_check_mark: |
|      ANT **(-G)**      | Diffusion |  :white_check_mark:   | Syn, Fcst, Refine |   :white_circle:   |
|   TimeGrad **(-G)**    | Diffusion |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|    TSDiff **(-G)**     | Diffusion | inference conditional | Syn, Fcst, Refine |   :white_circle:   |
|    MG-TSD **(-G)**     | Diffusion |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|     D3VAE **(-P)**     | Diffusion |  :white_check_mark:   |       Fcst        |   :white_circle:   |
| Latent ODE w. ODE-RNN  | Diff. Eq. |           -           |  Syn, Fcst, Imp   | :white_check_mark: |
|   Latent ODE w. RNN    | Diff. Eq. |           -           |  Syn, Fcst, Imp   | :white_check_mark: |
|       Latent SDE       | Diff. Eq. |           -           | Syn, (Fcst, Imp)  | :white_check_mark: |
|         SDEGAN         | Diff. Eq. |           -           |    Syn(irreg)     | :white_check_mark: |
|          LS4           | Diff. Eq. |           -           |        Syn        | :white_check_mark: | -->   

<!-- |          <!--          | SDformer **(-M)** |        VAE+GPT        | :white_check_mark: |        Syn         | :white_circle: | -->            
 <!--          |        TFM        |       Diff. Eq.       |         -          |        Fcst        | :white_circle: | -->           
 <!--          |       <!--        |         GANF          |        Flow        |         -          | AD             | :white_circle: | --> 




##  📝 Datasets
GenTS preset over 10 widely used time series generation datasets, from multiple domains and resoulutions. Some of them come naturally with missing values and class labels, supporting to benchmark different kinds of models.


| 🤖Name        | 🧭 Resolution     | 📦 Dimension |⭕️ Missing value      | 🏷️ Class label | 📪 Domain      |
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

<!-- ## Arena (TODO: experiments + webpage?)

Till XX 2025, the top three models for different tasks are:

| Model Rank | Synthesis | Forecasting | Imputation |
| :--------: | :-------: | :---------: | :--------: |
|    1st     |     -     |      -      |     -      |
|    2nd     |     -     |      -      |     -      |
|    3rd     |     -     |      -      |     -      |
 -->


## 🔧 Custormization
Our `BaseModel` and datamodules  `BaseData` are based on `lightning`,

One can easily join `GenTS` following the `lightning.LightningModule` and `lightning.DataModule`. Please refer to our document's [tutorial on custormization](https://willwang1113.github.io/GenTS/tutorials/customization.html) for details, or [this website](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html).


<!-- 

## TODO-list
- [x] Flow-based model (5.15)
- [x] ODE-based model (5.15)
- [x] Evaluation (5.31)
- [x] Model testing (6.15)
- [x] Benchmark datasets (6.15)
- [ ] tutorials (whole pipline for predicting, synthesis with missing values, imputation)
- [ ] doc files (dataset logic and model hyperparameter explain)
- [ ] pypi? optional
- [ ] Project webpage for benchmarking? [Example](https://huggingface.co/spaces/Salesforce/GIFT-Eval)

 -->




## 🙏 All Thanks To Our Contributors

<!-- <a href="https://github.com/WillWang1113/GenTS/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=WillWang1113/GenTS" />
</a> -->

<a href="https://github.com/WillWang1113/GenTS/graphs/contributors"><img src="https://contributors-img.web.app/image?repo=WillWang1113/GenTS" /></a>



## 📎 Citation
If you enjoy or benefit from using `GenTS`, a citation to this repository will be greatly appreciated！
```BibTeX
@misc{wang2025a,
title={GenTS: A Comprehensive Benchmark Library for Generative Time Series Models},
author={Chenxi Wang, Xiaorong Wang, Peiyang Li, Yi Wang},
year={2025},
url={https://github.com/WillWang1113/GenTS}
}
```
