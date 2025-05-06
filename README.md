# GenTS: A library for generative time series analysis


## Installation(TODO:details)
Python: 3.10

## Model zoo
|          Name          | Model Type |       Condition       |    Application    |      Finish?       |
| :--------------------: | :--------: | :-------------------: | :---------------: | :----------------: |
|       VanillaVAE       |    VAE     |           -           |        Syn        | :white_check_mark: |
|        TimeVAE         |    VAE     |           -           |        Syn        | :white_check_mark: |
|       TimeVQVAE        |    VAE     |      class label      |        Syn        | :white_check_mark: |
|         KoVAE          |    VAE     |           -           |    Syn(irreg)     | :white_check_mark: |
|       VanillaGAN       |    GAN     |           -           |        Syn        | :white_check_mark: |
|        TimeGAN         |    GAN     |           -           |        Syn        | :white_check_mark: |
|      AST **(-!)**      |    GAN     |  :white_check_mark:   |    Fcst(point)    | :white_check_mark: |
|       COSCI-GAN        |    GAN     |           -           |        Syn        | :white_check_mark: |
|         GT-GAN         |    GAN     |           -           |    Syn(irreg)     | :white_check_mark: |
|    PSA-GAN **(-G)**    |    GAN     |           -           |        Syn        | :white_check_mark: |
|         RCGAN          |    GAN     |           -           |        Syn        | :white_check_mark: |
|       VanillaMAF       |    Flow    |           -           |        Syn        | :white_check_mark: |
|      Fourier Flow      |    Flow    |           -           |        Syn        | :white_check_mark: |
|   LSTM-MAF **(-G)**    |    Flow    |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|      VanillaDDPM       | Diffusion  |           -           |        Syn        | :white_check_mark: |
|          CSDI          | Diffusion  |  :white_check_mark:   |   Impute, Fcst    | :white_check_mark: |
|      Diffusion-TS      | Diffusion  |  :white_check_mark:   | Syn, Fcst, Impute | :white_check_mark: |
|          TMDM          | Diffusion  |  :white_check_mark:   |       Fcst        | :white_check_mark: |
|        mr-diff         | Diffusion  |  :white_check_mark:   |       Fcst        | :white_check_mark: |
|          RATD          | Diffusion  |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|    FourierDiffusion    | Diffusion  |           -           |        Syn        | :white_check_mark: |
|       ImagenTime       | Diffusion  |  :white_check_mark:   |     Syn, Fcst     | :white_check_mark: |
|      D3M **(-M)**      | Diffusion  |           -           |     Fcst, Imp     |   :white_circle:   |
|  TimeWeaver **(-M)**   | Diffusion  |  :white_check_mark:   |        Syn        |   :white_circle:   |
| FTS-Diffusion **(-M)** | Diffusion  |           -           |        Syn        |   :white_circle:   |
|     FIDE **(-!)**      | Diffusion  |     block maxima      |        Syn        | :white_check_mark: |
|      ANT **(-G)**      | Diffusion  |  :white_check_mark:   | Syn, Fcst, Refine |   :white_circle:   |
|   TimeGrad **(-G)**    | Diffusion  |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|    TSDiff **(-G)**     | Diffusion  | inference conditional | Syn, Fcst, Refine |   :white_circle:   |
|    MG-TSD **(-G)**     | Diffusion  |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|     D3VAE **(-P)**     | Diffusion  |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|       Latent ODE       | Diff. Eq.  |           -           |        Syn        |   :white_circle:   |
|        ODE-RNN         | Diff. Eq.  |           -           |        Syn        |   :white_circle:   |
|       Neural SDE       | Diff. Eq.  |           -           |        Syn        |   :white_circle:   |
|          LS4           | Diff. Eq.  |           -           |        Syn        |   :white_circle:   |
|          TFM           | Diff. Eq.  |           -           |       Fcst        |   :white_circle:   |
|   SDformer **(-M)**    |  VAE+GPT   |  :white_check_mark:   |        Syn        |   :white_circle:   |
|          <!--          |    GANF    |         Flow          |         -         |         AD         | :white_circle: | --> |


*Notes*: 
- **(-G)** = GluonTS style code
- **(-P)** = PaddlePaddle instead of torch
- **(-M)** = Missing official codes
- **(-!)** = Official codes are functionally different from the paper


## Arena (TODO: experiments)

Till XX 2025, the top three models for five different tasks are:

| Model Rank | Synthesis | Forecasting | Imputation |
| :--------: | :-------: | :---------: | :--------: |
|    1st     |     -     |      -      |     -      |
|    2nd     |     -     |      -      |     -      |
|    3rd     |     -     |      -      |     -      |



## Custormization (TODO: writing details)

### How to build a new model?
Inheret ```BaseModel```, and make sure implement ```training_step```,```configure_optimizers```,```validation_step```, and ```_sample_impl```.

The former three are standard ```lightning``` methods for model training; The last one required for sampling.



## TODO-list
- [ ] Flow-based model (5.15)
- [ ] ODE-based model (5.15)
- [ ] Evaluation (TSGBench + new metrics J-FTSD, ICML2024) (5.31)
- [ ] Benchmark datasets (6.15)