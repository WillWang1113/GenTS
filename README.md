# GenTS: A library for generative time series analysis


## Installation
Python: 3.10

## Models 
SOTA TS models included are list as follows, where 

- $\triangle$: originally tensorflow implementation  
- $\dag$: original implementation is different from paper

|        Name         | Model Type |       Condition       |    Application    |      Finish?       |
| :-----------------: | :--------: | :-------------------: | :---------------: | :----------------: |
|     VanillaVAE      |    VAE     |           -           |        Syn        | :white_check_mark: |
|       TimeVAE       |    VAE     |           -           |        Syn        | :white_check_mark: |
|      TimeVQVAE      |    VAE     |      class label      |        Syn        | :white_check_mark: |
|        KoVAE        |    VAE     |           -           |        Syn        | :white_check_mark: |
|     VanillaGAN      |    GAN     |           -           |        Syn        | :white_check_mark: |
| TimeGAN$^\triangle$ |    GAN     |           -           |        Syn        | :white_check_mark: |
|     AST$^\dag$      |    GAN     |  :white_check_mark:   |    Fcst(point)    | :white_check_mark: |
|      COSCI-GAN      |    GAN     |           -           |        Syn        | :white_check_mark: |
|       GT-GAN        |    GAN     |           -           |        Syn        | :white_check_mark: |
|       PSA-GAN       |    GAN     |           -           |        Syn        | :white_check_mark: |
|        RCGAN        |    GAN     |           -           |        Syn        | :white_check_mark: |
|     VanillaMAF      |    Flow    |           -           |        Syn        | :white_check_mark: |
|    Fourier Flow     |    Flow    |           -           |        Syn        |   :white_circle:   |
|        GANF         |    Flow    |           -           |        AD         |   :white_circle:   |
|      LSTM-MAF       |    Flow    |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|         TFM         |    Flow    |           -           |       Fcst        |   :white_circle:   |
|     VanillaDDPM     | Diffusion  |           -           |        Syn        | :white_check_mark: |
|         ANT         | Diffusion  |  :white_check_mark:   | Syn, Fcst, Refine |   :white_circle:   |
|    Diffusion-TS     | Diffusion  |  :white_check_mark:   |     Syn, Fcst     |   :white_circle:   |
|        FIDE         | Diffusion  |           -           |        Syn        |   :white_circle:   |
|         D3M         | Diffusion  |           -           |     Fcst, Imp     |   :white_circle:   |
|    FTS-Diffusion    | Diffusion  |           -           |        Syn        |   :white_circle:   |
|     TimeWeaver      | Diffusion  |  :white_check_mark:   |        Syn        |   :white_circle:   |
|      TimeGrad       | Diffusion  |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|        CSDI         | Diffusion  |  :white_check_mark:   |     Imp, Fcst     |   :white_circle:   |
|        D3VAE        | Diffusion  |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|       TSDiff        | Diffusion  | inference conditional | Syn, Fcst, Refine |   :white_circle:   |
|        TMDM         | Diffusion  |  :white_check_mark:   |       Fcst        | :white_check_mark: |
|       mr-diff       | Diffusion  |  :white_check_mark:   |       Fcst        | :white_check_mark: |
|       MG-TSD        | Diffusion  |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|        RATD         | Diffusion  |  :white_check_mark:   |       Fcst        |   :white_circle:   |
|  FourierDiffusion   | Diffusion  |           -           |        Syn        | :white_check_mark: |
|     ImagenTime      | Diffusion  |  :white_check_mark:   |     Syn, Fcst     |   :white_circle:   |
|     Latent ODE      | Diff. Eq.  |           -           |        Syn        | :white_check_mark: |
|       ODE-RNN       | Diff. Eq.  |           -           |        Syn        |   :white_circle:   |
|     Neural SDE      | Diff. Eq.  |           -           |        Syn        |   :white_circle:   |
|         LS4         | Diff. Eq.  |           -           |        Syn        |   :white_circle:   |
|      SDformer       |  VAE+GPT   |  :white_check_mark:   |        Syn        |   :white_circle:   |



## Custormization

### How to build a new model?

## Similar projects on gtihub
- TSGM: https://github.com/AlexanderVNikitin/tsgm (NIPS2024 Datasets and Benchmarks Track)
- TSGBench: https://github.com/YihaoAng/TSGBench  (VLDB2024 Best Research Paper Award Nomination)
- Evaluation-of-Time-Series-Generative-Models: https://github.com/DeepIntoStreams/Evaluation-of-Time-Series-Generative-Models

### what they have:
- Basic models, VanillaGAN, VanillaVAE, etc.
- Evaluation pipline
- Datasets

### what they don't have
- SOTA models, especially diffusion-based models
- Newly proposed Time series Generation metrics, e.g. J-FTSD (ICML2024)
- Unified unconditional and conditional framework

## TODO
- [ ] Model reproduce
- [ ] Metrics under different applications
- [ ] Benchmark datasets (some models are designed for special cases?)