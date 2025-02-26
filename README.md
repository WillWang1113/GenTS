# GenTS: A library for generative time series analysis


## Installation
Python: 3.10

## Models 

|       Name       | Model Type |       Condition       |            Application             |      Finish?       |
| :--------------: | :--------: | :-------------------: | :--------------------------------: | :----------------: |
|    VanillaVAE    |    VAE     |           -           |             Synthesis              | :white_check_mark: |
|     TimeVAE      |    VAE     |           -           |             Synthesis              | :white_check_mark: |
|    TimeVQVAE     |    VAE     |           -           |             Synthesis              |   :white_circle:   |
|      KoVAE       |    VAE     |           -           |             Synthesis              |   :white_circle:   |
|    VanillaGAN    |    GAN     |           -           |             Synthesis              | :white_check_mark: |
|     TimeGAN      |    GAN     |           -           |             Synthesis              | :white_check_mark: |
|       AST        |    GAN     |  :white_check_mark:   |            Forecasting             |   :white_circle:   |
|    COSCI-GAN     |    GAN     |           -           |             Synthesis              |   :white_circle:   |
|      GT-GAN      |    GAN     |           -           |             Synthesis              |   :white_circle:   |
|     PSA-GAN      |    GAN     |           -           |             Synthesis              |   :white_circle:   |
|      GRUGAN      |    GAN     |           -           |             Imputation             |   :white_circle:   |
|      RCGAN       |    GAN     |           -           |             Synthesis              |   :white_circle:   |
|    VanillaMAF    |    Flow    |           -           |             Synthesis              | :white_check_mark: |
|   Fourier Flow   |    Flow    |           -           |             Synthesis              |   :white_circle:   |
|       GANF       |    Flow    |           -           |         Anomaly Detection          |   :white_circle:   |
|     LSTM-MAF     |    Flow    |  :white_check_mark:   |            Forecasting             |   :white_circle:   |
|       TFM        |    Flow    |           -           |            Forecasting             |   :white_circle:   |
|   VanillaDDPM    | Diffusion  |           -           |             Synthesis              | :white_check_mark: |
|       ANT        | Diffusion  |  :white_check_mark:   | Synthesis, Forecasting, Refinement |   :white_circle:   |
|   Diffusion-TS   | Diffusion  |  :white_check_mark:   |       Synthesis, Forecasting       |   :white_circle:   |
|       FIDE       | Diffusion  |           -           |             Synthesis              |   :white_circle:   |
|       D3M        | Diffusion  |           -           |      Forecasting, Imputation       |   :white_circle:   |
|  FTS-Diffusion   | Diffusion  |           -           |             Synthesis              |   :white_circle:   |
|    TimeWeaver    | Diffusion  |  :white_check_mark:   |             Synthesis              |   :white_circle:   |
|     TimeGrad     | Diffusion  |  :white_check_mark:   |            Forecasting             |   :white_circle:   |
|       CSDI       | Diffusion  |  :white_check_mark:   |      Imputation, Forecasting       |   :white_circle:   |
|      D3VAE       | Diffusion  |  :white_check_mark:   |            Forecasting             |   :white_circle:   |
|      TSDiff      | Diffusion  | inference conditional | Synthesis, Forecasting, Refinement |   :white_circle:   |
|       TMDM       | Diffusion  |  :white_check_mark:   |            Forecasting             |   :white_circle:   |
|     mr-diff      | Diffusion  |  :white_check_mark:   |            Forecasting             |   :white_circle:   |
|      MG-TSD      | Diffusion  |  :white_check_mark:   |            Forecasting             |   :white_circle:   |
|       RATD       | Diffusion  |  :white_check_mark:   |            Forecasting             |   :white_circle:   |
| FourierDiffusion | Diffusion  |           -           |             Synthesis              | :white_check_mark: |
|    ImagenTime    | Diffusion  |  :white_check_mark:   |       Synthesis, Forecasting       |   :white_circle:   |
|    Latent ODE    | Diff. Eq.  |           -           |        Continuous modelling        | :white_check_mark: |
|     ODE-RNN      | Diff. Eq.  |           -           |        Continuous modelling        |   :white_circle:   |
|    Neural SDE    | Diff. Eq.  |           -           |        Continuous modelling        |   :white_circle:   |
|       LS4        | Diff. Eq.  |           -           |        Continuous modelling        |   :white_circle:   |
|     SDformer     |  VAE+GPT   |  :white_check_mark:   |             Synthesis              |   :white_circle:   |



## Custormization

### How to build a new model?

## Similar projects on gtihub
- TSGM: https://github.com/AlexanderVNikitin/tsgm (NIPS2024 Datasets and Benchmarks Track)
- TSGBench: https://github.com/YihaoAng/TSGBench  (VLDB2024 Best Research Paper Award Nomination)
- Evaluation-of-Time-Series-Generative-Models: https://github.com/DeepIntoStreams/Evaluation-of-Time-Series-Generative-Models

### what they have:
- Basic models, VanillaGAN, VanillaVAE, etc.
- Evaluation pipline
- Datasets for unconditional generation

### what they don't have
- SOTA models, especially diffusion-based models
- Unified unconditional and conditional model
- conditional datasets

## TODO
- [ ] Model reproduce
- [ ] Metrics under different applications
- [ ] Benchmark datasets (some models are designed for special cases?)