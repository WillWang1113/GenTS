# GenTS: A library for generative time series analysis


## Installation
Python: 3.10


## Models 

<!-- ### Tasks included
|      Task       |      Conditional on      |
| :-------------: | :----------------------: |
|    Synthesis    | NA / high-level features |
|   Imputation    |      Observed data       |
|   Forecasting   |     Look-back window     |
| Superresolution |   Low-resolution data    | --> |

<!-- ### Models included -->
|    Name     | Model Type | Condition |            Application             | Finish? |
| :---------: | :--------: | :-------: | :--------------------------------: | :---------: |
| VanillaVAE  |    VAE     |     -     |             Synthesis              |   :white_check_mark: |
| VanillaGAN  |    GAN     |     -     |             Synthesis              |  Synthesis  |
| VanillaMAF  |    Flow    |     -     |             Synthesis              |  Synthesis  |
| VanillaDDPM | Diffusion  |     -     |             Synthesis              |  Synthesis  |
|   TimeVAE   |    VAE     |     -     |             Synthesis              |  Synthesis  |
|   TimeGAN   |    GAN     |     -     |             Synthesis              |  Synthesis  |
| Neural ODE  | Diff. Eq.  |     -     | Synthesis, Forecasting, Imputation |  Synthesis  |
| Neural ODE  | Diff. Eq.  |     -     |             Synthesis              |  Synthesis  |



## Custormization

### Conditions

### Models


## TODO
- [x] TimeVAE
- [x] TimeGAN
- [ ] TimeGrad
- [ ] Fourier Flow
- [ ] Neural ODE
- [ ] benchmark datasets
- [ ] condition data loader