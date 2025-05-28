###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys
import matplotlib

matplotlib.use("Agg")

import time
import argparse
import numpy as np
from random import SystemRandom

import torch


from src.utils.latentode_parsedataset import parse_datasets

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser("Latent ODE")
parser.add_argument("-n", type=int, default=100, help="Size of the dataset")
parser.add_argument("--niters", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument("-b", "--batch-size", type=int, default=50)
parser.add_argument("--viz", action="store_true", help="Show plots while training")

parser.add_argument(
    "--save", type=str, default="experiments/", help="Path for save checkpoints"
)
parser.add_argument(
    "--load",
    type=str,
    default=None,
    help="ID of the experiment to load for evaluation. If None, run a new experiment.",
)
parser.add_argument("-r", "--random-seed", type=int, default=1991, help="Random_seed")

parser.add_argument(
    "--dataset",
    type=str,
    default="periodic",
    help="Dataset to load. Available: physionet, activity, hopper, periodic",
)
parser.add_argument(
    "-s",
    "--sample-tp",
    type=float,
    default=None,
    help="Number of time points to sub-sample."
    "If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample",
)

parser.add_argument(
    "-c",
    "--cut-tp",
    type=int,
    default=None,
    help="Cut out the section of the timeline of the specified length (in number of points)."
    "Used for periodic function demo.",
)

parser.add_argument(
    "--quantization",
    type=float,
    default=0.1,
    help="Quantization on the physionet dataset."
    "Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min",
)

parser.add_argument(
    "--latent-ode", action="store_true", help="Run Latent ODE seq2seq model"
)
parser.add_argument(
    "--z0-encoder",
    type=str,
    default="odernn",
    help="Type of encoder for Latent ODE model: odernn or rnn",
)

parser.add_argument(
    "--classic-rnn",
    action="store_true",
    help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.",
)
parser.add_argument(
    "--rnn-cell",
    default="gru",
    help="RNN Cell type. Available: gru (default), expdecay",
)
parser.add_argument(
    "--input-decay",
    action="store_true",
    help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)",
)

parser.add_argument(
    "--ode-rnn",
    action="store_true",
    help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.",
)

parser.add_argument(
    "--rnn-vae",
    action="store_true",
    help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.",
)

parser.add_argument(
    "-l", "--latents", type=int, default=6, help="Size of the latent state"
)
parser.add_argument(
    "--rec-dims",
    type=int,
    default=20,
    help="Dimensionality of the recognition model (ODE or RNN).",
)

parser.add_argument(
    "--rec-layers",
    type=int,
    default=1,
    help="Number of layers in ODE func in recognition ODE",
)
parser.add_argument(
    "--gen-layers",
    type=int,
    default=1,
    help="Number of layers in ODE func in generative ODE",
)

parser.add_argument(
    "-u", "--units", type=int, default=100, help="Number of units per layer in ODE func"
)
parser.add_argument(
    "-g",
    "--gru-units",
    type=int,
    default=100,
    help="Number of units per layer in each of GRU update networks",
)

parser.add_argument(
    "--poisson",
    action="store_true",
    help="Model poisson-process likelihood for the density of events in addition to reconstruction.",
)
parser.add_argument(
    "--classif",
    action="store_true",
    help="Include binary classification loss -- used for Physionet dataset for hospiral mortality",
)

parser.add_argument(
    "--linear-classif",
    action="store_true",
    help="If using a classifier, use a linear classifier instead of 1-layer NN",
)
parser.add_argument(
    "--extrap",
    action="store_true",
    help="Set extrapolation mode. If this flag is not set, run interpolation mode.",
)

parser.add_argument(
    "-t", "--timepoints", type=int, default=100, help="Total number of time-points"
)
parser.add_argument(
    "--max-t",
    type=float,
    default=5.0,
    help="We subsample points in the interval [0, args.max_tp]",
)
parser.add_argument(
    "--noise-weight",
    type=float,
    default=0.01,
    help="Noise amplitude for generated traejctories",
)


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]

#####################################################################################################

if __name__ == "__main__":
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + ".ckpt")

    start = time.time()
    print("Sampling dataset of {} training examples".format(args.n))

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2) :]
    input_command = " ".join(input_command)


    ##################################################################
    data_obj = parse_datasets(args, device)
    for k in data_obj:
        print("{}:\n {}".format(k, data_obj[k]))
        
    train_dl = data_obj['train_dataloader']
    for i, batch in enumerate(train_dl):
        print(i)
        print(batch['data_to_predict'][0,:,-1])
        print(batch['mask_predicted_data'][0,:,-1])
        # for k in batch:
        #     print(k)
        #     print(batch[k])
        if i > 2:
            break