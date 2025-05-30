# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""

This script contains torch implementation for a vanilla RNN

"""

from __future__ import absolute_import, division, print_function

import numpy as np

import sys
import itertools
import pickle


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

import torch
from torch import nn
from torch.autograd import Variable


torch.manual_seed(1)


def save_model(model, filename):
    with open(filename, "wb") as saved_model_file:
        pickle.dump(model, saved_model_file)


def evaluate_RMSE(y_true, y_pred):
    y_pred_ = list(itertools.chain.from_iterable(y_pred))
    y_true_ = list(itertools.chain.from_iterable(y_true))

    return np.sqrt(np.mean((np.array(y_true_) - np.array(y_pred_)) ** 2))


def padd_arrays(X, max_length=None):
    if len(X[0].shape) == 1:
        X = [X[k].reshape((-1, 1)) for k in range(len(X))]

    if max_length is None:
        max_length = np.max(np.array([len(X[k]) for k in range(len(X))]))

    X_output = [
        np.expand_dims(
            np.vstack((X[k], np.zeros((max_length - X[k].shape[0], X[0].shape[1])))),
            axis=0,
        )
        for k in range(len(X))
    ]
    _mask = [
        np.expand_dims(
            np.vstack(
                (
                    np.ones((X[k].shape[0], X[k].shape[1])),
                    np.zeros((max_length - X[k].shape[0], X[0].shape[1])),
                )
            ),
            axis=0,
        )
        for k in range(len(X))
    ]

    return np.concatenate(X_output, axis=0), np.concatenate(_mask, axis=0)


def unpadd_arrays(X, masks):
    masks_lengths = np.sum(masks, axis=1)[:, 0]
    out_ = []

    for k in range(X.shape[0]):
        if len(X.shape) > 2:
            out_.append(X[k, : int(masks_lengths[k]), :])
        else:
            out_.append(X[k, : int(masks_lengths[k])])

    return out_


def get_data_split(X, Y, T, L, indexes):
    X_ = [X[u] for u in indexes]
    Y_ = [Y[u] for u in indexes]
    T_ = [T[u] for u in indexes]
    L_ = [L[u] for u in indexes]

    return X_, Y_, T_, L_


def model_loss_single(output, target, masks):
    single_loss = masks * (output - target) ** 2
    loss = torch.mean(torch.sum(single_loss, axis=0) / torch.sum(masks, axis=0))

    return loss


def single_losses(model):
    return model.masks * (model(model.X).view(-1, model.MAX_STEPS) - model.y) ** 2


def model_loss(output, target, masks):
    single_loss = masks * (output - target) ** 2
    loss = torch.sum(
        torch.sum(single_loss, axis=1) / torch.sum(torch.sum(masks, axis=1))
    )

    return loss


class RNN(nn.Module):
    def __init__(
        self, mode="RNN", INPUT_SIZE=30, OUTPUT_SIZE=1, HIDDEN_UNITS=100, NUM_LAYERS=1
    ):
        super(RNN, self).__init__()

        self.INPUT_SIZE = INPUT_SIZE
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.HIDDEN_UNITS = HIDDEN_UNITS
        self.NUM_LAYERS = NUM_LAYERS

        rnn_dict = {
            "RNN": nn.RNN(
                input_size=self.INPUT_SIZE,
                hidden_size=self.HIDDEN_UNITS,
                num_layers=self.NUM_LAYERS,
                batch_first=True,
            ),
            "LSTM": nn.LSTM(
                input_size=self.INPUT_SIZE,
                hidden_size=self.HIDDEN_UNITS,
                num_layers=self.NUM_LAYERS,
                batch_first=True,
            ),
            "GRU": nn.GRU(
                input_size=self.INPUT_SIZE,
                hidden_size=self.HIDDEN_UNITS,
                num_layers=self.NUM_LAYERS,
                batch_first=True,
            ),
        }

        self.mode = mode
        self.rnn = rnn_dict[self.mode]
        self.out = nn.Linear(self.HIDDEN_UNITS, self.OUTPUT_SIZE)

    def forward(self, x):
        if self.mode == "LSTM":
            r_out, (h_n, h_c) = self.rnn(
                x.reshape((-1, x.shape[1], 1)), None
            )  # None represents zero initial hidden state

        else:
            r_out, h_n = self.rnn(x.reshape((-1, x.shape[1], 1)), None)

        # choose r_out at the last time step
        out = self.out(r_out[:, :, :])

        return out.view(-1, x.shape[1])  # .transpose(2, 1)


class RNNmodel(nn.Module):
    def __init__(
        self,
        mode="RNN",
        EPOCH=5,
        BATCH_SIZE=150,
        MAX_STEPS=50,
        INPUT_SIZE=30,
        LR=0.01,
        OUTPUT_SIZE=1,
        HIDDEN_UNITS=20,
        NUM_LAYERS=1,
        N_STEPS=50,
    ):
        super(RNNmodel, self).__init__()

        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_STEPS = MAX_STEPS
        self.INPUT_SIZE = INPUT_SIZE
        self.LR = LR
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.HIDDEN_UNITS = HIDDEN_UNITS
        self.NUM_LAYERS = NUM_LAYERS
        self.N_STEPS = N_STEPS

        rnn_dict = {
            "RNN": nn.RNN(
                input_size=self.INPUT_SIZE,
                hidden_size=self.HIDDEN_UNITS,
                num_layers=self.NUM_LAYERS,
                batch_first=True,
            ),
            "LSTM": nn.LSTM(
                input_size=self.INPUT_SIZE,
                hidden_size=self.HIDDEN_UNITS,
                num_layers=self.NUM_LAYERS,
                batch_first=True,
            ),
            "GRU": nn.GRU(
                input_size=self.INPUT_SIZE,
                hidden_size=self.HIDDEN_UNITS,
                num_layers=self.NUM_LAYERS,
                batch_first=True,
            ),
        }

        self.mode = mode
        self.rnn = rnn_dict[self.mode]

        self.out = nn.Linear(self.HIDDEN_UNITS, self.OUTPUT_SIZE)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        if self.mode == "LSTM":
            r_out, (h_n, h_c) = self.rnn(
                x, None
            )  # None represents zero initial hidden state

        else:
            r_out, h_n = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(r_out[:, :, :])

        return out

    def fit(self, X, Y, verbosity=True):
        X_padded, _ = padd_arrays(X, max_length=self.MAX_STEPS)
        Y_padded, loss_masks = (
            np.squeeze(padd_arrays(Y, max_length=self.MAX_STEPS)[0], axis=2),
            np.squeeze(padd_arrays(Y, max_length=self.MAX_STEPS)[1], axis=2),
        )

        X = Variable(torch.tensor(X_padded), volatile=True).type(torch.FloatTensor)
        Y = Variable(torch.tensor(Y_padded), volatile=True).type(torch.FloatTensor)
        loss_masks = Variable(torch.tensor(loss_masks), volatile=True).type(
            torch.FloatTensor
        )

        self.X = X
        self.y = Y
        self.masks = loss_masks

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.LR
        )  # optimize all rnn parameters
        self.loss_fn = model_loss  # nn.MSELoss()

        # training and testing
        for epoch in range(self.EPOCH):
            for step in range(self.N_STEPS):
                batch_indexes = np.random.choice(
                    list(range(X.shape[0])), size=self.BATCH_SIZE, replace=True, p=None
                )

                x = torch.tensor(X[batch_indexes, :, :])
                y = torch.tensor(Y[batch_indexes])
                msk = torch.tensor(loss_masks[batch_indexes])

                b_x = Variable(
                    x.view(-1, self.MAX_STEPS, self.INPUT_SIZE)
                )  # reshape x to (batch, time_step, input_size)
                b_y = Variable(y)  # batch y
                b_m = Variable(msk)

                output = self(b_x).view(-1, self.MAX_STEPS)  # rnn output

                self.loss = self.loss_fn(output, b_y, b_m)  # MSE loss

                optimizer.zero_grad()  # clear gradients for this training step
                self.loss.backward(
                    retain_graph=True
                )  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                if (step % 50 == 0) and verbosity:
                    print("Epoch: ", epoch, "| train loss: %.4f" % self.loss.data)

    def predict(self, X, padd=False, numpy_output=False):
        if type(X) is list:
            X_, masks = padd_arrays(X, max_length=self.MAX_STEPS)

        else:
            X_, masks = padd_arrays([X], max_length=self.MAX_STEPS)

        X_test = Variable(torch.tensor(X_), volatile=True).type(torch.FloatTensor)
        predicts_ = self(X_test).view(-1, self.MAX_STEPS)

        if padd:
            prediction = unpadd_arrays(predicts_.detach().numpy(), masks)

        else:
            prediction = predicts_.detach().numpy()

        return prediction

    def sequence_loss(self):
        return single_losses(self)
