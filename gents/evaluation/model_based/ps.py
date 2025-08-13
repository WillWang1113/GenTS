import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


def predictive_score(ori_data: np.ndarray, generated_data: np.ndarray, device: str):
    """Predictive score.
    
    Predictive score is used for evaluating the usefulness of the generated time series on forecasting.
    
    The generated time series will be used for training a GRU forecasting model. Then, the trained model will be
    tested on the real data.
    
    The test MAE will be reported.

    Args:
        ori_data (np.ndarray): Real time series data.
        generated_data (np.ndarray): Generated time series data.
        device (str): Computing device.
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128

    # Initialize the predictor
    predictor_model = Predictor(input_dim=dim - 1, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(predictor_model.parameters())
    criterion = nn.L1Loss()

    # Training using Synthetic data
    for itt in range(iterations):
        # Set mini-batch
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]

        X_mb = np.array([generated_data[i][:-1, : (dim - 1)] for i in train_idx])
        T_mb = np.array([generated_time[i] - 1 for i in train_idx])
        Y_mb = np.array(
            [
                np.reshape(
                    generated_data[i][1:, (dim - 1)],
                    [len(generated_data[i][1:, (dim - 1)]), 1],
                )
                for i in train_idx
            ]
        )

        # Convert to PyTorch tensors
        X_mb = torch.FloatTensor(X_mb).to(device)
        Y_mb = torch.FloatTensor(Y_mb).to(device)

        # Train predictor
        optimizer.zero_grad()
        y_pred = predictor_model(X_mb)
        loss = criterion(y_pred, Y_mb)
        loss.backward()
        optimizer.step()

    ## Test the trained model on the original data
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]

    X_mb = np.array([ori_data[i][:-1, : (dim - 1)] for i in train_idx])
    T_mb = np.array([ori_time[i] - 1 for i in train_idx])
    Y_mb = np.array(
        [
            np.reshape(ori_data[i][1:, (dim - 1)], [len(ori_data[i][1:, (dim - 1)]), 1])
            for i in train_idx
        ]
    )

    # Convert to PyTorch tensors
    X_mb = torch.FloatTensor(X_mb).to(device)
    Y_mb = torch.FloatTensor(Y_mb).to(device)

    # Prediction
    with torch.no_grad():
        pred_Y_curr = predictor_model(X_mb).cpu()

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(Y_mb[i].numpy(), pred_Y_curr[i].numpy())

    predictive_score = MAE_temp / no

    return predictive_score
