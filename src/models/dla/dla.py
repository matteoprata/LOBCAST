# Forecasting the Mid-price Movements with High-Frequency LOB: A Dual-Stage Temporal Attention-Based Deep Learning Architecture
# Source: https://link.springer.com/content/pdf/10.1007/s13369-022-07197-3.pdf?pdf=button

import pytorch_lightning as pl
from torch import nn
import torch


class DLA(pl.LightningModule):
    def __init__(self, num_snapshots, num_features, hidden_size):
        super().__init__()

        self.W1 = nn.Linear(num_features, num_features, bias=False)

        self.softmax = nn.Softmax(dim=1)

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = nn.Linear(num_snapshots*hidden_size, 3)

    def forward(self, x):
        # x.shape = [batch_size, observation_length, num_features]

        X_tilde = self.W1(x)
        # alpha.shape = [batch_size, observation_length, num_features]

        alpha = self.softmax(X_tilde)
        # alpha.shape = [batch_size, observation_length, num_features]

        alpha = torch.mean(alpha, dim=2)
        # alpha.shape = [batch_size, observation_length]

        x_tilde = torch.einsum('ij,ijk->ijk', [alpha, x])
        # x_tilde.shape = [batch_size, observation_length, num_features]

        H, _ = self.gru(x_tilde)
        # o.shape = [batch_size, observation_length, hidden_size]

        H_tilde = self.W2(H)
        # o.shape = [batch_size, observation_length, hidden_size]

        beta = self.softmax(H_tilde)
        # o.shape = [batch_size, observation_length, hidden_size]

        beta = torch.mean(beta, dim=2)
        # beta.shape = [batch_size, observation_length]

        h_tilde = torch.einsum('ij,ijk->ijk', [beta, H])
        # h_tilde.shape = [batch_size, observation_length, hidden_size]

        h_tilde = torch.flatten(h_tilde, start_dim=1)
        # h_tilde.shape = [batch_size, hidden_size*observation_length]

        out = self.W3(h_tilde)
        # out.shape = [batch_size, 3]

        return out