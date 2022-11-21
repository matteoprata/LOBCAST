# Using Deep Learning to Detect Price Change Indications in Financial Markets
# Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8081663

import pytorch_lightning as pl
from torch import nn


class LSTM(pl.LightningModule):
    def __init__(self, num_classes, x_shape, hidden_layer_dim, num_layers, p_dropout):
        super().__init__()

        self.x_shape = x_shape                    # nfeat
        self.num_classes = num_classes            # 3
        self.hidden_layer_dim = hidden_layer_dim  # 32 - 64
        self.num_layers = num_layers              # 1
        # self.seq_length = seq_length            # number of snapshots

        self.lstm = nn.LSTM(
            input_size=x_shape,
            hidden_size=hidden_layer_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc_1 = nn.Linear(hidden_layer_dim, 64)   # fully connected 64 neurons
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)              # not specified
        self.fc = nn.Linear(64, num_classes)              # out layer

    def forward(self, x):
        x = x.float()

        output, (hn, _) = self.lstm(x)          # lstm with input, hidden, and internal state (batch, time-step, features)

        # before hn.shape = [1, batch_size, features]
        hn = hn.view(-1, self.hidden_layer_dim)  # reshaping the data for Dense layer next
        # after hn.shape = [batch_size, features]

        out = self.fc_1(hn)
        out = self.leakyReLU(out)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
 