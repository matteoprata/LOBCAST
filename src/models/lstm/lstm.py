import pytorch_lightning as pl
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, y_shape, x_shape, hidden_layer_dim, num_layers):
        super().__init__()

        self.x_shape = x_shape                    # nfeat
        self.y_shape = y_shape                    # 3
        self.hidden_layer_dim = hidden_layer_dim  # 32 - 64
        self.num_layers = num_layers              # 1
        # self.seq_length = seq_length            # horizon

        self.lstm = nn.LSTM(
            input_size=x_shape,
            hidden_size=hidden_layer_dim,
            num_layers=num_layers,
            batch_first=True
        )  # lstm
        
        self.fc_1 = nn.Linear(hidden_layer_dim, 64)   # fully connected 64 neurons
        self.dropout = nn.Dropout(p=0.2)              # not specified
        self.prelu = nn.PReLU()
        self.fc = nn.Linear(64, y_shape)              # out layer
    
    def forward(self, x):
        x = x.float()
        output, (hn, cn) = self.lstm(x)          # lstm with input, hidden, and internal state (batch, time-step, features)
        hn = hn.view(-1, self.hidden_layer_dim)  # reshaping the data for Dense layer next

        out = self.fc_1(hn)
        out = self.dropout(out)
        out = self.prelu(out) 
        out = self.fc(out)
        
        return out
 