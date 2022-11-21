# Using Deep Learning for price prediction by exploiting stationary limit order book features
# Source: https://www.sciencedirect.com/science/article/pii/S1568494620303410

import pytorch_lightning as pl
import torch
from torch import nn

class CNNLSTM(pl.LightningModule):
    def __init__(self, horizon, input_size, outshape, hidden_size, num_layers):
        super().__init__()

        self.input_size = input_size 
        self.num_layers = num_layers # 1
        self.hidden_size = hidden_size # 32
        self.horizon = horizon # horizon

        # Convolution 1
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=(5, 42), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU()

        # Convolution 2
        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(5, ))
        self.bn2 = nn.BatchNorm1d(16)
        self.prelu2 = nn.PReLU()

        # Convolution 3
        self.cnn3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(5, ))
        self.bn3 = nn.BatchNorm1d(32)
        self.prelu3 = nn.PReLU()

        # Convolution 4 
        self.cnn4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(5, ))
        self.bn4 = nn.BatchNorm1d(32)
        self.prelu4 = nn.PReLU()

        lstm_input = self.get_lstm_input_size(input_size, horizon)
        
        # lstm layers
        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # fully connected
        self.fc1 = nn.Linear(64, 32) # fully connected 64 neurons
        self.dropout = nn.Dropout(p=0.2) # not specified 
        self.prelu = nn.PReLU()
        
        self.fc2 = nn.Linear(32, outshape) # out layer

    def get_lstm_input_size(self, input_size, horizon):
        with torch.no_grad():
            sample_in = torch.ones(32, 1, horizon, input_size) # batch_size, 1, horizon, input_size
            sample_out = self.convolution_forward(sample_in)
        
        return sample_out.shape[-1]

    def forward(self, x):
        
        out = self.convolution_forward(x)

        # lstm
        _, (hn, _) = self.lstm(out) 
        
        hn = hn.view(-1, self.hidden_size) 

        out = self.fc1(hn)
        out = self.dropout(out)
        out = self.prelu(out) 
        out = self.fc2(out)

        return out

    def convolution_forward(self, x):

        # Convolution 1
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)

        # Convolution 2
        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        # Convolution 3
        out = self.cnn3(out)
        out = self.bn3(out)
        out = self.prelu3(out)

        # Convolution 4
        out = self.cnn4(out)
        out = self.bn4(out)
        out = self.prelu4(out)

        return out
