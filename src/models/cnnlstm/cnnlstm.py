# Using Deep Learning for price prediction by exploiting stationary limit order book features
# Source: https://www.sciencedirect.com/science/article/pii/S1568494620303410

import pytorch_lightning as pl
import torch
from torch import nn

class CNNLSTM(pl.LightningModule):
    def __init__(self, num_features, num_classes, batch_size, seq_len, hidden_size, num_layers, p_dropout):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_layers = num_layers  # 1
        self.hidden_size = hidden_size  # 64
        self.seq_len = seq_len  # number of snapshots (100)

        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 42), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU()

        # Convolution 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(5,))
        self.bn2 = nn.BatchNorm1d(16)
        self.prelu2 = nn.PReLU()

        # Convolution 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(5,))
        self.bn3 = nn.BatchNorm1d(32)
        self.prelu3 = nn.PReLU()

        # Convolution 4 
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(5,))
        self.bn4 = nn.BatchNorm1d(32)
        self.prelu4 = nn.PReLU()

        self.lstm_input = self.get_lstm_input_size(num_features, seq_len)
        
        # lstm layers
        self.lstm = nn.LSTM(input_size=self.lstm_input, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # fully connected
        self.fc1 = nn.Linear(hidden_size, 32)  # fully connected
        self.dropout = nn.Dropout(p=p_dropout)  # not specified
        self.prelu = nn.PReLU()
        
        self.fc2 = nn.Linear(32, self.num_classes)  # out layer

    def get_lstm_input_size(self, num_features, seq_len):
        with torch.no_grad():
            sample_in = torch.ones(self.batch_size, 1, seq_len, num_features) # batch_size, 1, seq_len, num_features
            sample_out = self.convolution_forward(sample_in)

        return sample_out.shape[-1]

    def forward(self, x):
        # Adding the channel dimension
        x = x[:, None, :]  # x.shape = [batch_size, 1, 100, 40]

        # print('x.shape:', x.shape)
        
        out = self.convolution_forward(x)
        # print('After convolution_forward:', out.shape)

        # lstm
        _, (hn, _) = self.lstm(out)
        # print('After lstm:', hn.shape)

        # flatten
        hn = hn.view(-1, self.hidden_size)
        # print('After flatten:', hn.shape)

        out = self.fc1(hn)
        # print('After fc1:', out.shape)

        out = self.dropout(out)
        out = self.prelu(out)

        out = self.fc2(out)
        # print('After fc2:', out.shape)

        return out

    def convolution_forward(self, x):
        # print('Starting convolution_forward')

        # print('x.shape:', x.shape)

        # Convolution 1
        out = self.conv1(x)
        # print('After convolution1:', out.shape)

        out = self.bn1(out)
        # print('After bn1:', out.shape)

        out = self.prelu1(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        # print('After prelu1:', out.shape)

        # Convolution 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        # print('After convolution2, bn2, prelu2:', out.shape)

        # Convolution 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.prelu3(out)
        # print('After convolution3, bn3, prelu3:', out.shape)

        # Convolution 4
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.prelu4(out)
        # print('After convolution4, bn4, prelu4:', out.shape)

        # print('Ending convolution_forward')

        return out
