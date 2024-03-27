# Forecasting Stock Prices from the Limit Order Book using Convolutional Neural Networks
# Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8010701

import pytorch_lightning as pl
from torch import nn

from src.models.lobcast_model import LOBCAST_model, LOBCAST_module
from src.hyper_parameters import HPTunable


class CNN1(LOBCAST_model):

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)

        n_features = input_dim[1]

        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, n_features), padding=(3, 0), dilation=(2, 1))
        self.relu1 = nn.LeakyReLU()

        # Convolution 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(4,))
        self.relu2 = nn.LeakyReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        # Convolution 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(3,), padding=2)
        self.relu3 = nn.LeakyReLU()

        # Convolution 4
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(3,), padding=2)
        self.relu4 = nn.LeakyReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        # Fully connected 1
        self.fc1 = nn.Linear(26*32, 32)
        self.relu5 = nn.LeakyReLU()

        # Fully connected 2
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # Adding the channel dimension
        x = x[:, None, :]  # x.shape = [batch_size, 1, 100, 40]

        # print('x.shape:', x.shape)

        # Convolution 1
        out = self.conv1(x)
        out = self.relu1(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        # print('After convolution1:', out.shape)

        # Convolution 2
        out = self.conv2(out)
        out = self.relu2(out)
        # print('After convolution2:', out.shape)

        # Max pool 1
        out = self.maxpool1(out)
        # print('After maxpool1:', out.shape)

        # Convolution 3
        out = self.conv3(out)
        out = self.relu3(out)
        # print('After convolution3:', out.shape)

        # Convolution 4
        out = self.conv4(out)
        out = self.relu4(out)
        # print('After convolution4:', out.shape)

        # Max pool 2
        out = self.maxpool2(out)
        # print('After maxcpool2:', out.shape)

        # flatten
        out = out.view(out.size(0), -1)
        # print('After flatten:', out.shape)

        # Linear function 1
        out = self.fc1(out)
        out = self.relu5(out)
        # print('After linear1:', out.shape)

        # Linear function (readout)
        out = self.fc2(out)
        # print('After linear2:', out.shape)

        return out


CNN_lm = LOBCAST_module(CNN1)
