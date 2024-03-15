# Using Deep Learning for price prediction by exploiting stationary limit order book features
# Source: https://www.sciencedirect.com/science/article/pii/S1568494620303410

import pytorch_lightning as pl
from torch import nn

from src.models.lobcast_model import LOBCAST_model, LOBCAST_module


CONFIG = {
    "temp": [249],
}


class CNN2(LOBCAST_model):
    def __init__(self, input_dim, output_dim,  temp=249):
        super().__init__(input_dim, output_dim)

        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(10, 42), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU()

        # Convolution 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(10,))  # 3
        self.bn2 = nn.BatchNorm1d(16)
        self.prelu2 = nn.PReLU()

        # Convolution 3
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(8,))  # 1
        self.bn3 = nn.BatchNorm1d(32)
        self.prelu3 = nn.PReLU()

        # Convolution 4
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(6,))  # 1
        self.bn4 = nn.BatchNorm1d(32)
        self.prelu4 = nn.PReLU()

        # Convolution 5
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(4,))  # 1
        self.bn5 = nn.BatchNorm1d(32)
        self.prelu5 = nn.PReLU()

        # Fully connected 1
        self.fc1 = nn.Linear(temp * 32, 32)
        self.prelu6 = nn.PReLU()

        # Fully connected 2
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # Adding the channel dimension
        x = x[:, None, :]  # x.shape = [batch_size, 1, 100, 40]

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

        # Convolution 5
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.prelu5(out)
        # print('After convolution5, bn5, prelu5:', out.shape)

        # flatten
        out = out.view(out.size(0), -1)
        # print('After flatten:', out.shape)

        # Linear function 1
        out = self.fc1(out)
        out = self.prelu6(out)
        # print('After fc1:', out.shape)

        # Linear function (readout)
        out = self.fc2(out)
        # print('After fc2:', out.shape)

        return out


CNN2_ml = LOBCAST_module("CNN2", CNN2, CONFIG)
