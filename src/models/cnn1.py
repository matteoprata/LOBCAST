import pytorch_lightning as pl
from torch import nn

class CNN(pl.LightningModule):
    def __init__(self, horizon, n_feat, outshape, temp):
        super().__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, n_feat), padding=(3, 0), dilation=(2, 1))
        self.relu1 = nn.LeakyReLU()

        # Convolution 2
        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(4, ))
        self.relu2 = nn.LeakyReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        # Convolution 3
        self.cnn3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(3, ), padding=2)
        self.relu3 = nn.LeakyReLU()

        # Convolution 4
        self.cnn4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(3, ), padding=2)
        self.relu4 = nn.LeakyReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        # Fully connected 1
        self.fc1 = nn.Linear(temp * 32, 32)
        self.relu5 = nn.LeakyReLU()

        # Fully connected 2
        self.fc2 = nn.Linear(32, outshape)

    def forward(self, x):
        # Convolution 1
        # print('cnn1', x.shape)
        out = self.cnn1(x)
        out = self.relu1(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)

        # Convolution 2
        # print('cnn2', out.shape)
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 1
        # print('max1', out.shape)
        out = self.maxpool1(out)

        # Convolution 3
        # print('cnn3', out.shape)
        out = self.cnn3(out)
        out = self.relu3(out)

        # Convolution 4
        # print('cnn4', out.shape)
        out = self.cnn4(out)
        out = self.relu4(out)

        # Max pool 2
        # print('max2', out.shape)
        out = self.maxpool2(out)

        # flatten
        # print('flatten', out.shape)
        out = out.view(out.size(0), -1)

        # Linear function 1
        # print('fc1', out.shape)
        out = self.fc1(out)
        out = self.relu5(out)

        # Linear function (readout)
        # print('fc2', out.shape)
        out = self.fc2(out)

        return out