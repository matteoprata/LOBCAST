import pytorch_lightning as pl
from torch import nn


class CNN2(pl.LightningModule):
    def __init__(self, horizon, n_feat, outshape, temp):
        super().__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(10, 42), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU()

        # Convolution 2
        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(10, )) # 3
        self.bn2 = nn.BatchNorm1d(16)
        self.prelu2 = nn.PReLU()

        # Convolution 3
        self.cnn3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(8, )) # 1
        self.bn3 = nn.BatchNorm1d(32)
        self.prelu3 = nn.PReLU()

        # Convolution 4 
        self.cnn4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(6, )) # 1
        self.bn4 = nn.BatchNorm1d(32)
        self.prelu4 = nn.PReLU()

        # Convolution 5
        self.cnn5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(4, )) # 1
        self.bn5 = nn.BatchNorm1d(32)
        self.prelu5 = nn.PReLU()

        # Fully connected 1
        self.fc1 = nn.Linear(temp*32, 32)
        self.prelu6 = nn.PReLU()

        # Fully connected 2
        self.fc2 = nn.Linear(32, outshape)

    def forward(self, x):
        # Convolution 1
        # print('cnn1', x.shape)
        out = self.cnn1(x)
        # print('bn1', out.shape)
        out = self.bn1(out)
        # print('prelu1', out.shape)
        out = self.prelu1(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)

        # Convolution 2
        # print('cnn2', out.shape)
        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        # Convolution 3
        # print('cnn3', out.shape)
        out = self.cnn3(out)
        out = self.bn3(out)
        out = self.prelu3(out)

        # Convolution 4
        # print('cnn4', out.shape)
        out = self.cnn4(out)
        out = self.bn4(out)
        out = self.prelu4(out)

        # Convolution 5
        # print('cnn5', out.shape)
        out = self.cnn5(out)
        out = self.bn5(out)
        out = self.prelu5(out)

        # flatten
        # print('flatten', out.shape)
        out = out.view(out.size(0), -1)

        # Linear function 1
        # print('fc1', out.shape)
        out = self.fc1(out)
        out = self.prelu6(out)

        # Linear function (readout)
        # print('fc2', out.shape)
        out = self.fc2(out)

        return out