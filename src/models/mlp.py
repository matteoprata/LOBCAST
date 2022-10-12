import pytorch_lightning as pl
from torch import nn


class MLP(pl.LightningModule):
    def __init__(self, inshape, outshape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inshape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, outshape),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)
