# Using Deep Learning to Detect Price Change Indications in Financial Markets
# Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8081663

import pytorch_lightning as pl
from torch import nn

CONFIG = {
    "hidden_layer_dim": [88],
    "p_dropout": [99]
}


class LOBCAST_model(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim


class MLP(LOBCAST_model):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_layer_dim=128,
            p_dropout=.1
    ):
        super(MLP, self).__init__(input_dim, output_dim)

        self.linear1 = nn.Linear(self.input_dim, hidden_layer_dim)
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear2 = nn.Linear(hidden_layer_dim, self.output_dim)

    def forward(self, x):
        # [batch_size x 40 x observation_length]
        x = x.view(x.size(0), -1).float()
        out = self.linear1(x)
        out = self.leakyReLU(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out
