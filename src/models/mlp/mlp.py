# Using Deep Learning to Detect Price Change Indications in Financial Markets
# Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8081663

import pytorch_lightning as pl
from torch import nn


class MLP(pl.LightningModule):

    def __init__(
            self,
            num_features,
            num_classes,
            hidden_layer_dim=128,
            p_dropout=.1
    ):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(num_features, hidden_layer_dim)
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear2 = nn.Linear(hidden_layer_dim, num_classes)

    def forward(self, x):
        # [batch_size x 40 x window]
        x = x.view(x.size(0), -1).float()

        out = self.linear1(x)
        out = self.leakyReLU(out)
        out = self.dropout(out)

        out = self.linear2(out)

        return out
