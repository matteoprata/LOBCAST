# Deep Adaptive Input Normalization for Time Series Forecasting
# Source: https://arxiv.org/pdf/1902.07892.pdf

import pytorch_lightning as pl
import torch.nn as nn
from src.models.dain.dain_layer import DAIN_Layer


class DAIN(pl.LightningModule):

    def __init__(
        self,
        backward_window,
        num_features,
        num_classes,
        mlp_hidden,
        p_dropout,
        mode,
        mean_lr,
        gate_lr,
        scale_lr,
    ):
        super(DAIN, self).__init__()

        self.backward_window = backward_window
        self.num_features = num_features
        self.num_classes = num_classes

        self.mlp_hidden = mlp_hidden
        self.p_dropout = p_dropout

        self.dean = DAIN_Layer(
            mode=mode,
            mean_lr=mean_lr,
            gate_lr=gate_lr,
            scale_lr=scale_lr,
            input_dim=num_features
        )

        self.base = nn.Sequential(
            nn.Linear(backward_window*num_features, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, x):
        # print('First:', x.shape)

        x = x.transpose(1, 2)
        # print('After transpose:', x.shape)

        x = self.dean(x)
        # print('After dean:', x.shape)

        x = x.contiguous().view(x.size(0), self.backward_window*self.num_features)
        # print('After contiguous:', x.shape)

        x = self.base(x)
        # print('After base:', x.shape)

        return x
