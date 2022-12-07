# Temporal Attention augmented Bilinear Network for Financial Time-Series Data Analysis
# Source: https://ieeexplore.ieee.org/abstract/document/8476227

import pytorch_lightning as pl
from torch import nn
import torch
from src.models.tabl.bl_layer import BL_layer
from src.models.tabl.tabl_layer import TABL_layer

class BTABL(pl.LightningModule):
    def __init__(self, d2, d1, t1, t2, d3, t3):
        super().__init__()

        self.BL = BL_layer(d2, d1, t1, t2)
        self.TABL = TABL_layer(d3, d2, t2, t3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        x = self.BL(x)
        x = self.dropout(x)
        x = self.TABL(x)
        x = torch.squeeze(x)

        return x